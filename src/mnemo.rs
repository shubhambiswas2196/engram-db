use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use memmap2::Mmap;
use crc32fast::Hasher;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnemoRecord {
    pub id: u64,
    pub content: String,
    pub vector: Vec<f32>,
    pub timestamp: u64,
    pub ttl: Option<u64>,
    pub metadata: Option<HashMap<String, String>>,
}

const MAGIC_BYTES: &[u8; 4] = b"MNMO";
const SYNC_MARKER: &[u8; 4] = b"\xFA\xFA\xFA\xFA";
const HEADER_SIZE: u64 = 64;
const CURRENT_VERSION: u16 = 3; // Version 3: Native Vectors & TTL

// Record flags
const FLAG_HAS_TTL: u8 = 0b00000001;
const FLAG_HAS_METADATA: u8 = 0b00000010;

pub struct MnemoEngine {
    path: PathBuf,
    writer: File,
    pub index: HashMap<u64, u64>, // ID -> Record Start Offset
    last_id: u64,
    mmap: Option<Mmap>,
    vector_cache: HashMap<u64, Vec<f32>>,
}

impl MnemoEngine {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let path = base_path.as_ref().to_path_buf().join("store.mnemo");
        
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let mut index = HashMap::new();
        let mut last_id = 0;
        let mut is_valid = false;

        let file_len = file.metadata()?.len();
        if file_len >= HEADER_SIZE {
            let mut magic = [0u8; 4];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut magic)?;
            if &magic == MAGIC_BYTES {
                is_valid = true;
            }
        }

        if !is_valid {
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(MAGIC_BYTES)?;
            file.write_all(&CURRENT_VERSION.to_le_bytes())?;
            file.write_all(&[0u8; 58])?; 
            file.flush()?;
        } else {
            file.seek(SeekFrom::Start(4))?;
            let mut version_bytes = [0u8; 2];
            file.read_exact(&mut version_bytes)?;
            let version = u16::from_le_bytes(version_bytes);
            
            let (recovered_index, recovered_last_id) = Self::scan_records(&mut file, version)?;
            index = recovered_index;
            last_id = recovered_last_id;
        }

        let mmap = if file.metadata()?.len() > HEADER_SIZE {
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };

        Ok(Self {
            path,
            writer: file,
            index,
            last_id,
            mmap,
            vector_cache: HashMap::new(),
        })
    }

    pub fn append_with_vector(&mut self, content: &str, vector: Vec<f32>, metadata: Option<HashMap<String, String>>, ttl: Option<u64>) -> Result<u64> {
        self.mmap = None;
        
        let id = self.last_id + 1;
        let content_bytes = content.as_bytes();
        let content_len = content_bytes.len() as u32;
        let vector_len = vector.len() as u32;
        
        let mut flags: u8 = 0;
        if ttl.is_some() { flags |= FLAG_HAS_TTL; }
        if metadata.is_some() { flags |= FLAG_HAS_METADATA; }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // 1. Sync Marker
        let record_start_offset = self.writer.seek(SeekFrom::End(0))?;
        self.writer.write_all(SYNC_MARKER)?;
        
        // 2. ID
        self.writer.write_all(&id.to_le_bytes())?;
        
        // 3. Metadata & TTL
        self.writer.write_all(&[flags])?;
        self.writer.write_all(&timestamp.to_le_bytes())?;
        
        if let Some(t) = ttl {
            self.writer.write_all(&t.to_le_bytes())?;
        }
        
        if let Some(ref m) = metadata {
            let meta_bytes = serde_json::to_vec(m)?;
            self.writer.write_all(&(meta_bytes.len() as u32).to_le_bytes())?;
            self.writer.write_all(&meta_bytes)?;
        }

        // 4. Content
        self.writer.write_all(&content_len.to_le_bytes())?;
        self.writer.write_all(content_bytes)?;
        
        // 5. Vector
        self.writer.write_all(&vector_len.to_le_bytes())?;
        for &val in &vector {
            self.writer.write_all(&val.to_le_bytes())?;
        }
        
        // 6. Checksum (Simple implementation for now)
        let mut hasher = Hasher::new();
        hasher.update(content_bytes);
        let checksum = hasher.finalize();
        self.writer.write_all(&checksum.to_le_bytes())?;
        
        self.writer.flush()?;

        self.index.insert(id, record_start_offset);
        self.last_id = id;
        self.vector_cache.insert(id, vector);

        Ok(id)
    }

    pub fn read_record(&mut self, id: u64) -> Result<Option<MnemoRecord>> {
        let file_len = self.writer.metadata()?.len();
        if let Some(ref map) = self.mmap {
            if map.len() < file_len as usize {
                let file = File::open(&self.path)?;
                self.mmap = Some(unsafe { Mmap::map(&file)? });
            }
        } else {
            let file = File::open(&self.path)?;
            self.mmap = Some(unsafe { Mmap::map(&file)? });
        }

        let offset = match self.index.get(&id) {
            Some(o) => *o as usize,
            None => return Ok(None),
        };

        if let Some(ref map) = self.mmap {
            let mut pos = offset;
            
            // Sync
            if &map[pos..pos+4] != SYNC_MARKER { return Ok(None); }
            pos += 4;
            
            // ID
            let rid = u64::from_le_bytes(map[pos..pos+8].try_into()?);
            pos += 8;
            if rid != id { return Ok(None); }
            
            // Flags
            let flags = map[pos];
            pos += 1;
            
            // Timestamp
            let timestamp = u64::from_le_bytes(map[pos..pos+8].try_into()?);
            pos += 8;
            
            // TTL
            let ttl = if flags & FLAG_HAS_TTL != 0 {
                let t = u64::from_le_bytes(map[pos..pos+8].try_into()?);
                pos += 8;
                Some(t)
            } else { None };
            
            // Metadata
            let metadata = if flags & FLAG_HAS_METADATA != 0 {
                let mlen = u32::from_le_bytes(map[pos..pos+4].try_into()?) as usize;
                pos += 4;
                let mvec = &map[pos..pos+mlen];
                pos += mlen;
                Some(serde_json::from_slice(mvec)?)
            } else { None };
            
            // Content
            let clen = u32::from_le_bytes(map[pos..pos+4].try_into()?) as usize;
            pos += 4;
            let content = std::str::from_utf8(&map[pos..pos+clen])?.to_string();
            pos += clen;
            
            // Vector
            let vlen = u32::from_le_bytes(map[pos..pos+4].try_into()?) as usize;
            pos += 4;
            let mut vector = Vec::with_capacity(vlen);
            for _ in 0..vlen {
                vector.push(f32::from_le_bytes(map[pos..pos+4].try_into()?));
                pos += 4;
            }
            
            Ok(Some(MnemoRecord { id, content, vector, timestamp, ttl, metadata }))
        } else {
            Ok(None)
        }
    }

    fn scan_records(file: &mut File, version: u16) -> Result<(HashMap<u64, u64>, u64)> {
        let mut index = HashMap::new();
        let mut last_id = 0;
        let file_len = file.metadata()?.len();
        
        file.seek(SeekFrom::Start(HEADER_SIZE))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        let mut pos = 0;
        while pos + 17 <= buffer.len() { // SYNC(4) + ID(8) + FLAGS(1) + TS(8) = 21, let's say 17 for safety loop
            if &buffer[pos..pos+4] == SYNC_MARKER {
                let record_start = HEADER_SIZE + pos as u64;
                let id = u64::from_le_bytes(buffer[pos+4..pos+12].try_into()?);
                
                index.insert(id, record_start);
                if id > last_id { last_id = id; }
                
                // Advance past fixed parts to find lengths and jump
                let mut inner_pos = pos + 12; // After ID
                let flags = buffer[inner_pos];
                inner_pos += 9; // Skip Flags(1) + TS(8)
                
                if flags & FLAG_HAS_TTL != 0 { inner_pos += 8; }
                if flags & FLAG_HAS_METADATA != 0 {
                    let mlen = u32::from_le_bytes(buffer[inner_pos..inner_pos+4].try_into()?) as usize;
                    inner_pos += 4 + mlen;
                }
                
                // Content
                let clen = u32::from_le_bytes(buffer[inner_pos..inner_pos+4].try_into()?) as usize;
                inner_pos += 4 + clen;
                
                // Vector
                let vlen = u32::from_le_bytes(buffer[inner_pos..inner_pos+4].try_into()?) as usize;
                inner_pos += 4 + (vlen * 4);
                
                // Checksum
                inner_pos += 4;
                
                pos = inner_pos;
            } else {
                pos += 1;
            }
        }
        
        Ok((index, last_id))
    }
}

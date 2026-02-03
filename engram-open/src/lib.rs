use anyhow::{Context, Result};
use fastembed::{InitOptions, TextEmbedding, EmbeddingModel};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use hnsw_rs::prelude::*;

// Integration of Mnemo Engine
mod mnemo;
use mnemo::{MnemoEngine, MnemoRecord};

// Core Struct (Pure Rust)
pub struct EngramDBInternal {
    model: TextEmbedding,
    store: MnemoEngine,
    path: PathBuf,
    hnsw: Hnsw<'static, f32, DistCosine>,
}

impl EngramDBInternal {
    pub fn new(path: String) -> Result<Self> {
        let path_buf = PathBuf::from(path);
        if !path_buf.exists() {
            fs::create_dir_all(&path_buf)?;
        }

        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))
            .context("Failed to initialize embedding model")?;

        // Initialize Mnemo Engine
        let mut store = MnemoEngine::new(&path_buf)?;

        // Initialize HNSW
        println!("🧠 Engram: Initializing HNSW index...");
        let mut hnsw = Hnsw::new(32, 1000000, 16, 200, DistCosine);

        // Rebuild HNSW index from Mnemo storage on startup
        let ids: Vec<u64> = store.index.keys().cloned().collect();
        for id in ids {
            if let Some(record) = store.read_record(id)? {
                hnsw.insert((&record.vector, id as usize));
            }
        }

        Ok(Self {
            model,
            store,
            path: path_buf,
            hnsw,
        })
    }

    pub fn store(&mut self, text: String, metadata: Option<HashMap<String, String>>) -> Result<()> {
        let documents = vec![text.as_str()];
        let embeddings = self.model.embed(documents, None)?;
        let embedding = embeddings[0].clone();

        // 1. Persist to Binary Log
        let id = self.store.append_with_vector(&text, embedding.clone(), metadata, None)?;

        // 2. Add to HNSW Index
        self.hnsw.insert((&embedding, id as usize));

        Ok(())
    }

    pub fn recall(&mut self, query: String, limit: i32) -> Result<Vec<(String, Option<HashMap<String, String>>)>> {
        let binding = self.model.embed(vec![query], None)?;
        let query_embedding = &binding[0];

        // HNSW Search: limit is the number of neighbors, 100 is the search depth (ef)
        let results = self.hnsw.search(query_embedding, limit as usize, 100);
        
        let mut memories = Vec::new();
        for res in results {
            let id = res.d_id as u64;
            if let Some(record) = self.store.read_record(id)? {
                memories.push((record.content, record.metadata));
            }
        }

        Ok(memories)
    }
}

// --- Python Bindings ---
#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "EngramDB")]
    struct PyEngramDB {
        inner: Arc<Mutex<EngramDBInternal>>,
    }

    #[pymethods]
    impl PyEngramDB {
        #[new]
        fn new(path: String) -> PyResult<Self> {
            let db = EngramDBInternal::new(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            Ok(PyEngramDB {
                inner: Arc::new(Mutex::new(db)),
            })
        }

        fn store(&self, text: String, metadata: Option<HashMap<String, String>>) -> PyResult<()> {
            let mut db = self.inner.lock().unwrap();
            db.store(text, metadata).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }

        fn recall(&self, query: String, limit: usize) -> PyResult<Vec<(String, Option<HashMap<String, String>>)>> {
            let mut db = self.inner.lock().unwrap();
            db.recall(query, limit as i32).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }

        fn search_raw(&self, query_vector: Vec<f32>, limit: usize) -> PyResult<Vec<(String, Option<HashMap<String, String>>)>> {
             let mut db = self.inner.lock().unwrap();
             
             // Directly search HNSW
             let results = db.hnsw.search(&query_vector, limit, 100);
             
             let mut memories = Vec::new();
             for res in results {
                 let id = res.d_id as u64;
                 if let Some(record) = db.store.read_record(id).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))? {
                      memories.push((record.content, record.metadata));
                 }
             }
             Ok(memories)
        }

        fn embed_only(&self, text: String) -> PyResult<Vec<f32>> {
             let db = self.inner.lock().unwrap();
             let documents = vec![text.as_str()];
             let embeddings = db.model.embed(documents, None).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
             Ok(embeddings[0].clone())
        }

        fn count(&self) -> PyResult<usize> {
            let db = self.inner.lock().unwrap();
            Ok(db.store.index.len())
        }
    }

    #[pymodule]
    fn engram(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyEngramDB>()?;
        Ok(())
    }
}

// --- Node.js / TypeScript Bindings ---
#[cfg(feature = "node")]
use napi_derive::napi;

#[cfg(feature = "node")]
#[napi]
pub struct EngramDB {
    inner: Arc<Mutex<EngramDBInternal>>,
}

#[cfg(feature = "node")]
#[napi]
impl EngramDB {
    #[napi(constructor)]
    pub fn new(path: String) -> napi::Result<Self> {
        let db = EngramDBInternal::new(path).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(EngramDB {
            inner: Arc::new(Mutex::new(db)),
        })
    }

    #[napi]
    pub fn store(&self, text: String, metadata: Option<HashMap<String, String>>) -> napi::Result<()> {
        let mut db = self.inner.lock().unwrap();
        db.store(text, metadata).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn recall(&self, query: String, limit: i32) -> napi::Result<Vec<serde_json::Value>> {
        let mut db = self.inner.lock().unwrap();
        let results = db.recall(query, limit).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        
        let mut js_results = Vec::new();
        for (content, metadata) in results {
            let mut obj = serde_json::Map::new();
            obj.insert("content".to_string(), serde_json::Value::String(content));
            obj.insert("metadata".to_string(), serde_json::to_value(metadata).unwrap_or(serde_json::Value::Null));
            js_results.push(serde_json::Value::Object(obj));
        }
        Ok(js_results)
    }

    #[napi]
    pub fn count(&self) -> napi::Result<u32> {
        let db = self.inner.lock().unwrap();
        Ok(db.store.index.len() as u32)
    }
}

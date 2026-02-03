# 🧠 Engram: Fast, Local-First Memory for LLMs

Engram is a high-performance, local-first memory database designed specifically for AI agents and RAG (Retrieval-Augmented Generation) applications. It is built in Rust for speed and provides native Python bindings for ease of use.

**Zero Config. Zero API Keys. Zero Latency.**

## ✨ Features

- **🚀 Blazing Fast**: Core engine written in Rust with HNSW indexing for sub-10ms retrieval.
- **🔒 Privacy First**: Everything stays on your machine. No data is sent to external embedding providers.
- **📦 All-in-One**: Integrated vector storage, metadata management, and local embeddings (via FastEmbed).
- **🔋 Battery Included**: Auto-downloads and manages optimized ONNX embedding models locally.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/engram
cd engram

### Python
```bash
pip install maturin
maturin develop --features python
```

### Node.js
```bash
npx napi build --release --features node
```

## 📖 Quick Start (RAG in 30 Seconds)

```python
import engram

# 1. Initialize the database (Saves to a local folder)
db = engram.EngramDB("./my_knowledge_base")

# 2. Store documents with metadata
db.store(
    "Engram is a memory database written in Rust.", 
    {"source": "docs", "priority": "high"}
)

# 3. Recall based on semantic meaning
results = db.recall("How is Engram built?", limit=1)

for content, metadata in results:
    print(f"Retrieved: {content}")
    print(f"Metadata: {metadata}")
```

## 🏗️ Architecture

Engram uses a custom binary storage engine called **Mnemo** combined with **HNSW** (Hierarchical Navigable Small World) for ultra-fast vector search.

1. **Mnemo Engine**: A low-level, append-only binary log that ensures your data is persisted safely to disk.
2. **HNSW Index**: An in-memory graph structure rebuilt from disk on startup for lightning-fast nearest neighbor search.
3. **Local Embeddings**: Uses `fastembed-rs` to run optimized ONNX models like `all-MiniLM-L6-v2` locally on your CPU/GPU.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

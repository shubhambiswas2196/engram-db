import os
import engram
from typing import List

# Simple RAG Helper
class MinimalRAG:
    def __init__(self, db_path: str = "./engram_memory"):
        print(f"🧠 Initializing Engram at {db_path}...")
        self.db = engram.EngramDB(db_path)

    def add_document(self, text: str, source_name: str):
        # In a real app, you'd chunk this text. 
        # For this example, we store it directly.
        chunks = [text[i:i+500] for i in range(0, len(text), 400)]
        for i, chunk in enumerate(chunks):
            self.db.store(chunk, {"source": source_name, "chunk_id": str(i)})
        print(f"✅ Added '{source_name}' to memory.")

    def query(self, user_query: str):
        print(f"🔍 Searching memory for: '{user_query}'")
        results = self.db.recall(user_query, limit=3)
        
        if not results:
            print("❌ No matching memories found.")
            return

        print("\n--- Top Relevant Results ---")
        for content, meta in results:
            print(f"[{meta.get('source')}] {content[:100]}...")
        print("----------------------------\n")

if __name__ == "__main__":
    # Create the RAG instance
    rag = MinimalRAG()

    # 1. Ingest some sample data
    rag.add_document(
        "The HNSW (Hierarchical Navigable Small World) algorithm is a state-of-the-art "
        "method for approximate nearest neighbor search in high-dimensional spaces.",
        "Algorithm Docs"
    )
    
    rag.add_document(
        "Engram uses Rust to provide a high-performance memory engine that works "
        "entirely without external API keys or cloud dependencies.",
        "Engram Features"
    )

    # 2. Perform a query
    rag.query("How does Engram work?")
    rag.query("Tell me about HNSW.")

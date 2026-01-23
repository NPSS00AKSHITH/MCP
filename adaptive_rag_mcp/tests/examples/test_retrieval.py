"""Test script for dense retrieval.

Run from project root:
    python examples/test_retrieval.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage import ChunkStore
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.pipeline import RetrievalPipeline


def main():
    print("=" * 60)
    print("Dense Retrieval Test")
    print("=" * 60)
    
    # Initialize components with test paths
    chunk_store = ChunkStore(db_path="./data/test_chunks.db")
    ingestion = IngestionPipeline(chunk_store=chunk_store)
    
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    vector_store = FAISSVectorStore(
        dimensions=embedder.dimensions,
        index_path="./data/test_faiss",
    )
    retrieval = RetrievalPipeline(embedder=embedder, vector_store=vector_store)
    
    # Test 1: Ingest and index a document
    print("\n📄 Test 1: Ingesting and indexing document")
    
    result = ingestion.ingest_text(
        text="""
        Retrieval-Augmented Generation (RAG) is an AI framework that combines 
        information retrieval with text generation. It enhances large language 
        models by providing them with relevant context from external knowledge 
        sources.
        
        RAG works by first retrieving relevant documents based on a query, 
        then using those documents as context for generating a response. 
        This approach helps reduce hallucinations and provides more accurate, 
        up-to-date information.
        
        Key components of RAG include:
        1. Document chunking and storage
        2. Vector embeddings for semantic search
        3. Retrieval mechanisms (dense, sparse, or hybrid)
        4. Language model for response generation
        
        Popular embedding models include sentence-transformers like 
        all-MiniLM-L6-v2 for fast inference, or larger models like 
        all-mpnet-base-v2 for better quality.
        """,
        doc_id="rag_overview",
        file_name="rag_overview.txt",
    )
    print(f"   Ingested: {result.total_chunks} chunks")
    
    # Index the document
    chunks = ingestion.get_chunks("rag_overview")
    indexed = retrieval.index_chunks(chunks)
    print(f"   Indexed: {indexed} chunks")
    
    # Test 2: Search
    print("\n🔍 Test 2: Searching for 'What is RAG?'")
    results = retrieval.search("What is RAG?", k=3)
    print(f"   Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        preview = r.content[:100].replace('\n', ' ')
        print(f"   {i}. Score: {r.score:.4f} - {preview}...")
    
    # Test 3: Search with different query
    print("\n🔍 Test 3: Searching for 'embedding models'")
    results = retrieval.search("embedding models", k=2)
    print(f"   Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        preview = r.content[:100].replace('\n', ' ')
        print(f"   {i}. Score: {r.score:.4f} - {preview}...")
    
    # Test 4: Get stats
    print("\n📊 Test 4: Retrieval statistics")
    stats = retrieval.get_stats()
    print(f"   Model: {stats['embedder_model']}")
    print(f"   Dimensions: {stats['embedder_dimensions']}")
    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Index path: {stats['index_path']}")
    
    # Test 5: Ingest another document and search across both
    print("\n📄 Test 5: Adding second document")
    result2 = ingestion.ingest_text(
        text="""
        FAISS (Facebook AI Similarity Search) is a library for efficient 
        similarity search of dense vectors. It supports various index types
        including flat indexes, IVF indexes, and HNSW for different 
        performance/accuracy tradeoffs.
        
        FAISS is particularly useful for RAG systems because it can handle
        millions of vectors with fast query times.
        """,
        doc_id="faiss_intro",
        file_name="faiss_intro.txt",
    )
    chunks2 = ingestion.get_chunks("faiss_intro")
    indexed2 = retrieval.index_chunks(chunks2)
    print(f"   Indexed additional {indexed2} chunks")
    
    results = retrieval.search("vector similarity search library", k=3)
    print(f"   Search across both docs found {len(results)} results:")
    for i, r in enumerate(results, 1):
        preview = r.content[:80].replace('\n', ' ')
        print(f"   {i}. [doc: {r.doc_id}] Score: {r.score:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Dense retrieval tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

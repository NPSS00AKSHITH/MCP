"""Comparison example: Dense vs Sparse vs Hybrid search.

Run from project root:
    python examples/test_hybrid.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage import ChunkStore
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.sparse_retriever import BM25Retriever
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.hybrid import HybridRetriever, SearchMode


def main():
    print("=" * 70)
    print("Dense vs Sparse vs Hybrid Search Comparison")
    print("=" * 70)
    
    # Initialize components with test paths
    chunk_store = ChunkStore(db_path="./data/hybrid_test_chunks.db")
    ingestion = IngestionPipeline(chunk_store=chunk_store)
    
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    vector_store = FAISSVectorStore(
        dimensions=embedder.dimensions,
        index_path="./data/hybrid_test_faiss",
    )
    sparse_retriever = BM25Retriever(index_path="./data/hybrid_test_bm25")
    dense_pipeline = RetrievalPipeline(embedder=embedder, vector_store=vector_store)
    
    hybrid = HybridRetriever(
        dense_retriever=dense_pipeline,
        sparse_retriever=sparse_retriever,
    )
    
    # Create test documents with different characteristics
    print("\n📚 Creating test documents...")
    
    docs = [
        {
            "doc_id": "ml_basics",
            "text": """Machine learning is a subset of artificial intelligence that 
            enables systems to learn from data. Deep learning uses neural networks 
            with multiple layers to model complex patterns.""",
        },
        {
            "doc_id": "nlp_guide",
            "text": """Natural Language Processing (NLP) is a field of AI focused on 
            understanding human language. Transformers like BERT and GPT revolutionized 
            NLP by using attention mechanisms.""",
        },
        {
            "doc_id": "rag_overview",
            "text": """Retrieval-Augmented Generation (RAG) combines information retrieval 
            with language models. It retrieves relevant documents and uses them as context 
            for generating grounded responses.""",
        },
        {
            "doc_id": "python_basics",
            "text": """Python is a high-level programming language known for its 
            readability. It supports multiple paradigms including object-oriented, 
            procedural, and functional programming.""",
        },
        {
            "doc_id": "vector_search",
            "text": """Vector databases enable semantic search by storing embeddings. 
            FAISS and Qdrant are popular choices. Cosine similarity measures vector 
            closeness for retrieval.""",
        },
    ]
    
    for doc in docs:
        result = ingestion.ingest_text(
            text=doc["text"],
            doc_id=doc["doc_id"],
            file_name=f"{doc['doc_id']}.txt",
        )
        chunks = ingestion.get_chunks(doc["doc_id"])
        hybrid.index_chunks(chunks, save=False)
    
    # Save indexes
    hybrid.dense.vector_store.save()
    hybrid.sparse.save()
    
    print(f"   ✓ Indexed {len(docs)} documents")
    
    # Test queries that demonstrate differences between modes
    test_queries = [
        {
            "query": "What is RAG?",
            "description": "Semantic query - dense should excel",
        },
        {
            "query": "BERT GPT transformers",
            "description": "Keyword query - sparse should excel",
        },
        {
            "query": "How do neural networks learn from data?",
            "description": "Mixed query - hybrid should excel",
        },
    ]
    
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        print(f"\n{'='*70}")
        print(f"Query {i}: \"{query}\"")
        print(f"Type: {test['description']}")
        print("=" * 70)
        
        for mode in [SearchMode.DENSE, SearchMode.SPARSE, SearchMode.HYBRID]:
            results = hybrid.search(query, k=3, mode=mode)
            
            print(f"\n📍 {mode.value.upper()} Results:")
            if not results:
                print("   (no results)")
                continue
            
            for rank, r in enumerate(results[:3], 1):
                score_info = f"score={r.score:.4f}"
                if r.dense_score is not None and r.sparse_score is not None:
                    score_info += f" (dense={r.dense_score:.3f}, sparse={r.sparse_score:.3f})"
                elif r.dense_score is not None:
                    score_info += f" (dense only)"
                elif r.sparse_score is not None:
                    score_info += f" (sparse only)"
                
                print(f"   {rank}. [{r.doc_id}] {score_info}")
    
    # Show stats
    print(f"\n{'='*70}")
    print("📊 Index Statistics")
    print("=" * 70)
    stats = hybrid.get_stats()
    print(f"   Dense vectors: {stats['dense']['total_vectors']}")
    print(f"   Sparse documents: {stats['sparse']['total_documents']}")
    print(f"   RRF constant k: {stats['rrf_k']}")
    
    print("\n" + "=" * 70)
    print("✅ Hybrid search comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

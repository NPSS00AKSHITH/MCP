"""Test script for document ingestion.

Run from project root:
    python examples/test_ingestion.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage import ChunkStore


def main():
    print("=" * 60)
    print("Document Ingestion Test")
    print("=" * 60)
    
    # Use a test database
    store = ChunkStore(db_path="./data/test_chunks.db")
    pipeline = IngestionPipeline(chunk_store=store)
    
    # Test 1: Ingest sample markdown file
    print("\n📄 Test 1: Ingesting sample_document.md")
    sample_path = Path(__file__).parent / "sample_document.md"
    
    if sample_path.exists():
        result = pipeline.ingest_file(sample_path)
        print(f"   Success: {result.success}")
        print(f"   Doc ID: {result.doc_id}")
        print(f"   File: {result.file_name}")
        print(f"   Type: {result.file_type}")
        print(f"   Chunks: {result.total_chunks}")
        print(f"   Characters: {result.total_characters}")
        
        if result.error:
            print(f"   Error: {result.error}")
    else:
        print(f"   ❌ File not found: {sample_path}")
    
    # Test 2: Ingest raw text
    print("\n📝 Test 2: Ingesting raw text")
    
    raw_text = """
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience. It focuses on developing 
    computer programs that can access data and use it to learn for themselves.
    
    The process begins with observations or data, such as examples, direct 
    experience, or instruction. The goal is to allow computers to learn 
    automatically without human intervention and adjust actions accordingly.
    
    Key applications include image recognition, speech recognition, 
    natural language processing, and recommendation systems.
    """
    
    result = pipeline.ingest_text(
        text=raw_text,
        doc_id="ml_intro_001",
        file_name="machine_learning_intro.txt",
        metadata={"topic": "machine learning", "level": "beginner"}
    )
    
    print(f"   Success: {result.success}")
    print(f"   Doc ID: {result.doc_id}")
    print(f"   Chunks: {result.total_chunks}")
    
    # Test 3: List documents
    print("\n📋 Test 3: Listing all documents")
    docs = pipeline.list_documents()
    print(f"   Total documents: {len(docs)}")
    for doc in docs:
        print(f"   - {doc['file_name']} ({doc['total_chunks']} chunks)")
    
    # Test 4: Get chunks for a document
    print("\n🔍 Test 4: Getting chunks for first document")
    if docs:
        doc_id = docs[0]['doc_id']
        chunks = pipeline.get_chunks(doc_id)
        print(f"   Document: {docs[0]['file_name']}")
        print(f"   Total chunks: {len(chunks)}")
        if chunks:
            print(f"\n   First chunk preview:")
            print(f"   Chunk ID: {chunks[0].chunk_id}")
            print(f"   Index: {chunks[0].chunk_index}")
            content_preview = chunks[0].content[:200].replace('\n', ' ')
            print(f"   Content: {content_preview}...")
    
    # Test 5: Get stats
    print("\n📊 Test 5: Storage statistics")
    stats = pipeline.get_stats()
    print(f"   Documents: {stats['document_count']}")
    print(f"   Chunks: {stats['chunk_count']}")
    print(f"   Total characters: {stats['total_characters']}")
    print(f"   Database: {stats['database_path']}")
    
    print("\n" + "=" * 60)
    print("✅ Ingestion tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

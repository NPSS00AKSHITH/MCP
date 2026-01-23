import os
from src.ingestion.pipeline import get_pipeline
from src.retrieval.pipeline import get_retrieval_pipeline
from src.server.tools.retrieval_tools import index_document
# Force loading .env before other imports might need settings (though pydantic does it lazily usually)
from dotenv import load_dotenv
load_dotenv() 

def test_workflow():
    # Absolute path to the PDF based on user info
    pdf_path = r"d:\MCP\memorize-final.pdf" 
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return

    print("Step 1: Ingesting Document...")
    ingestion = get_pipeline()
    result = ingestion.ingest_file(pdf_path)
    
    if not result.success:
        print(f"Ingestion failed: {result.error}")
        return
        
    print(f"Ingestion successful! Doc ID: {result.doc_id}")
    print(f"File Name: {result.file_name}")
    print(f"Total Chunks: {result.total_chunks}")
    
    print("\nStep 2: Indexing Document...")
    # Index the ingested chunks into the vector store
    idx_res = index_document({"doc_id": result.doc_id})
    if not idx_res["success"]:
        print(f"Indexing failed: {idx_res.get('error')}")
        return
        
    print(f"Indexing successful! Chunks indexed: {idx_res['chunks_indexed']}")
    
    print("\nStep 3: Searching...")
    retrieval = get_retrieval_pipeline()
    
    # Generic query
    query = "strategies for memorization"
    print(f"Query: '{query}'")
    
    search_results = retrieval.search(query, k=3)
    
    print(f"\nFound {len(search_results)} results:")
    for i, res in enumerate(search_results):
        print(f"\nResult {i+1}:")
        print(f"Score: {res.score:.4f}")
        print(f"Content: {res.content[:200]}...")
        
if __name__ == "__main__":
    test_workflow()

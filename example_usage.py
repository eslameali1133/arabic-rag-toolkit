"""
Example usage of rag_utils.py in a new project.
Copy rag_utils.py next to this file (or install it as a local package).
"""
 
from rag_utils import ingest_documents, query_documents
 

DOCS_DIR = "/Users/eslamali/Desktop/projects/Python/Gen AI Projects/GenAI_RAG/Docs_dir"
VECTOR_DB_PATH =  "/Users/eslamali/Desktop/projects/Python/Gen AI Projects/GenAI_RAG/vector_db"
COLLECTION_NAME = "document_collection"
 
# --- Step 1: Ingest (run once, or whenever documents change) ---
"""ingest_documents(
    docs_dir_path=DOCS_DIR,
    vector_db_path=VECTOR_DB_PATH,
    collection_name=COLLECTION_NAME,
    # Everything else uses sensible defaults (Arabic+English OCR,
    # multilingual-e5-large embeddings, 1000/150 chunking).
)"""
 
# --- Step 2: Query (call this as many times as you want) ---
response = query_documents(
    query="ما هي محاسن استخدام الحاسب الآلي؟",
    vector_db_path=VECTOR_DB_PATH,
    collection_name=COLLECTION_NAME,
)
 
print(response["answer"])
print("-" * 80)
for src in response["sources"]:
    print(src)


    
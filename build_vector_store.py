
from app.knowledge_base import (
    get_vector_store,
    index_faq_documents,
    CHROMA_DB_DIR,
)


indexed_count = index_faq_documents()

vector_store = get_vector_store()

stored_data = vector_store.get()

print("FAQ documents indexed:", indexed_count)
print("Records stored in ChromaDB:", len(stored_data["ids"]))
print("Database location:", CHROMA_DB_DIR)

print("\nSample stored record:")
print("ID:", stored_data["ids"][0])
print("Metadata:", stored_data["metadatas"][0])
print("Content:", stored_data["documents"][0])

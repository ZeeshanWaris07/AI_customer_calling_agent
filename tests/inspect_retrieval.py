
from app.knowledge_base import get_vector_store


query = "My order arrived broken. What should I do?"

vector_store = get_vector_store()

results = vector_store.similarity_search_with_score(
    query,
    k=15,
)

print("Customer:", query)

for rank, (document, distance) in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"FAQ ID: {document.metadata['faq_id']}")
    print(f"Category: {document.metadata['category']}")
    print(f"Distance: {distance:.4f}")

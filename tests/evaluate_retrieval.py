
from app.knowledge_base import get_vector_store


test_cases = [
    {
        "query": "How many days do I have to send a product back?",
        "expected": "FAQ002",
    },
    {
        "query": "What payment options can I use?",
        "expected": "FAQ004",
    },
    {
        "query": "How long does shipping normally take?",
        "expected": "FAQ001",
    },
    {
        "query": "My order arrived broken. What should I do?",
        "expected": "FAQ006",
    },
    {
        "query": "I received the wrong item. How can I report it?",
        "expected": "FAQ013",
    },
    {
        "query": "Can I exchange something I purchased?",
        "expected": "FAQ012",
    },
    {
        "query": "How can I change my delivery address?",
        "expected": "FAQ010",
    },
    {
        "query": "Do you offer free laptop installation?",
        "expected": None,
    },
]


vector_store = get_vector_store()

successful_retrievals = 0
supported_queries = 0

for test in test_cases:
    query = test["query"]
    expected = test["expected"]

    print("\n" + "=" * 60)
    print("Customer:", query)

    results = vector_store.similarity_search_with_score(
        query,
        k=3,
    )

    retrieved_ids = []

    for rank, (document, distance) in enumerate(results, start=1):
        faq_id = document.metadata["faq_id"]
        retrieved_ids.append(faq_id)

        print(f"\nRank: {rank}")
        print(f"FAQ ID: {faq_id}")
        print(f"Category: {document.metadata['category']}")
        print(f"Distance: {distance:.4f}")

    if expected is None:
        print("\nUnsupported query: inspect nearest results.")
        continue

    supported_queries += 1

    if expected in retrieved_ids:
        successful_retrievals += 1
        rank = retrieved_ids.index(expected) + 1

        print(f"\nPASS: Expected FAQ found at rank {rank}")

    else:
        print(f"\nFAIL: Expected {expected} was not retrieved.")

accuracy = successful_retrievals / supported_queries * 100

print("\n" + "=" * 60)
print("RETRIEVAL EVALUATION")
print(f"Successful retrievals: {successful_retrievals}/{supported_queries}")
print(f"Recall@3: {accuracy:.2f}%")

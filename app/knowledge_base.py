
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.database import load_faqs


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

FAQ_COLLECTION_NAME = "shopease_faqs"


def load_faq_documents() -> list[Document]:
    faqs = load_faqs()

    if not faqs:
        raise ValueError("FAQ knowledge base is empty.")

    documents = []

    for faq in faqs:
        question = faq["question"].strip()
        answer = faq["answer"].strip()

        if not question or not answer:
            raise ValueError(
                f"FAQ {faq['id']} has an empty question or answer."
            )

        content = (
            f"Question: {question}\n\n"
            f"Answer: {answer}"
        )

        document = Document(
            page_content=content,
            metadata={
                "faq_id": faq["id"],
                "category": faq["category"],
            },
        )

        documents.append(document)

    return documents


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:

    model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )


    return model


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=FAQ_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DB_DIR),
    )


def index_faq_documents() -> int:
    documents = load_faq_documents()

    document_ids = [
        document.metadata["faq_id"]
        for document in documents
    ]

    vector_store = get_vector_store()

    vector_store.add_documents(
        documents=documents,
        ids=document_ids,
    )

    return len(documents)

def get_faq_retriever():
    return get_vector_store().as_retriever(
        search_type = 'similarity',
        search_kwargs = {
            'k' : 3
        },
    )

def retrieve_faqs(query:str) -> list[Document]:
    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty")

    retriever = get_faq_retriever()

    print("retriever returned")

    return retriever.invoke(query)
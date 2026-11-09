
import traceback

print("[1] Importing LangChain embeddings...", flush=True)

try:
    from langchain_huggingface import HuggingFaceEmbeddings

    print("[2] Import successful.", flush=True)

    print("[3] Initializing HuggingFaceEmbeddings...", flush=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    print("[4] LangChain wrapper initialized!", flush=True)

    print("[5] Generating query embedding...", flush=True)

    vector = embeddings.embed_query(
        "Can I return my purchased product?"
    )

    print("[6] Embedding generated!", flush=True)
    print("Dimensions:", len(vector))

except Exception:
    traceback.print_exc()

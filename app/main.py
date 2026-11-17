from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent import AgentRequest, AgentResponse, process_message
from app.knowledge_base import (
    index_faq_documents,
    get_faq_retriever,
    get_vector_store,
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("[STARTUP] Loading FAQ resources...", flush=True)

    # Load FAQs and generate embeddings.
    total = index_faq_documents()

    print(
        f"[STARTUP] Successfully indexed {total} FAQs.",
        flush=True,
    )

    # Initialize the retriever.
    get_faq_retriever()

    print(
        "[STARTUP] FAQ retriever is ready.",
        flush=True,
    )

    yield

    print("[SHUTDOWN] Closing application.", flush=True)


app = FastAPI(
    title="ShopEase API",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://YOUR-FRONTEND.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "ShopEase AI Customer Calling Agent is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/chat", response_model=AgentResponse)
def chat(request: AgentRequest) -> AgentResponse:
    return process_message(request)
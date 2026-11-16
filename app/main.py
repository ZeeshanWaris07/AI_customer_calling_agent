
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent import AgentRequest, AgentResponse, process_message
from contextlib import asynccontextmanager
from app.knowledge_base import get_faq_retriever

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("[STARTUP] Loading FAQ resources...", flush=True)

    get_faq_retriever()

    print("[STARTUP] FAQ resources loaded successfully.", flush=True)

    yield

    print("[SHUTDOWN] Closing application.", flush=True)

app = FastAPI(
    title="ShopEase API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
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

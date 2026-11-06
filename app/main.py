
from fastapi import FastAPI

from app.agent import AgentRequest, AgentResponse, process_message


app = FastAPI(
    title="ShopEase AI Customer Calling Agent",
    description="An AI-powered customer support assistant.",
    version="1.0.0",
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


import os
import re

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from app.knowledge_base import retrieve_faqs
from app.prompts import FAQ_ANSWER_PROMPT,INTENT_CLASSIFICATION_PROMPT
from app.database import load_faqs, get_order_by_id


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


Intent = Literal[
    "greeting",
    "faq",
    "order_status",
    "complaint",
    "human_handoff",
    "unknown",
]

Outcome = Literal[
    "resolved",
    "escalated",
    "needs_clarification",
]


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    session_id: Optional[str] = None
    intent: Intent
    response: str = Field(min_length=1)
    outcome: Outcome


class IntentClassification(BaseModel):
    intent: Intent


class FAQAnswer(BaseModel):
    supported_by_context: bool
    answer: str = Field(min_length=1)
    source_faq_ids: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_intent_classifier():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from the .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        api_key=api_key,
        temperature=0,
    )

    return llm.with_structured_output(
        IntentClassification,
        method="json_schema",
    )


def classify_intent(message: str) -> IntentClassification:
    message = message.strip()

    if not message:
        raise ValueError("Customer message cannot be empty.")

    prompt = INTENT_CLASSIFICATION_PROMPT.format_messages(
        message = message
    )

    result = get_intent_classifier().invoke(prompt)

    if isinstance(result, IntentClassification):
        return result

    return IntentClassification.model_validate(result)




def handle_order_status(message: str) -> tuple[str, Outcome]:
    match = re.search(
        r"\bORD-\d{4}\b",
        message,
        flags=re.IGNORECASE,
    )

    if not match:
        return (
            "Please provide your order ID, such as ORD-1001, "
            "so I can check the order status.",
            "needs_clarification",
        )

    order_id = match.group().upper()

    order = get_order_by_id(order_id)

    if order is None:
        return (
            f"I couldn't find order {order_id} in our records. "
            "Please check the order ID.",
            "needs_clarification",
        )

    response = (
        f"Order {order_id} is currently "
        f"{order['status'].lower()}."
    )

    estimated_delivery = order["estimated_delivery"]

    if estimated_delivery:
        response += (
            f" Its estimated delivery date is "
            f"{estimated_delivery.isoformat()}."
        )

    return response, "resolved"



def process_message(request: AgentRequest) -> AgentResponse:
    message = request.message.strip()

    classification = classify_intent(message)
    intent = classification.intent

    if intent == "greeting":
        response = (
            "Hello! I'm the ShopEase automated assistant. "
            "I can help with store policies and order-status questions."
        )
        outcome: Outcome = "resolved"

    elif intent == "faq":
        response,outcome = handle_faq_question(message)

    elif intent == "order_status":
        response, outcome = handle_order_status(message)

    elif intent in ("complaint", "human_handoff"):
        response = (
            "This request requires human customer support. "
            "This prototype does not yet notify a live support agent, "
            "so please contact the store's support team directly."
        )
        outcome = "escalated"

    else:
        response = (
            "I'm not sure how to help with that request. "
            "Are you asking about a store policy or an order, "
            "or would you like human support?"
        )
        outcome = "needs_clarification"

    return AgentResponse(
        session_id=request.session_id,
        intent=intent,
        response=response,
        outcome=outcome,
    )


@lru_cache(maxsize=1)
def get_faq_answerer():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from the .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        api_key=api_key,
        temperature=0,
    )

    return llm.with_structured_output(
        FAQAnswer,
        method="json_schema",
    )



FAQ_FALLBACK_RESPONSE = (
    "I couldn't find enough information in the approved FAQ "
    "to answer that confidently. Would you like me to connect "
    "you with customer support?"
)


def handle_faq_question(
    question: str,
) -> tuple[str, Outcome]:

    documents = retrieve_faqs(question)


    if not documents:
        return FAQ_FALLBACK_RESPONSE, "needs_clarification"

    retrieved_ids = {
        document.metadata["faq_id"]
        for document in documents
    }

    context = "\n\n---\n\n".join(
        (
            f"FAQ ID: {document.metadata['faq_id']}\n"
            f"Category: {document.metadata['category']}\n"
            f"{document.page_content}"
        )
        for document in documents
    )

    answerer = get_faq_answerer()

    result = answerer.invoke(
        FAQ_ANSWER_PROMPT.format_messages(
            question=question,
            context=context,
        )
    )

    if not isinstance(result, FAQAnswer):
        result = FAQAnswer.model_validate(result)

    used_ids = set(result.source_faq_ids)

    if (
        not result.supported_by_context
        or not result.answer.strip()
        or not used_ids
        or not used_ids.issubset(retrieved_ids)
    ):
        return FAQ_FALLBACK_RESPONSE, "needs_clarification"

    return result.answer.strip(), "resolved"

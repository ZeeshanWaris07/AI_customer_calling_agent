
import app.agent as agent


def mock_intent(monkeypatch, intent):
    monkeypatch.setattr(
        agent,
        "classify_intent",
        lambda message: agent.IntentClassification(intent=intent),
    )


def test_greeting(monkeypatch):
    mock_intent(monkeypatch, "greeting")

    result = agent.process_message(
        agent.AgentRequest(message="Hello")
    )

    assert result.intent == "greeting"
    assert result.outcome == "resolved"


def test_faq(monkeypatch):
    mock_intent(monkeypatch, "faq")

    result = agent.process_message(
        agent.AgentRequest(message="What is your return policy?")
    )

    assert result.intent == "faq"
    assert "14 days" in result.response
    assert result.outcome == "resolved"


def test_order_status(monkeypatch):
    mock_intent(monkeypatch, "order_status")

    result = agent.process_message(
        agent.AgentRequest(
            message="Check order ORD-1001"
        )
    )

    assert result.intent == "order_status"
    assert "shipped" in result.response.lower()
    assert result.outcome == "resolved"


def test_missing_order_id(monkeypatch):
    mock_intent(monkeypatch, "order_status")

    result = agent.process_message(
        agent.AgentRequest(message="Where is my order?")
    )

    assert result.outcome == "needs_clarification"
    assert "order ID" in result.response


def test_complaint_escalation(monkeypatch):
    mock_intent(monkeypatch, "complaint")

    result = agent.process_message(
        agent.AgentRequest(message="My product is damaged")
    )

    assert result.intent == "complaint"
    assert result.outcome == "escalated"


def test_human_handoff(monkeypatch):
    mock_intent(monkeypatch, "human_handoff")

    result = agent.process_message(
        agent.AgentRequest(message="I want a human")
    )

    assert result.intent == "human_handoff"
    assert result.outcome == "escalated"


def test_unknown_intent(monkeypatch):
    mock_intent(monkeypatch, "unknown")

    result = agent.process_message(
        agent.AgentRequest(message="Tell me a joke")
    )

    assert result.outcome == "needs_clarification"

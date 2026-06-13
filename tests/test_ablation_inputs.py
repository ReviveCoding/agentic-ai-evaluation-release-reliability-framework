from agentic_eval_framework.models.ablation import build_ablation_text


ROW = {
    "dialogue_context": "[SYSTEM] how can I help",
    "user_utterance": "book a meeting tomorrow",
    "service": "Calendar_1",
    "intent": "CreateEvent",
    "service_description": "manage calendar events",
    "intent_description": "create a new event",
    "required_slots": ["date"],
    "known_slots": {"date": "tomorrow"},
    "missing_slots": [],
    "risk_flags": [],
}


def test_utterance_only_excludes_structured_oracle_fields():
    text = build_ablation_text(ROW, "utterance_context_only")
    assert "Calendar_1" not in text
    assert "CreateEvent" not in text
    assert "book a meeting tomorrow" in text


def test_metadata_only_excludes_utterance():
    text = build_ablation_text(ROW, "metadata_names_only")
    assert "Calendar_1" in text
    assert "CreateEvent" in text
    assert "book a meeting tomorrow" not in text

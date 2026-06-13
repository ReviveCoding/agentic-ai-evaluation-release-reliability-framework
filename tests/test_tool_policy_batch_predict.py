from pathlib import Path

from agentic_eval_framework.models.transformer_tool_policy import ToolPolicyModel


def test_rule_backend_batch_predict_without_model(tmp_path):
    model = ToolPolicyModel(tmp_path / "missing")
    rows = [
        {"service": "Calendar_1", "intent": "GetEvents", "missing_slots": []},
        {"service": "Weather_1", "intent": "GetWeather", "missing_slots": []},
    ]
    assert model.predict(rows) == ["calendar_lookup", "weather_lookup"]

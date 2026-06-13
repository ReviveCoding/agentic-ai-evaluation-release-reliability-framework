from __future__ import annotations

from agentic_eval_framework.retrieval.hierarchical_retriever import HierarchicalBM25Retriever
from agentic_eval_framework.retrieval.recency_query import recency_weighted_query


def _docs():
    return [
        {
            "doc_id": "intent::Banks_1::TransferMoney",
            "doc_type": "intent",
            "service": "Banks_1",
            "intent": "TransferMoney",
            "title": "Banks TransferMoney",
            "text": "transfer money from checking account to another person",
        },
        {
            "doc_id": "intent::Weather_1::GetWeather",
            "doc_type": "intent",
            "service": "Weather_1",
            "intent": "GetWeather",
            "title": "Weather GetWeather",
            "text": "weather forecast temperature rain wind for a city and date",
        },
        {
            "doc_id": "service::Weather_1",
            "doc_type": "service",
            "service": "Weather_1",
            "intent": "",
            "title": "Weather",
            "text": "weather forecast service",
        },
    ]


def test_recency_query_repeats_current_turn_over_stale_context():
    payload = {
        "dialogue_context": "[USER] Transfer money [SYSTEM] Transfer completed",
        "user_request": "What is the weather in Boston tomorrow?",
        "known_slots": {"city": "Boston", "date": "tomorrow"},
    }
    query = recency_weighted_query(payload)
    assert query.count("What is the weather") == 4
    assert "Transfer completed" in query


def test_predicted_tool_filter_and_current_turn_rank_weather_first():
    retriever = HierarchicalBM25Retriever(_docs(), reranker_path="missing.joblib")
    results = retriever.search(
        "transfer money transfer completed weather Boston tomorrow",
        top_k=2,
        tool_name="weather_lookup",
        current_query="weather Boston tomorrow",
        use_reranker=False,
    )
    assert results[0]["service"] == "Weather_1"
    assert results[0]["tool_compatible"] is True

class _ToolPriorOnlyReranker:
    """A fake reranker that would select the predicted-tool candidate if active."""

    @staticmethod
    def score(vectors):
        return [float(vector[5]) for vector in vectors]


def test_conflicting_current_query_disables_predicted_tool_prior():
    retriever = HierarchicalBM25Retriever(_docs(), reranker_path="missing.joblib")
    retriever.reranker = _ToolPriorOnlyReranker()
    results = retriever.search(
        "please check the forecast",
        top_k=2,
        tool_name="search_places",
        current_query="please check the forecast",
        use_reranker=True,
        include_features=True,
    )
    assert results[0]["service"] == "Weather_1"
    assert results[0]["tool_filter_active"] is False
    assert all(result["tool_compatible"] is False for result in results)

class _HostilePredictedToolReranker:
    # Scores the stale predicted-tool feature above every query signal.

    @staticmethod
    def score(vectors):
        return [1000.0 * float(vector[5]) for vector in vectors]


def test_query_tool_override_beats_hostile_predicted_tool_reranker():
    docs = _docs() + [
        {
            "doc_id": "intent::Hotels_1::SearchHotel",
            "doc_type": "intent",
            "service": "Hotels_1",
            "intent": "SearchHotel",
            "title": "Hotels SearchHotel",
            "text": "find hotel rooms and lodging",
        }
    ]
    retriever = HierarchicalBM25Retriever(docs, reranker_path="missing.joblib")
    retriever.reranker = _HostilePredictedToolReranker()
    results = retriever.search(
        "please check the forecast",
        top_k=2,
        tool_name="search_places",
        current_query="please check the forecast",
        use_reranker=True,
        include_features=True,
    )
    assert results[0]["service"] == "Weather_1"
    assert results[0]["query_tool_override_active"] is True
    assert results[0]["query_tool_match"] is True
    assert results[0]["effective_tool"] == "weather_lookup"

from agentic_eval_framework.retrieval.hierarchical_retriever import HierarchicalBM25Retriever
from agentic_eval_framework.retrieval.targets import compatible_doc_ids


def _docs():
    return [
        {"doc_id": "service::Hotels_1", "doc_type": "service", "service": "Hotels_1", "intent": "", "title": "Hotels_1", "text": "hotel lodging stays"},
        {"doc_id": "intent::Hotels_1::SearchHotel", "doc_type": "intent", "service": "Hotels_1", "intent": "SearchHotel", "title": "SearchHotel", "text": "find a hotel with wifi"},
        {"doc_id": "service::Hotels_4", "doc_type": "service", "service": "Hotels_4", "intent": "", "title": "Hotels_4", "text": "hotel lodging reservations"},
        {"doc_id": "intent::Hotels_4::SearchHotel", "doc_type": "intent", "service": "Hotels_4", "intent": "SearchHotel", "title": "SearchHotel", "text": "search hotels by city and amenities"},
        {"doc_id": "service::Weather_1", "doc_type": "service", "service": "Weather_1", "intent": "", "title": "Weather", "text": "weather forecast rain"},
    ]


def test_hierarchical_retrieval_is_deterministic_and_semantic():
    docs = _docs()
    retriever = HierarchicalBM25Retriever(docs)
    first = retriever.search("find a hotel in Boston with wifi", top_k=4)
    second = retriever.search("find a hotel in Boston with wifi", top_k=4)
    assert [item["doc_id"] for item in first] == [item["doc_id"] for item in second]
    scenario = {"service": "Hotels_4", "intent": "SearchHotel"}
    compatible = compatible_doc_ids(scenario, docs)
    assert "intent::Hotels_1::SearchHotel" in compatible
    assert "intent::Hotels_4::SearchHotel" in compatible
    assert first[0]["doc_id"] in compatible

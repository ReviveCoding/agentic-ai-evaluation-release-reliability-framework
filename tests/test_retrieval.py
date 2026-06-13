from agentic_eval_framework.retrieval.bm25_retriever import BM25Retriever


def test_bm25_retriever():
    docs = [{"doc_id": "a", "title": "Calendar", "text": "calendar events meetings"}, {"doc_id": "b", "title": "Weather", "text": "weather forecast rain"}]
    r = BM25Retriever(docs)
    results = r.search("what meetings tomorrow", top_k=1)
    assert results[0]["doc_id"] == "a"

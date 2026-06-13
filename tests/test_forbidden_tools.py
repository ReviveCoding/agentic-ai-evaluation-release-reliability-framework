from agentic_eval_framework.evaluators.forbidden_tools import forbidden_tool_rate


def test_forbidden_tool_rate():
    assert forbidden_tool_rate(["search_places", "final_answer"], ["safety_check"]) == 0.0
    assert forbidden_tool_rate(["safety_check", "final_answer"], ["safety_check"]) == 0.5

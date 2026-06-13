from __future__ import annotations

from agentic_eval_framework.tools.ask_clarification import ask_clarification
from agentic_eval_framework.tools.calendar_lookup import calendar_lookup
from agentic_eval_framework.tools.calendar_write import calendar_write
from agentic_eval_framework.tools.final_answer import final_answer
from agentic_eval_framework.tools.media_search import media_search
from agentic_eval_framework.tools.safety_check import safety_check
from agentic_eval_framework.tools.search_docs import search_docs
from agentic_eval_framework.tools.search_places import search_places
from agentic_eval_framework.tools.weather_lookup import weather_lookup

TOOL_REGISTRY = {
    "search_places": search_places,
    "calendar_lookup": calendar_lookup,
    "calendar_write": calendar_write,
    "media_search": media_search,
    "weather_lookup": weather_lookup,
    "ask_clarification": ask_clarification,
    "safety_check": safety_check,
    "final_answer": final_answer,
    "search_docs": search_docs,
}

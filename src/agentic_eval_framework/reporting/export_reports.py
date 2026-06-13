from __future__ import annotations

from agentic_eval_framework.models.evaluate_tool_policy import evaluate_policies
from agentic_eval_framework.reporting.dataset_card import export_dataset_card
from agentic_eval_framework.reporting.data_validation_report import export_data_validation_report
from agentic_eval_framework.reporting.model_card import export_model_card
from agentic_eval_framework.reporting.failure_attribution_report import export_failure_attribution_report
from agentic_eval_framework.reporting.latency_report import export_latency_report
from agentic_eval_framework.reporting.model_comparison_report import export_model_comparison_report
from agentic_eval_framework.reporting.release_gate_report import export_release_gate_report
from agentic_eval_framework.reporting.trace_lineage_report import export_trace_lineage_report
from agentic_eval_framework.retrieval.evaluate_retrieval import evaluate_retrieval


def export_all_reports() -> None:
    evaluate_policies()
    evaluate_retrieval()
    export_dataset_card()
    export_data_validation_report()
    export_model_card()
    export_model_comparison_report()
    export_failure_attribution_report()
    export_trace_lineage_report()
    export_release_gate_report()
    export_latency_report()

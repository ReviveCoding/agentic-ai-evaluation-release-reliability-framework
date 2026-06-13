from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_eval_framework.data.build_tool_policy_dataset import build_text
from agentic_eval_framework.models.model_manifest import load_model_manifest


class ToolPolicyModel:
    """Inference wrapper for transformer or sklearn tool-policy models.

    The framework supports a lightweight sklearn fallback for CI/smoke tests and
    a Hugging Face sequence-classification model for GPU-capable local training.
    """

    def __init__(self, model_dir: str | Path = "models/tool_policy") -> None:
        self.model_dir = Path(model_dir)
        self.backend = "rule"
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}
        self.pipeline: Any = None
        self.vectorizer: Any = None
        self.classifier: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: Any = None
        self.manifest: dict[str, Any] = {}
        self.model_fingerprint = "untrained-rule-fallback"
        self.recommended_min_confidence: float | None = None
        self._load()

    def _load(self) -> None:
        meta_path = self.model_dir / "model_meta.json"
        if not meta_path.exists():
            return
        self.manifest = load_model_manifest(self.model_dir)
        self.model_fingerprint = str(self.manifest.get("model_fingerprint", "legacy-model-without-manifest"))
        recommended = self.manifest.get("recommended_min_confidence")
        self.recommended_min_confidence = float(recommended) if recommended is not None else None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.backend = meta.get("backend", "rule")
        self.label_to_id = meta.get("label_to_id", {})
        self.id_to_label = {int(v): k for k, v in self.label_to_id.items()}
        if self.backend == "sklearn":
            import joblib
            self.vectorizer = joblib.load(self.model_dir / "vectorizer.joblib")
            self.classifier = joblib.load(self.model_dir / "classifier.joblib")
        elif self.backend == "transformer":
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.model.eval()
            except Exception as exc:  # pragma: no cover - optional runtime dependency
                raise RuntimeError(f"Could not load transformer model from {self.model_dir}: {exc}") from exc

    def predict_one(self, row: dict[str, Any]) -> str:
        text = build_text(row)
        if self.backend == "sklearn" and self.vectorizer is not None and self.classifier is not None:
            X = self.vectorizer.transform([text])
            return str(self.classifier.predict(X)[0])
        if self.backend == "transformer" and self.model is not None and self.tokenizer is not None:
            import torch
            batch = self.tokenizer([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                logits = self.model(**batch).logits
            pred_id = int(logits.argmax(dim=-1).item())
            return self.id_to_label[pred_id]
        # Safe fallback used before training exists.
        from agentic_eval_framework.models.rule_policy import RulePolicy
        return RulePolicy().predict_one(row)


    def predict_with_confidence(
        self, rows: list[dict[str, Any]], batch_size: int = 64
    ) -> tuple[list[str], list[float]]:
        if not rows:
            return [], []
        texts = [build_text(r) for r in rows]
        if self.backend == "sklearn" and self.vectorizer is not None and self.classifier is not None:
            X = self.vectorizer.transform(texts)
            probabilities = self.classifier.predict_proba(X)
            indices = probabilities.argmax(axis=1)
            labels = [str(self.classifier.classes_[int(i)]) for i in indices]
            confidences = [float(probabilities[j, int(i)]) for j, i in enumerate(indices)]
            return labels, confidences
        if self.backend == "transformer" and self.model is not None and self.tokenizer is not None:
            import torch
            labels: list[str] = []
            confidences_out: list[float] = []
            for start in range(0, len(texts), max(1, int(batch_size))):
                chunk = texts[start : start + max(1, int(batch_size))]
                batch = self.tokenizer(
                    chunk, truncation=True, padding=True, max_length=256, return_tensors="pt"
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.inference_mode():
                    probabilities = torch.softmax(self.model(**batch).logits, dim=-1)
                confidences, pred_ids = probabilities.max(dim=-1)
                labels.extend(self.id_to_label[int(i)] for i in pred_ids.detach().cpu().tolist())
                confidences_out.extend(float(x) for x in confidences.detach().cpu().tolist())
            return labels, confidences_out
        from agentic_eval_framework.models.rule_policy import RulePolicy
        labels = RulePolicy().predict(rows)
        return labels, [1.0] * len(labels)

    def predict(self, rows: list[dict[str, Any]], batch_size: int = 64) -> list[str]:
        labels, _ = self.predict_with_confidence(rows, batch_size=batch_size)
        return labels

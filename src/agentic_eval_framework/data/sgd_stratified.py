from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

from agentic_eval_framework.data.build_tool_policy_dataset import (
    normalize_slot_values,
    service_to_tool,
)
from agentic_eval_framework.data.parse_sgd import (
    intent_required_slots,
    load_dialogues,
    load_schemas,
    user_frames,
)
from agentic_eval_framework.utils.io import write_json, write_jsonl


def _service_base(service: str) -> str:
    return str(service or "Unknown").split("_")[0].lower()


def _intent_description(schema: dict[str, Any], intent_name: str) -> str:
    for intent in schema.get("intents", []):
        if intent.get("name") == intent_name:
            return str(intent.get("description", ""))
    return ""


def dialogue_rows(
    dialogue: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    source_split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    context: list[str] = []
    dialogue_id = str(dialogue.get("dialogue_id", "dialogue"))
    default_service = (dialogue.get("services") or ["Unknown_1"])[0]
    for turn_idx, turn in enumerate(dialogue.get("turns", [])):
        if turn.get("speaker") == "USER":
            for frame_idx, frame in enumerate(user_frames(turn)):
                service = str(frame.get("service", default_service))
                state = frame.get("state", {}) or {}
                intent = str(state.get("active_intent", "NONE"))
                known_slots = normalize_slot_values(state.get("slot_values", {}) or {})
                schema = schemas.get(service, {})
                required = intent_required_slots(schema, intent)
                missing = [
                    slot
                    for slot in required
                    if slot not in known_slots or known_slots.get(slot) in (None, "")
                ]
                tool_label = service_to_tool(service, intent, missing)
                rows.append(
                    {
                        "example_id": f"{source_split}::{dialogue_id}_{turn_idx}_{frame_idx}",
                        "source_dialogue_id": dialogue_id,
                        "source_split": source_split,
                        "dialogue_context": " ".join(context[-4:]),
                        "user_utterance": str(turn.get("utterance", "")),
                        "service": service,
                        "service_base": _service_base(service),
                        "intent": intent,
                        "service_description": str(schema.get("description", "")),
                        "intent_description": _intent_description(schema, intent),
                        "known_slots": known_slots,
                        "missing_slots": missing,
                        "required_slots": required,
                        "tool_label": tool_label,
                        "expected_next_action": (
                            "tool_call"
                            if tool_label not in {"ask_clarification", "safety_check"}
                            else tool_label
                        ),
                        "risk_flags": ["sensitive_service"]
                        if tool_label == "safety_check"
                        else [],
                        "split": source_split,
                    }
                )
        context.append(f"[{turn.get('speaker', 'UNK')}] {turn.get('utterance', '')}")
    return rows


def rows_for_dialogues(
    dialogues: Iterable[dict[str, Any]],
    schemas: dict[str, dict[str, Any]],
    source_split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dialogue in dialogues:
        rows.extend(dialogue_rows(dialogue, schemas, source_split))
    return rows


@dataclass(frozen=True)
class DialogueGroup:
    dialogue_id: str
    dialogue: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    labels: frozenset[str]
    services: frozenset[str]


def build_dialogue_groups(
    dialogues: list[dict[str, Any]],
    schemas: dict[str, dict[str, Any]],
    source_split: str,
) -> list[DialogueGroup]:
    groups: list[DialogueGroup] = []
    for dialogue in dialogues:
        rows = dialogue_rows(dialogue, schemas, source_split)
        if not rows:
            continue
        groups.append(
            DialogueGroup(
                dialogue_id=str(dialogue.get("dialogue_id", "dialogue")),
                dialogue=dialogue,
                rows=tuple(rows),
                labels=frozenset(str(row["tool_label"]) for row in rows),
                services=frozenset(str(row["service_base"]) for row in rows),
            )
        )
    return groups


def _rarity_scores(groups: list[DialogueGroup]) -> dict[str, float]:
    label_frequency = Counter(label for group in groups for label in group.labels)
    service_frequency = Counter(service for group in groups for service in group.services)
    scores: dict[str, float] = {}
    for group in groups:
        label_score = sum(1.0 / max(1, label_frequency[label]) for label in group.labels)
        service_score = sum(1.0 / max(1, service_frequency[s]) for s in group.services)
        scores[group.dialogue_id] = label_score + 0.25 * service_score
    return scores


def select_balanced_dialogues(
    groups: list[DialogueGroup],
    max_dialogues: int | None,
    seed: int = 42,
) -> list[DialogueGroup]:
    if max_dialogues is None or max_dialogues >= len(groups):
        return list(groups)
    if max_dialogues <= 0:
        raise ValueError("max_dialogues must be positive")

    rng = random.Random(seed)
    rarity = _rarity_scores(groups)
    by_label: dict[str, list[DialogueGroup]] = defaultdict(list)
    by_service: dict[str, list[DialogueGroup]] = defaultdict(list)
    for group in groups:
        for label in group.labels:
            by_label[label].append(group)
        for service in group.services:
            by_service[service].append(group)

    selected: dict[str, DialogueGroup] = {}

    def choose(candidates: list[DialogueGroup]) -> None:
        available = [g for g in candidates if g.dialogue_id not in selected]
        if not available or len(selected) >= max_dialogues:
            return
        rng.shuffle(available)
        available.sort(key=lambda g: rarity[g.dialogue_id], reverse=True)
        selected[available[0].dialogue_id] = available[0]

    # Guarantee at least three dialogue groups per tool label so every label can
    # be seeded into train, calibration, and evaluation. Then add service diversity.
    for label in sorted(by_label, key=lambda key: len(by_label[key])):
        for _ in range(min(3, len(by_label[label]))):
            choose(by_label[label])
    for service in sorted(by_service, key=lambda key: len(by_service[key])):
        choose(by_service[service])

    remaining = [g for g in groups if g.dialogue_id not in selected]
    rng.shuffle(remaining)
    remaining.sort(key=lambda g: rarity[g.dialogue_id], reverse=True)
    for group in remaining:
        if len(selected) >= max_dialogues:
            break
        selected[group.dialogue_id] = group

    result = list(selected.values())
    rng.shuffle(result)
    return result


def _target_sizes(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    train = int(total * ratios[0])
    calibration = int(total * ratios[1])
    evaluation = total - train - calibration
    return {"train": train, "calibration": calibration, "eval": evaluation}


def stratified_group_split(
    groups: list[DialogueGroup],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[DialogueGroup]]:
    if not groups:
        raise ValueError("No dialogue groups are available")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1")

    split_names = ("train", "calibration", "eval")
    targets = _target_sizes(len(groups), ratios)
    label_group_counts = Counter(label for group in groups for label in group.labels)
    impossible = sorted(label for label, count in label_group_counts.items() if count < 3)
    if impossible:
        raise ValueError(
            "Cannot place every label in train/calibration/eval; labels appear in fewer than "
            f"three dialogue groups: {impossible}. Increase --max-dialogues."
        )

    total_label_rows = Counter(
        str(row["tool_label"]) for group in groups for row in group.rows
    )
    total_service_rows = Counter(
        str(row["service_base"]) for group in groups for row in group.rows
    )
    target_label_rows = {
        split: {label: count * ratios[index] for label, count in total_label_rows.items()}
        for index, split in enumerate(split_names)
    }
    target_service_rows = {
        split: {service: count * ratios[index] for service, count in total_service_rows.items()}
        for index, split in enumerate(split_names)
    }

    rng = random.Random(seed)
    assignments: dict[str, list[DialogueGroup]] = {name: [] for name in split_names}
    assigned: set[str] = set()
    label_rows: dict[str, Counter[str]] = {name: Counter() for name in split_names}
    service_rows: dict[str, Counter[str]] = {name: Counter() for name in split_names}

    def add(split: str, group: DialogueGroup) -> None:
        assignments[split].append(group)
        assigned.add(group.dialogue_id)
        label_rows[split].update(str(row["tool_label"]) for row in group.rows)
        service_rows[split].update(str(row["service_base"]) for row in group.rows)

    # Seed every split with every label, starting with the rarest labels.
    for label in sorted(label_group_counts, key=lambda key: label_group_counts[key]):
        for split in split_names:
            if label_rows[split][label] > 0:
                continue
            candidates = [
                group
                for group in groups
                if group.dialogue_id not in assigned and label in group.labels
            ]
            if not candidates:
                raise ValueError(
                    f"Unable to seed label {label!r} into split {split!r}; increase the subset size."
                )
            rng.shuffle(candidates)
            candidates.sort(
                key=lambda group: (
                    len(group.labels),
                    sum(label_group_counts[item] for item in group.labels),
                )
            )
            add(split, candidates[0])

    rarity = _rarity_scores(groups)
    remaining = [group for group in groups if group.dialogue_id not in assigned]
    rng.shuffle(remaining)
    remaining.sort(key=lambda group: rarity[group.dialogue_id], reverse=True)

    def assignment_cost(split: str, group: DialogueGroup) -> float:
        if len(assignments[split]) >= targets[split]:
            return float("inf")
        projected_size = len(assignments[split]) + 1
        size_cost = abs(projected_size - targets[split]) / max(1, targets[split])
        group_label_counts = Counter(str(row["tool_label"]) for row in group.rows)
        group_service_counts = Counter(str(row["service_base"]) for row in group.rows)
        label_cost = sum(
            abs(
                label_rows[split][label]
                + group_label_counts[label]
                - target_label_rows[split][label]
            )
            / max(1.0, target_label_rows[split][label])
            for label in group_label_counts
        )
        service_cost = sum(
            abs(
                service_rows[split][service]
                + group_service_counts[service]
                - target_service_rows[split][service]
            )
            / max(1.0, target_service_rows[split][service])
            for service in group_service_counts
        )
        return 2.0 * size_cost + label_cost + 0.2 * service_cost

    for group in remaining:
        split = min(split_names, key=lambda name: assignment_cost(name, group))
        if assignment_cost(split, group) == float("inf"):
            split = min(split_names, key=lambda name: len(assignments[name]))
        add(split, group)

    for split in split_names:
        if len(assignments[split]) != targets[split]:
            raise AssertionError(
                f"Split {split} has {len(assignments[split])} dialogues, expected {targets[split]}"
            )
    return assignments


def flatten_groups(groups: Iterable[DialogueGroup], logical_split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        for original in group.rows:
            row = dict(original)
            row["split"] = logical_split
            row["example_id"] = f"{logical_split}::{original['source_dialogue_id']}::{original['example_id'].split('::')[-1]}"
            rows.append(row)
    return rows


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["tool_label"]) for row in rows).items()))


def _service_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["service_base"]) for row in rows).items()))


def _validate_label_coverage(
    split_rows: dict[str, list[dict[str, Any]]],
    minimum_rows_per_label: int,
) -> None:
    train_labels = set(_label_counts(split_rows["train"]))
    for split_name in ("calibration", "eval"):
        labels = set(_label_counts(split_rows[split_name]))
        missing = sorted(train_labels - labels)
        if missing:
            raise ValueError(f"{split_name} split is missing train labels: {missing}")
    for split_name, rows in split_rows.items():
        counts = Counter(str(row["tool_label"]) for row in rows)
        too_small = {label: count for label, count in counts.items() if count < minimum_rows_per_label}
        if too_small:
            raise ValueError(
                f"{split_name} split has labels below minimum_rows_per_label="
                f"{minimum_rows_per_label}: {too_small}. Increase --max-dialogues."
            )


def build_stratified_sgd_datasets(
    raw_dir: str | Path,
    processed_dir: str | Path,
    max_train_dialogues: int | None = 250,
    max_ood_dialogues: int | None = 250,
    seed: int = 42,
    minimum_rows_per_label: int = 3,
) -> dict[str, Any]:
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    train_schemas = load_schemas(raw / "train")
    train_dialogues = load_dialogues(raw / "train")
    train_groups_all = build_dialogue_groups(train_dialogues, train_schemas, "official_train")
    selected_train_groups = select_balanced_dialogues(
        train_groups_all, max_train_dialogues, seed=seed
    )
    assignments = stratified_group_split(selected_train_groups, seed=seed)
    internal_rows = {
        split: flatten_groups(groups, split) for split, groups in assignments.items()
    }
    _validate_label_coverage(internal_rows, minimum_rows_per_label)

    write_jsonl(processed / "tool_policy_train.jsonl", internal_rows["train"])
    write_jsonl(
        processed / "tool_policy_calibration.jsonl", internal_rows["calibration"]
    )
    write_jsonl(processed / "tool_policy_eval.jsonl", internal_rows["eval"])

    ood_summary: dict[str, Any] = {}
    for source_split, logical_name in (("dev", "ood_dev"), ("test", "ood_test")):
        schemas = load_schemas(raw / source_split)
        dialogues = load_dialogues(raw / source_split)
        groups = build_dialogue_groups(dialogues, schemas, f"official_{source_split}")
        selected = select_balanced_dialogues(groups, max_ood_dialogues, seed=seed + (1 if source_split == "dev" else 2))
        rows = flatten_groups(selected, logical_name)
        write_jsonl(processed / f"tool_policy_{logical_name}.jsonl", rows)
        ood_summary[logical_name] = {
            "dialogues": len(selected),
            "rows": len(rows),
            "label_counts": _label_counts(rows),
            "service_counts": _service_counts(rows),
        }

    manifest = {
        "seed": seed,
        "source": "Schema-Guided Dialogue",
        "architecture": {
            "official_train": "stratified 80/10/10 model_train/calibration/id_eval",
            "official_dev": "OOD validation only",
            "official_test": "OOD final test only",
        },
        "max_train_dialogues": max_train_dialogues,
        "max_ood_dialogues": max_ood_dialogues,
        "minimum_rows_per_label": minimum_rows_per_label,
        "internal": {
            split: {
                "dialogues": len(assignments[split]),
                "rows": len(rows),
                "label_counts": _label_counts(rows),
                "service_counts": _service_counts(rows),
            }
            for split, rows in internal_rows.items()
        },
        "ood": ood_summary,
    }
    write_json(processed / "sgd_split_manifest.json", manifest)
    return manifest



def build_service_docs_from_training_rows(
    raw_dir: str | Path,
    training_rows: list[dict[str, Any]],
    out_path: str | Path,
) -> list[dict[str, Any]]:
    """Build schema documents while mining examples only from model-training rows.

    Official dev/test schemas are included because SGD exposes target schemas at
    evaluation time, but utterance examples come only from the internal model
    training split. This prevents retrieval leakage from calibration, ID test, or
    OOD dialogues.
    """
    from agentic_eval_framework.retrieval.targets import intent_doc_id, service_doc_id

    raw = Path(raw_dir)
    merged: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        merged.update(load_schemas(raw / split))

    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in training_rows:
        key = (str(row.get("service", "")), str(row.get("intent", "")))
        utterance = str(row.get("user_utterance", "")).strip()
        bucket = examples[key]
        if utterance and utterance not in bucket and len(bucket) < 20:
            bucket.append(utterance)

    docs: list[dict[str, Any]] = []
    for service_name, schema in sorted(merged.items()):
        slots = {
            str(slot.get("name")): str(slot.get("description", ""))
            for slot in schema.get("slots", [])
        }
        slot_text = "; ".join(
            f"{name}: {description}" for name, description in slots.items()
        )
        intent_text = "; ".join(
            f"{intent.get('name')}: {intent.get('description', '')}"
            for intent in schema.get("intents", [])
        )
        docs.append(
            {
                "doc_id": service_doc_id(service_name),
                "doc_type": "service",
                "service": service_name,
                "intent": "",
                "title": service_name,
                "text": f"{schema.get('description', '')} Intents: {intent_text}. Slots: {slot_text}.",
            }
        )
        for intent in schema.get("intents", []):
            intent_name = str(intent.get("name", ""))
            required = list(intent.get("required_slots", []) or [])
            optional = list(intent.get("optional_slots", []) or [])
            relevant_slots = "; ".join(
                f"{slot}: {slots.get(slot, '')}" for slot in required + optional
            )
            example_text = " | ".join(examples.get((service_name, intent_name), [])[:8])
            docs.append(
                {
                    "doc_id": intent_doc_id(service_name, intent_name),
                    "doc_type": "intent",
                    "service": service_name,
                    "intent": intent_name,
                    "title": f"{service_name} {intent_name}",
                    "text": (
                        f"Service: {schema.get('description', '')}. Action: {intent_name}. "
                        f"{intent.get('description', '')}. Required inputs: {', '.join(required) or 'none'}. "
                        f"Optional inputs: {', '.join(optional) or 'none'}. "
                        f"Slot definitions: {relevant_slots}. Example requests: {example_text}."
                    ),
                }
            )

    tool_docs = [
        ("tool::search_places", "Search hotels, restaurants, travel, events, and place-like services."),
        ("tool::calendar_lookup", "Read calendar events and calendar availability without changing state."),
        ("tool::calendar_write", "Stage a calendar create, update, or delete action for confirmation."),
        ("tool::media_search", "Search movies, music, shows, and media-like services."),
        ("tool::weather_lookup", "Retrieve weather forecasts by location and date."),
        ("tool::search_docs", "Retrieve personal or enterprise documents and policies."),
        ("tool::ask_clarification", "Ask for missing required information before taking action."),
        ("tool::safety_check", "Review sensitive or policy-constrained requests before acting."),
        ("tool::final_answer", "Complete the task with a grounded answer."),
    ]
    for doc_id, body in tool_docs:
        docs.append(
            {
                "doc_id": doc_id,
                "doc_type": "tool",
                "service": "framework",
                "intent": "",
                "title": doc_id,
                "text": body,
            }
        )
    write_jsonl(out_path, docs)
    return docs

def audit_raw_mapping(
    raw_dir: str | Path,
    out_json: str | Path = "outputs/sgd_mapping_audit.json",
    out_md: str | Path = "reports/sgd_mapping_audit.md",
) -> dict[str, Any]:
    raw = Path(raw_dir)
    per_split: dict[str, Any] = {}
    combined_label_counts: Counter[str] = Counter()
    combined_service_counts: Counter[str] = Counter()
    service_tool_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for split in ("train", "dev", "test"):
        schemas = load_schemas(raw / split)
        rows = rows_for_dialogues(load_dialogues(raw / split), schemas, f"official_{split}")
        labels = Counter(str(row["tool_label"]) for row in rows)
        services = Counter(str(row["service_base"]) for row in rows)
        intents = Counter(str(row["intent"]) for row in rows)
        combined_label_counts.update(labels)
        combined_service_counts.update(services)
        for row in rows:
            service_tool_counts[str(row["service_base"])][str(row["tool_label"])] += 1
        per_split[split] = {
            "rows": len(rows),
            "label_counts": dict(sorted(labels.items())),
            "service_counts": dict(sorted(services.items())),
            "intent_counts": dict(sorted(intents.items())),
        }

    domain_expectations = {
        "calendar": {"calendar_lookup", "calendar_write"},
        "appointment": {"calendar_lookup", "calendar_write"},
        "weather": {"weather_lookup"},
        "bank": {"safety_check"},
        "banks": {"safety_check"},
        "payment": {"safety_check"},
    }
    failures: list[str] = []
    for service, expected_tools in domain_expectations.items():
        if combined_service_counts[service] <= 0:
            continue
        observed = set(service_tool_counts[service])
        if not observed & expected_tools:
            failures.append(
                f"service={service} expected one of {sorted(expected_tools)} but observed {sorted(observed)}"
            )

    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "per_split": per_split,
        "combined_label_counts": dict(sorted(combined_label_counts.items())),
        "combined_service_counts": dict(sorted(combined_service_counts.items())),
        "service_to_tool_counts": {
            service: dict(sorted(counts.items()))
            for service, counts in sorted(service_tool_counts.items())
        },
    }
    write_json(out_json, result)
    lines = [
        "# SGD Service-to-Tool Mapping Audit",
        "",
        f"Status: **{result['status']}**",
        f"Failures: {failures or ['none']}",
        "",
        "## Combined tool-label counts",
    ]
    lines.extend(f"- `{label}`: {count}" for label, count in result["combined_label_counts"].items())
    lines += ["", "## Service to tool mapping"]
    for service, counts in result["service_to_tool_counts"].items():
        lines.append(f"- `{service}`: {counts}")
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

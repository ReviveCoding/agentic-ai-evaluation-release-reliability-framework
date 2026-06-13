from agentic_eval_framework.data.parse_sgd import SAMPLE_DIALOGUES, SAMPLE_SCHEMAS
from agentic_eval_framework.data.sgd_stratified import (
    DialogueGroup,
    build_dialogue_groups,
    select_balanced_dialogues,
    stratified_group_split,
)


def test_balanced_selector_is_deterministic_and_bounded():
    schemas = {schema["service_name"]: schema for schema in SAMPLE_SCHEMAS}
    groups = build_dialogue_groups(SAMPLE_DIALOGUES, schemas, "train")
    first = select_balanced_dialogues(groups, min(4, len(groups)), seed=7)
    second = select_balanced_dialogues(groups, min(4, len(groups)), seed=7)
    assert [group.dialogue_id for group in first] == [group.dialogue_id for group in second]
    assert len(first) == min(4, len(groups))


def test_group_split_preserves_label_coverage_and_dialogue_disjointness():
    labels = ["search_places", "media_search", "ask_clarification"]
    groups = []
    for index in range(60):
        label = labels[index % len(labels)]
        row = {
            "example_id": f"raw::{index}",
            "source_dialogue_id": f"d{index}",
            "service_base": "restaurants" if label == "search_places" else "media",
            "tool_label": label,
        }
        groups.append(
            DialogueGroup(
                dialogue_id=f"d{index}",
                dialogue={"dialogue_id": f"d{index}"},
                rows=(row,),
                labels=frozenset({label}),
                services=frozenset({row["service_base"]}),
            )
        )
    splits = stratified_group_split(groups, seed=11)
    ids = [{group.dialogue_id for group in splits[name]} for name in ("train", "calibration", "eval")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for name in ("train", "calibration", "eval"):
        observed = {label for group in splits[name] for label in group.labels}
        assert observed == set(labels)

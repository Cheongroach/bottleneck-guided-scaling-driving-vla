"""Describe the published weak teacher without treating agreement as correctness."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from contracts import ACTION_NAMES, CAMERAS


def mean(values):
    return sum(values)/len(values) if values else None


def distribution(values, names):
    counts = Counter(values)
    n = sum(counts.values())
    entropy = -sum((v/n)*math.log(v/n) for v in counts.values()) if n else None
    return dict(counts={name: counts[i] for i, name in enumerate(names)},
                unique_classes=len(counts),
                majority_fraction=max(counts.values())/n if n else None,
                normalized_entropy=entropy/math.log(len(names)) if n else None)


def summarize(rows):
    teachers = [r for r in rows if r.get("teacher") is not None]
    trajectories = [r for r in teachers if r["teacher"].get("trajectory") is not None]
    errors = [[math.dist(p, q) for p, q in zip(r["teacher"]["trajectory"], r["trajectory"])]
              for r in trajectories]
    labels = Counter(r["action"] for r in rows)
    majority = max(labels, key=labels.get) if labels else None
    text_counts = Counter(r.get("teacher_text", "") for r in teachers)
    return dict(
        samples=len(rows), parsed_teachers=len(teachers),
        ground_truth_action=distribution([r["action"] for r in rows], ACTION_NAMES),
        teacher_action=distribution([r["teacher"]["action"] for r in teachers], ACTION_NAMES),
        teacher_ground_truth_action_agreement=mean(
            [r["teacher"]["action"] == r["action"] for r in teachers]),
        ground_truth_majority_action=ACTION_NAMES[majority] if majority is not None else None,
        ground_truth_majority_action_accuracy=labels[majority]/len(rows) if rows else None,
        teacher_evidence_camera_fraction={
            camera: mean([r["teacher"]["evidence"][i] > 0 for r in teachers])
            for i, camera in enumerate(CAMERAS)},
        teacher_evidence_pattern_count=len({
            tuple(r["teacher"]["evidence"]) for r in teachers}),
        teacher_empty_evidence_fraction=mean([
            not any(r["teacher"]["evidence"]) for r in teachers]),
        teacher_unique_output_texts=len(text_counts),
        teacher_most_repeated_output_fraction=max(text_counts.values())/len(teachers) if teachers else None,
        teacher_trajectory=dict(
            count=len(trajectories),
            ADE_m=mean([mean(e) for e in errors]),
            FDE_m=mean([e[-1] for e in errors]),
            negative_forward_waypoint_fraction=mean([
                p[0] < 0 for r in trajectories for p in r["teacher"]["trajectory"]]),
            ground_truth_negative_forward_waypoint_fraction=mean([
                p[0] < 0 for r in trajectories for p in r["trajectory"]]),
            all_zero_fraction=mean([
                all(p == [0, 0] for p in r["teacher"]["trajectory"]) for r in trajectories])))


def analyze(records_path):
    raw = records_path.read_bytes()
    rows = json.loads(raw)
    assert rows and all(r["split"] in ("train", "val") for r in rows)
    assert len({r["token"] for r in rows}) == len(rows), "Duplicate sample tokens"
    return dict(
        records_sha256=hashlib.sha256(raw).hexdigest(),
        overall=summarize(rows),
        splits={split: summarize([r for r in rows if r["split"] == split])
                for split in ("train", "val")},
        interpretation=(
            "Parsed teacher labels measure structural validity. Action agreement compares "
            "the weak teacher with the predefined ego-trajectory action label, and is not "
            "an independent safety score. Evidence has no human ground truth. Repeated "
            "outputs or concentrated classes can inflate explanation agreement metrics."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.records)
    temporary = args.output.with_suffix(".upload")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))

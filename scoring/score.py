"""Score an ARC submission.json (Kaggle format) against the official ARC-AGI solutions.

Submission format: {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}, one entry per
test input in task order. A test input counts as solved if any attempt matches exactly; a
task's score is the fraction of its test inputs solved (ARC Prize convention).
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

ARC_DATA = Path(__file__).resolve().parents[1] / "arc-agi" / "data"

Grid = list[list[int]]
Submission = dict[str, list[dict[str, Grid]]]


class Score(TypedDict):
    """Scores for one split, overall and per task."""

    split: str
    tasks_in_split: int
    tasks_submitted: int
    score_on_submitted: float
    score_on_split: float
    per_task: dict[str, float]


def score(submission: Submission, split_dir: Path) -> Score:
    """Score a submission against every task file in `split_dir`.

    Args:
        submission: Attempts per task id, one entry per test input.
        split_dir: Directory of official task files with solutions.

    Returns:
        Overall and per-task scores.

    Raises:
        ValueError: If the submission names tasks not in the split.
    """
    task_files = sorted(split_dir.glob("*.json"))
    per_task: dict[str, float] = {}
    for path in task_files:
        task_id = path.stem
        if task_id not in submission:
            continue
        tests = json.loads(path.read_text())["test"]
        attempts = submission[task_id]
        solved = sum(
            any(grid == test["output"] for grid in attempts[i].values())
            for i, test in enumerate(tests)
            if i < len(attempts)
        )
        per_task[task_id] = solved / len(tests)

    unknown = set(submission) - {p.stem for p in task_files}
    if unknown:
        raise ValueError(
            f"{len(unknown)} submitted tasks not in {split_dir}: {sorted(unknown)[:5]}"
        )

    total = sum(per_task.values())
    return {
        "split": split_dir.name,
        "tasks_in_split": len(task_files),
        "tasks_submitted": len(per_task),
        "score_on_submitted": total / len(per_task) if per_task else 0.0,
        "score_on_split": total / len(task_files),
        "per_task": per_task,
    }


def main() -> None:
    """Score a submission file and print the summary."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    ap.add_argument("--out", type=Path, help="Write the full result, including per-task scores.")
    args = ap.parse_args()

    result = score(json.loads(args.submission.read_text()), ARC_DATA / args.split)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main()

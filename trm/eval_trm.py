"""Evaluate a TRM checkpoint on ARC-AGI-1 with the upstream voting evaluator, on cuda/mps/cpu."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
import yaml
from torch._C._distributed_c10d import HashStore

ROOT = Path(__file__).resolve().parent
TRM_DIR = ROOT / "TinyRecursiveModels"
sys.path.insert(0, str(TRM_DIR))

from dataset.build_arc_dataset import inverse_aug  # noqa: E402
from dataset.common import PuzzleDatasetMetadata  # noqa: E402
from evaluators.arc import ARC  # noqa: E402
from models.losses import IGNORE_LABEL_ID  # noqa: E402
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1  # noqa: E402


def pick_device(name: str) -> torch.device:
    """Resolve `auto` to the best available backend."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_test_split(
    data_dir: Path, max_tasks: int | None, max_aug: int | None, seed: int
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer], list[str]]:
    """Return examples for a subset of eval tasks: (inputs, labels, puzzle_ids, task_names)."""
    split = data_dir / "test"
    arr = {
        k: np.load(split / f"all__{k}.npy")
        for k in ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")
    }
    identifier_map = json.loads((data_dir / "identifiers.json").read_text())

    groups = np.arange(len(arr["group_indices"]) - 1)
    if max_tasks is not None:
        groups = np.sort(np.random.default_rng(seed).choice(groups, max_tasks, replace=False))

    example_idx: list[int] = []
    example_pid: list[int] = []
    task_names: list[str] = []
    for g in groups:
        p_start, p_end = arr["group_indices"][g], arr["group_indices"][g + 1]
        # Puzzle 0 of each group is the unaugmented task.
        if max_aug is not None:
            p_end = min(p_end, p_start + max_aug)
        for p in range(p_start, p_end):
            e_start, e_end = arr["puzzle_indices"][p], arr["puzzle_indices"][p + 1]
            example_idx.extend(range(e_start, e_end))
            example_pid.extend([arr["puzzle_identifiers"][p]] * (e_end - e_start))
        task_names.append(inverse_aug(identifier_map[arr["puzzle_identifiers"][p_start]])[0])

    idx = np.asarray(example_idx)
    return arr["inputs"][idx], arr["labels"][idx], np.asarray(example_pid), task_names


def build_model(
    ckpt_dir: Path,
    ckpt_name: str,
    metadata: PuzzleDatasetMetadata,
    batch_size: int,
    device: torch.device,
) -> TinyRecursiveReasoningModel_ACTV1:
    """Build TRM from the checkpoint's config and load its weights."""
    cfg = yaml.safe_load((ckpt_dir / "all_config.yaml").read_text())["arch"]
    for k in ("loss", "name"):
        cfg.pop(k)
    cfg.update(
        batch_size=batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )
    if device.type == "cpu":
        cfg["forward_dtype"] = "float32"

    with torch.device(device):
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
    state = torch.load(ckpt_dir / ckpt_name, map_location="cpu", weights_only=True)
    state = {k.removeprefix("_orig_mod.model."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main() -> None:
    """Run the evaluation and write metrics.json."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "arc1concept-aug-1000")
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=ROOT / "checkpoints" / "trm_arc_prize_verification" / "arc_v1_public",
    )
    ap.add_argument("--ckpt-name", default="step_518071")
    ap.add_argument("--max-tasks", type=int, default=None, help="Random subset of eval tasks.")
    ap.add_argument(
        "--max-aug",
        type=int,
        default=None,
        help="Augmented copies per task to vote over (1 = original only).",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "trm_arc1")
    args = ap.parse_args()

    device = pick_device(args.device)
    metadata = PuzzleDatasetMetadata(
        **json.loads((args.data_dir / "test" / "dataset.json").read_text())
    )
    inputs, labels, pids, task_names = load_test_split(
        args.data_dir, args.max_tasks, args.max_aug, args.seed
    )
    print(f"device={device} tasks={len(task_names)} examples={len(inputs)}", flush=True)

    model = build_model(args.ckpt_dir, args.ckpt_name, metadata, args.batch_size, device)

    # The upstream evaluator gathers through torch.distributed; an in-process store needs no port.
    dist.init_process_group("gloo", store=HashStore(), rank=0, world_size=1)
    evaluator = ARC(str(args.data_dir), metadata)
    evaluator.test_puzzles = {n: evaluator.test_puzzles[n] for n in task_names}

    labels = labels.astype(np.int32)
    labels[labels == metadata.ignore_label_id] = IGNORE_LABEL_ID
    n, bs = len(inputs), args.batch_size
    exact, t0 = 0, time.time()
    with torch.inference_mode():
        for start in range(0, n, bs):
            sl = slice(start, min(start + bs, n))
            pad = bs - (sl.stop - sl.start)
            arrays = {
                "inputs": np.pad(
                    inputs[sl].astype(np.int32),
                    ((0, pad), (0, 0)),
                    constant_values=metadata.pad_id,
                ),
                "labels": np.pad(labels[sl], ((0, pad), (0, 0)), constant_values=IGNORE_LABEL_ID),
                "puzzle_identifiers": np.pad(
                    pids[sl].astype(np.int32),
                    (0, pad),
                    constant_values=metadata.blank_identifier_id,
                ),
            }
            batch = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
            with torch.device(device):
                carry = model.initial_carry(batch)
            # In eval mode every sequence runs exactly halt_max_steps ACT steps.
            while True:
                carry, outputs = model(carry=carry, batch=batch)
                if carry.halted.all():
                    break
            preds = outputs["logits"].argmax(-1)
            mask = batch["labels"] != IGNORE_LABEL_ID
            valid = mask.any(-1)
            exact += int((valid & ((preds == batch["labels"]) | ~mask).all(-1)).sum())
            # The evaluator upcasts to float64, which MPS lacks, so hand it CPU tensors.
            evaluator.update_batch(
                {k: v.cpu() for k, v in batch.items()},
                {"preds": preds.cpu(), "q_halt_logits": outputs["q_halt_logits"].float().cpu()},
            )
            done = sl.stop
            rate = done / (time.time() - t0)
            print(
                f"{done}/{n} examples  {rate:.1f} ex/s  eta {(n - done) / rate / 60:.1f} min",
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = evaluator.result(str(args.out_dir), rank=0, world_size=1)
    results["per_augmentation_exact_acc"] = exact / n
    results.update(
        tasks=len(task_names),
        examples=n,
        max_aug=args.max_aug,
        device=str(device),
        seconds=round(time.time() - t0, 1),
    )
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

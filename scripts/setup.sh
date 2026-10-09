#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

git submodule update --init --recursive

cd trm
uv sync --all-packages --frozen -q

data=data/arc1concept-aug-1000
if [ ! -f "$data/test/dataset.json" ]; then
    (cd TinyRecursiveModels && uv run --frozen --project .. python -m dataset.build_arc_dataset \
        --input-file-prefix kaggle/combined/arc-agi --output-dir "../$data" \
        --subsets training evaluation concept --test-set-name evaluation)
fi

ckpt=checkpoints/trm_arc_prize_verification
if [ ! -f "$ckpt/arc_v1_public/step_518071" ]; then
    uv run --frozen hf download arcprize/trm_arc_prize_verification --include 'arc_v1_public/*' \
        --local-dir "$ckpt"
fi

#!/usr/bin/env bash
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
base=$HOME/$1 url=$2 branch=$3 action=$4
shift 4
runs=$base-runs

[ -d "$base/.git" ] || git clone -q "$url" "$base"
git -C "$base" fetch -q origin
git -C "$base" checkout -q --detach "origin/$branch"
git -C "$base" submodule update -q --init --recursive

case $action in
    setup)
        cd "$base" && exec scripts/setup.sh ;;
    prune)
        for wt in "$runs"/*/; do
            [ -d "$wt" ] || continue
            name=$(basename "$wt")
            if tmux has-session -t "=$name" 2>/dev/null; then
                echo "kept $name (running)"
            else
                git -C "$base" worktree remove --force "$wt" && echo "removed $name"
            fi
        done
        git -C "$base" worktree prune ;;
    start)
        commit=$1 run=$2
        shift 2
        echo "active runs: $(tmux ls -F '#S' 2>/dev/null | paste -sd' ' - || true)"
        wt=$runs/$run
        git -C "$base" worktree add -q --detach "$wt" "$commit"
        git -C "$wt" submodule update -q --init --recursive
        for d in data checkpoints results; do
            mkdir -p "$base/trm/$d"
            ln -s "$base/trm/$d" "$wt/trm/$d"
        done
        (cd "$wt" && uv sync --all-packages --frozen -q)

        logs=$base/trm/results/logs
        mkdir -p "$logs"
        log=$logs/$run.log
        echo "run=$run commit=$commit" > "$log"
        cmd=$(printf '%q ' "$@")
        mon="bash $base/tools/monitor.sh $logs/$run.csv & mon=\$!"
        tmux new-session -d -s "$run" -c "$wt/trm" -e "PATH=$PATH" \
            "set -o pipefail; $mon; $cmd 2>&1 | tee -a $log; echo exit=\$? >> $log; kill \$mon"
        echo "started $run at ${commit:0:12}: just logs $run" ;;
    *)
        echo "unknown action $action" >&2; exit 1 ;;
esac

#!/usr/bin/env bash
set -euo pipefail
host=$1 dir=$2 task=$3 target=$4
shift 4
cd "$(git rev-parse --show-toplevel)"

die() { echo "$*" >&2; exit 1; }

ref='' latest='' push=''
while [ $# -gt 0 ]; do
    case $1 in
        --ref) ref=$2; shift 2 ;;
        --latest) latest=1; shift ;;
        --push) push=1; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
[ -z "$ref" ] || [ -z "$latest" ] || die "--ref and --latest are mutually exclusive"

run=$task-$(date -u +%Y%m%d-%H%M%S)
eval_cmd() { cmd=(uv run --frozen python eval_trm.py --out-dir "results/$run"); }

case $target in
    local)
        [ -z "$ref$latest$push" ] || die "--ref, --latest and --push only apply to remote"
        case $task in
            setup) exec scripts/setup.sh ;;
            eval) eval_cmd; cd trm && exec "${cmd[@]}" "$@" ;;
            *) die "$task has no local form" ;;
        esac ;;
    remote) ;;
    *) die "target must be local or remote, got '$target'" ;;
esac

branch=$(git branch --show-current)
[ -n "$branch" ] || die "detached HEAD; check out a branch first"
[ -z "$push" ] || git push -q origin "$branch"
git fetch -q origin

on_remote() {
    local args
    printf -v args '%q ' "$dir" "$(git remote get-url origin)" "$branch" "$@"
    ssh "$host" bash -s -- "$args" < scripts/remote.sh
}

case $task in
    setup|prune)
        [ -z "$ref$latest" ] || die "$task always uses origin/$branch"
        on_remote "$task" ;;
    eval)
        if [ -n "$latest" ]; then
            commit=$(git rev-parse "origin/$branch")
        elif [ -n "$ref" ]; then
            commit=$(git rev-parse --verify "$ref^{commit}")
        else
            commit=$(git rev-parse HEAD)
            if [ -n "$(git status --porcelain)" ]; then
                echo "warning: uncommitted changes are not part of this run:" >&2
                git status --short >&2
            fi
        fi
        [ -n "$(git branch -r --contains "$commit")" ] || die "$commit is not on origin; push it or pass --push"
        run=$run-$(git rev-parse --short "$commit")
        eval_cmd
        on_remote start "$commit" "$run" "${cmd[@]}" "$@" ;;
    *) die "unknown task $task" ;;
esac

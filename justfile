set dotenv-load
set positional-arguments

host := env("REMOTE_HOST")
remote_dir := env("REMOTE_DIR")

export PATH := env("HOME") + "/.local/bin:" + env("PATH")

# just eval local|remote [--ref REF | --latest] [--push] [ARGS]; ARGS go to eval_trm.py.
eval target *args:
    scripts/run.sh {{host}} {{remote_dir}} eval "$@"

# Fetch what git doesn't carry (submodules, env, dataset, checkpoint): just setup local|remote [--push]
setup target *args:
    scripts/run.sh {{host}} {{remote_dir}} setup "$@"

# Remove remote run worktrees whose tmux session has ended. Results are kept.
prune:
    scripts/run.sh {{host}} {{remote_dir}} prune remote

# Copy trm/results back from the GPU box.
pull:
    rsync -az {{host}}:{{remote_dir}}/trm/results/ trm/results/

# Follow a remote run's log; defaults to the latest.
logs run="":
    ssh -t {{host}} "cd {{remote_dir}}/trm/results/logs && tail -n 100 -f {{ if run == "" { "\\$(ls -t *.log | head -1)" } else { run + ".log" } }}"

attach run:
    ssh -t {{host}} tmux attach -t {{run}}

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 OUTPUT_ROOT WORKLOAD [WORKLOAD ...]" >&2
    exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$1"
shift
parallelism="${PAPER_AGENT_PARALLEL:-2}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    stty -echo
    IFS= read -r OPENAI_API_KEY
    stty echo
    export OPENAI_API_KEY
fi

cd "$root_dir"

pids=()
names=()

wait_batch() {
    local failed=0
    local index
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[paper-agents-parallel] ${names[$index]} complete"
        else
            echo "[paper-agents-parallel] ${names[$index]} failed" >&2
            failed=1
        fi
    done
    pids=()
    names=()
    return "$failed"
}

for workload in "$@"; do
    .venv-agent/bin/python scripts/run_paper_agents.py \
        --output-root "$output_root" \
        --iterations 50 \
        --il-budget 10.0 \
        --workloads "$workload" &
    pids+=("$!")
    names+=("$workload")
    if [[ ${#pids[@]} -ge $parallelism ]]; then
        wait_batch
    fi
done

if [[ ${#pids[@]} -gt 0 ]]; then
    wait_batch
fi

#!/usr/bin/env bash
# Load the Cloudmist (OpenAI-compatible) credential + endpoint for photon_tuner.
#
#   source scripts/env/activate_cloudmist.sh
#   .venv-agent/bin/python main.py --provider cloudmist --base-url "$OPENAI_BASE_URL" --model "$EPIC_LLM_MODEL" --workload vit_tiny --iterations 1
#
# The key itself never lives in the repo tree as a tracked file: it is read from
# ``.env`` at the project root (gitignored, mode 600). If you prefer a shell
# export, set OPENAI_API_KEY yourself and this script will keep it.
_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_ENV_FILE="$_PROJECT_ROOT/.env"

if [ -f "$_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$_ENV_FILE"
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[activate_cloudmist] OPENAI_API_KEY is empty — create .env first:" >&2
    echo "    printf \"export OPENAI_API_KEY='sk-...'\\n\" > .env && chmod 600 .env" >&2
    return 1 2>/dev/null || exit 1
fi

export OPENAI_BASE_URL="https://v2.cloudmist.cloud/v1"
export EPIC_LLM_PROVIDER="${EPIC_LLM_PROVIDER:-cloudmist}"
export EPIC_LLM_MODEL="${EPIC_LLM_MODEL:-deepseek-v4.1-flash}"

echo "[activate_cloudmist] base_url=$OPENAI_BASE_URL model=$EPIC_LLM_MODEL (credential loaded)"

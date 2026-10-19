"""Optional live smoke test: one real round-trip to the configured LLM.

Not part of the default unittest discovery (filename does not match
``test_*.py``); run manually when network/API access is available::

    python3 tests/live_llm_smoke.py

Uses the provider/model from ``configs/llm.json`` and the API key resolved from
the environment or ``~/.dsh/.credentials.yaml``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.common.io import read_json
from photon_tuner.llm.config import LLMConfig
from photon_tuner.llm.clients import create_llm_client


def main() -> int:
    payload = read_json(Path(__file__).resolve().parents[1] / "configs" / "llm.json")
    config = LLMConfig(**{key: payload[key] for key in payload if key in LLMConfig.__dataclass_fields__})
    client = create_llm_client(config)
    print(f"provider={config.provider} model={client.model} base_url={client.base_url}")

    response = client.complete_json_messages(
        [
            {"role": "system", "content": "Reply with strict JSON only."},
            {"role": "user", "content": '{"task":"ping","instruction":"Return {\"ok\": true, \"echo\": \"pong\"}"}'},
        ]
    )
    print("response:", response)
    assert response.get("ok") is True, f"unexpected response: {response}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

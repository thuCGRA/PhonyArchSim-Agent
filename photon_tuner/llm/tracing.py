"""Per-call LLM tracing: persist every request/response pair plus readable digests.

The recorder writes, into the directory it is pointed at:

* ``prompt_calls.json`` -- one slim entry per LLM call (no payload bodies),
* ``raw_json/*.json`` -- the exact request and response bodies,
* ``system_prompts.md`` -- deduplicated system messages,
* ``view_md/*.md`` -- one readable dialogue per agent role,
* ``README.md`` -- an index over the raw calls.

Tracing the raw bodies makes a run auditable: what the model was actually asked
and answered can be checked afterwards instead of trusting a summary produced
after the fact.
"""

from __future__ import annotations

import copy
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


#: One dialogue file per agent role, in execution order.
DIALOGUE_GROUPS = [
    {
        "key": "model_level",
        "title": "Model-Level Agent Dialogue",
        "filename": "view_md/model_level_agent_dialogue.md",
        "final_task": "model_level_analysis",
    },
    {
        "key": "layer_level",
        "title": "Layer-Level Agent Dialogue",
        "filename": "view_md/layer_level_agent_dialogue.md",
        "final_task": "layer_level_analysis",
    },
    {
        "key": "optical_level",
        "title": "Optical-Level Agent Dialogue",
        "filename": "view_md/optical_level_agent_dialogue.md",
        "final_task": "optical_level_analysis",
    },
    {
        "key": "solution_generation",
        "title": "Solution-Generation Agent Dialogue",
        "filename": "view_md/solution_generation_agent_dialogue.md",
        "final_task": "solution_generation",
    },
]

#: Maps a final-analysis task name back to its agent role for dialogue routing.
FINAL_TASK_TO_AGENT = {
    "model_level_analysis": "model_level",
    "layer_level_analysis": "layer_level",
    "optical_level_analysis": "optical_level",
    "solution_generation": "solution_generation",
}

#: Record keys that must NOT be persisted in the slim index (too large/verbose).
_SLIM_EXCLUDED_KEYS = frozenset(
    {"raw_input", "raw_output", "system_prompt", "response_content", "started_at_monotonic"}
)


class LLMTraceRecorder:
    """Incrementally persist raw LLM request/response data and readable dialogues."""

    def __init__(self, output_dir: Path, index_width: int = 4) -> None:
        self.output_dir = Path(output_dir)
        self.index_width = index_width
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "raw_json").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "view_md").mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = self._restore_previous_calls()
        self.resume_generation = self._resume_generation_for(self.records)

    # -- public hooks (called by the LLM clients) --------------------------- #

    def begin_call(self, request_body: Dict[str, Any]) -> int:
        """Snapshot an outgoing request, append a pending record, return its index."""
        index = len(self.records)
        snapshot = copy.deepcopy(request_body)
        messages = _extract_messages(snapshot)
        context = self._context_for(messages)
        task = str(context.get("task", "unknown"))
        agent_name = str(context.get("agent_name") or FINAL_TASK_TO_AGENT.get(task) or "unknown")
        display_index = f"{index:0{self.index_width}d}"
        stem = task if task.startswith(agent_name) else f"{agent_name}_{task}"
        prefix = self._file_prefix(display_index)
        record = {
            "index": index,
            "display_index": display_index,
            "resume_generation": self.resume_generation,
            "agent_name": agent_name,
            "task": task,
            "status": "pending",
            "started_at": _utc_timestamp(),
            "started_at_monotonic": time.monotonic(),
            "ended_at": None,
            "duration_s": None,
            "message_count": len(messages),
            "system_prompt": _join_system_messages(messages),
            "raw_input": snapshot,
            "raw_output": None,
            "response_content": None,
            "error": None,
            "input_json": f"raw_json/{prefix}_input_{stem}.json",
            "output_json": f"raw_json/{prefix}_output_{stem}.json",
        }
        self.records.append(record)
        self._dump_json(self.output_dir / record["input_json"], snapshot)
        self._refresh_derived_files()
        return index

    def end_call(
        self,
        call_index: int,
        raw_output: Dict[str, Any],
        response_content: Optional[str] = None,
        error: Optional[BaseException | str] = None,
    ) -> None:
        """Attach a response (or failure) to the pending record opened by ``begin_call``."""
        record = self.records[call_index]
        snapshot = copy.deepcopy(raw_output)
        started = record.get("started_at_monotonic")
        if isinstance(started, (int, float)):
            record["duration_s"] = round(time.monotonic() - float(started), 6)
        record["ended_at"] = _utc_timestamp()
        record["raw_output"] = snapshot
        record["usage"] = _usage_from(snapshot)
        record["model"] = _model_from(snapshot)
        record["response_content"] = response_content
        if error is None:
            record["status"] = "ok"
        else:
            record["status"] = "error"
            record["error"] = _describe_error(error)
        self._dump_json(self.output_dir / record["output_json"], snapshot)
        self._refresh_derived_files()

    # -- context / file-naming helpers -------------------------------------- #

    def _context_for(self, messages: Sequence[Any]) -> Dict[str, Any]:
        """Pick the task/agent payload that best describes this call."""
        parsed: List[Dict[str, Any]] = []
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            payload = _parse_json_content(message.get("content"))
            if isinstance(payload, dict):
                parsed.append(payload)
        parsed.reverse()

        last = parsed[-1] if parsed else {}
        last_task = last.get("task")
        if last_task and last_task != "tool_observation":
            context = dict(last)
            context["message_count"] = len(messages)
            return context

        react = next((payload for payload in parsed if payload.get("task") == "react_tool_use"), None)
        if react is not None:
            context = dict(react)
            context["message_count"] = len(messages)
            context["latest_user_message"] = last
            return context

        for payload in reversed(parsed):
            if payload.get("task"):
                return payload
        return {}

    def _file_prefix(self, display_index: str) -> str:
        if self.resume_generation <= 0:
            return display_index
        return f"{display_index}_resume{self.resume_generation:02d}"

    # -- persistence / rebuild of derived files ----------------------------- #

    def _refresh_derived_files(self) -> None:
        slim = [self._slim(record) for record in self.records]
        self._dump_json(self.output_dir / "prompt_calls.json", slim)
        self._write_system_prompt_summary()
        self._write_dialogues()
        self._write_index(slim)

    def _slim(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in record.items() if key not in _SLIM_EXCLUDED_KEYS}

    def _write_system_prompt_summary(self) -> None:
        seen: Dict[str, str] = {}
        lines = ["# LLM System Prompts", ""]
        for record in self.records:
            prompt = record["system_prompt"]
            if not prompt or prompt in seen:
                continue
            seen[prompt] = record["display_index"]
            lines.extend([f"## call {record['display_index']}: {record['agent_name']} / {record['task']}", ""])
            _append_rendered_blocks(lines, prompt)
        self._write_text(self.output_dir / "system_prompts.md", "\n".join(lines).rstrip() + "\n")

    def _write_dialogues(self) -> None:
        for group in DIALOGUE_GROUPS:
            members = [
                record
                for record in self.records
                if record["agent_name"] == group["key"] or record["task"] == group["final_task"]
            ]
            if not members:
                continue
            self._write_text(
                self.output_dir / group["filename"],
                render_agent_dialogue_markdown(
                    title=group["title"], final_task=group["final_task"], records=members
                ),
            )

    def _write_index(self, slim: List[Dict[str, Any]]) -> None:
        lines = [
            "# LLM Trace",
            "",
            "`raw_json/` stores raw request/response JSON for every LLM call. "
            "`view_md/` stores readable per-agent dialogue expansions.",
            "",
            "## Readable Dialogues",
            "",
            "| agent | dialogue |",
            "|---|---|",
        ]
        for group in DIALOGUE_GROUPS:
            lines.append(f"| {group['key']} | `{group['filename']}` |")
        lines.extend(["", "## Raw Calls", "",
                      "| call | status | agent | task | duration_s | raw input | raw output |",
                      "|---:|---|---|---|---:|---|---|"])
        for record in slim:
            lines.append(
                f"| {record['display_index']} | {record['status']} | {record['agent_name']} | {record['task']} | "
                f"{_format_duration_cell(record.get('duration_s'))} | "
                f"`{record['input_json']}` | `{record['output_json']}` |"
            )
        self._write_text(self.output_dir / "README.md", "\n".join(lines) + "\n")

    # -- resume support ----------------------------------------------------- #

    def _restore_previous_calls(self) -> List[Dict[str, Any]]:
        index_path = self.output_dir / "prompt_calls.json"
        if not index_path.exists():
            return []
        try:
            slim = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(slim, list):
            return []
        restored: List[Dict[str, Any]] = []
        for item in slim:
            if not isinstance(item, dict):
                continue
            raw_input = self._read_json(str(item.get("input_json", "")))
            raw_output = self._read_json(str(item.get("output_json", "")))
            messages = _extract_messages(raw_input) if isinstance(raw_input, dict) else []
            content = None
            if isinstance(raw_output, dict) and isinstance(raw_output.get("content"), str):
                content = raw_output["content"]
            restored.append(
                {
                    **item,
                    "started_at_monotonic": None,
                    "system_prompt": _join_system_messages(messages),
                    "raw_input": raw_input if isinstance(raw_input, dict) else {},
                    "raw_output": raw_output if isinstance(raw_output, dict) else None,
                    "response_content": content,
                }
            )
        return restored

    def _resume_generation_for(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0
        highest = 0
        unfinished = False
        for record in records:
            try:
                highest = max(highest, int(record.get("resume_generation", 0)))
            except (TypeError, ValueError):
                pass
            if record.get("status") == "pending":
                unfinished = True
        return highest + 1 if unfinished else highest

    # -- low-level IO -------------------------------------------------------- #

    def _read_json(self, relative: str) -> Any:
        if not relative:
            return None
        try:
            return json.loads((self.output_dir / relative).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _dump_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def render_agent_dialogue_markdown(
    title: str,
    final_task: str,
    records: List[Dict[str, Any]],
) -> str:
    """Render one agent's calls as a readable markdown dialogue file."""
    completed = [record for record in records if record["task"] == final_task or record["status"] == "error"]
    if not completed:
        completed = [records[-1]]

    lines = [
        f"# {title}",
        "",
        "## Calls Included",
        "",
        "| call | status | task | raw input | raw output |",
        "|---:|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['display_index']} | {record['status']} | {record['task']} | "
            f"`{record['input_json']}` | `{record['output_json']}` |"
        )

    for number, record in enumerate(completed, start=1):
        messages = _extract_messages(record["raw_input"])
        if not isinstance(messages, list):
            messages = []
        lines.extend([
            "",
            f"## Dialogue {number}: call {record['display_index']}",
            "",
            f"- task: `{record['task']}`",
            f"- status: `{record['status']}`",
            f"- message_count: `{len(messages)}`",
            f"- raw_input: `{record['input_json']}`",
            f"- raw_output: `{record['output_json']}`",
            "",
        ])
        if record.get("error"):
            lines.extend(["### Error", ""])
            _append_json_block(lines, record["error"])
        lines.extend(["### Dialogue Messages", ""])
        for position, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            lines.extend([f"#### Message {position:02d}: {message.get('role', 'unknown')}", ""])
            _append_rendered_blocks(lines, str(message.get("content", "")))
        lines.extend([f"#### Message {len(messages):02d}: assistant output", ""])
        answer = record.get("response_content")
        if isinstance(answer, str) and answer.strip():
            _append_rendered_blocks(lines, answer)
        else:
            _append_json_block(lines, record.get("raw_output"))
    return "\n".join(lines).rstrip() + "\n"


# -- markdown rendering helpers -------------------------------------------- #


def _append_rendered_blocks(lines: List[str], content: str) -> None:
    if "```" in content:
        _append_text_block(lines, content)
        return
    for kind, value in _split_embedded_json(content):
        if kind == "json":
            _append_json_block(lines, value)
        else:
            text = str(value).strip()
            if text:
                _append_text_block(lines, text)


def _append_text_block(lines: List[str], text: str) -> None:
    body = str(text).strip()
    if body:
        fence = _fence_for(body)
        lines.extend([f"{fence}text", body, fence, ""])


def _append_json_block(lines: List[str], payload: Any) -> None:
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    fence = _fence_for(body)
    lines.extend([f"{fence}json", body, fence, ""])


def _split_embedded_json(content: str) -> List[tuple[str, Any]]:
    decoder = json.JSONDecoder()
    pieces: List[tuple[str, Any]] = []
    plain_start = 0
    cursor = 0
    while cursor < len(content):
        if content[cursor] not in "{[":
            cursor += 1
            continue
        try:
            parsed, end = decoder.raw_decode(content[cursor:])
        except json.JSONDecodeError:
            cursor += 1
            continue
        if plain_start < cursor:
            pieces.append(("text", content[plain_start:cursor]))
        pieces.append(("json", parsed))
        cursor += end
        plain_start = cursor
    if plain_start < len(content):
        pieces.append(("text", content[plain_start:]))
    return pieces or [("text", content)]


def _fence_for(content: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


# -- small utilities -------------------------------------------------------- #


def _extract_messages(raw_input: Dict[str, Any]) -> List[Any]:
    messages = raw_input.get("messages")
    if messages is None and isinstance(raw_input.get("request_body"), dict):
        messages = raw_input["request_body"].get("messages")
    return messages if isinstance(messages, list) else []


def _join_system_messages(messages: Sequence[Any]) -> str:
    return "\n\n".join(
        message["content"]
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    )


def _parse_json_content(content: Any) -> Any:
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _usage_from(raw_output: Dict[str, Any]) -> Dict[str, Any]:
    response = raw_output.get("response") if isinstance(raw_output, dict) else None
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage")
    return copy.deepcopy(usage) if isinstance(usage, dict) else {}


def _model_from(raw_output: Dict[str, Any]) -> str:
    """Return the LLM model name from a raw response body (for auditability)."""
    response = raw_output.get("response") if isinstance(raw_output, dict) else None
    if not isinstance(response, dict):
        return ""
    model = response.get("model")
    return str(model) if model else ""


def _describe_error(error: BaseException | str) -> Dict[str, str]:
    if isinstance(error, BaseException):
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": "Error", "message": str(error)}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _format_duration_cell(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"

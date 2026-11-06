"""检查点快照契约的测试。

``snapshot`` / ``restore`` 是父 LangGraph 节点将 ``IterationContext`` 持久化到
``SqliteSaver`` 所用的序列化边界。带版本号的快照必须无损往返，并拒绝（给出明确
提示）加载不兼容代码版本写出的快照。
"""

import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from photon_tuner.core.pareto import ParetoArchive
from photon_tuner.core.schemas import EvaluationResult
from photon_tuner.runtime.state import (
    SNAPSHOT_SCHEMA_VERSION,
    IterationContext,
    restore,
    snapshot,
)


@dataclass
class _FakeTuningState:
    """``TuningState`` 的最小替身（避免导入 langgraph）。"""

    best: Any = None
    history: List[dict] = field(default_factory=list)
    iteration: int = 0


def _make_ctx() -> IterationContext:
    current = EvaluationResult.from_dict(
        {"hardware": {"arch": {"power": 5}}, "metrics": {"loss": 1.0}, "iteration": 0}
    )
    return IterationContext(
        iteration=1,
        output_dir=Path("/tmp/iter"),
        current=current,
        state=_FakeTuningState(),
        forbidden={"fp1"},
        pareto=ParetoArchive(),
        search_strategy="joint",
        state_strategy="best",
    )


class TestSnapshotSchemaVersion(unittest.TestCase):
    def test_snapshot_carries_version(self):
        data = snapshot(_make_ctx())
        self.assertEqual(data["schema_version"], SNAPSHOT_SCHEMA_VERSION)

    def test_snapshot_roundtrip(self):
        ctx = _make_ctx()
        restored = restore(snapshot(ctx), _FakeTuningState)
        self.assertEqual(restored.iteration, ctx.iteration)
        self.assertEqual(restored.forbidden, ctx.forbidden)
        self.assertEqual(restored.current.metrics["loss"], 1.0)

    def test_restore_rejects_missing_version(self):
        data = snapshot(_make_ctx())
        del data["schema_version"]
        with self.assertRaises(ValueError):
            restore(data, _FakeTuningState)

    def test_restore_rejects_mismatched_version(self):
        data = snapshot(_make_ctx())
        data["schema_version"] = SNAPSHOT_SCHEMA_VERSION + 99
        with self.assertRaises(ValueError):
            restore(data, _FakeTuningState)


if __name__ == "__main__":
    unittest.main()

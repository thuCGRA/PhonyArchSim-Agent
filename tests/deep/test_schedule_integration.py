"""RAG 驱动调度的集成测试：真实编排路径下模式确实改变"哪些阶段下场"。

与 ``tests/test_scheduler.py``（纯函数契约）互补：这里证明调度决策真的传递到了
编排器的阶段跳过、证据偏移与落盘产物，而不只是算出一个好看的字典。
"""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.llm.config import LLMConfig
from photon_tuner.rag.vector_store import HistoryVectorStore
from photon_tuner.runtime.orchestrator import DeepAgentTuner
from photon_tuner.runtime.scheduler import ScheduleSignals, decide
from photon_tuner.simulators.simphony import SimphonyAdapter
from photon_tuner.testing.fake_model import FakeToolCallingModel

ROOT = Path(__file__).resolve().parents[2]


def _build(tmp, **kwargs):
    ss = read_json(ROOT / "configs" / "search_space.json")
    adapter = SimphonyAdapter(
        simphony_root=ROOT / "photon_tuner" / "simulators" / "simphony",
        output_root=Path(tmp),
        search_space=ss,
        mode="analytic",
        workload_name="vit_tiny",
    )
    tuner = DeepAgentTuner(
        adapter=adapter,
        llm_config=LLMConfig(provider="mock"),
        devicelib_root=str(ROOT / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"),
        output_root=Path(tmp) / "runs",
        search_space=ss,
        store=HistoryVectorStore(Path(tmp) / "history.db"),
        model=FakeToolCallingModel(),
        **kwargs,
    )
    hardware = normalize_hardware_to_design_space(default_hardware(), ss)
    return tuner, hardware


class TestScheduleIntegration(unittest.TestCase):
    def test_adaptive_schedule_skips_stages_and_records_trajectory(self):
        """冷启动跑完整串行链；先验成熟后进入错峰流水线或收敛聚焦。"""
        with tempfile.TemporaryDirectory() as tmp:
            tuner, hw = _build(tmp, schedule_strategy="rag")
            summary = tuner.tune(initial_hardware=hw, max_iterations=3)

            self.assertEqual(summary["schedule_strategy"], "rag")
            history = summary["schedule_history"]
            self.assertEqual(len(history), 3)
            # 每轮都要有可审计的决策记录（模式 + 触发信号快照）。
            for item in history:
                self.assertIn(item["mode"], {"serial", "pipeline", "focus"})
                self.assertIn("signals", item)
                self.assertEqual(set(item["signals"]["pipeline_evidence_ages"]),
                                 {"architecture", "topology", "device"})

            run_dir = Path(summary["run_dir"])
            summary_record = read_json(run_dir / "summary.json")
            self.assertEqual(summary_record["best"], summary["best"])
            self.assertEqual(summary_record["best_hardware"], summary["best_hardware"])
            recorded = summary_record["scheduler_config"]
            self.assertEqual(recorded["improved_delta_pct"], 5.0)
            self.assertEqual(recorded["convergence_window"], 10)
            self.assertEqual(recorded["convergence_delta"], 0.10)
            self.assertIsNone(recorded["focus_exit_delta"])
            for i in (1, 2, 3):
                self.assertTrue((run_dir / f"iter_{i:03d}" / "analysis" / "schedule.json").is_file())

            # 第 1 轮记忆为空（ρ=0）→ serial：三层均按本轮证据顺序执行。
            first = read_json(run_dir / "iter_001" / "analysis" / "schedule.json")
            self.assertEqual(first["mode"], "serial")
            self.assertEqual(first["stages_to_run"], ["architecture", "topology", "device"])

            # 流水线运行三层；聚焦模式只运行指定层。
            modes = [item["mode"] for item in history]
            if "pipeline" in modes:
                idx = modes.index("pipeline") + 1
                decision = read_json(run_dir / f"iter_{idx:03d}" / "analysis" / "schedule.json")
                self.assertEqual(decision["stages_to_run"], ["architecture", "topology", "device"])
                self.assertEqual(decision["evidence_offsets"],
                                 {"architecture": 0, "topology": 1, "device": 2})
            if "focus" in modes:
                idx = modes.index("focus") + 1
                self.assertIn("pipeline", modes[:idx - 1],
                              "adaptive scheduling must establish Pipeline before Focus")
                analysis = run_dir / f"iter_{idx:03d}" / "analysis"
                skipped = sorted(p.name for p in analysis.glob("*_skipped.json"))
                self.assertTrue(skipped, "聚焦模式必须留下被跳过阶段的标记")
                decision = read_json(analysis / "schedule.json")
                for stage in ("architecture", "topology", "device"):
                    if stage not in decision["stages_to_run"]:
                        self.assertIn(f"{stage}_skipped.json", skipped)

    def test_device_evidence_controls_pipeline_readiness(self):
        """Missing or stale device history must trigger a complete refresh."""
        with tempfile.TemporaryDirectory() as tmp:
            tuner, _ = _build(tmp, schedule_strategy="rag")
            for device_round, expected_age, expected_mode in (
                (None, None, "serial"), (5, None, "serial"),
                (2, 3, "serial"), (3, 2, "pipeline"), (4, 1, "pipeline"),
            ):
                with self.subTest(device_round=device_round):
                    ctx = SimpleNamespace(iteration=5, evidence_history={
                        "architecture": {"4": {"_iteration": 4}},
                        "topology": {"3": {"_iteration": 3}},
                        "device": {} if device_round is None else {
                            str(device_round): {"_iteration": device_round},
                        },
                    })
                    ages = tuner._pipeline_evidence_ages(ctx)
                    self.assertEqual(ages, {"architecture": 1, "topology": 2,
                                            "device": expected_age})
                    decision = decide(ScheduleSignals(
                        iteration=5, improved_case_count=3,
                        improved_case_layers=["architecture", "topology", "device"],
                        objective_history=[100, 70, 49, 34.3],
                        pipeline_evidence_ages=ages,
                    ))
                    self.assertEqual(decision.mode, expected_mode)

    def test_fixed_schedule_never_skips_analysis_stages(self):
        """消融对照：固定 serial 模式下三层每轮都跑，没有任何跳过标记。"""
        with tempfile.TemporaryDirectory() as tmp:
            tuner, hw = _build(tmp, schedule_strategy="fixed", fixed_schedule_mode="serial")
            summary = tuner.tune(initial_hardware=hw, max_iterations=2)
            run_dir = Path(summary["run_dir"])
            for i in (1, 2):
                analysis = run_dir / f"iter_{i:03d}" / "analysis"
                self.assertEqual(list(analysis.glob("*_skipped.json")), [])
                self.assertEqual(read_json(analysis / "schedule.json")["mode"], "serial")
            self.assertEqual({item["mode"] for item in summary["schedule_history"]}, {"serial"})

    def test_skipped_stage_records_real_evidence_age(self):
        """被跳过阶段的 ``evidence_age`` 必须是**真实**轮差，不能恒为 1。

        这是一个真实踩过的坑：证据缓存曾漏存来源轮次，于是年龄恒被算成 1，
        而实际可能用了更旧的诊断——"容忍 1–2 轮陈旧"的声明就变成了空话。
        """
        with tempfile.TemporaryDirectory() as tmp:
            tuner, hw = _build(tmp, schedule_strategy="fixed", fixed_schedule_mode="focus")
            summary = tuner.tune(initial_hardware=hw, max_iterations=4)
            run_dir = Path(summary["run_dir"])

            ages = []
            for marker in sorted(run_dir.glob("iter_*/analysis/*_skipped.json")):
                payload = read_json(marker)
                # 早于任何证据的轮次（iter 1）允许 age 为 None。
                if payload.get("evidence_age") is not None:
                    ages.append((payload["stage"], payload["iteration"], payload["evidence_age"]))
                    self.assertGreaterEqual(payload["evidence_age"], 1, payload)
                    self.assertIsNotNone(payload.get("evidence_iteration"), payload)

            self.assertTrue(ages, "聚焦模式下应当出现被跳过的阶段")
            # 证据来源必须真的被记录：轮到其他层时，证据年龄会递增。
            self.assertTrue(
                any(age >= 2 for _, _, age in ages),
                f"未观察到 age≥2 的证据（说明年龄被写死为 1）：{ages}",
            )

    def test_candidate_payload_carries_recent_attempts(self):
        """候选智能体必须能看到最近几轮的结果与步长。

        诊断真实失败模式时发现：载荷里只有当前点，智能体看不到"前三次都把面积
        放大一倍且都变差"，于是持续提出跨 5–6 参数的大改；而传统基线只做单参数
        原子移动——在乘性目标下这是决定性的差异。
        """
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            tuner, _ = _build(tmp, schedule_strategy="fixed")
            ctx = SimpleNamespace(
                state=SimpleNamespace(history=[
                    {"iteration": 1, "is_best": False,
                     "metrics": {"objective": 100.0, "latency_ms": 1.0,
                                 "energy_mj": 2.0, "area_mm2": 50.0},
                     "action": {"label": "initial"}},
                    {"iteration": 2, "is_best": True,
                     "metrics": {"objective": 80.0, "latency_ms": 1.0,
                                 "energy_mj": 2.0, "area_mm2": 40.0},
                     "action": {"name": "step_hardware_parameter"}},
                ])
            )
            attempts = tuner._recent_attempts(ctx)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["iteration"], 1)
            self.assertIsNone(attempts[0]["delta_pct_vs_previous"])
            # 第二项必须给出相对上一轮的变化（100 -> 80 = +20%）
            self.assertAlmostEqual(attempts[1]["delta_pct_vs_previous"], 20.0, places=6)
            self.assertTrue(attempts[1]["accepted"])
            self.assertEqual(attempts[1]["area_mm2"], 40.0)
            # 乘积目标的对数分解：这一轮面积与能耗都没变，只有目标变了——
            # 三项 log_delta 必须都给出数值，且 dominant_factor 指向变动最大的项。
            self.assertEqual(set(attempts[1]["log_delta"]), {"latency_ms", "energy_mj", "area_mm2"})
            self.assertAlmostEqual(attempts[1]["log_delta"]["area_mm2"], math.log(0.8), places=3)

    def test_history_records_what_was_applied(self):
        """历史记录必须带"改了什么"。

        只记"结果变差了"而不记"改了什么"，候选智能体无法从失败中学习：
        ``recent_attempts[*].edit`` 曾恒为 None。
        """
        with tempfile.TemporaryDirectory() as tmp:
            tuner, hw = _build(tmp, schedule_strategy="fixed")
            summary = tuner.tune(initial_hardware=hw, max_iterations=2)
            history = summary["history"]
            self.assertTrue(history)
            self.assertIn("applied", history[-1])
            # 至少有一轮真的应用了编辑（假模型总会提出至少一个动作）。
            self.assertTrue(
                any(entry.get("applied") for entry in history),
                "没有任何一轮记录到已应用的编辑",
            )

    def test_candidate_agent_receives_recent_attempts_and_scope(self):
        """端到端：候选智能体实际收到的载荷里必须有 recent_attempts。

        只测 ``_recent_attempts()`` 的返回值不够——历史上出过"函数算对了但没人
        调用"的缺口。这里拦截 ``_invoke``，直接检查真正发给模型的那个字典。
        """
        with tempfile.TemporaryDirectory() as tmp:
            tuner, hw = _build(tmp, schedule_strategy="fixed")
            captured = []
            original = tuner._invoke

            def spy(agent, output_dir, payload, output_model):
                if getattr(agent, "name", "") == "candidate_agent":
                    captured.append(dict(payload))
                return original(agent, output_dir, payload, output_model)

            tuner._invoke = spy
            tuner.tune(initial_hardware=hw, max_iterations=2)

            self.assertTrue(captured, "候选阶段没有被调用")
            for payload in captured:
                self.assertIn("recent_attempts", payload)

    def test_edit_granularity_limit_defers_extra_edits(self):
        """``--max-edits 1``：一次评估只承载一个编辑假设，其余推迟。

        多参数同时改在乘积目标下会相互抵消且无法归因；把它限成单步后，
        agent 与基线才在同一粒度上比较"谁更会选方向"。
        """
        with tempfile.TemporaryDirectory() as tmp:
            tuner, hw = _build(tmp, schedule_strategy="fixed", max_edits_per_proposal=1)
            summary = tuner.tune(initial_hardware=hw, max_iterations=3)
            run_dir = Path(summary["run_dir"])

            applied_counts = []
            for proposal in sorted(run_dir.glob("iter_*/analysis/candidate_proposal.json")):
                payload = read_json(proposal)
                edits = len(payload.get("applied_parameters") or []) + len(
                    payload.get("applied_link_device_types") or []
                )
                applied_counts.append(edits)

            self.assertTrue(applied_counts)
            self.assertLessEqual(max(applied_counts), 1, applied_counts)
            # 被切掉的编辑必须出现在 deferred 里，而不是静默消失。
            self.assertTrue(
                any(read_json(p).get("deferred_edits")
                    for p in sorted(run_dir.glob("iter_*/analysis/candidate_proposal.json"))),
                "被推迟的编辑没有被记录",
            )

    def test_sequential_search_strategy_maps_to_fixed_serial(self):
        """``--search-strategy sequential`` 保持历史语义：固定串行、三层轮转。"""
        with tempfile.TemporaryDirectory() as tmp:
            tuner, hw = _build(tmp, schedule_strategy="fixed")
            summary = tuner.tune(initial_hardware=hw, max_iterations=3, search_strategy="sequential")
            self.assertEqual([item["mode"] for item in summary["schedule_history"]], ["serial"] * 3)
            self.assertEqual(
                [item["focus_layer"] for item in summary["schedule_history"]],
                ["architecture", "topology", "device"],
            )


if __name__ == "__main__":
    unittest.main()

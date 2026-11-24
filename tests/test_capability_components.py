import unittest

from photon_tuner.capabilities import build_capability_manifest, validate_agent_capability_surface
from photon_tuner.skills import (
    AtomicSkillSpec,
    SkillCategory,
    SkillRegistry,
    SkillRuntime,
    SkillStep,
    build_default_skill_registry,
)
from photon_tuner.toolbox.analysis_tools import _REGISTRY as ANALYSIS_REGISTRY
from photon_tuner.toolbox.catalog import (
    TOPOLOGY_TOOL_DESCRIPTIONS,
    ToolAccess,
    capability_manifest,
    tool_descriptors,
    validate_catalog,
)
from photon_tuner.toolbox.device_tools import _REGISTRY as DEVICE_REGISTRY
from photon_tuner.toolbox.support_tools import _REGISTRY as SUPPORT_REGISTRY
from photon_tuner.workflows import WorkflowRuntime, build_default_workflow_registry


class CapabilityComponentsTest(unittest.TestCase):
    def test_tool_catalog_covers_all_40_project_tools(self):
        runtime_names = (
            set(ANALYSIS_REGISTRY)
            | set(DEVICE_REGISTRY)
            | set(SUPPORT_REGISTRY)
            | set(TOPOLOGY_TOOL_DESCRIPTIONS)
        )
        descriptors = tool_descriptors()

        self.assertEqual(len(runtime_names), 40)
        self.assertEqual({item.name for item in descriptors}, runtime_names)
        self.assertEqual(sum(item.access == ToolAccess.WRITE for item in descriptors), 6)
        self.assertEqual(sum(item.access == ToolAccess.CONTROL for item in descriptors), 1)
        self.assertEqual(sum(item.access == ToolAccess.READ for item in descriptors), 33)
        self.assertEqual(sum(len(items) for items in capability_manifest().values()), 40)
        validate_catalog(runtime_names)

    def test_tools_and_atomic_skills_have_distinct_names_and_categories(self):
        tools = {item.name for item in tool_descriptors()}
        skills = build_default_skill_registry()

        self.assertEqual(len(skills.list()), 15)
        self.assertFalse(tools & {skill.name for skill in skills.list()})
        self.assertEqual(
            {skill.category for skill in skills.list()},
            {
                SkillCategory.DIAGNOSIS,
                SkillCategory.DECISION,
                SkillCategory.VERIFICATION,
                SkillCategory.LEARNING,
                SkillCategory.SCHEDULING,
            },
        )
        skills.validate_tools(tools)

    def test_atomic_skill_applies_domain_rule_after_tool_evidence(self):
        calls = []

        def invoke(name, arguments):
            calls.append((name, dict(arguments)))
            return {"energy_scope": {"memory_share": 0.72}}

        skill = build_default_skill_registry().get("assess_memory_energy_dominance")
        result = SkillRuntime(invoke).execute(skill, {"threshold": 0.6})

        self.assertEqual(calls, [("summarize_layer_ranks", {"top_layers": 1})])
        self.assertTrue(result.conclusion["is_memory_dominated"])
        self.assertEqual(result.conclusion["memory_share"], 0.72)
        self.assertIn("summarize_layer_ranks", result.evidence)

    def test_skill_registry_is_open_for_project_extensions(self):
        registry = SkillRegistry()
        extension = AtomicSkillSpec(
            name="assess_custom_metric",
            goal="判断自定义指标是否超过阈值",
            description="Project extension example.",
            category=SkillCategory.VERIFICATION,
            preconditions=("metric exists",),
            required_evidence=("custom metric",),
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object"},
            steps=(SkillStep("summarize_metrics"),),
            decision_rule="metric > threshold",
            failure_policy="fail closed",
            evaluator=lambda inputs, evidence: {"accepted": True},
        )

        registry.register(extension)

        self.assertIs(registry.get(extension.name), extension)
        self.assertEqual(registry.list(SkillCategory.VERIFICATION), [extension])

    def test_composite_workflow_branches_between_atomic_skills(self):
        calls = []

        def invoke(name, arguments):
            calls.append(name)
            if name == "validate_coupling":
                return {"consistent": True, "violations": []}
            if name == "summarize_design_identity":
                return {"fingerprint": "abc", "seen": False, "known_fingerprint_count": 3}
            raise AssertionError(name)

        skills = build_default_skill_registry()
        workflows = build_default_workflow_registry()
        workflows.validate_skills(skills)
        result = WorkflowRuntime(skills, SkillRuntime(invoke)).execute(
            workflows.get("candidate_preflight")
        )

        self.assertEqual(calls, ["validate_coupling", "summarize_design_identity"])
        self.assertEqual(result.terminal_node, "novelty")
        self.assertTrue(result.conclusions["novelty"]["novel"])

    def test_manifest_and_role_bindings_are_complete_and_read_only(self):
        validate_agent_capability_surface()
        manifest = build_capability_manifest()

        self.assertEqual(manifest["summary"]["tool_count"], 40)
        self.assertEqual(manifest["summary"]["agent_visible_tool_count"], 34)
        self.assertEqual(manifest["summary"]["atomic_skill_count"], 15)
        self.assertEqual(manifest["summary"]["workflow_count"], 3)
        self.assertEqual(set(manifest["role_bindings"]), {
            "bottleneck", "architecture", "topology", "device", "candidate", "evaluate",
        })


if __name__ == "__main__":
    unittest.main()

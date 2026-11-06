"""角色提示词的 golden 回归测试。

提示词位于 ``configs/prompts/prompts.yaml``，是构建角色智能体所依据的契约。这些
断言锁定每个提示词的结构骨架，使任何意外措辞改动若丢掉必需小节（Role / Mission /
grounding / output contract）都会大声失败。
"""

import unittest

from photon_tuner.agents.prompts import load_prompt

ROLE_NAMES = ("model", "architecture", "topology", "device", "solution", "evaluate")


class TestPromptGolden(unittest.TestCase):
    def test_all_roles_load_non_empty(self):
        for role in ROLE_NAMES:
            text = load_prompt(role)
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 100, role)

    def test_every_prompt_declares_role(self):
        for role in ROLE_NAMES:
            self.assertIn("Role:", load_prompt(role), role)

    def test_analysis_prompts_declare_mission(self):
        for role in ("model", "architecture", "topology", "device"):
            self.assertIn("Mission:", load_prompt(role), role)

    def test_model_and_architecture_declare_grounding(self):
        for role in ("model", "architecture"):
            self.assertIn("Grounding", load_prompt(role), role)

    def test_solution_prompt_locks_edit_contract(self):
        text = load_prompt("solution")
        for token in (
            "modify_hardware_parameter",
            "modify_topology_template",
            "modify_device_parameter",
            '"name"',
        ):
            self.assertIn(token, text)

    def test_topology_prompt_locks_optical_budget_contract(self):
        text = load_prompt("topology")
        for token in ("power_budget", "margin", "laser_output", "recommended_template"):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()

"""由原子技能组成的默认组合工作流。"""

from photon_tuner.workflows.core import (
    CompositeWorkflowSpec,
    WorkflowNode,
    WorkflowRegistry,
    WorkflowTransition,
)


DEFAULT_WORKFLOWS = (
    CompositeWorkflowSpec(
        name="candidate_preflight",
        goal="在消耗仿真预算前完成候选合法性和新颖性检查",
        entry="legality",
        nodes=(
            WorkflowNode(
                name="legality",
                skill_name="assess_candidate_legality",
                transitions=(WorkflowTransition("novelty", "legal", True),),
            ),
            WorkflowNode(name="novelty", skill_name="assess_design_novelty"),
        ),
    ),
    CompositeWorkflowSpec(
        name="optical_path_diagnosis",
        goal="形成光功率预算和主导损耗组件的联合诊断",
        entry="power_budget",
        nodes=(
            WorkflowNode(
                name="power_budget",
                skill_name="assess_optical_power_budget",
                default_target="dominant_loss",
            ),
            WorkflowNode(name="dominant_loss", skill_name="identify_dominant_optical_loss"),
        ),
    ),
    CompositeWorkflowSpec(
        name="memory_aware_search_review",
        goal="依据有效先验和目标收敛状态决定是否具备切换工作流的条件",
        entry="memory",
        nodes=(
            WorkflowNode(
                name="memory",
                skill_name="assess_memory_maturity",
                transitions=(WorkflowTransition("convergence", "mature", True),),
            ),
            WorkflowNode(name="convergence", skill_name="assess_objective_convergence"),
        ),
    ),
)


def build_default_workflow_registry() -> WorkflowRegistry:
    return WorkflowRegistry(DEFAULT_WORKFLOWS)

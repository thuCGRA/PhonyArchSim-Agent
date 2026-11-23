# Agent 能力清单

> 本文件由 `scripts/analysis/capability_inventory.py` 自动生成。

## 总览

| 层次 | 数量 | 说明 |
| --- | ---: | --- |
| 确定性工具 | 40 | 资源接口；写工具仅供宿主调用 |
| 原子技能 | 15 | 单一专业问题；不调用其他技能 |
| 组合工作流 | 3 | 技能间条件分支和循环 |

## 工具箱（按资源分类）

| 分类 | 数量 | 工具 |
| --- | ---: | --- |
| design_space | 2 | searchable_parameters、summarize_search_space |
| devices | 4 | inspect_device_config、list_device_candidates、summarize_device_frontier、summarize_device_provenance |
| editing | 4 | modify_device_parameter、modify_hardware_parameter、modify_topology_template、step_hardware_parameter |
| reports | 14 | describe_metric_units、inspect_layer_details、summarize_area_breakdown、summarize_device_area_breakdown、summarize_device_energy_breakdown、summarize_energy_breakdown、summarize_hardware_config、summarize_insertion_loss、summarize_insertion_loss_detail、summarize_layer_ranks、summarize_metrics、summarize_operator_groups、summarize_topology_evidence、summarize_workload_features |
| simulation | 2 | evaluate_design、preview_evaluation_schema |
| state_memory | 8 | commit_verified_case、compare_search_states、retrieve_similar_cases、select_analysis_base、summarize_case_memory、summarize_design_identity、summarize_pareto_frontier、summarize_schedule_signals |
| topology | 5 | inspect_light_path、inspect_topology_template、list_topology_templates、summarize_optical_structure、summarize_topology |
| validation | 1 | validate_coupling |

## 原子技能（按专业任务分类）

| 分类 | 数量 | 技能 |
| --- | ---: | --- |
| decision | 3 | assess_objective_improvement、choose_analysis_baseline、select_device_frontier_candidate |
| diagnosis | 6 | assess_memory_energy_dominance、assess_optical_power_budget、identify_dominant_area_device、identify_dominant_energy_device、identify_dominant_optical_loss、identify_latency_hotspot |
| learning | 2 | assess_case_reusability、assess_memory_maturity |
| scheduling | 1 | assess_objective_convergence |
| verification | 3 | assess_candidate_legality、assess_design_novelty、assess_parameter_effectiveness |

## 组合工作流

| 名称 | 单一工作流目标 | 技能节点 |
| --- | --- | --- |
| candidate_preflight | 在消耗仿真预算前完成候选合法性和新颖性检查 | assess_candidate_legality → assess_design_novelty |
| memory_aware_search_review | 依据有效先验和目标收敛状态决定是否具备切换工作流的条件 | assess_memory_maturity → assess_objective_convergence |
| optical_path_diagnosis | 形成光功率预算和主导损耗组件的联合诊断 | assess_optical_power_budget → identify_dominant_optical_loss |

## Agent 角色白名单

| 角色 | 能力数 | 能力 |
| --- | ---: | --- |
| bottleneck | 20 | summarize_hardware_config、summarize_metrics、summarize_energy_breakdown、summarize_area_breakdown、summarize_insertion_loss、summarize_layer_ranks、summarize_operator_groups、compare_search_states、select_analysis_base、describe_metric_units、summarize_workload_features、summarize_pareto_frontier、summarize_schedule_signals、assess_memory_energy_dominance、identify_latency_hotspot、identify_dominant_energy_device、identify_dominant_area_device、choose_analysis_baseline、assess_objective_improvement、assess_objective_convergence |
| architecture | 23 | inspect_layer_details、summarize_layer_ranks、summarize_operator_groups、summarize_hardware_config、summarize_metrics、compare_search_states、select_analysis_base、summarize_energy_breakdown、summarize_area_breakdown、summarize_device_energy_breakdown、summarize_device_area_breakdown、summarize_insertion_loss、searchable_parameters、summarize_search_space、summarize_workload_features、summarize_pareto_frontier、assess_memory_energy_dominance、identify_latency_hotspot、identify_dominant_energy_device、identify_dominant_area_device、choose_analysis_baseline、assess_objective_improvement、assess_parameter_effectiveness |
| topology | 18 | summarize_optical_structure、summarize_insertion_loss_detail、summarize_insertion_loss、summarize_device_energy_breakdown、summarize_device_area_breakdown、summarize_topology、inspect_light_path、list_topology_templates、inspect_topology_template、validate_coupling、inspect_device_config、summarize_topology_evidence、searchable_parameters、summarize_device_provenance、assess_optical_power_budget、identify_dominant_optical_loss、assess_parameter_effectiveness、assess_candidate_legality |
| device | 11 | list_device_candidates、inspect_device_config、summarize_device_frontier、summarize_device_energy_breakdown、summarize_device_area_breakdown、summarize_insertion_loss_detail、summarize_device_provenance、identify_dominant_energy_device、identify_dominant_area_device、identify_dominant_optical_loss、select_device_frontier_candidate |
| candidate | 18 | searchable_parameters、summarize_hardware_config、summarize_metrics、compare_search_states、list_topology_templates、inspect_topology_template、validate_coupling、summarize_design_identity、retrieve_similar_cases、summarize_case_memory、summarize_pareto_frontier、assess_objective_improvement、assess_parameter_effectiveness、assess_candidate_legality、assess_design_novelty、assess_case_reusability、assess_memory_maturity、assess_objective_convergence |
| evaluate | 45 | summarize_hardware_config、summarize_metrics、summarize_energy_breakdown、summarize_area_breakdown、summarize_insertion_loss、summarize_insertion_loss_detail、summarize_layer_ranks、summarize_operator_groups、summarize_optical_structure、summarize_device_energy_breakdown、summarize_device_area_breakdown、summarize_topology、inspect_light_path、list_topology_templates、inspect_topology_template、validate_coupling、list_device_candidates、inspect_device_config、compare_search_states、select_analysis_base、summarize_search_space、describe_metric_units、summarize_workload_features、summarize_device_provenance、preview_evaluation_schema、summarize_design_identity、retrieve_similar_cases、summarize_case_memory、summarize_pareto_frontier、summarize_schedule_signals、assess_memory_energy_dominance、identify_latency_hotspot、identify_dominant_energy_device、identify_dominant_area_device、assess_optical_power_budget、identify_dominant_optical_loss、choose_analysis_baseline、assess_objective_improvement、select_device_frontier_candidate、assess_parameter_effectiveness、assess_candidate_legality、assess_design_novelty、assess_case_reusability、assess_memory_maturity、assess_objective_convergence |

"""搜索空间有效性验证（确定性消融证据）。

验证流程是一个确定性框架（不涉及 LLM）：对每个 (参数, 候选值, 工作负载) 组合，
它会扰动基线硬件、注入 SimPhony、读回注入后的子架构、进行仿真，并记录哪些
指标发生了变化。原始证据由 :class:`~photon_tuner.evidence.store.EvidenceStore`
持久化，供后续解读流程（如 RAG 增强）读取。
"""

from photon_tuner.evidence.store import EvidenceStore
from photon_tuner.evidence.probe import SearchSpaceProbe

__all__ = ["EvidenceStore", "SearchSpaceProbe"]

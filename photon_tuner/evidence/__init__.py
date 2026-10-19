"""Search-space validity verification (deterministic ablation evidence).

The verification loop is a *deterministic* harness (no LLM): for every
(parameter, candidate value, workload) it perturbs the baseline hardware,
injects it into SimPhony, reads back the injected sub-architecture, simulates,
and records which metrics moved.  The intelligence lives in the separate
:class:`~photon_tuner.agents.evidence.EvidenceAnalystAgent`, which reads the
raw evidence afterwards.
"""

from photon_tuner.evidence.store import EvidenceStore
from photon_tuner.evidence.probe import SearchSpaceProbe

__all__ = ["EvidenceStore", "SearchSpaceProbe"]

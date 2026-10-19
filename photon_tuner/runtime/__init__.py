"""Runtime for the photonic multi-agent DSE framework.

Two layers live together here:

- the *agent engine* (``react_loop`` / ``llm_completion`` / ``results`` /
  ``tool_specs_for_prompt``) -- the reusable machinery a single agent runs on;
- the *top-level orchestration* (``tuner.MultiAgentTuner``) -- the one entry
  point that strings the four agents, the simulator, and the history store into
  the tuning loop.
"""

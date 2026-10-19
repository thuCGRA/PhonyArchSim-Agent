"""Photonic DSE framework entry point.

Run ``python main.py --iterations 4`` from the project root (default: offline
mock LLM + analytic SimPhony mirror).  See README.md for the real-simulator and
openai-compatible configurations.
"""

from photon_tuner.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

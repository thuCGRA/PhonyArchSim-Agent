"""光子 DSE 框架的统一测试入口。

在项目根目录运行::

    .venv-agent/bin/python tests/run_tests.py   # 领域 + deep-agent 全部测试
    python3 tests/run_tests.py                  # 仅领域测试（缺 langgraph 时自动跳过 deep）

目录约定：
- ``tests/*.py``：领域测试（纯 stdlib + PyYAML，无第三方依赖）。
- ``tests/deep/``：deep-agent 运行时测试，依赖 ``langgraph``；当它不可导入时整体跳过。
"""

import sys
import unittest
from pathlib import Path


def _load_test_dir(loader: unittest.TestLoader, pkg_prefix: str, directory: Path) -> unittest.TestSuite:
    """按 ``pkg_prefix.<文件名>`` 加载某目录顶层的全部 ``test_*.py``（不递归子目录）。"""
    suite = unittest.TestSuite()
    for path in sorted(directory.glob("test_*.py")):
        suite.addTests(loader.loadTestsFromName(f"{pkg_prefix}.{path.stem}"))
    return suite


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    tests_dir = Path(__file__).resolve().parent
    loader = unittest.TestLoader()

    # 领域测试（顶层 test_*.py）。
    suite = _load_test_dir(loader, "tests", tests_dir)

    # deep-agent 测试（tests/deep/），仅在 langgraph 可导入时纳入。
    try:
        import langgraph  # noqa: F401
        have_langgraph = True
    except Exception:
        have_langgraph = False

    if have_langgraph:
        suite.addTests(_load_test_dir(loader, "tests.deep", tests_dir / "deep"))
    else:
        print(
            "[run_tests] langgraph 不可导入 — 跳过 tests/deep/ "
            "（用 .venv-agent/bin/python 运行可包含 deep 测试）"
        )

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    if not have_langgraph or result.skipped:
        print("Full verification incomplete: some tests were skipped.", file=sys.stderr)
    else:
        print("""All automated tests passed. Next, from the project root in Bash:

1. Configure the key for your provider in this terminal:
   eval "$(.venv-agent/bin/python configs/api_keys.py)"
2. Edit provider, model and base_url in configs/llm.json (created from configs/llm.example.json).
3. Verify the real SimPhony simulator with a local mock model (no API request):
   .venv-agent/bin/python scripts/dev/verify_runtime.py simulator
4. Verify your selected online model and one candidate evaluation (uses API credits):
   .venv-agent/bin/python scripts/dev/verify_runtime.py model""")

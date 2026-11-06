"""光子 DSE 框架的 SimPhony 模拟器适配器。

该适配器桥接智能体使用的光子学 ``hardware`` 字典与 SimPhony 模拟器。

智能体流水线**只使用 real 模式**：CLI 不再暴露 ``--mode``，仿真固定为真实
:class:`onnarchsim.simulator.ONNArchSimulator`（需 PyTorch / TorchONN / pyutils），
且因其只装在 Python-3.10 虚拟环境中，智能体进程通过 ``transport="mcp"`` 跨进程调用。

``analytic`` 路径（纯 Python 的 SimPhony 工作流公式级镜像，见
:mod:`photon_tuner.simulators.analytic`）**不再是智能体的可选执行模式**，仅作为
模拟器库的内部能力保留，供离线单元测试与实验脚本直接调用。

两种模式都会把相同的产物写入运行目录：``hardware.json``、``report.json``、
各工作流明细 JSON 文件以及 ``metrics.json``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from photon_tuner.common.io import write_json
from photon_tuner.core.objective import metrics_with_objective
from photon_tuner.core.schemas import EvaluationResult
from photon_tuner.core.search_space import materialize_hardware
from photon_tuner.core.workload import DEFAULT_WORKLOAD, total_macs
from photon_tuner.simulators.analytic import evaluate as analytic_evaluate
from photon_tuner.simulators.base import SimulatorAdapter
from photon_tuner.simulators.netlist import first_key


# 按模型族划分的映射配置。``transformer`` 覆盖随附的 BERT（QBertViTBase）与
# MobileViT 模型（QConv2d + QLinear + QMatMul）；``cnn`` 覆盖仅含卷积/全连接层的模型。
MODEL_TYPE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "transformer": {
        "nn_conversion_cfg": "configs/nn_mapping/simple_transformer.yml",
        "model2arch_map_cfg": "configs/architecture_mapping/simple_lt_transformer.yml",
    },
    "cnn": {
        "nn_conversion_cfg": "configs/nn_mapping/simple_cnn.yml",
        "model2arch_map_cfg": "configs/architecture_mapping/simple_lt_cnn.yml",
    },
}


class SimphonyAdapter(SimulatorAdapter):
    """用 SimPhony（或其镜像）求值光子学硬件的适配器。"""

    def __init__(
        self,
        simphony_root: Path | str,
        output_root: Path | str,
        search_space: Optional[Dict[str, Any]] = None,
        mode: str = "auto",
        # real 模式（ONNArchSimulator）的构造输入。
        nn_model: Any = None,
        nn_model_factory: Optional[str] = None,
        onn_conversion_cfg: Optional[str] = None,
        nn_conversion_cfg: Optional[str] = None,
        model2arch_map_cfg: Optional[str] = None,
        model_type: str = "transformer",
        arch_cfg_file: str = "configs/design/architectures/LT_hetero.yml",
        arch_version: str = "v1",
        input_shape: tuple = (2, 3, 224, 224),
        # analytic 模式的工作负载选择（见 configs/workloads.json）。
        workload_name: Optional[str] = None,
        # 跨进程桥接：当 ``transport="mcp"`` 时，真实模拟器在 Python-3.10 venv 中运行，
        # 本适配器通过 deep MCP 客户端经 stdio 与其通信，而不是导入 onnarchsim。
        transport: str = "inprocess",
        mcp_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.simphony_root = Path(simphony_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.search_space = search_space or {}
        self.mode = mode
        self.transport = transport
        self.mcp_env = mcp_env or {}
        self._mcp_client = None
        self.nn_model = nn_model
        self.nn_model_factory = nn_model_factory
        self.onn_conversion_cfg = onn_conversion_cfg
        mapping = MODEL_TYPE_MAPPINGS.get(model_type, MODEL_TYPE_MAPPINGS["transformer"])
        self.nn_conversion_cfg = nn_conversion_cfg or mapping["nn_conversion_cfg"]
        self.model2arch_map_cfg = model2arch_map_cfg or mapping["model2arch_map_cfg"]
        self.model_type = model_type
        self.arch_cfg_file = arch_cfg_file
        self.arch_version = arch_version
        self.input_shape = input_shape
        # 在 ``real`` 模式下工作负载取自实际的 ONN 模型，因此该参数只用于离线解析镜像。
        self.workload_name = workload_name or DEFAULT_WORKLOAD

    # -- SimulatorAdapter 接口 --------------------------------------------- #

    def evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        if self.transport == "mcp":
            return self._evaluate_via_mcp(hardware, iteration)
        hardware = materialize_hardware(hardware)
        run_dir = self.output_root / f"iter_{iteration:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "hardware.json", hardware)

        report = self._run_report(hardware, run_dir)
        self._write_report_files(run_dir, report)

        energy_pj = float(report["metrics"]["energy_pj"])
        area_um2 = float(report["metrics"]["area_um2"])
        latency_s = float(report["metrics"]["latency_s"])
        macs = total_macs(report["workload"]["partition_cycles"][first_key(
            report["workload"]["partition_cycles"]
        )])
        # 吞吐归一化：把能耗归一化到每 MAC（pJ_per_mac），把面积归一化到每单位吞吐
        # （MAC/s），使并行权重阵列（单输出向量，吞吐低）与矩阵阵列在跨范式 Pareto
        # 上可比——权重阵列器件少、面积/能耗低，但同样吞吐需更多时间。
        throughput = macs / latency_s if latency_s > 0 else 0.0
        metrics = {
            "latency_s": latency_s,
            "energy_pj": energy_pj,
            "area_um2": area_um2,
            "total_macs": macs,
            "throughput_mac_per_s": throughput,
            "pJ_per_mac": energy_pj / macs if macs else energy_pj,
            "area_um2_per_mac_per_s": area_um2 / throughput if throughput > 0 else area_um2,
            "core_insertion_loss": float(
                report["insertion_loss"][first_key(report["insertion_loss"])][
                    "core_insertion_loss"
                ]
            ),
        }
        metrics = metrics_with_objective(metrics, hardware, self.search_space)
        write_json(run_dir / "metrics.json", metrics)

        return EvaluationResult(
            run_dir=run_dir,
            hardware=hardware,
            metrics=metrics,
            detail_files={
                "report": run_dir / "report.json",
                "energy": run_dir / "energy_detail.json",
                "area": run_dir / "area_detail.json",
                "insertion_loss": run_dir / "insertion_loss.json",
                "workload": run_dir / "workload.json",
                "memory": run_dir / "memory.json",
            },
            raw_files={
                "hardware": run_dir / "hardware.json",
                "metrics": run_dir / "metrics.json",
            },
        )

    def _evaluate_via_mcp(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        """通过进程外 SimPhony MCP 服务器求值。

        复现进程内打包运行的目录布局：``hardware.json`` / ``metrics.json`` 以及细粒度
        报告文件，这些文件由服务器带内返回，以便宿主机把它们写入智能体读取工具所期望的
        同一 ``run_dir``。
        """
        from photon_tuner.bridge.mcp.simphony_tool import SimphonyMCPClient

        hardware = materialize_hardware(hardware)
        run_dir = self.output_root / f"iter_{iteration:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "hardware.json", hardware)

        if self._mcp_client is None:
            # 为本次运行的进程外服务器分配私有暂存根目录，避免两个并发宿主机在共享的
            # ``.runs_mcp/real`` 路径上冲突（服务器在带内返回前会把报告/明细文件写在那里）。
            # 显式的 SIMPHONY_OUTPUT_ROOT 优先。
            env = dict(self.mcp_env)
            env.setdefault("SIMPHONY_OUTPUT_ROOT", str(self.output_root))
            self._mcp_client = SimphonyMCPClient.default(env=env)

        data = self._mcp_client.evaluate(hardware, iteration)
        metrics = {k: float(v) for k, v in data["metrics"].items()}
        write_json(run_dir / "metrics.json", metrics)

        # 进程外服务器返回完整的 SimPhony 报告；写入与进程内路径相同的各工作流明细文件，
        # 使智能体的读取工具（energy/area/insertion-loss/...）在 real 模式下照常工作，
        # 而不是因空的 detail_files 抛出 FileNotFoundError。
        report = data.get("report") or {}
        if report:
            self._write_report_files(run_dir, report)

        return EvaluationResult(
            run_dir=run_dir,
            hardware=hardware,
            metrics=metrics,
            detail_files=(
                {
                    "report": run_dir / "report.json",
                    "energy": run_dir / "energy_detail.json",
                    "area": run_dir / "area_detail.json",
                    "insertion_loss": run_dir / "insertion_loss.json",
                    "workload": run_dir / "workload.json",
                    "memory": run_dir / "memory.json",
                }
                if report
                else {}
            ),
            raw_files={
                "hardware": run_dir / "hardware.json",
                "metrics": run_dir / "metrics.json",
            },
        )

    def close(self) -> None:
        """关闭 MCP 桥接（若已打开）。"""
        if self._mcp_client is not None:
            self._mcp_client.close()
            self._mcp_client = None

    def schema(self) -> Dict[str, Any]:
        return {
            "adapter": "simphony",
            "mode": self.mode,
            "workload": self.workload_name,
            "search_space": self.search_space,
            "tunable_hardware_fields": {
                "optical_structure": [
                    "tiles",
                    "cores_per_tile",
                    "core_height",
                    "core_width",
                    "num_wavelength",
                    "work_freq",
                    "dataflow",
                    "precision_in_bit",
                    "precision_w_bit",
                    "precision_out_bit",
                ],
                "device_choices": ["devices.core.*", "devices.node.*"],
            },
            "detail_schema": {
                "report": "report.json — per-device / per-chip / per-layer energy, area, insertion loss, latency, memory breakdowns.",
                "energy": "energy_detail.json — per-device dynamic/static energy (pJ).",
                "area": "area_detail.json — per-device area (um^2).",
                "insertion_loss": "insertion_loss.json — node/core longest optical path (dB).",
                "workload": "workload.json — workload name/kind plus per-layer partition cycles and layer sizes.",
                "memory": "memory.json — per-layer memory latency/energy and hierarchy spec.",
            },
        }

    # -- 内部辅助函数 -------------------------------------------------------- #

    def _run_report(self, hardware: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        if self.mode == "analytic":
            return self._run_analytic(hardware)
        if self.mode == "real":
            return self._run_real_report(hardware, run_dir)
        # auto：优先使用真实模拟器，失败则回退到解析镜像。
        try:
            return self._run_real_report(hardware, run_dir)
        except (ImportError, ModuleNotFoundError):
            return self._run_analytic(hardware)

    def _run_analytic(self, hardware: Dict[str, Any]) -> Dict[str, Any]:
        """解析镜像的入口。

        作为唯一的调用点，使所有路径（``analytic`` 与 ``auto`` 回退）都转发所选的工作负载。
        """
        return analytic_evaluate(
            hardware,
            str(self.simphony_root),
            workload_name=self.workload_name,
        )

    def _run_real_report(self, hardware: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """运行真实的 ONNArchSimulator 并返回其细粒度报告。"""
        _, report = self.evaluate_with_readback(hardware, run_dir)
        return report

    def evaluate_with_readback(
        self, hardware: Dict[str, Any], run_dir: Path
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """运行一次 real 模式求值，返回 ``(injected_sub_arch, report)``。

        注入的子架构在 ``_inject_hardware`` 之后从模拟器读回，以便调用方确认被搜索的
        参数确实生效（这是有效性校验的核心）。
        """
        try:
            from onnarchsim.simulator import ONNArchSimulator
        except ImportError as exc:
            raise ImportError(
                "real SimPhony mode requires torchonn/pyutils (not installed); "
                "use --mode analytic for the offline mirror."
            ) from exc

        nn_model = _resolve_nn_model(self.nn_model, self.nn_model_factory)
        if nn_model is None:
            raise ImportError(
                "real SimPhony mode requires an ONN/nn model: pass it to the "
                "adapter, or set --nn-model-factory (module:function)."
            )

        # SimPhony 相对进程 CWD 解析 ``configs/nn_mapping/...``、``./cacti`` 与
        # ``./configs/memory``，因此从 SimPhony 检出版本根目录运行模拟器，同时让所有
        # 运行输出（``run_dir``）保持绝对路径。
        cwd = os.getcwd()
        os.chdir(self.simphony_root)
        try:
            simulator = ONNArchSimulator(
                nn_model=nn_model,
                onn_conversion_cfg=self.onn_conversion_cfg,
                nn_conversion_cfg=self.nn_conversion_cfg,
                onn_model=None,
                model2arch_map_cfg=self.model2arch_map_cfg,
                devicelib_root=str(self.simphony_root / "configs" / "devices"),
                device_cfg_files=["*/*.yml"],
                arch_cfg_file=str(self.simphony_root / self.arch_cfg_file),
                arch_version=self.arch_version,
                input_shape=self.input_shape,
                log_path=str(run_dir / "simphony.log"),
            )
            _inject_hardware(simulator, hardware, str(self.simphony_root))
            injected = read_sub_arch(simulator)
            report = simulator.run_full_report(
                output_path=str(run_dir / "report.json")
            )
        finally:
            os.chdir(cwd)
        # ``run_full_report`` 已返回报告；JSON 也会写出。
        if not isinstance(report, dict):
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        return injected, report

    def _write_report_files(self, run_dir: Path, report: Dict[str, Any]) -> None:
        """把完整报告拆分为各工作流明细文件。"""
        write_json(run_dir / "report.json", report)
        write_json(run_dir / "energy_detail.json", report.get("energy", {}))
        write_json(run_dir / "area_detail.json", report.get("area", {}))
        write_json(run_dir / "insertion_loss.json", report.get("insertion_loss", {}))
        write_json(run_dir / "workload.json", report.get("workload", {}))
        write_json(run_dir / "memory.json", report.get("memory", {}))


def _inject_hardware(
    simulator: Any,
    hardware: Dict[str, Any],
    simphony_root: str,
) -> None:
    """把搜索得到的 hardware 字典注入新建的 ONNArchSimulator。

    ONNArchSimulator 从固定 YAML 读取架构、完全忽略搜索得到的 ``hardware`` 字典，因此
    若无此步骤，real 模式求值将始终仿真同一个 LT 基线。这里用解析版 ``build_sub_arch``
    （与离线模式同一代码路径）重建子架构，并替换模拟器已解析的子架构以及逐层 miniblock
    和 layer->sub-arch 映射，使 tiles / cores / 几何 / 频率 / 精度 / 内存 / 器件以及
    L4 拓扑覆盖全部生效。
    """
    from photon_tuner.simulators.netlist import (
        build_mesh_sub_arch,
        build_sub_arch,
        build_weight_array_sub_arch,
        load_device_catalog,
        sub_arch_key,
    )

    catalog = load_device_catalog(f"{simphony_root}/configs/devices")
    core_kind = str(hardware.get("arch", {}).get("topology", {}).get("core_kind", ""))
    if core_kind.startswith("mzi_mesh_"):
        sub_arch = build_mesh_sub_arch(hardware, simphony_root, catalog, core_kind)
    elif core_kind in ("mrr_weight_bank", "pcm_crossbar", "wdm_link"):
        sub_arch = build_weight_array_sub_arch(hardware, simphony_root, catalog, core_kind)
    elif core_kind == "dptc_crossbar":
        sub_arch = build_sub_arch(hardware, simphony_root, catalog)
    else:
        # 未登记的 core_kind 曾经静默落回 build_sub_arch → 得到一个 crossbar 的评估结果，
        # 使 agent 以为自己在评测另一个拓扑。宁可报错也不要产出错误的数值。
        raise ValueError(
            f"unknown core_kind {core_kind!r}: no netlist builder and no evaluation path; "
            f"refusing to silently fall back to the DPTC crossbar"
        )

    # 子架构在报告/字典里的键取实际拓扑名；SimPhony 的层映射表里那个静态名
    # （LEGACY_SUB_ARCH_NAME，实为 "LT"）只是查表键，不能带到结果里冒充基线。
    legacy_name = next(iter(getattr(simulator, "sub_arch_dictionary", {}) or {}), None)
    if legacy_name is None:
        raise ValueError("ONNArchSimulator has no sub-architecture to inject into")
    key = sub_arch_key(hardware)
    if key != legacy_name:
        # 重键：子架构字典、层→子架构映射必须同步改名，否则下游 simu_* 的循环
        # 会取不到层列表。
        reordered = {}
        for old_key, value in simulator.sub_arch_dictionary.items():
            reordered[key if old_key == legacy_name else old_key] = value
        simulator.sub_arch_dictionary = reordered
        if legacy_name in simulator.sub_arch_to_layer_mapping:
            simulator.sub_arch_to_layer_mapping[key] = (
                simulator.sub_arch_to_layer_mapping.pop(legacy_name)
            )
    name = key
    simulator.sub_arch_dictionary[name] = sub_arch

    tiles = int(sub_arch["tiles"])
    cores = int(sub_arch["cores_per_tile"])
    core_cfg = sub_arch["core"]
    # 分区 miniblock 是 *矩阵* 平铺维度。对于网格核心，它是 n_modes x n_modes
    # （其酉矩阵大小），而非 height/width 中存储的 MZI 数量包围盒（height=n_mzi,
    # width=1）；matrix_height/width 保存正确的平铺维度，缺省时回退到 height/width。
    height = int(core_cfg.get("matrix_height", core_cfg.get("height")))
    width = int(core_cfg.get("matrix_width", core_cfg.get("width")))
    for layer in simulator.sub_arch_to_layer_mapping.get(name, []):
        # miniblock 顺序为 [tiles, cores_per_tile, height, width]。
        simulator.arch_blocks[layer] = [tiles, cores, height, width]
        simulator.layer_to_sub_arch_mapping[layer] = sub_arch


def read_sub_arch(simulator: Any) -> Dict[str, Any]:
    """从已构建的模拟器返回注入的子架构字典。

    ``_inject_hardware`` 用 ``build_sub_arch`` 的输出替换 ``simulator.sub_arch_dictionary[name]``；
    该辅助函数把该状态读回，以便校验流程确认被搜索的参数确实生效。
    """
    dictionary = getattr(simulator, "sub_arch_dictionary", {}) or {}
    name = next(iter(dictionary), None)
    if name is None:
        return {}
    return dict(dictionary[name])


def _resolve_nn_model(nn_model: Any, factory: Optional[str]) -> Any:
    """返回 ONN/nn 模型；未直接传入对象时，从 ``module:function`` 工厂字符串加载。"""
    if nn_model is not None:
        return nn_model
    if not factory:
        return None
    if ":" not in factory:
        raise ValueError("--nn-model-factory must be 'module.path:function_name'")
    module_name, function_name = factory.split(":", 1)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, function_name)()

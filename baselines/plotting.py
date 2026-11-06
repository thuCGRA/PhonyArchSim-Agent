"""零依赖 SVG 绘图（不依赖 matplotlib，可在 .venv-agent 中直接运行）。

产出可直接嵌入论文/报告的矢量图。只实现两条常用曲线：折线图（可带对数 y 轴）与
分组柱状图。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

# 统一配色（tab10 前 7 色）。
_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]


def _esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt(value: float) -> str:
    """紧凑的刻度/数值标签（3 位有效数字）。"""
    if value == 0:
        return "0"
    magnitude = abs(value)
    if 1e-2 <= magnitude < 1e5:
        return f"{value:.3g}"
    return f"{value:.2e}"


def _log_ticks(lo: float, hi: float, intervals: int = 5) -> List[float]:
    """在 log10 空间生成 ``intervals + 1`` 个等距刻度（返回 log10 值）。

    刻度在 log 空间等分，而不是只取整数数量级——后者在数据跨度不足一个数量级时
    （例如 130→434，log10 仅 2.11→2.64）会返回空列表，使 y 轴没有任何网格与标签。
    """
    if hi <= lo:
        hi = lo + 1.0
    return [lo + (hi - lo) * i / intervals for i in range(intervals + 1)]


def _linear_ticks(lo: float, hi: float, intervals: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + 1.0
    return [lo + (hi - lo) * i / intervals for i in range(intervals + 1)]


def _x_ticks(x_min: float, x_max: float, target: int = 8) -> List[int]:
    """整数 x 刻度；跨度小时逐点，跨度大时等距抽样并保证末点入选。"""
    lo, hi = int(math.floor(x_min)), int(math.ceil(x_max))
    if hi <= lo:
        return [lo]
    if hi - lo <= target:
        return list(range(lo, hi + 1))
    step = max(1, int(round((hi - lo) / target)))
    ticks = list(range(lo, hi + 1, step))
    if ticks[-1] != hi:
        ticks.append(hi)
    return ticks


def _legend_svg(
    entries: Sequence[Tuple[str, str]],
    *,
    plot_x: float,
    plot_y: float,
    plot_w: float,
    plot_h: float,
    swatch: str,
) -> List[str]:
    """在绘图区右上角绘制带半透明白底的图例（避免被曲线穿透）。"""
    if not entries:
        return []
    char_w = 6.9
    box_w = min(34 + max(len(str(name)) for name, _ in entries) * char_w + 12, plot_w * 0.6)
    row_h = 20
    box_h = row_h * len(entries) + 10
    x0 = plot_x + plot_w - box_w - 8
    y0 = plot_y + 8

    out = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" '
        f'fill="white" fill-opacity="0.88" stroke="#ddd"/>'
    ]
    for i, (name, color) in enumerate(entries):
        cy = y0 + 10 + row_h * i + row_h / 2
        if swatch == "line":
            out.append(
                f'<line x1="{x0 + 8:.1f}" y1="{cy:.1f}" x2="{x0 + 34:.1f}" y2="{cy:.1f}" '
                f'stroke="{color}" stroke-width="3"/>'
            )
        else:
            out.append(f'<rect x="{x0 + 8:.1f}" y="{cy - 6:.1f}" width="26" height="12" fill="{color}"/>')
        out.append(f'<text x="{x0 + 40:.1f}" y="{cy + 4:.1f}" font-size="12">{_esc(name)}</text>')
    return out


def _write_empty(out_path: str, width: int, height: int, title: str, message: str) -> str:
    """无可绘制数据时仍产出一个合法文件，避免调用方拿到悬空路径。"""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]
    if title:
        parts.append(
            f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="18" '
            f'font-weight="bold">{_esc(title)}</text>'
        )
    parts.append(
        f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-size="14" '
        f'fill="#999">{_esc(message)}</text>'
    )
    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return out_path


def line_chart_svg(
    series: Sequence[Dict[str, Any]],
    out_path: str,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    log_y: bool = False,
    width: int = 800,
    height: int = 520,
) -> str:
    """折线图。每个 ``series`` 项: ``{"name", "x": [...], "y": [...]}``。

    ``log_y=True`` 时 y 轴取对数（目标函数常跨数量级）。返回写入的文件路径。
    """
    ml, mr, mt, mb = 78, 24, 52, 62
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    plot_bottom = mt + plot_h

    all_x = [v for s in series for v in s.get("x", [])]
    all_y = [v for s in series for v in s.get("y", []) if v is not None]
    if not all_x or not all_y:
        return _write_empty(out_path, width, height, title, "no data to plot")

    x_min, x_max = min(all_x), max(all_x)
    if x_max == x_min:
        x_max = x_min + 1
    y_min, y_max = min(all_y), max(all_y)
    if y_max == y_min:
        padding = abs(y_min) * 0.05 or 1.0
        y_min, y_max = y_min - padding, y_max + padding

    if log_y:
        y_min = max(y_min, 1e-300)
        y_max = max(y_max, y_min * 1.0000001)
        lo, hi = math.log10(y_min), math.log10(y_max)
        if hi <= lo:
            hi = lo + 1.0

        def ty(value: float) -> float:
            return mt + (1.0 - (math.log10(max(value, 1e-300)) - lo) / (hi - lo)) * plot_h

        y_ticks = [(10.0 ** t, _fmt(10.0 ** t)) for t in _log_ticks(lo, hi)]
    else:
        def ty(value: float) -> float:
            return plot_bottom - (value - y_min) / (y_max - y_min) * plot_h

        y_ticks = [(v, _fmt(v)) for v in _linear_ticks(y_min, y_max)]

    def tx(value: float) -> float:
        return ml + (value - x_min) / (x_max - x_min) * plot_w

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]
    if title:
        parts.append(
            f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="18" '
            f'font-weight="bold">{_esc(title)}</text>'
        )

    # y 网格线 + 刻度标签。
    for value, label in y_ticks:
        yy = ty(value)
        if not (mt - 0.5 <= yy <= plot_bottom + 0.5):
            continue
        parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{width - mr}" y2="{yy:.2f}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{ml - 8}" y="{yy + 4:.2f}" text-anchor="end" font-size="11">{label}</text>')

    # x 网格线 + 刻度标签（整数求值序号）。
    for x in _x_ticks(x_min, x_max):
        if not (x_min <= x <= x_max):
            continue
        xx = tx(x)
        parts.append(
            f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{plot_bottom}" stroke="#f4f4f4"/>'
        )
        parts.append(
            f'<text x="{xx:.2f}" y="{plot_bottom + 18}" text-anchor="middle" font-size="11">{int(x)}</text>'
        )

    # 轴标签。
    if ylabel:
        parts.append(
            f'<text x="20" y="{mt + plot_h / 2:.1f}" text-anchor="middle" font-size="14" '
            f'transform="rotate(-90 20 {mt + plot_h / 2:.1f})">{_esc(ylabel)}</text>'
        )
    if xlabel:
        parts.append(
            f'<text x="{ml + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" '
            f'font-size="14">{_esc(xlabel)}</text>'
        )

    # 曲线。
    legend_entries: List[Tuple[str, str]] = []
    for idx, s in enumerate(series):
        color = _PALETTE[idx % len(_PALETTE)]
        legend_entries.append((str(s.get("name", f"s{idx}")), color))
        pts = [(tx(x), ty(y)) for x, y in zip(s.get("x", []), s.get("y", [])) if y is not None]
        if not pts:
            continue
        if len(pts) == 1:
            # 单次求值：画点而不是整条曲线消失。
            parts.append(f'<circle cx="{pts[0][0]:.2f}" cy="{pts[0][1]:.2f}" r="3.2" fill="{color}"/>')
            continue
        d = "M " + " L ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<circle cx="{pts[0][0]:.2f}" cy="{pts[0][1]:.2f}" r="3" fill="{color}"/>')
        parts.append(f'<circle cx="{pts[-1][0]:.2f}" cy="{pts[-1][1]:.2f}" r="3.4" fill="{color}"/>')

    parts.extend(
        _legend_svg(
            legend_entries, plot_x=ml, plot_y=mt, plot_w=plot_w, plot_h=plot_h, swatch="line"
        )
    )
    parts.append(f'<rect x="{ml}" y="{mt}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#ccc"/>')
    parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return out_path


def bar_chart_svg(
    categories: Sequence[str],
    series: Sequence[Dict[str, Any]],
    out_path: str,
    *,
    title: str = "",
    ylabel: str = "",
    log_y: bool = False,
    width: int = 800,
    height: int = 480,
) -> str:
    """分组柱状图。``series`` 项: ``{"name", "values": [...]}``，``values`` 与
    ``categories`` 等长。"""
    ml, mr, mt, mb = 78, 24, 52, 62
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    plot_bottom = mt + plot_h

    n_groups = max(1, len(categories))
    n_series = len(series)
    all_vals = [v for s in series for v in s.get("values", []) if v is not None and v > 0]
    if not all_vals or n_series == 0:
        return _write_empty(out_path, width, height, title, "no data to plot")

    if log_y:
        y_bottom = min(all_vals)
        y_top = max(all_vals)
        if y_top <= y_bottom:
            y_top = y_bottom * 1.1
        lo, hi = math.log10(y_bottom), math.log10(y_top)
        if hi <= lo:
            hi = lo + 1.0

        def ty(value: float) -> float:
            return mt + (1.0 - (math.log10(max(value, 1e-300)) - lo) / (hi - lo)) * plot_h

        y_ticks = [(10.0 ** t, _fmt(10.0 ** t)) for t in _log_ticks(lo, hi)]
    else:
        # 基线固定为 0（可比），顶端留 5% 余量避免最高柱顶住边框。
        y_bottom = 0.0
        y_top = max(all_vals) * 1.05
        span = y_top - y_bottom

        def ty(value: float) -> float:
            return plot_bottom - (value - y_bottom) / span * plot_h

        y_ticks = [(v, _fmt(v)) for v in _linear_ticks(y_bottom, y_top)]

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    if title:
        parts.append(
            f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="18" '
            f'font-weight="bold">{_esc(title)}</text>'
        )
    if ylabel:
        parts.append(
            f'<text x="20" y="{mt + plot_h / 2:.1f}" text-anchor="middle" font-size="14" '
            f'transform="rotate(-90 20 {mt + plot_h / 2:.1f})">{_esc(ylabel)}</text>'
        )

    # y 网格线 + 刻度标签。
    for value, label in y_ticks:
        yy = ty(value)
        if not (mt - 0.5 <= yy <= plot_bottom + 0.5):
            continue
        parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{width - mr}" y2="{yy:.2f}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{ml - 8}" y="{yy + 4:.2f}" text-anchor="end" font-size="11">{label}</text>')

    group_w = plot_w / n_groups
    bar_w = max(2.0, group_w * 0.8 / n_series)

    for gi, cat in enumerate(categories):
        cx = ml + group_w * (gi + 0.5)
        parts.append(
            f'<text x="{cx:.1f}" y="{plot_bottom + 18}" text-anchor="middle" '
            f'font-size="11">{_esc(cat)}</text>'
        )
        for si, s in enumerate(series):
            values = s.get("values", [])
            if gi >= len(values):
                continue
            value = values[gi]
            if value is None:
                continue
            color = _PALETTE[si % len(_PALETTE)]
            # 柱高 = 底边 - 柱顶（旧实现写成“顶 - 底”，恒为负，被 max(1.0, ...) 压成 1px）。
            bar_h = max(1.0, plot_bottom - ty(value))
            bx = cx - (n_series / 2 - si) * bar_w - bar_w / 2
            parts.append(
                f'<rect x="{bx:.2f}" y="{plot_bottom - bar_h:.2f}" width="{bar_w:.2f}" '
                f'height="{bar_h:.2f}" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{bx + bar_w / 2:.2f}" y="{plot_bottom - bar_h - 4:.2f}" '
                f'text-anchor="middle" font-size="9">{_fmt(value)}</text>'
            )

    parts.append(f'<rect x="{ml}" y="{mt}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#ccc"/>')
    legend_entries = [
        (str(s.get("name", f"s{si}")), _PALETTE[si % len(_PALETTE)]) for si, s in enumerate(series)
    ]
    parts.extend(
        _legend_svg(
            legend_entries, plot_x=ml, plot_y=mt, plot_w=plot_w, plot_h=plot_h, swatch="rect"
        )
    )
    parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return out_path

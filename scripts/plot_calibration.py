"""Render the calibration reliability diagram from the benchmark, as SVG.

    python scripts/plot_calibration.py

Writes docs/calibration-light.svg and docs/calibration-dark.svg. Deterministic
(the benchmark is), zero-dependency (SVG is emitted by hand — in keeping with the
rest of the project), and theme-paired for embedding via <picture> in the README.

The chart is a reliability diagram: predicted confidence (x) against observed
accuracy (y), with the y=x diagonal as perfect calibration. The RAW series sits
below the diagonal (overconfident); the CALIBRATED series hugs it. Colours and
contrast follow the data-viz palette (validated slots: blue #2a78d6 calibrated,
orange #eb6834 raw).
"""

from __future__ import annotations

from pathlib import Path

from bus_factor.benchmark import run_benchmark

# --- palette (validated categorical slots + chrome), per mode -----------------
_THEMES = {
    "light": {
        "surface": "#fcfcfb", "border": "rgba(11,11,11,0.10)",
        "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "raw": "#eb6834", "cal": "#2a78d6", "good": "#006300",
    },
    "dark": {
        "surface": "#1a1a19", "border": "rgba(255,255,255,0.12)",
        "primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "raw": "#d95926", "cal": "#3987e5", "good": "#0ca30c",
    },
}
_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Plot geometry.
_W, _H = 860, 520
_PX, _PY = 70, 96          # plot origin (bottom-left) offsets from canvas edges
_PW, _PH = 430, 340        # plot width / height (square-ish)


def _reliability_points(items: list[dict], n_bins: int = 5) -> list[tuple[float, float, int]]:
    """Bin items by confidence; return (mean_confidence, accuracy, count) per bin."""
    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for it in items:
        idx = min(n_bins - 1, int(it["confidence"] * n_bins))
        bins[idx].append(it)
    pts = []
    for b in bins:
        if not b:
            continue
        conf = sum(i["confidence"] for i in b) / len(b)
        acc = sum(1 for i in b if i["correct"]) / len(b)
        pts.append((conf, acc, len(b)))
    return pts


def _sx(v: float) -> float:
    return _PX + v * _PW


def _sy(v: float) -> float:
    return (_H - _PY) - v * _PH


def _poly(points: list[tuple[float, float, int]], color: str) -> list[str]:
    out = []
    coords = " ".join(f"{_sx(c):.1f},{_sy(a):.1f}" for c, a, _ in points)
    out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
               f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
    for c, a, n in points:
        r = 4 + min(6.0, n * 0.5)
        out.append(f'<circle cx="{_sx(c):.1f}" cy="{_sy(a):.1f}" r="{r:.1f}" '
                   f'fill="{color}" stroke="{{surface}}" stroke-width="2"/>')
    return out


def _render(mode: str, reports: dict) -> str:
    t = _THEMES[mode]
    raw_pts = _reliability_points(reports["leave_one_out"].items)
    cal_pts = _reliability_points(reports["leave_one_out_calibrated"].items)
    raw_s = reports["leave_one_out"].summary
    cal_s = reports["leave_one_out_calibrated"].summary

    s: list[str] = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
             f'font-family="{_FONT}" width="{_W}" height="{_H}">')
    s.append(f'<rect x="1" y="1" width="{_W-2}" height="{_H-2}" rx="14" '
             f'fill="{t["surface"]}" stroke="{t["border"]}" stroke-width="1"/>')

    # Titles.
    s.append(f'<text x="{_PX}" y="40" fill="{t["primary"]}" font-size="20" '
             f'font-weight="700">Confidence calibration under leave-one-out</text>')
    s.append(f'<text x="{_PX}" y="63" fill="{t["secondary"]}" font-size="13">'
             f'Raw retrieval confidence is overconfident; isotonic calibration puts it '
             f'on the diagonal.</text>')

    # Gridlines + ticks (0, .25, .5, .75, 1).
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = _sy(g)
        x = _sx(g)
        s.append(f'<line x1="{_PX}" y1="{y:.1f}" x2="{_PX+_PW}" y2="{y:.1f}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        s.append(f'<text x="{_PX-10}" y="{y+4:.1f}" fill="{t["muted"]}" font-size="11" '
                 f'text-anchor="end">{g:.2f}</text>')
        s.append(f'<text x="{x:.1f}" y="{_H-_PY+22}" fill="{t["muted"]}" font-size="11" '
                 f'text-anchor="middle">{g:.2f}</text>')

    # Axes.
    s.append(f'<line x1="{_PX}" y1="{_sy(0):.1f}" x2="{_PX+_PW}" y2="{_sy(0):.1f}" '
             f'stroke="{t["axis"]}" stroke-width="1.5"/>')
    s.append(f'<line x1="{_PX}" y1="{_sy(0):.1f}" x2="{_PX}" y2="{_sy(1):.1f}" '
             f'stroke="{t["axis"]}" stroke-width="1.5"/>')
    s.append(f'<text x="{_PX+_PW/2:.0f}" y="{_H-_PY+44}" fill="{t["secondary"]}" '
             f'font-size="12.5" text-anchor="middle">predicted confidence</text>')
    s.append(f'<text transform="translate({_PX-44},{_sy(0.5):.0f}) rotate(-90)" '
             f'fill="{t["secondary"]}" font-size="12.5" text-anchor="middle">'
             f'observed accuracy</text>')

    # Perfect-calibration diagonal.
    s.append(f'<line x1="{_sx(0):.1f}" y1="{_sy(0):.1f}" x2="{_sx(1):.1f}" y2="{_sy(1):.1f}" '
             f'stroke="{t["muted"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{_sx(0.62)+6:.0f}" y="{_sy(0.62)-8:.0f}" fill="{t["muted"]}" '
             f'font-size="11" transform="rotate(-34 {_sx(0.62):.0f} {_sy(0.62):.0f})">'
             f'perfect calibration</text>')

    # Series.
    for line in _poly(raw_pts, t["raw"]):
        s.append(line.replace("{surface}", t["surface"]))
    for line in _poly(cal_pts, t["cal"]):
        s.append(line.replace("{surface}", t["surface"]))

    # Legend (right column).
    lx = _PX + _PW + 40
    ly = 120
    s.append(f'<rect x="{lx-14}" y="{ly-26}" width="256" height="104" rx="10" '
             f'fill="none" stroke="{t["border"]}" stroke-width="1"/>')
    s.append(f'<circle cx="{lx}" cy="{ly-4}" r="6" fill="{t["cal"]}"/>')
    s.append(f'<text x="{lx+16}" y="{ly}" fill="{t["primary"]}" font-size="13" '
             f'font-weight="600">Calibrated</text>')
    s.append(f'<circle cx="{lx}" cy="{ly+24}" r="6" fill="{t["raw"]}"/>')
    s.append(f'<text x="{lx+16}" y="{ly+28}" fill="{t["primary"]}" font-size="13" '
             f'font-weight="600">Raw (uncalibrated)</text>')
    s.append(f'<line x1="{lx-6}" y1="{ly+50}" x2="{lx+6}" y2="{ly+50}" '
             f'stroke="{t["muted"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{lx+16}" y="{ly+54}" fill="{t["secondary"]}" font-size="12">'
             f'perfect calibration</text>')

    # Headline stats (below the legend box; box bottom = ly+78 = 198).
    stx = lx - 14
    sty = ly + 118          # 238
    s.append(f'<text x="{stx}" y="{sty}" fill="{t["muted"]}" font-size="11" '
             f'letter-spacing="0.5">CALIBRATION ERROR (ECE)</text>')
    s.append(f'<text x="{stx}" y="{sty+27}" fill="{t["primary"]}" font-size="21" '
             f'font-weight="700">{raw_s["ece"]:.2f} '
             f'<tspan fill="{t["muted"]}" font-size="15" font-weight="400">&#8594;</tspan> '
             f'<tspan fill="{t["good"]}">{cal_s["ece"]:.2f}</tspan></text>')

    sty2 = sty + 70          # 308
    s.append(f'<text x="{stx}" y="{sty2}" fill="{t["muted"]}" font-size="11" '
             f'letter-spacing="0.5">ACCURACY WHEN IT ANSWERS</text>')
    s.append(f'<text x="{stx}" y="{sty2+27}" fill="{t["primary"]}" font-size="21" '
             f'font-weight="700">{raw_s["accuracy"]:.0%} '
             f'<tspan fill="{t["muted"]}" font-size="15" font-weight="400">&#8594;</tspan> '
             f'<tspan fill="{t["good"]}">{cal_s["selective_accuracy"]:.0%}</tspan></text>')
    s.append(f'<text x="{stx}" y="{sty2+50}" fill="{t["secondary"]}" font-size="12">'
             f'abstains on {1-cal_s["coverage"]:.0%} it can’t recover '
             f'(coverage {cal_s["coverage"]:.0%})</text>')

    # Footer.
    s.append(f'<text x="{_PX}" y="{_H-18}" fill="{t["muted"]}" font-size="11">'
             f'reproduce: python scripts/synthetic_benchmark.py  ·  '
             f'{raw_s["n"]}-question holdout, offline extractive baseline</text>')

    s.append("</svg>")
    return "\n".join(s)


def main() -> int:
    reports = run_benchmark()
    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    for mode in ("light", "dark"):
        path = out_dir / f"calibration-{mode}.svg"
        path.write_text(_render(mode, reports), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

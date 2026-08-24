"""Generate locked-metric Milestone 11 SVG charts and the repository QR code."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans TC',sans-serif"
INK = "#172033"
MUTED = "#566176"
BLUE = "#2563EB"
TEAL = "#0F9D8A"
ORANGE = "#E8792E"
PURPLE = "#7656C8"
GRID = "#DCE2EC"
PALE = "#F5F7FB"
WHITE = "#FFFFFF"


def text(x: float, y: float, value: str, size: int = 24, *, color: str = INK,
         weight: int = 400, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        f'{escape(value)}</text>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str,
         radius: float = 0, stroke: str = "none", stroke_width: float = 0) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = GRID,
         width: float = 2, dash: str | None = None) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"{dashed}/>'
    )


def svg_document(title: str, description: str, body: list[str], width: int = 1600,
                 height: int = 900) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            rect(0, 0, width, height, WHITE),
            *body,
            "</svg>",
        ]
    ) + "\n"


def domain_shift_chart() -> None:
    result = json.loads((ROOT / "results/cross_domain_evaluation.json").read_text())
    values = {
        "Accuracy": (
            result["cnr_ext_test"]["accuracy"],
            result["pklot"]["overall"]["accuracy"],
        ),
        "Occupied F1": (
            result["cnr_ext_test"]["f1_occupied"],
            result["pklot"]["overall"]["f1_occupied"],
        ),
    }
    body = [
        text(100, 92, "Domain shift exposed by zero-shot transfer", 42, weight=500),
        text(100, 132, "CNR-EXT source-domain test vs. PKLot target-domain evaluation", 23, color=MUTED),
        rect(1120, 82, 24, 24, BLUE, 4), text(1158, 103, "CNR-EXT test", 20),
        rect(1345, 82, 24, 24, ORANGE, 4), text(1383, 103, "PKLot zero-shot", 20),
    ]
    chart_left, chart_right, chart_top, chart_bottom = 160, 1490, 200, 760
    for tick in range(0, 11, 2):
        value = tick / 10
        y = chart_bottom - value * (chart_bottom - chart_top)
        body += [line(chart_left, y, chart_right, y), text(chart_left - 22, y + 8, f"{value:.1f}", 19, color=MUTED, anchor="end")]
    body += [
        line(chart_left, chart_top, chart_left, chart_bottom, INK, 2),
        line(chart_left, chart_bottom, chart_right, chart_bottom, INK, 2),
        text(45, 480, "Score", 22, color=MUTED),
    ]
    centers = [520, 1120]
    bar_width = 165
    for center, (metric, (source, target)) in zip(centers, values.items()):
        for offset, score, color in [(-100, source, BLUE), (100, target, ORANGE)]:
            height = score * (chart_bottom - chart_top)
            x = center + offset - bar_width / 2
            y = chart_bottom - height
            body += [rect(x, y, bar_width, height, color, 8), text(center + offset, y - 18, f"{score:.3f}", 25, weight=500, anchor="middle")]
        body += [
            text(center, 812, metric, 25, weight=500, anchor="middle"),
            text(center, 849, f"absolute drop {source - target:.3f}", 21, color=MUTED, anchor="middle"),
        ]
    (IMAGES / "domain_shift.svg").write_text(
        svg_document(
            "Domain shift visualization",
            "Grouped bars show substantial accuracy and occupied-F1 drops from CNR-EXT to zero-shot PKLot.",
            body,
        ),
        encoding="utf-8",
    )


def v1_v2_chart() -> None:
    result = json.loads((ROOT / "results/v2_fresh_final_comparison.json").read_text())
    v1 = result["models"]["v1_finetuned_resnet18"]
    v2 = result["models"]["v2a_balanced_resnet18"]
    metrics = [
        ("Accuracy", v1["overall"]["accuracy"], v2["overall"]["accuracy"]),
        ("Occupied F1", v1["overall"]["f1_occupied"], v2["overall"]["f1_occupied"]),
        ("Macro-site F1", v1["macro_site_f1_occupied"], v2["macro_site_f1_occupied"]),
        ("UFPR04 occupied recall", v1["by_site"]["UFPR04"]["recall_occupied"], v2["by_site"]["UFPR04"]["recall_occupied"]),
    ]
    body = [
        text(100, 92, "V2-A closes the documented UFPR04 recall gap", 42, weight=500),
        text(100, 132, "One-time comparison on the same 154,669 fresh-final samples", 23, color=MUTED),
        rect(1120, 82, 24, 24, PURPLE, 12), text(1158, 103, "V1", 20),
        rect(1280, 82, 24, 24, TEAL, 12), text(1318, 103, "V2-A", 20),
    ]
    x0, x1 = 520, 1480
    minimum, maximum = 0.84, 1.005
    for tick in [0.85, 0.90, 0.95, 1.00]:
        x = x0 + (tick - minimum) / (maximum - minimum) * (x1 - x0)
        body += [line(x, 205, x, 760, GRID, 2), text(x, 800, f"{tick:.2f}", 19, color=MUTED, anchor="middle")]
    body += [text((x0 + x1) / 2, 846, "Score (shared fresh-final protocol)", 22, color=MUTED, anchor="middle")]
    for index, (label, before, after) in enumerate(metrics):
        y = 275 + index * 135
        bx = x0 + (before - minimum) / (maximum - minimum) * (x1 - x0)
        ax = x0 + (after - minimum) / (maximum - minimum) * (x1 - x0)
        body += [
            text(450, y + 8, label, 23, weight=500, anchor="end"),
            text(450, y + 39, f"Δ {after - before:+.6f}", 18, color=MUTED, anchor="end"),
            line(bx, y, ax, y, MUTED, 8),
            f'<circle cx="{bx}" cy="{y}" r="14" fill="{PURPLE}"/>',
            f'<circle cx="{ax}" cy="{y}" r="14" fill="{TEAL}"/>',
            text(bx - 18, y - 27, f"{before:.6f}", 19, color=PURPLE, anchor="end"),
            text(ax + 18, y + 44, f"{after:.6f}", 19, color=TEAL, anchor="start"),
        ]
    (IMAGES / "v1_v2_improvement.svg").write_text(
        svg_document(
            "V1 versus V2 improvement",
            "Dumbbell chart compares V1 and V2-A accuracy, occupied F1, macro-site F1, and UFPR04 occupied recall.",
            body,
        ),
        encoding="utf-8",
    )


def workflow_diagram() -> None:
    body = [
        text(80, 80, "Leakage-aware cross-domain research workflow", 42, weight=500),
        text(80, 120, "Data stays external; manifests, protocol locks, code, and reports stay reproducible in Git", 22, color=MUTED),
    ]
    nodes = [
        (90, 190, 310, 105, "External SSD", "CNRPark+EXT + PKLot images", BLUE),
        (485, 190, 310, 105, "Portable manifests", "relative paths + group metadata", BLUE),
        (880, 190, 310, 105, "Leakage audits", "date / frame / site boundaries", BLUE),
        (1275, 190, 235, 105, "Protocol locks", "splits + hashes", BLUE),
        (90, 410, 310, 120, "Source-domain track", "SimpleCNN → ResNet18\nCNR-EXT one-time test", PURPLE),
        (485, 410, 310, 120, "Domain-shift test", "Frozen ResNet18 → PKLot\nzero-shot evaluation", ORANGE),
        (880, 410, 310, 120, "Target adaptation V1", "PKLot fine-tuning\nheld-out error analysis", ORANGE),
        (1275, 410, 235, 120, "Robustness V2", "balanced sampling\nResNet18 vs EfficientNet", TEAL),
        (485, 660, 310, 115, "Validation-only selection", "V2-A ResNet18 locked", TEAL),
        (880, 660, 310, 115, "Fresh-final comparison", "V1 vs V2-A once\n154,669 samples", TEAL),
        (1275, 660, 235, 115, "Production demo", "V2-A · threshold 0.5", TEAL),
    ]
    for x, y, width, height, heading, detail, color in nodes:
        body += [rect(x, y, width, height, PALE, 14, color, 3), text(x + 22, y + 38, heading, 23, color=color, weight=500)]
        for line_index, value in enumerate(detail.split("\n")):
            body.append(text(x + 22, y + 73 + line_index * 28, value, 18, color=MUTED))
    arrows = [
        (400, 242, 485, 242), (795, 242, 880, 242), (1190, 242, 1275, 242),
        (245, 295, 245, 410), (640, 295, 640, 410), (1035, 295, 1035, 410), (1392, 295, 1392, 410),
        (400, 470, 485, 470), (795, 470, 880, 470), (1190, 470, 1275, 470),
        (1392, 530, 1392, 590), (1392, 590, 640, 590), (640, 590, 640, 660),
        (795, 718, 880, 718), (1190, 718, 1275, 718),
    ]
    for x1, y1, x2, y2 in arrows:
        body += [line(x1, y1, x2, y2, MUTED, 3), f'<circle cx="{x2}" cy="{y2}" r="5" fill="{MUTED}"/>']
    body += [text(90, 845, "Immutable evaluation boundaries", 18, color=MUTED), line(400, 839, 560, 839, MUTED, 3)]
    (IMAGES / "research_workflow.svg").write_text(
        svg_document(
            "Research architecture and workflow",
            "Diagram traces external data through portable manifests, leakage audits, source and target experiments, locked selection, final comparison, and V2-A demo.",
            body,
        ),
        encoding="utf-8",
    )


def repository_qr() -> None:
    import qrcode

    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=10, border=4)
    qr.add_data("https://github.com/mieuxwei/parking-occupancy-detection")
    qr.make(fit=True)
    qr.make_image(fill_color=INK, back_color=WHITE).save(IMAGES / "repository_qr.png")


def demo_gif() -> None:
    from PIL import Image

    paths = [IMAGES / "v2_demo_initial.png", IMAGES / "v2_demo_inference_result.png"]
    if not all(path.is_file() for path in paths):
        return
    frames = [Image.open(path).convert("RGB") for path in paths]
    frames[0].save(
        IMAGES / "v2_parking_occupancy_demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[1600, 2600],
        loop=0,
        optimize=True,
    )


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    domain_shift_chart()
    v1_v2_chart()
    workflow_diagram()
    repository_qr()
    demo_gif()
    for name in ["domain_shift.svg", "v1_v2_improvement.svg", "research_workflow.svg", "repository_qr.png", "v2_parking_occupancy_demo.gif"]:
        path = IMAGES / name
        if path.is_file():
            print(f"{path.relative_to(ROOT)}\t{path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

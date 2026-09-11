"""Shared publication style for the FAAT paper figures.
Self-contained (no dependency on the project skill module) so the figure
scripts are robust and portable. CVPR aesthetic: serif, ~8pt, muted palette,
FAAT highlighted.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "floats" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- palette (muted, colorblind-friendly-ish; FAAT = bold red) ----
COLORS = {
    "FAAT":      "#c0392b",
    "Ours":      "#c0392b",
    "BadNets":   "#5b8db8",
    "BadNets-C": "#5b8db8",
    "Blended":   "#e69138",
    "Blended-C": "#e69138",
    "MultiBpp-RGB": "#6a994e",
    "MultiBpp-B":   "#38703a",
    "Narcissus": "#8e7cc3",
    "BppAttack": "#a64d79",
    "KST":       "#c0392b",
    "SDT":       "#d9534f",
    "random":    "#bbbbbb",
    "res_square":"#3a6ea5",
    "forget":    "#7aa648",
}
DASH = "—"   # placeholder glyph for not-yet-filled cells


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "savefig.dpi": 600,
        "figure.dpi": 120,
        "pdf.fonttype": 42,   # embed TrueType
        "ps.fonttype": 42,
    })


def save(fig, name):
    """Save a figure to floats/figures/<name>.pdf (+ png preview)."""
    setup()
    fig.tight_layout()
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {pdf.name} (+png)")


def color(name):
    return COLORS.get(name, "#555555")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1f2a33"
GREY = "#7a869a"
RED = "#c0392b"
TEAL = "#0f7b6c"
AMBER = "#b7791f"
BLUE = "#2f6f9f"


def setup():
    plt.rcParams.update({
        "figure.figsize": (11, 5.2),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e7ec",
        "grid.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
    })
    return plt

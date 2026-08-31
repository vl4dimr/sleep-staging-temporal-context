"""
Paleta y estilo comunes a todas las figuras del manuscrito
==========================================================
Criterio: escala de grises mas UN solo acento. Ninguna informacion se
codifica unicamente por color, de modo que la figura sigue siendo legible
impresa en blanco y negro o por un lector con deficiencia de color.
"""

from matplotlib import rcParams

INK = "#1f1f1f"        # texto
LINE = "#9a9a9a"       # bordes de caja
ARROW = "#6e6e6e"      # flechas
GRID = "#e2e2e2"       # rejilla
WHITE = "#ffffff"

ACCENT = "#3d5a80"     # acento unico (azul pizarra, igual que la Figura 1)
TINT = "#e9edf3"       # version muy clara del acento, para rellenos
NEUTRAL = "#b0b0b0"    # serie de referencia en las graficas
NEUTRAL_D = "#787878"  # gris oscuro cuando hace falta una tercera serie


def apply():
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rcParams["pdf.fonttype"] = 42      # Elsevier: fuentes incrustadas
    rcParams["ps.fonttype"] = 42
    rcParams["axes.linewidth"] = 0.8
    rcParams["xtick.major.width"] = 0.8
    rcParams["ytick.major.width"] = 0.8


def clean_axes(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#707070")
    ax.tick_params(colors=INK, labelsize=8, length=3)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)

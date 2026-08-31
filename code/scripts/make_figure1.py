"""
Figura 1: pipeline de datos y arquitectura
==========================================
Formato apaisado a ancho de pagina completa (190 mm, doble columna Elsevier),
con TODO el detalle metodologico: la version compacta anterior habia perdido
informacion necesaria para reproducir el trabajo.

Esquinas redondeadas al 45% de la altura de cada caja y relleno pastel por
etapa. El color acompana al texto, nunca lo sustituye: las decisiones de
protocolo se marcan ademas con borde grueso, de modo que la figura sigue
siendo legible impresa en blanco y negro.

Las cifras se leen del modelo y de los resultados.

Salida: figuras_paper/Figura1_pipeline.{png,pdf}
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))
from models.sequence_sleep_net import SequenceSleepNet
from style import apply, INK, ARROW

apply()

OUT = ROOT / "figuras_paper"
ds = json.load(open(ROOT / "data" / "colab" / "dataset_info.json", encoding="utf-8"))
cv = json.load(open(ROOT / "resultados_colab" / "resultados" / "cv_base_gru.json",
                    encoding="utf-8"))

P_FULL = cv[0]["n_parameters"]
E_FULL = cv[0]["n_parameters_encoder"]
P_COMPACT = SequenceSleepNet(seq_len=21, variant="tiny", seq_encoder="gru",
                             use_attention=False).count_parameters()
N_SCORED = sum(r["n"] for r in cv)
L = 21
counts, S = ds["class_counts"], ds["classes"]
TOT = sum(counts)

# --- paleta "premium": tonos apagados de baja saturacion ------------------
# Rellenos casi neutros con un matiz apenas perceptible y bordes profundos
# desaturados (pizarra, salvia, arcilla, ciruela, oro viejo). La sobriedad
# viene de bajar el croma, no de oscurecer: los rellenos siguen siendo claros
# para que el texto negro conserve contraste.
PASTEL = {
    "blue":   ("#e9edf3", "#3d5a80"),   # pizarra
    "mint":   ("#eaefe8", "#5f7a61"),   # salvia
    "peach":  ("#f5ebe3", "#a9714b"),   # arcilla
    "lilac":  ("#edeaf1", "#6f6288"),   # ciruela apagada
    "sand":   ("#f4efe2", "#a08a4f"),   # oro viejo
    "green":  ("#e9efec", "#4a7862"),   # verde bosque suave
    "strip":  ("#f0f2f6", "#3d5a80"),   # franjas de resumen
}
NAVY = "#24354f"      # titulos de panel: azul tinta

# Reticula ajustada: el texto llena la caja, sin aire sobrante. El
# interlineado queda en ~1.25 veces el cuerpo de letra, que es lo que hace
# que el bloque se lea como una unidad y no como lineas sueltas.
# Las constantes van en fraccion del eje, asi que al reducir la altura de la
# figura hay que escalarlas en la misma proporcion; de lo contrario el texto,
# que va en puntos, desbordaria. S es ese factor.
SCALE = 1.44
TITLE_H, LINE_H, PAD = 0.072 * SCALE, 0.056 * SCALE, 0.019 * SCALE
ROUND = 0.0           # sin redondeo: esquinas rectas


def radius(w, h):
    """Radio de esquina. Debe calcularse sobre el lado menor: si supera la
    mitad de ese lado, los arcos de las dos esquinas se cruzan y la caja se
    dibuja como una lente."""
    return min(ROUND, 0.49) * min(w, h)


def box(ax, x, y, w, title, lines, tone="grey", key=False,
        fs_t=7.8, fs_b=6.6):
    h = PAD + TITLE_H + len(lines) * LINE_H + PAD
    fill, edge = PASTEL[tone]
    # bordes uniformes y finos, como la referencia: las decisiones de
    # protocolo se nombran en la leyenda de la figura, no con borde grueso
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius(w, h):.4f}",
        linewidth=1.1, edgecolor=edge, facecolor=fill,
        zorder=2))
    ax.text(x + w / 2, y + h - PAD - 0.004, title, ha="center", va="top",
            fontsize=fs_t, weight="bold", color=INK, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2,
                y + h - PAD - TITLE_H - i * LINE_H - LINE_H * 0.45, ln,
                ha="center", va="center", fontsize=fs_b, color=INK, zorder=3)
    return h


def strip(ax, x, y, w, h, lines, tone="grey", fs=6.6):
    """Franja informativa sin titulo."""
    fill, edge = PASTEL[tone]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius(w, h):.4f}",
        linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=2))
    for i, (txt, it) in enumerate(lines):
        ax.text(x + w / 2, y + h - (i + 0.62) * (h / len(lines)), txt,
                ha="center", va="center", fontsize=fs, color=INK,
                style="italic" if it else "normal", zorder=3)


def arrow(ax, p1, p2, label=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=ARROW, zorder=1,
                                 shrinkA=2.5, shrinkB=2.5))
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.030, label,
                ha="center", va="bottom", fontsize=6.0, style="italic",
                color=ARROW, zorder=3)


fig = plt.figure(figsize=(7.48, 3.42))         # 190 mm de ancho
gsp = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.90], hspace=0.10,
                       left=0.010, right=0.990, top=0.925, bottom=0.020)

# =========================== PANEL A ======================================
ax = fig.add_subplot(gsp[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 1.085, "(a)  Data pipeline", fontsize=11, weight="bold",
        color=NAVY, va="top")

W, GAP = 0.1735, 0.0257
xs = [0.012 + i * (W + GAP) for i in range(5)]
BOX_H = PAD + TITLE_H + 5 * LINE_H + PAD
yA = 0.965 - BOX_H          # las cajas arrancan pegadas al borde superior

specs = [
    ("Sleep-EDF-78", ["Sleep Cassette subset",
                      f"{ds['n_people']} subjects x 2 nights",
                      f"{ds['n_recordings']} recordings",
                      "Fpz-Cz at 100 Hz",
                      "checked vs EDF header"], "blue", False),
    ("Sleep-period crop", ["+/- 30 min around the",
                           "first and last",
                           "non-wake epoch",
                           "",
                           "else W is 2/3 of data"], "mint", True),
    ("Conditioning", ["band-pass 0.5-45 Hz",
                      "per-epoch z-score",
                      "",
                      "removes inter-subject",
                      "gain differences"], "peach", False),
    ("Split by PERSON", ["SC4001 and SC4002 are",
                         "the same individual",
                         "both nights, same fold",
                         "",
                         f"5-fold CV, {ds['n_people']} subjects"], "lilac", True),
    ("Windowing", [f"L = {L} consecutive epochs",
                   "never crossing recording",
                   "boundaries",
                   "train stride 7, eval disjoint",
                   "each epoch scored once"], "sand", True),
]
h = None
for x, (t, ls, tone, k) in zip(xs, specs):
    h = box(ax, x, yA, W, t, ls, tone=tone, key=k)
for i in range(4):
    arrow(ax, (xs[i] + W, yA + h / 2), (xs[i + 1], yA + h / 2))

hs = 0.150 * SCALE
ys = yA - 0.050 * SCALE - hs
xlast = xs[4] + W / 2
arrow(ax, (xlast, yA), (xlast, ys + hs))
strip(ax, 0.012, ys, 0.976, hs,
      [(f"Resulting corpus: {ds['n_epochs']:,} epochs of 30 s        "
        + "      ".join(f"{s} {100*c/TOT:.1f}%" for s, c in zip(S, counts)), False),
       (f"the five test folds sum to {N_SCORED:,}, exactly the corpus size: "
        f"no epoch is scored twice and none is omitted", True)], tone="strip")

# (sin nota de "thick borders": los bordes son uniformes; las decisiones de
#  protocolo se enumeran en la leyenda de la figura dentro del manuscrito)

# =========================== PANEL B ======================================
ax = fig.add_subplot(gsp[1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 1.10, "(b)  Two-stage model", fontsize=11, weight="bold",
        color=NAVY, va="top")

yBOX = 0.965 - BOX_H        # cajas del modelo, pegadas al borde superior
yMID = yBOX + BOX_H / 2     # eje horizontal del flujo

# entrada
xin = 0.008
hin = 0.105 * SCALE
for i in range(6):
    ax.add_patch(FancyBboxPatch(
        (xin + i * 0.0195, yMID - hin / 2), 0.015, hin,
        boxstyle=f"round,pad=0.002,rounding_size={radius(0.015, hin):.4f}",
        linewidth=0.8, edgecolor="#9a9a9a", facecolor="#ffffff", zorder=2))
ax.text(xin + 0.055, yMID + hin / 2 + 0.055, f"L = {L} epochs", ha="center",
        fontsize=6.8, color=INK)
ax.text(xin + 0.055, yMID - hin / 2 - 0.055, "3000 samples", ha="center",
        fontsize=6.2, style="italic", color=INK)
ax.text(xin + 0.055, yMID - hin / 2 - 0.125, "each", ha="center",
        fontsize=6.2, style="italic", color=INK)

# hueco amplio entre codificador y capa de secuencia: ahi va la etiqueta
# "L embeddings", que con cajas mas anchas pisaba los bordes
x1, w1 = 0.138, 0.268
x2, w2 = 0.472, 0.258
x3, w3 = 0.760, 0.228

arrow(ax, (0.120, yMID), (x1, yMID))
box(ax, x1, yBOX, w1, "Per-epoch encoder (shared)",
    ["stem conv (k=49, s=4) + max-pool",
     "4 x [ separable conv | SE | attention ]",
     "channels w, 2w, 4w, 4w",
     "global average pooling",
     f"w = 32 / 24 / 16;  {E_FULL:,} par. at w = 32"],
    tone="peach", fs_b=6.3)
arrow(ax, (x1 + w1, yMID), (x2, yMID), "L embeddings")
box(ax, x2, yBOX, w2, "Sequence layer",
    ["bidirectional GRU, 64 units",
     "or multi-head attention over",
     "the L embeddings",
     "or NONE: per-epoch",
     "classification (control)"],
    tone="lilac", key=True, fs_b=6.3)
arrow(ax, (x2 + w2, yMID), (x3, yMID))
box(ax, x3, yBOX, w3, "Shared classifier",
    ["dropout, dense, dropout,",
     "dense, softmax",
     "",
     f"{L} labels in one pass",
     "W / N1 / N2 / N3 / REM"],
    tone="green", fs_b=6.3)

# franja de configuraciones
# sin flecha: esta franja resume las configuraciones comparadas, no es un
# paso del flujo (una flecha la haria parecer salida de la capa de secuencia)
hc = 0.165 * SCALE
yc = yBOX - 0.050 * SCALE - hc
fill, edge = PASTEL["strip"]
ax.add_patch(FancyBboxPatch(
    (0.012, yc), 0.976, hc,
    boxstyle=f"round,pad=0.004,rounding_size={radius(0.976, hc):.4f}",
    linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=2))
cols = [(0.175, "full", f"{P_FULL:,} parameters",
         "w = 32, BiGRU, full encoder"),
        (0.500, "compact", f"{P_COMPACT:,} parameters",
         f"w = 16, BiGRU, no intra-epoch attention "
         f"({100*(1-P_COMPACT/P_FULL):.0f}% smaller)"),
        (0.825, "control", "no sequence layer",
         "isolates the effect of temporal context")]
for cx, name, val, desc in cols:
    ax.text(cx, yc + hc - 0.038 * SCALE, name, ha="center", fontsize=7.4,
            weight="bold", color=INK, zorder=3)
    ax.text(cx, yc + hc - 0.088 * SCALE, val, ha="center", fontsize=6.6,
            color=INK, zorder=3)
    ax.text(cx, yc + hc - 0.132 * SCALE, desc, ha="center", fontsize=6.1,
            color=INK, style="italic", zorder=3)
for sx in (0.3375, 0.6625):
    ax.plot([sx, sx], [yc + 0.020 * SCALE, yc + hc - 0.020 * SCALE], color=edge,
            lw=0.8, zorder=3)

OUT.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    p = OUT / f"Figura1_pipeline.{ext}"
    fig.savefig(p, dpi=600 if ext == "png" else None, bbox_inches="tight",
                facecolor="white")
    print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
plt.close(fig)
print(f"\nfull {P_FULL:,} | compact {P_COMPACT:,} | encoder {E_FULL:,}")

"""
Figuras 2 y 3 del manuscrito
============================
  Fig. 2  Variabilidad entre folds y tamanos de efecto en escala comun.
          Es la figura que sostiene el argumento metodologico.
  Fig. 3  Frontera precision-tamano y efecto del contexto por clase.

Paleta minima compartida con la Figura 1 (style.py): un acento y dos grises.
Las series se distinguen ademas por marcador y por posicion, de modo que la
figura sigue siendo legible en blanco y negro.

Toda cifra procede de los JSON de resultados.

Salida: figuras_paper/Figura{2,3}_*.{png,pdf}
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from style import apply, clean_axes, INK, ACCENT, NEUTRAL, NEUTRAL_D, ARROW

apply()

RES = ROOT / "resultados_colab" / "resultados"
OUT = ROOT / "figuras_paper"
STAGES = ["W", "N1", "N2", "N3", "REM"]

cv = json.load(open(RES / "cv_base_gru.json", encoding="utf-8"))
red = json.load(open(RES / "cv_tiny_gru_nolta.json", encoding="utf-8"))
t1 = json.load(open(RES / "tabla1_variantes.json", encoding="utf-8"))

K = np.array([r["kappa"] for r in cv])
KR = np.array([r["kappa"] for r in red])
D = KR - K
SD = K.std(ddof=1)
P_FULL, P_RED = cv[0]["n_parameters"], red[0]["n_parameters"]
CTX = t1["base_gru"]["kappa"] - t1["base_none"]["kappa"]
BETWEEN = K.max() - K.min()


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
    plt.close(fig)


# ============================== FIGURA 2 ==================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

ax = axes[0]
x = np.arange(len(K))
w = 0.36
ax.bar(x - w / 2, K, w, color=ACCENT, label=f"full, {P_FULL:,} par.",
       edgecolor="white", linewidth=0.5)
ax.bar(x + w / 2, KR, w, color=NEUTRAL, label=f"compact, {P_RED:,} par.",
       edgecolor="white", linewidth=0.5)
ax.axhline(K.mean(), color=ACCENT, ls="--", lw=0.9)
ax.axhline(KR.mean(), color=NEUTRAL_D, ls=":", lw=1.0)
ax.set_xticks(x, [str(i) for i in range(len(K))])
ax.set_xlabel("cross-validation fold", fontsize=8, color=INK)
ax.set_ylabel("Cohen's $\\kappa$", fontsize=8, color=INK)
ax.set_ylim(0.60, 0.80)
ax.legend(frameon=False, fontsize=6.8, loc="upper center",
          bbox_to_anchor=(0.5, 1.20), ncol=2, handlelength=1.1,
          columnspacing=1.2)
clean_axes(ax)
ax.text(-0.155, 1.22, "(a)", transform=ax.transAxes, fontsize=9.5,
        weight="bold", color=INK)

ax = axes[1]
vals = [BETWEEN, CTX, abs(D.mean())]
labs = ["between-fold\nrange\nsame model",
        "temporal\ncontext\nadded",
        "79% fewer\nparameters\nremoved"]
cols = [NEUTRAL, ACCENT, NEUTRAL_D]
bars = ax.bar(range(3), vals, 0.52, color=cols, edgecolor="white",
              linewidth=0.5)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=7.5,
                weight="bold", color=INK)
ax.set_xticks(range(3), labs, fontsize=6.8)
ax.set_ylabel("effect size, $\\Delta\\kappa$", fontsize=8, color=INK)
ax.set_ylim(0, max(vals) * 1.32)
clean_axes(ax)
ax.text(-0.155, 1.22, "(b)", transform=ax.transAxes, fontsize=9.5,
        weight="bold", color=INK)

fig.text(0.5, -0.16,
         "Fig. 2. (a) Cohen's $\\kappa$ per fold for both models under "
         "subject-level cross-validation; dashed and dotted lines are the\n"
         "respective means. (b) The same three quantities on a common scale: "
         "subject sampling moves the result more than\nremoving four fifths "
         "of the parameters does, and about as much as adding temporal "
         "context (measured on fold 0).",
         ha="center", fontsize=6.9, style="italic", color=ARROW)
fig.tight_layout()
save(fig, "Figura2_variabilidad")

# ============================== FIGURA 3 ==================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

ax = axes[0]
series = [("none", "o", NEUTRAL, "no context"),
          ("attn", "^", NEUTRAL_D, "attention"),
          ("gru", "s", ACCENT, "BiGRU")]
for e, mk, c, lab in series:
    pts = [(t1[f"{v}_{e}"]["n_parameters"], t1[f"{v}_{e}"]["kappa"])
           for v in ("tiny", "small", "base")]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=c,
            ms=5.5, lw=1.0, label=lab, markeredgecolor="white",
            markeredgewidth=0.8)
for v, dy in (("tiny", -12), ("small", 6), ("base", 6)):
    p = t1[f"{v}_gru"]
    ax.annotate(v, (p["n_parameters"], p["kappa"]), xytext=(4, dy),
                textcoords="offset points", fontsize=6.6, color=INK)
ax.axvline(100_000, color=NEUTRAL_D, ls=":", lw=0.9)
ax.text(104_000, 0.664, "100k parameters", fontsize=6.0, color=NEUTRAL_D,
        rotation=90, va="bottom")
ax.set_xscale("log")
ax.set_xlabel("total parameters, log scale", fontsize=8, color=INK)
ax.set_ylabel("Cohen's $\\kappa$", fontsize=8, color=INK)
ax.set_ylim(0.66, 0.78)
ax.legend(frameon=False, fontsize=6.8, loc="upper left",
          bbox_to_anchor=(0.02, 0.99), handlelength=1.4)
clean_axes(ax)
ax.text(-0.155, 1.07, "(a)", transform=ax.transAxes, fontsize=9.5,
        weight="bold", color=INK)

ax = axes[1]
bn, bg = t1["base_none"], t1["base_gru"]
x = np.arange(5)
w = 0.36
vn = [bn["f1_class"][s] for s in STAGES]
vg = [bg["f1_class"][s] for s in STAGES]
ax.bar(x - w / 2, vn, w, color=NEUTRAL, label="no context",
       edgecolor="white", linewidth=0.5)
ax.bar(x + w / 2, vg, w, color=ACCENT, label="with BiGRU",
       edgecolor="white", linewidth=0.5)
for i, (p, q) in enumerate(zip(vn, vg)):
    ax.annotate(f"{q-p:+.02f}", (i + w / 2, q), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=6.4,
                color=ACCENT, weight="bold")
ax.set_xticks(x, STAGES, fontsize=7.5)
ax.set_xlabel("sleep stage", fontsize=8, color=INK)
ax.set_ylabel("F1", fontsize=8, color=INK)
ax.set_ylim(0, 1.08)
ax.legend(frameon=False, fontsize=6.8, loc="lower right", handlelength=1.1)
clean_axes(ax)
ax.text(-0.155, 1.07, "(b)", transform=ax.transAxes, fontsize=9.5,
        weight="bold", color=INK)

fig.text(0.5, -0.16,
         f"Fig. 3. Both panels are evaluated on fold 0 and are not "
         f"cross-validated; differences below the between-fold standard\n"
         f"deviation of {SD:.3f} should not be interpreted. (a) Agreement "
         f"against model size: the three sequence conditions form\nseparated "
         f"bands, whereas widening the encoder moves the result little. "
         f"(b) The gain from temporal context\nconcentrates on REM and N1, "
         f"the stages whose scoring depends most on neighbouring epochs.",
         ha="center", fontsize=6.9, style="italic", color=ARROW)
fig.tight_layout()
save(fig, "Figura3_frontera")

print(f"\nbetween-fold {BETWEEN:.4f} | context {CTX:+.4f} | "
      f"compression {D.mean():+.4f} | SD {SD:.4f}")

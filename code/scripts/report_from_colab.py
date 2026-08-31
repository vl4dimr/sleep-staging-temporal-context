"""
Tablas y figuras del paper a partir de los resultados de Colab
==============================================================
Consume los tres JSON que produce el notebook:

    cv_base_gru.json        validacion cruzada por persona
    tabla1_variantes.json   3 variantes x 3 codificadores de secuencia
    tabla2_ablacion.json    componentes del codificador por epoch

y emite:

    reporte/tablas.md       tablas en Markdown
    reporte/tablas.tex      las mismas en LaTeX
    reporte/fig_cv.png      distribucion por fold con media e IC
    reporte/fig_contexto.png efecto del contexto temporal, por clase
    reporte/fig_confusion.png matriz de confusion del mejor modelo
    reporte/fig_frontera.png  precision frente a numero de parametros

Toda cifra procede de los JSON. Si falta un fichero, se omite esa seccion
y se dice explicitamente; nunca se rellena con valores inventados.

Uso:
    python scripts/report_from_colab.py --dir <carpeta con los json>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

STAGES = ["W", "N1", "N2", "N3", "REM"]

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
rcParams["savefig.dpi"] = 300
rcParams["axes.linewidth"] = 0.8

INK, GRID = "#1a1a1a", "#dcdcdc"
C1, C2, C3 = "#0B6E99", "#B5446E", "#8a8a8a"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def load(d, name):
    p = d / name
    if not p.exists():
        print(f"  [falta] {name} -> se omite la seccion correspondiente")
        return None
    return json.load(open(p, encoding="utf-8"))


# --------------------------------------------------------------- tabla CV
def section_cv(cv, md, tex):
    if not cv:
        return None
    k = np.array([r["kappa"] for r in cv])
    a = np.array([r["acc"] for r in cv])
    f = np.array([r["f1"] for r in cv])
    n = len(cv)
    # IC95 de la media por t de Student (n pequeno)
    tcrit = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
             7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n - 1, 1.96)
    ci = tcrit * k.std(ddof=1) / np.sqrt(n)

    md.append(f"### Tabla 1. Validación cruzada por persona ({n} folds)\n")
    md.append("| Fold | Personas en test | Epochs | Accuracy | κ | Macro-F1 |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for r in cv:
        md.append(f"| {r.get('fold','?')} | — | {r.get('n',0):,} | "
                  f"{r['acc']:.4f} | {r['kappa']:.4f} | {r['f1']:.4f} |")
    md.append(f"| **Media ± DE** | | | **{a.mean():.4f} ± {a.std(ddof=1):.4f}** | "
              f"**{k.mean():.4f} ± {k.std(ddof=1):.4f}** | "
              f"**{f.mean():.4f} ± {f.std(ddof=1):.4f}** |")
    md.append("")
    md.append(f"IC 95% de κ: **[{k.mean()-ci:.4f}, {k.mean()+ci:.4f}]**  ·  "
              f"rango entre folds [{k.min():.4f}, {k.max():.4f}]  ·  "
              f"amplitud {k.max()-k.min():.4f}\n")

    tex.append(r"\begin{tabular}{lrrr}\hline")
    tex.append(r"Fold & Acc. & $\kappa$ & Macro-F1 \\ \hline")
    for r in cv:
        tex.append(f"{r.get('fold','?')} & {r['acc']:.4f} & {r['kappa']:.4f} "
                   f"& {r['f1']:.4f} " + r"\\")
    tex.append(r"\hline Mean $\pm$ SD & "
               f"{a.mean():.4f} $\\pm$ {a.std(ddof=1):.4f} & "
               f"{k.mean():.4f} $\\pm$ {k.std(ddof=1):.4f} & "
               f"{f.mean():.4f} $\\pm$ {f.std(ddof=1):.4f} " + r"\\ \hline")
    tex.append(r"\end{tabular}")
    return dict(k=k, a=a, f=f, ci=ci)


def fig_cv(cv, out):
    if not cv:
        return
    k = np.array([r["kappa"] for r in cv])
    a = np.array([r["acc"] for r in cv])
    x = np.arange(len(k))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - 0.2, a, 0.38, color=C1, label="Accuracy")
    ax.bar(x + 0.2, k, 0.38, color=C2, label="Cohen's $\\kappa$")
    ax.axhline(a.mean(), color=C1, ls="--", lw=1.2, alpha=0.8)
    ax.axhline(k.mean(), color=C2, ls="--", lw=1.2, alpha=0.8)
    ax.text(len(k)-0.4, a.mean()+0.008, f"media {a.mean():.3f}",
            color=C1, fontsize=8, ha="right")
    ax.text(len(k)-0.4, k.mean()-0.022, f"media {k.mean():.3f}",
            color=C2, fontsize=8, ha="right")
    ax.set_xticks(x, [f"fold {i}" for i in range(len(k))])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("valor", fontsize=10, color=INK)
    ax.set_title("Validación cruzada por persona: variabilidad entre folds",
                 fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
    clean(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_cv.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("   fig_cv.png")


# ------------------------------------------------------------ tabla 1 vars
ENC_LABEL = {"none": "sin contexto", "gru": "BiGRU", "attn": "atención"}


def section_variants(t1, md, tex):
    if not t1:
        return
    md.append("### Tabla 2. Variantes del codificador y del modelo de secuencia\n")
    md.append("| Variante | Contexto | Parámetros | Codificador | Accuracy | κ | Macro-F1 |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    tex.append(r"\begin{tabular}{llrrrrr}\hline")
    tex.append(r"Variant & Context & Params & Encoder & Acc. & $\kappa$ & Macro-F1 \\ \hline")
    for v in ("base", "small", "tiny"):
        for e in ("none", "gru", "attn"):
            r = t1.get(f"{v}_{e}")
            if not r:
                continue
            sub = "**" if r["n_parameters"] < 100_000 else ""
            md.append(f"| {v} | {ENC_LABEL[e]} | {sub}{r['n_parameters']:,}{sub} | "
                      f"{r['n_parameters_encoder']:,} | {r['acc']:.4f} | "
                      f"{r['kappa']:.4f} | {r['f1']:.4f} |")
            tex.append(f"{v} & {ENC_LABEL[e]} & {r['n_parameters']} & "
                       f"{r['n_parameters_encoder']} & {r['acc']:.4f} & "
                       f"{r['kappa']:.4f} & {r['f1']:.4f} " + r"\\")
    tex.append(r"\hline\end{tabular}")
    md.append("\nEn **negrita**, las configuraciones por debajo de 100.000 parámetros.\n")

    # efecto del contexto por variante
    md.append("#### Aportación del contexto temporal\n")
    md.append("| Variante | κ sin contexto | κ con BiGRU | Δκ | Coste en parámetros |")
    md.append("|---|---:|---:|---:|---:|")
    for v in ("base", "small", "tiny"):
        n_, g_ = t1.get(f"{v}_none"), t1.get(f"{v}_gru")
        if not (n_ and g_):
            continue
        dp = g_["n_parameters"] - n_["n_parameters"]
        md.append(f"| {v} | {n_['kappa']:.4f} | {g_['kappa']:.4f} | "
                  f"**{g_['kappa']-n_['kappa']:+.4f}** | +{dp:,} "
                  f"({100*dp/n_['n_parameters']:.0f}%) |")
    md.append("")


def fig_context(t1, out):
    if not t1:
        return
    pairs = [(v, t1.get(f"{v}_none"), t1.get(f"{v}_gru"))
             for v in ("base", "small", "tiny")]
    pairs = [(v, a, b) for v, a, b in pairs if a and b]
    if not pairs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    x = np.arange(len(pairs))
    ax.bar(x-0.2, [p[1]["kappa"] for p in pairs], 0.38, color=C3,
           label="sin contexto")
    ax.bar(x+0.2, [p[2]["kappa"] for p in pairs], 0.38, color=C1,
           label="con BiGRU")
    for i, (_, a, b) in enumerate(pairs):
        ax.annotate(f"{b['kappa']-a['kappa']:+.3f}",
                    (i+0.2, b["kappa"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5,
                    color=C1, weight="bold")
    ax.set_xticks(x, [p[0] for p in pairs])
    ax.set_ylabel("Cohen's $\\kappa$", fontsize=10, color=INK)
    ax.set_title("Efecto del contexto temporal", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 0.9)
    clean(ax)

    ax = axes[1]
    base_n, base_g = t1.get("base_none"), t1.get("base_gru")
    if base_n and base_g:
        w = 0.38
        x = np.arange(len(STAGES))
        vn = [base_n["f1_class"][s] for s in STAGES]
        vg = [base_g["f1_class"][s] for s in STAGES]
        ax.bar(x-0.2, vn, w, color=C3, label="sin contexto")
        ax.bar(x+0.2, vg, w, color=C1, label="con BiGRU")
        for i, (p, q) in enumerate(zip(vn, vg)):
            ax.annotate(f"{q-p:+.3f}", (i+0.2, q), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8, color=C1)
        ax.set_xticks(x, STAGES)
        ax.set_ylabel("F1 por clase", fontsize=10, color=INK)
        ax.set_title("Dónde actúa el contexto (variante base)",
                     fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(0, 1.0)
        clean(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_contexto.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("   fig_contexto.png")


# ------------------------------------------------------------- ablacion
def section_ablation(t2, md, tex):
    if not t2:
        return
    ref = t2.get("completo")
    md.append("### Tabla 3. Ablación de los componentes del codificador\n")
    md.append("| Configuración | Parámetros | Accuracy | κ | Δκ |")
    md.append("|---|---:|---:|---:|---:|")
    tex.append(r"\begin{tabular}{lrrrr}\hline")
    tex.append(r"Configuration & Params & Acc. & $\kappa$ & $\Delta\kappa$ \\ \hline")
    for name, r in t2.items():
        d = "—" if (not ref or name == "completo") else f"{r['kappa']-ref['kappa']:+.4f}"
        md.append(f"| {name} | {r['n_parameters']:,} | {r['acc']:.4f} | "
                  f"{r['kappa']:.4f} | {d} |")
        tex.append(f"{name} & {r['n_parameters']} & {r['acc']:.4f} & "
                   f"{r['kappa']:.4f} & {d} " + r"\\")
    tex.append(r"\hline\end{tabular}")
    md.append("")


# ------------------------------------------------------------- confusion
def fig_confusion(t1, out):
    best = None
    for name, r in (t1 or {}).items():
        if r.get("cm") and (best is None or r["kappa"] > best[1]["kappa"]):
            best = (name, r)
    if not best:
        return
    name, r = best
    cm = np.array(r["cm"], dtype=float)
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{int(cm[i,j]):,}\n{pct[i,j]:.1f}%", ha="center",
                    va="center", fontsize=7.5,
                    color="white" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(5), STAGES); ax.set_yticks(range(5), STAGES)
    ax.set_xlabel("Predicho", fontsize=10, color=INK)
    ax.set_ylabel("Real", fontsize=10, color=INK)
    ax.set_title(f"Matriz de confusión — {name}\n"
                 f"accuracy {r['acc']*100:.1f}%  ·  $\\kappa$ {r['kappa']:.3f}"
                 f"  ·  n = {r['n']:,}", fontsize=10.5, color=INK, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("% de la clase real", fontsize=9)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("   fig_confusion.png")


def fig_frontier(t1, out):
    if not t1:
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    marks = {"none": ("o", C3), "gru": ("s", C1), "attn": ("^", C2)}
    for e in ("none", "gru", "attn"):
        pts = [(t1[f"{v}_{e}"]["n_parameters"], t1[f"{v}_{e}"]["kappa"], v)
               for v in ("tiny", "small", "base") if f"{v}_{e}" in t1]
        if not pts:
            continue
        m, c = marks[e]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], m + "-", color=c,
                ms=9, lw=1.4, alpha=0.85, label=ENC_LABEL[e],
                markeredgecolor="white", markeredgewidth=1.5)
        for x, y, v in pts:
            ax.annotate(v, (x, y), textcoords="offset points", xytext=(7, -10),
                        fontsize=8, color=INK)
    ax.axvline(100_000, color="#d0a000", ls=":", lw=1.6)
    ax.text(101_000, ax.get_ylim()[0]+0.01, "100K parámetros", fontsize=8.5,
            color="#8a6d00", rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Parámetros del modelo completo (escala log)", fontsize=10, color=INK)
    ax.set_ylabel("Cohen's $\\kappa$", fontsize=10, color=INK)
    ax.set_title("Frontera precisión–tamaño", fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    clean(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_frontera.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("   fig_frontera.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="carpeta con los JSON descargados de Drive")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    out = Path(args.out) if args.out else d.parent / "reporte"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo de {d}")
    cv = load(d, "cv_base_gru.json")
    t1 = load(d, "tabla1_variantes.json")
    t2 = load(d, "tabla2_ablacion.json")
    if not any((cv, t1, t2)):
        raise SystemExit("No se encontro ningun JSON de resultados.")

    md, tex = ["# Resultados — todas las cifras proceden de artefactos medidos\n"], []
    stats = section_cv(cv, md, tex)
    section_variants(t1, md, tex)
    section_ablation(t2, md, tex)

    (out / "tablas.md").write_text("\n".join(md), encoding="utf-8")
    (out / "tablas.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\nFiguras:")
    fig_cv(cv, out)
    fig_context(t1, out)
    fig_confusion(t1, out)
    fig_frontier(t1, out)

    print("\n" + "\n".join(md))
    if stats:
        k = stats["k"]
        print(f"\nPARA EL ABSTRACT:  kappa = {k.mean():.3f} +/- {k.std(ddof=1):.3f} "
              f"(IC95 {k.mean()-stats['ci']:.3f}-{k.mean()+stats['ci']:.3f}), "
              f"accuracy = {stats['a'].mean()*100:.1f}%, "
              f"validacion cruzada de {len(k)} folds por sujeto")
    print(f"\nEscrito en {out}")


if __name__ == "__main__":
    main()

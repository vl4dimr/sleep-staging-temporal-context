"""
Validacion cruzada del modelo reducido, en CPU local
====================================================
Evalua `tiny + BiGRU sin atencion intra-epoch` (76.085 parametros) sobre LOS
MISMOS 5 folds por persona que se usaron en Colab para `base + BiGRU`
(362.085 parametros), de modo que la comparacion sea pareada fold a fold.

Por que se puede hacer en CPU: quitar la atencion intra-epoch elimina no solo
el 59% de los parametros sino la parte cara del computo (atencion sobre 47
pasos, en cuatro bloques). Medido: 5,8x mas rapido que el modelo completo.

Los folds se reconstruyen con GroupKFold, que es determinista: mismos datos,
mismo K y mismo agrupamiento producen exactamente las mismas particiones que
en Colab. El script lo VERIFICA contra los tamanos guardados antes de
entrenar, y aborta si no coinciden.

Unica diferencia respecto a Colab: alli se uso precision mixta float16 y aqui
float32. No introduce sesgo sistematico, pero queda anotado en el JSON.

Salida (formato identico al del notebook, para report_from_colab.py):
    resultados_colab/resultados/cv_tiny_gru_nolta.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from models.sequence_sleep_net import SequenceSleepNet

STAGES = ["W", "N1", "N2", "N3", "REM"]


def recording_bounds(subj, idx):
    s = subj[idx]
    ch = np.flatnonzero(s[1:] != s[:-1]) + 1
    st = np.concatenate(([0], ch))
    en = np.concatenate((ch, [len(idx)]))
    return [(int(idx[a]), int(idx[b - 1]) + 1) for a, b in zip(st, en)]


class SeqDS(Dataset):
    """Identico al del notebook: ventanas dentro de una misma grabacion; en
    evaluacion cada epoch se puntua exactamente una vez."""

    def __init__(self, X, y, bounds, L, mode, stride=1, augment=False):
        self.X, self.y, self.L, self.mode, self.aug = X, y, L, mode, augment
        self.items = []
        for a, b in bounds:
            if b - a < L:
                continue
            if mode == "train":
                self.items += [(p, 0) for p in range(a, b - L + 1, stride)]
            else:
                p = a
                while p + L <= b:
                    self.items.append((p, 0))
                    p += L
                if p < b:
                    self.items.append((b - L, p - (b - L)))
        self.dropped = sum(b - a for a, b in bounds if b - a < L)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, vf = self.items[i]
        x = torch.from_numpy(self.X[p:p + self.L].astype(np.float32))
        y = torch.from_numpy(self.y[p:p + self.L])
        if self.aug:
            x = x * (torch.rand(1).item() * 0.45 + 0.8)
            x = x + torch.randn_like(x) * 0.05
            if torch.rand(1).item() < 0.5:
                x = -x
        m = torch.ones(self.L, dtype=torch.bool)
        if vf:
            m[:vf] = False
        return x, y, m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, T = [], []
    for x, y, m in loader:
        lg = model(x.to(device))
        p = lg.argmax(-1).cpu()
        P.append(p[m].numpy())
        T.append(y[m].numpy())
    p, t = np.concatenate(P), np.concatenate(T)
    return dict(
        acc=float(accuracy_score(t, p)),
        kappa=float(cohen_kappa_score(t, p)),
        f1=float(f1_score(t, p, average="macro")),
        f1_class={n: float(v) for n, v in
                  zip(STAGES, f1_score(t, p, average=None, labels=range(5)))},
        cm=confusion_matrix(t, p, labels=range(5)).tolist(),
        n=int(len(t)),
        majority=float(np.bincount(t, minlength=5).max() / len(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--seq-len", type=int, default=21)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    data = ROOT / "data" / "colab" / "sleepedf_all_fp16.npz"
    out_dir = ROOT / "resultados_colab" / "resultados"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cv_tiny_gru_nolta.json"

    print(f"Cargando {data.name} ...", flush=True)
    Z = np.load(data, allow_pickle=True)
    X_all, y_all = Z["X"], Z["y"].astype(np.int64)
    subject, person = Z["subject"], Z["person"]
    print(f"  {len(y_all):,} epochs | {len(set(subject.tolist()))} grabaciones "
          f"| {len(set(person.tolist()))} personas")

    gkf = GroupKFold(n_splits=args.folds)
    folds = list(gkf.split(np.arange(len(y_all)), y_all, groups=person))

    # --- verificar que los folds coinciden con los de Colab ---
    ref_path = out_dir / "cv_base_gru.json"
    ref = json.load(open(ref_path)) if ref_path.exists() else None
    print("\nfolds reconstruidos:")
    for i, (tr_i, va_i) in enumerate(folds):
        tag = ""
        if ref and i < len(ref):
            ok = len(va_i) == ref[i]["n"]
            tag = "  coincide con Colab" if ok else \
                  f"  NO COINCIDE (Colab: {ref[i]['n']})"
            if not ok:
                raise SystemExit(
                    "Los folds no coinciden con los de Colab; la comparacion "
                    "pareada seria invalida. Aborto.")
        print(f"  fold {i}: {len(tr_i):7,} train / {len(va_i):6,} test | "
              f"{len(set(person[va_i].tolist())):2d} personas{tag}")

    results = []
    if out_json.exists():
        results = json.load(open(out_json))
        print(f"\nReanudando: {len(results)} folds ya completados")

    t_all = time.time()
    for i, (tr_i, va_i) in enumerate(folds):
        if any(r["fold"] == i for r in results):
            continue
        print(f"\n=== fold {i + 1}/{len(folds)} ===", flush=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        tr = SeqDS(X_all, y_all, recording_bounds(subject, tr_i), args.seq_len,
                   "train", stride=args.stride, augment=True)
        va = SeqDS(X_all, y_all, recording_bounds(subject, va_i), args.seq_len,
                   "eval")
        tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
        vl = DataLoader(va, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=0)

        model = SequenceSleepNet(variant="tiny", seq_encoder="gru",
                                 seq_len=args.seq_len, dropout=args.dropout,
                                 use_attention=False).to(device)
        n_tot = model.count_parameters()
        n_enc = model.encoder_parameters()
        if i == 0 or not results:
            print(f"  {n_tot:,} parametros (codificador {n_enc:,})")
            print(f"  {len(tr):,} ventanas de entrenamiento, "
                  f"{len(va):,} de evaluacion")

        cnt = np.bincount(y_all[tr_i], minlength=5).astype(np.float64)
        w = torch.tensor(cnt.sum() / (5 * np.maximum(cnt, 1)),
                         dtype=torch.float32)
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best, best_r, hist = -1.0, None, []
        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            for x, y, m in tl:
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x).reshape(-1, 5), y.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            r = evaluate(model, vl, device)
            r["epoch"] = ep
            r["sec"] = time.time() - t0
            hist.append({k: r[k] for k in ("epoch", "acc", "kappa", "f1", "sec")})
            flag = ""
            if r["kappa"] > best:
                best, best_r = r["kappa"], r
                flag = "  <- mejor"
            print(f"  ep {ep:2d}/{args.epochs}  acc {r['acc']:.4f}  "
                  f"kappa {r['kappa']:.4f}  f1 {r['f1']:.4f}  "
                  f"({r['sec']:.0f}s){flag}", flush=True)

        best_r.update(n_parameters=int(n_tot), n_parameters_encoder=int(n_enc),
                      history=hist, dropped_epochs=int(va.dropped), fold=i,
                      precision="float32 (Colab uso float16 mixto)")
        results.append(best_r)
        results.sort(key=lambda r: r["fold"])
        json.dump(results, open(out_json, "w"), indent=2)
        print(f"  fold {i} -> acc {best_r['acc']:.4f}  "
              f"kappa {best_r['kappa']:.4f}  (guardado)", flush=True)

    # --- resumen y comparacion pareada ---
    k = np.array([r["kappa"] for r in results])
    a = np.array([r["acc"] for r in results])
    print(f"\n{'='*62}")
    print(f"MODELO REDUCIDO — {results[0]['n_parameters']:,} parametros, "
          f"{len(results)} folds, {(time.time()-t_all)/60:.0f} min")
    print(f"  accuracy  {a.mean():.4f} +/- {a.std(ddof=1):.4f}")
    print(f"  kappa     {k.mean():.4f} +/- {k.std(ddof=1):.4f}")

    if ref and len(ref) == len(results):
        kb = np.array([r["kappa"] for r in ref])
        d = k - kb
        print(f"\nComparacion pareada contra base+BiGRU "
              f"({ref[0]['n_parameters']:,} parametros):")
        for i, (x, y) in enumerate(zip(kb, k)):
            print(f"  fold {i}:  base {x:.4f}   reducido {y:.4f}   "
                  f"delta {y-x:+.4f}")
        print(f"  delta medio: {d.mean():+.4f} +/- {d.std(ddof=1):.4f}")
        # t pareada: se evalua el resultado, no se deja la interpretacion
        # al lector (una nota generica se lee mal cuando |t| supera el umbral)
        tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) else 0
        crit_t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                  6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(len(d) - 1, 1.96)
        sig = abs(tstat) > crit_t
        print(f"  t pareada({len(d)-1} gl) = {tstat:.3f}   "
              f"umbral al 5% = {crit_t:.3f}")
        print(f"  => diferencia {'SIGNIFICATIVA' if sig else 'NO significativa'} "
              f"al 5%" + ("" if sig else " (no se puede descartar equivalencia)"))
        if sig:
            print(f"     magnitud {abs(d.mean()):.4f}, frente a una desviacion "
                  f"entre folds de {kb.std(ddof=1):.4f}: "
                  f"{abs(d.mean())/kb.std(ddof=1):.2f} veces la variabilidad "
                  f"entre sujetos")
        print(f"  reduccion de parametros: "
              f"{100*(1-results[0]['n_parameters']/ref[0]['n_parameters']):.0f}%")
    print(f"\nEscrito en {out_json}")


if __name__ == "__main__":
    main()

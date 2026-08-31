"""
Empaqueta el corpus para Colab
==============================
Une train/val/test en un unico fichero y convierte las senales a float16.

Por que float16: las senales estan normalizadas con z-score, asi que sus
valores viven aproximadamente en [-8, 8]. float16 da ~3 cifras decimales en
ese rango, muy por encima de la resolucion util de un EEG ya normalizado.
Se reduce el tamano a la mitad sin perdida relevante, y en la GPU se
reconvierte a float32 antes de entrenar.

Por que un unico fichero: la validacion cruzada necesita repartir las 78
personas en folds, y para eso hacen falta todas juntas. La particion
train/val/test original se conserva en el array 'split' por si se quiere
reproducir el experimento de referencia.

Salida: data/colab/sleepedf_all_fp16.npz  con X, y, subject, split
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed"
DST = ROOT / "data" / "colab"
STAGES = ["W", "N1", "N2", "N3", "REM"]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    Xs, ys, subs, splits = [], [], [], []

    for split in ("train", "val", "test"):
        z = np.load(SRC / f"{split}_v2.npz", allow_pickle=True)
        X, y, s = z["X"], z["y"], z["subject"]
        print(f"{split}: {len(y):,} epochs, {len(set(s.tolist()))} grabaciones")
        Xs.append(X.astype(np.float16))
        ys.append(y.astype(np.int8))
        subs.append(s.astype("U8"))
        splits.append(np.full(len(y), split, dtype="U5"))
        del X, y, s, z

    X = np.concatenate(Xs, axis=0); del Xs
    y = np.concatenate(ys, axis=0)
    subject = np.concatenate(subs)
    split = np.concatenate(splits)

    person = np.array([s[3:5] for s in subject], dtype="U2")
    print(f"\nTOTAL {len(y):,} epochs | {len(set(subject.tolist()))} grabaciones "
          f"| {len(set(person.tolist()))} personas")
    c = np.bincount(y.astype(np.int64), minlength=5)
    print("  " + "  ".join(f"{n} {v:,} ({100*v/len(y):.1f}%)"
                           for n, v in zip(STAGES, c)))

    # comprobacion: cada grabacion sigue siendo un bloque contiguo
    changes = int((subject[1:] != subject[:-1]).sum()) + 1
    n_rec = len(set(subject.tolist()))
    assert changes == n_rec, (
        f"las grabaciones no son contiguas ({changes} bloques vs {n_rec}); "
        "las ventanas de secuencia serian invalidas")
    print("  bloques contiguos por grabacion: OK")

    out = DST / "sleepedf_all_fp16.npz"
    np.savez(out, X=X, y=y, subject=subject, person=person, split=split)
    mb = out.stat().st_size / 1024 / 1024
    print(f"\nEscrito {out}  ({mb:.0f} MB)")

    meta = {"n_epochs": int(len(y)), "n_recordings": int(n_rec),
            "n_people": int(len(set(person.tolist()))),
            "dtype_X": "float16", "sampling_rate": 100, "epoch_seconds": 30,
            "classes": STAGES, "class_counts": c.tolist(),
            "protocol": ("Sleep-EDF-78 cassette; canal Fpz-Cz; recorte al "
                         "periodo de sueno +/-30 min; filtro 0.5-45 Hz; "
                         "z-score por epoch; grabaciones contiguas en orden "
                         "temporal; 'person' = caracteres 3-4 del id, de modo "
                         "que las dos noches de un sujeto comparten persona")}
    json.dump(meta, open(DST / "dataset_info.json", "w"), indent=2)
    print(f"Escrito {DST / 'dataset_info.json'}")
    print("\nSube a Google Drive la carpeta data/colab/ completa.")


if __name__ == "__main__":
    main()

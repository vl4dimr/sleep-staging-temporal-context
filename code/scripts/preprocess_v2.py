"""
Preprocess Sleep-EDF (v2) - protocolo estandar de la literatura
===============================================================
Corrige tres defectos de preprocess_data.py:

  1. RECORTE AL PERIODO DE SUENO (+/- 30 min). Sin esto, las horas de
     vigilia grabadas antes/despues de dormir hacen que W sea ~69% del
     corpus y el modelo no supera la clase mayoritaria.
     Protocolo de Supratak et al. (2017), usado tambien por AttnSleep
     y SleepTransformer, lo que hace los numeros comparables.

  2. FILTRADO 0.5-45 Hz (elimina deriva y ruido de red).

  3. NORMALIZACION z-score por epoch. Los datos originales quedaban en
     voltios crudos (std ~ 2.7e-5), lo que desestabiliza el entrenamiento.

Mantiene el split por SUJETO de la version anterior (ya era correcto):
ningun sujeto aparece en mas de una particion.

Salida: train_v2.npz / val_v2.npz / test_v2.npz con claves X, y, subject
"""

import re
import pickle
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mne
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

STAGE_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # N3 y N4 se fusionan segun AASM
    "Sleep stage R": 4,
}
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

EPOCH_DURATION = 30      # segundos
TARGET_FS = 100          # Hz
WAKE_MARGIN_EPOCHS = 60  # 60 epochs de 30 s = 30 min a cada lado
L_FREQ, H_FREQ = 0.5, 45.0
SEED = 42


def person_of(recording_id):
    """
    Sleep-EDF nombra los ficheros SC4ssNE0, donde 'ss' identifica a la PERSONA
    y 'N' la noche. Por tanto SC4001 y SC4002 son la misma persona en dos
    noches distintas.

    Repartir por identificador de grabacion (SC4001 / SC4002) mete las dos
    noches de un mismo individuo en particiones distintas, lo que es fuga de
    sujeto: el modelo ve el EEG de esa persona al entrenar y se le evalua
    sobre ella. Esta funcion devuelve la persona, que es la unidad correcta
    para el split.
    """
    return recording_id[3:5]      # 'SC4001' -> '00'


def find_pairs(raw_dir):
    """Empareja cada PSG con su hipnograma por ID de grabacion y noche."""
    pairs = []
    for psg_path in sorted(raw_dir.glob("*PSG*.edf")):
        m = re.match(r"(SC\d{4})([EFGH])(\d)", psg_path.name)
        if not m:
            continue
        subject, letter, night = m.group(1), m.group(2), m.group(3)
        # El hipnograma comparte los 6 primeros caracteres (SC4001) y la noche
        cands = sorted(raw_dir.glob(f"{subject}{night}*-Hypnogram.edf"))
        if not cands:
            cands = sorted(raw_dir.glob(f"{subject}*-Hypnogram.edf"))
        if cands:
            pairs.append({"subject": subject, "night": night,
                          "psg": psg_path, "hyp": cands[0]})
    return pairs


def process_subject(pair):
    """Devuelve (epochs, labels) ya recortados, filtrados y normalizados."""
    subject = pair["subject"]
    try:
        raw = mne.io.read_raw_edf(str(pair["psg"]), preload=True, verbose=False)

        eeg_ch = next((c for c in raw.ch_names if "Fpz-Cz" in c), None)
        if eeg_ch is None:
            eeg_ch = next((c for c in raw.ch_names if "EEG" in c), None)
        if eeg_ch is None:
            return None, f"{subject}: sin canal EEG"
        raw.pick([eeg_ch])

        raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
        if raw.info["sfreq"] != TARGET_FS:
            raw.resample(TARGET_FS, verbose=False)

        data = raw.get_data()[0]
        fs = int(raw.info["sfreq"])
        spe = EPOCH_DURATION * fs  # muestras por epoch

        ann = mne.read_annotations(str(pair["hyp"]))

        # --- construir la secuencia de epochs en orden temporal ---
        epochs, labels = [], []
        for a in ann:
            desc = a["description"]
            if desc not in STAGE_MAPPING:
                continue          # descarta '?' y 'Movement time'
            stage = STAGE_MAPPING[desc]
            start = int(a["onset"] * fs)
            end = int((a["onset"] + a["duration"]) * fs)
            for s in range(start, end - spe + 1, spe):
                if s + spe <= len(data):
                    epochs.append(data[s:s + spe])
                    labels.append(stage)

        if not epochs:
            return None, f"{subject}: sin epochs validos"

        X = np.asarray(epochs, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)

        # --- 1. RECORTE al periodo de sueno +/- 30 min ---
        non_wake = np.flatnonzero(y != 0)
        if non_wake.size == 0:
            return None, f"{subject}: sin epochs de sueno"
        lo = max(0, non_wake[0] - WAKE_MARGIN_EPOCHS)
        hi = min(len(y), non_wake[-1] + WAKE_MARGIN_EPOCHS + 1)
        X, y = X[lo:hi], y[lo:hi]

        # --- 3. z-score por epoch ---
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        X = (X - mu) / np.maximum(sd, 1e-8)

        return {"subject": subject, "X": X.astype(np.float32), "y": y}, None

    except Exception as e:
        return None, f"{subject}: {type(e).__name__}: {str(e)[:60]}"


def main():
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(raw_dir)
    print(f"Pares PSG-Hipnograma encontrados: {len(pairs)}")
    if not pairs:
        raise SystemExit("ERROR: no se encontraron pares.")

    # --- split por PERSONA (no por grabacion), antes de procesar nada ---
    people = sorted({person_of(p["subject"]) for p in pairs})
    tr_p, tmp_p = train_test_split(people, test_size=0.30, random_state=SEED)
    va_p, te_p = train_test_split(tmp_p, test_size=0.50, random_state=SEED)
    tr_p, va_p, te_p = set(tr_p), set(va_p), set(te_p)
    assert not (tr_p & va_p) and not (tr_p & te_p) and not (va_p & te_p), \
        "fuga de persona entre particiones"

    # las grabaciones heredan la particion de su persona: las dos noches de
    # un mismo individuo caen siempre juntas
    tr_s = {p["subject"] for p in pairs if person_of(p["subject"]) in tr_p}
    va_s = {p["subject"] for p in pairs if person_of(p["subject"]) in va_p}
    te_s = {p["subject"] for p in pairs if person_of(p["subject"]) in te_p}

    print(f"Personas: {len(people)} total -> "
          f"{len(tr_p)} train / {len(va_p)} val / {len(te_p)} test")
    print(f"Grabaciones: {len(pairs)} total -> "
          f"{len(tr_s)} train / {len(va_s)} val / {len(te_s)} test")

    buckets = {"train": [], "val": [], "test": []}
    errors = []

    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(process_subject, p): p for p in pairs}
        for i, fut in enumerate(as_completed(futs), 1):
            res, err = fut.result()
            if err:
                errors.append(err)
            else:
                s = res["subject"]
                split = "train" if s in tr_s else ("val" if s in va_s else "test")
                buckets[split].append(res)
            if i % 20 == 0 or i == len(pairs):
                n = {k: sum(len(r["y"]) for r in v) for k, v in buckets.items()}
                print(f"  {i}/{len(pairs)} | train {n['train']:,} | "
                      f"val {n['val']:,} | test {n['test']:,} | errores {len(errors)}")

    meta = {"recordings_train": sorted(tr_s), "recordings_val": sorted(va_s),
            "recordings_test": sorted(te_s),
            "people_train": sorted(tr_p), "people_val": sorted(va_p),
            "people_test": sorted(te_p), "errors": errors,
            "protocol": ("sleep-period +/-30min, 0.5-45Hz, z-score per epoch; "
                         "split by PERSON (both nights of a subject stay in "
                         "the same partition)"),
            "class_names": STAGE_NAMES}

    print("\nDistribucion de clases tras el recorte:")
    for split, recs in buckets.items():
        if not recs:
            print(f"  {split}: VACIO")
            continue
        X = np.concatenate([r["X"] for r in recs], axis=0)
        y = np.concatenate([r["y"] for r in recs], axis=0)
        subj = np.concatenate([np.full(len(r["y"]), r["subject"]) for r in recs])
        c = np.bincount(y, minlength=5)
        pct = " ".join(f"{n} {v:6d} ({100*v/len(y):4.1f}%)"
                       for n, v in zip(STAGE_NAMES, c))
        print(f"  {split:5s} {len(y):7,} epochs | {pct}")
        np.savez_compressed(out_dir / f"{split}_v2.npz", X=X, y=y, subject=subj)
        meta[f"n_{split}"] = int(len(y))
        meta[f"counts_{split}"] = c.tolist()
        del X, y, subj

    with open(out_dir / "metadata_v2.pkl", "wb") as f:
        pickle.dump(meta, f)

    if errors:
        print(f"\n{len(errors)} sujetos descartados:")
        for e in errors[:10]:
            print("   ", e)
    print("\nListo -> data/processed/{train,val,test}_v2.npz")


if __name__ == "__main__":
    main()

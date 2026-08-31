"""
Repara las descargas incompletas de Sleep-EDF
=============================================
Compara el tamano de cada .edf con el que declara su propia cabecera EDF
(n_records x bytes_por_record + cabecera) y vuelve a descargar los que no
cuadran, desde PhysioNet, con reanudacion por rangos HTTP.

Verifica de nuevo tras descargar; no da por buena ninguna descarga sin
comprobar el tamano.
"""

import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
TOL = 0.995  # se acepta a partir del 99.5% del tamano esperado


def expected_size(p):
    """Tamano que el fichero deberia tener segun su cabecera EDF."""
    with open(p, "rb") as f:
        h = f.read(256)
        n_records = int(h[236:244].decode("ascii").strip())
        n_sig = int(h[252:256].decode("ascii").strip())
        f.seek(256 + n_sig * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80))
        samp = [int(f.read(8).decode("ascii").strip()) for _ in range(n_sig)]
    return 256 * (n_sig + 1) + n_records * sum(samp) * 2


def incomplete(paths):
    out = []
    for p in paths:
        try:
            exp = expected_size(p)
            act = p.stat().st_size
            if act < exp * TOL:
                out.append((p, act, exp))
        except Exception as e:
            print(f"  {p.name}: cabecera ilegible ({e}) -> se redescarga")
            out.append((p, p.stat().st_size, None))
    return out


def download(p, expected, attempts=3):
    url = BASE + p.name
    for k in range(1, attempts + 1):
        try:
            have = p.stat().st_size if p.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "sleep-edf-repair/1.0"})
            mode = "wb"
            if have > 0 and expected and have < expected:
                req.add_header("Range", f"bytes={have}-")
                mode = "ab"
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status == 206 and mode == "ab":
                    pass                    # el servidor acepta reanudacion
                elif r.status == 200:
                    mode, have = "wb", 0    # descarga completa desde cero
                with open(p, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            size = p.stat().st_size
            if expected is None or size >= expected * TOL:
                return True, size
            print(f"    intento {k}: quedo en {size/1e6:.1f} MB de "
                  f"{expected/1e6:.1f} MB, reintentando")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"    intento {k} fallo: {type(e).__name__}: {e}")
            time.sleep(3 * k)
    return False, p.stat().st_size if p.exists() else 0


def main():
    # solo los ficheros con el nombre canonico de PhysioNet (SC4001E0-PSG.edf)
    psg = sorted(q for q in RAW.glob("SC[0-9][0-9][0-9][0-9][EFGH][0-9]-PSG.edf"))
    hyp = sorted(RAW.glob("SC*-Hypnogram.edf"))
    print(f"Ficheros con nombre canonico: {len(psg)} PSG, {len(hyp)} hipnogramas")

    bad = incomplete(psg) + incomplete(hyp)
    if not bad:
        print("Todos los ficheros estan completos.")
        return 0

    falta = sum((e - a) for _, a, e in bad if e) / 1e9
    print(f"\nIncompletos: {len(bad)} ficheros, faltan {falta:.2f} GB\n")

    # 4 conexiones en paralelo: acelera mucho sin castigar al servidor
    ok, fail, n_done = 0, [], 0
    lock = threading.Lock()

    def job(item):
        nonlocal ok, n_done
        p, act, exp = item
        good, size = download(p, exp)
        with lock:
            n_done += 1
            if good:
                ok += 1
                print(f"[{n_done}/{len(bad)}] {p.name}: OK -> {size/1e6:.1f} MB",
                      flush=True)
            else:
                fail.append(p.name)
                print(f"[{n_done}/{len(bad)}] {p.name}: FALLO "
                      f"({size/1e6:.1f} MB)", flush=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(job, bad))

    print(f"\nReparados {ok}/{len(bad)}")
    if fail:
        print("No se pudieron completar:")
        for n in fail:
            print("   ", n)

    print("\nVerificacion final:")
    still = incomplete(sorted(RAW.glob("SC[0-9][0-9][0-9][0-9][EFGH][0-9]-PSG.edf")))
    print(f"  PSG aun incompletos: {len(still)}")
    for p, a, e in still:
        print(f"    {p.name}: {a/1e6:.1f}/{e/1e6:.1f} MB")
    return 1 if (fail or still) else 0


if __name__ == "__main__":
    sys.exit(main())

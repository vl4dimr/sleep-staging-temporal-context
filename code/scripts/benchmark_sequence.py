"""
Eficiencia de los modelos de secuencia (los del manuscrito)
===========================================================
Mide parametros, tamano en disco (fp32 e INT8 dinamico) y latencia de
inferencia por epoch de 30 s, para las configuraciones que aparecen en el
paper.

La latencia se reporta por epoch de 30 s clasificado, no por ventana: una
ventana de L epochs produce L etiquetas en una sola pasada, asi que el coste
amortizado por epoch es el que importa para el despliegue.

Se mide en la CPU de esta maquina, cuyo nombre queda registrado. No se
extrapola a hardware no medido.

Salida: results_v2/efficiency_sequence.json
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from models.sequence_sleep_net import SequenceSleepNet

L = 21
CONFIGS = [
    ("base+BiGRU",            dict(variant="base",  seq_encoder="gru")),
    ("base sin contexto",     dict(variant="base",  seq_encoder="none")),
    ("small+BiGRU",           dict(variant="small", seq_encoder="gru")),
    ("tiny+BiGRU",            dict(variant="tiny",  seq_encoder="gru")),
    ("tiny+atencion",         dict(variant="tiny",  seq_encoder="attn")),
    ("compacto (tiny+BiGRU sin atencion intra)",
     dict(variant="tiny", seq_encoder="gru", use_attention=False)),
]


def cpu_name():
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                text=True, stderr=subprocess.DEVNULL)
            return out.strip().splitlines()[0].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def measure(model, tmp, threads=1, n_warm=5, n_run=20):
    torch.save(model.state_dict(), tmp)
    fp32 = tmp.stat().st_size / 1024 / 1024
    tmp.unlink()

    q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.GRU},
                                            dtype=torch.qint8)
    torch.save(q.state_dict(), tmp)
    int8 = tmp.stat().st_size / 1024 / 1024
    tmp.unlink()

    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    model.eval()
    x = torch.randn(1, L, 3000)
    with torch.no_grad():
        for _ in range(n_warm):
            model(x)
        ts = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000.0)
    torch.set_num_threads(prev)
    a = np.asarray(ts)
    return dict(size_fp32_mb=round(fp32, 3), size_int8_mb=round(int8, 3),
                window_ms_mean=float(a.mean()), window_ms_std=float(a.std()),
                per_epoch_ms=float(a.mean() / L),
                per_epoch_ms_p95=float(np.percentile(a, 95) / L))


def main():
    out_dir = ROOT / "results_v2"
    out_dir.mkdir(exist_ok=True)
    tmp = out_dir / "_tmp_seq.pt"
    host = {"cpu": cpu_name(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.cuda.is_available()}
    print("Midiendo en:", host["cpu"])
    print(f"Ventana de {L} epochs; latencia por epoch = latencia de ventana / {L}\n")

    res = {"host": host, "seq_len": L,
           "note": ("Latencia medida en la CPU indicada, un solo hilo, batch=1. "
                    "No se ha medido en hardware embebido ni en GPU."),
           "configs": {}}
    print(f"{'configuracion':44s}{'params':>9s}{'fp32 MB':>9s}"
          f"{'INT8 MB':>9s}{'ms/epoch':>10s}")
    for label, kw in CONFIGS:
        m = SequenceSleepNet(seq_len=L, **kw).eval()
        r = measure(m, tmp)
        r["parameters"] = m.count_parameters()
        r["parameters_encoder"] = m.encoder_parameters()
        res["configs"][label] = r
        print(f"{label:44s}{r['parameters']:9,}{r['size_fp32_mb']:9.2f}"
              f"{r['size_int8_mb']:9.2f}{r['per_epoch_ms']:10.2f}")

    json.dump(res, open(out_dir / "efficiency_sequence.json", "w"), indent=2)
    print(f"\nGuardado en {out_dir / 'efficiency_sequence.json'}")


if __name__ == "__main__":
    main()

"""
SequenceSleepNet: EfficientSleepNet + contexto temporal
=======================================================
Motivacion medida, no supuesta. Clasificando cada epoch de 30 s de forma
aislada, EfficientSleepNet se estanca en kappa ~ 0.69 y la regularizacion
fuerte no lo mueve (0.6907 -> 0.6885): el techo no es sobreajuste, es que
un epoch aislado no contiene la informacion necesaria.

Los estadios del sueno siguen reglas de transicion (N1 es transitorio, REM
sigue a N2, no se pasa de vigilia a N3), de modo que los epochs vecinos son
informativos. Por eso DeepSleepNet, SeqSleepNet y AttnSleep clasifican
secuencias.

Arquitectura:

    L epochs de 30 s
        |
        v
    EfficientSleepNet (codificador compartido, sin cabeza clasificadora)
        |  -> un embedding de D dimensiones por epoch
        v
    codificador de secuencia (BiGRU o atencion) sobre los L embeddings
        |
        v
    clasificador compartido -> L predicciones (una por epoch)

Es many-to-many: una pasada produce las L etiquetas, asi que el coste por
epoch apenas sube respecto al modelo aislado. El codificador por epoch sigue
siendo el mismo, de modo que las variantes Base/Small/Tiny se conservan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficient_sleep_net import (EfficientSleepNet, EfficientSleepNetSmall,
                                  EfficientSleepNetTiny)

EPOCH_ENCODERS = {"base": EfficientSleepNet,
                  "small": EfficientSleepNetSmall,
                  "tiny": EfficientSleepNetTiny}


class EpochEncoder(nn.Module):
    """
    EfficientSleepNet truncado: se queda con el tronco (stem + bloques) y el
    pooling, y devuelve el embedding en lugar de los logits. Reutiliza tal
    cual los pesos y la definicion del modelo por epoch.
    """

    def __init__(self, variant="base", **kwargs):
        super().__init__()
        net = EPOCH_ENCODERS[variant](**kwargs)
        self.stem = net.stem
        self.blocks = net.blocks
        # el ancho de salida es base_filters*4, deducido del modelo construido
        self.out_dim = net.classifier[3].in_features

    def forward(self, x):                     # x: (B, 1, 3000)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return F.adaptive_avg_pool1d(x, 1).flatten(1)   # (B, out_dim)


class SequenceSleepNet(nn.Module):
    """
    Clasifica una secuencia de L epochs consecutivos de la misma grabacion.

    seq_encoder:
      'gru'  BiGRU de una capa. Barato y suele bastar para dependencias
             locales, que es lo que gobierna las transiciones de sueno.
      'attn' atencion multi-cabeza con codificacion posicional aprendida.
      'none' sin contexto: equivale al modelo aislado. Sirve como control
             del ablation, entrenado exactamente con el mismo codigo.
    """

    def __init__(self, variant="base", num_classes=5, seq_len=21,
                 seq_encoder="gru", hidden=64, num_heads=4, dropout=0.3,
                 **encoder_kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.seq_encoder_kind = seq_encoder

        self.encoder = EpochEncoder(variant, dropout=dropout, **encoder_kwargs)
        d = self.encoder.out_dim

        if seq_encoder == "gru":
            self.seq = nn.GRU(d, hidden, num_layers=1, batch_first=True,
                              bidirectional=True)
            ctx_dim = hidden * 2
        elif seq_encoder == "attn":
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.norm = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
            ctx_dim = d
        elif seq_encoder == "none":
            self.seq = None
            ctx_dim = d
        else:
            raise ValueError(f"seq_encoder desconocido: {seq_encoder}")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ctx_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        """
        x: (B, L, 3000) o (B, L, 1, 3000)
        return: (B, L, num_classes) - una prediccion por epoch de la ventana
        """
        if x.dim() == 4:
            x = x.squeeze(2)
        B, L, T = x.shape

        # el codificador se aplica a los B*L epochs de golpe
        e = self.encoder(x.reshape(B * L, 1, T)).reshape(B, L, -1)

        if self.seq_encoder_kind == "gru":
            ctx, _ = self.seq(e)
        elif self.seq_encoder_kind == "attn":
            h = self.norm(e + self.pos[:, :L])
            a, _ = self.attn(h, h, h, need_weights=False)
            ctx = e + a
        else:
            ctx = e

        return self.head(ctx)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encoder_parameters(self):
        """Parametros del codificador por epoch, que es lo que se desplegaria
        en un dispositivo si el contexto se procesara aparte."""
        return sum(p.numel() for p in self.encoder.parameters()
                   if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(2, 21, 3000)
    print(f"{'variante':8s} {'contexto':6s} {'total':>10s} {'codificador':>12s} "
          f"{'salida':>16s}")
    for v in ("base", "small", "tiny"):
        for se in ("none", "gru", "attn"):
            m = SequenceSleepNet(variant=v, seq_encoder=se, seq_len=21).eval()
            with torch.no_grad():
                y = m(x)
            print(f"{v:8s} {se:6s} {m.count_parameters():10,} "
                  f"{m.encoder_parameters():12,} {str(tuple(y.shape)):>16s}")

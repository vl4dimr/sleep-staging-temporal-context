"""
EfficientSleepNet: Lightweight Deep Learning for Real-time Sleep Stage Classification
=====================================================================================
A novel architecture combining depthwise separable convolutions with temporal attention
mechanisms for edge deployment on wearable devices.

Author: [Your Name]
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution 1D
    Reduces parameters by 8-9x compared to standard convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention
    Adaptively recalibrates channel-wise feature responses
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)


class LightweightTemporalAttention(nn.Module):
    """
    Lightweight Temporal Attention Mechanism
    Captures long-range temporal dependencies with minimal parameters
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Single linear for Q, K, V projection (more efficient)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        B, T, C = x.shape

        # Residual connection
        residual = x
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = x + residual

        # Back to (B, C, T)
        return x.transpose(1, 2)


class StandardConv1d(nn.Module):
    """
    Standard (non-separable) Conv1d block, used only as an ablation baseline
    to quantify the parameter cost of NOT using depthwise separable convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class EfficientSleepBlock(nn.Module):
    """
    Efficient Sleep Block: DSConv + SE + Temporal Attention

    The use_se / use_attention / use_dsconv flags exist for the ablation study.
    Their defaults reproduce the full model exactly.
    """
    def __init__(self, in_channels, out_channels, kernel_size=7,
                 use_se=True, use_attention=True, use_dsconv=True):
        super().__init__()
        conv_cls = DepthwiseSeparableConv1d if use_dsconv else StandardConv1d
        self.dsconv = conv_cls(
            in_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size//2
        )
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        self.attention = LightweightTemporalAttention(out_channels) if use_attention else nn.Identity()

        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.dsconv(x)
        x = self.se(x)
        x = self.attention(x)
        return x + residual


class EfficientSleepNet(nn.Module):
    """
    EfficientSleepNet: Main Model Architecture

    Input: Single-channel EEG (30s @ 100Hz = 3000 samples)
    Output: 5 sleep stages (W, N1, N2, N3, REM)

    Design principles:
    - Depthwise separable convolutions for parameter efficiency
    - Squeeze-and-excitation for channel attention
    - Lightweight temporal attention for long-range dependencies
    - Progressive downsampling for computational efficiency
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 5,
        base_filters: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.3,
        use_se: bool = True,
        use_attention: bool = True,
        use_dsconv: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes

        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=49, stride=4, padding=24),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        # Four EfficientSleepBlocks with progressive channel expansion.
        # NOTE: depth is fixed at 4 blocks; the model variants (Base/Small/Tiny)
        # differ in channel width (base_filters), not in depth. The num_blocks
        # argument is retained for API compatibility but does not change depth.
        self.num_blocks_effective = 4
        blk = lambda i, o: EfficientSleepBlock(
            i, o, kernel_size=7,
            use_se=use_se, use_attention=use_attention, use_dsconv=use_dsconv
        )
        self.blocks = nn.ModuleList([
            blk(base_filters, base_filters*2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            blk(base_filters*2, base_filters*4),
            nn.MaxPool1d(kernel_size=2, stride=2),
            blk(base_filters*4, base_filters*4),
            nn.MaxPool1d(kernel_size=2, stride=2),
            blk(base_filters*4, base_filters*4),
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_filters*4, base_filters*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_filters*2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, 1, 3000) - 30s EEG @ 100Hz
        Returns:
            logits: Output tensor of shape (batch, 5)
        """
        # Stem
        x = self.stem(x)

        # Efficient blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.classifier(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self):
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024


class EfficientSleepNetTiny(EfficientSleepNet):
    """
    Tiny variant: 73,525 parameters (base_filters=16).
    Depth is the same 4 blocks as Base; only channel width differs.
    """
    def __init__(self, num_classes=5, **kwargs):
        kwargs.setdefault("dropout", 0.2)
        super().__init__(
            input_channels=1,
            num_classes=num_classes,
            base_filters=16,
            num_blocks=3,
            **kwargs
        )


class EfficientSleepNetSmall(EfficientSleepNet):
    """
    Small variant: 162,989 parameters (base_filters=24).
    Depth is the same 4 blocks as Base; only channel width differs.
    """
    def __init__(self, num_classes=5, **kwargs):
        kwargs.setdefault("dropout", 0.25)
        super().__init__(
            input_channels=1,
            num_classes=num_classes,
            base_filters=24,
            num_blocks=4,
            **kwargs
        )


# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    Combines hard label loss with soft label loss from teacher
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft label loss (knowledge distillation)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("EfficientSleepNet Model Summary")
    print("=" * 60)

    # Create models
    models = {
        "EfficientSleepNet (Base)": EfficientSleepNet(),
        "EfficientSleepNet (Small)": EfficientSleepNetSmall(),
        "EfficientSleepNet (Tiny)": EfficientSleepNetTiny(),
    }

    # Test input: 30 seconds of EEG at 100 Hz
    x = torch.randn(2, 1, 3000)

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"\n{name}:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters:   {model.count_parameters():,}")
        print(f"  Model size:   {model.get_model_size_mb():.2f} MB")

    print("\n" + "=" * 60)
    print("All models tested successfully!")
    print("=" * 60)

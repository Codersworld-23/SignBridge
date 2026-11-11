"""
PyTorch Neural Network Suite for SignBridge.
Implements 4 progressive architectures for the academic benchmark & ablation study:
1. Model 1 (Baseline): SignBridgeBiLSTM (Bidirectional LSTM with Attention Pooling)
2. Model 2 (Local-Global): SignBridgeCNNGRU (1D Temporal CNN + Bi-GRU)
3. Model 3 (Non-Recurrent): SignBridgeTCN (Dilated Residual Temporal Convolutional Network)
4. Model 4 (State-of-the-Art): SignBridgeTransformer (Multi-Head Self-Attention Transformer Encoder)
"""

import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model 1: Baseline Bidirectional LSTM with Attention Pooling
# ============================================================================
class SignBridgeBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 258,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 262,
        dropout: float = 0.3
    ):
        super(SignBridgeBiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.spatial_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        x_proj = self.spatial_proj(x_flat).view(b, t, self.hidden_dim)

        lstm_out, _ = self.lstm(x_proj)  # (b, t, hidden_dim * 2)

        # Attention pooling
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (b, hidden_dim * 2)
        return self.classifier(context)


# ============================================================================
# Model 2: 1D Temporal CNN + Bi-GRU
# ============================================================================
class SignBridgeCNNGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = 258,
        hidden_dim: int = 128,
        num_classes: int = 262,
        dropout: float = 0.3
    ):
        super(SignBridgeCNNGRU, self).__init__()

        # Temporal Convolutions (Input: B, D, T)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D) -> transpose to (B, D, T) for 1D conv
        x_conv = x.transpose(1, 2)
        c1 = self.conv1(x_conv)
        c2 = self.conv2(c1)

        # Back to (B, T, hidden_dim) for recurrent modeling
        x_seq = c2.transpose(1, 2)
        gru_out, _ = self.gru(x_seq)  # (B, T, hidden_dim * 2)

        # Global average pool over sequence
        pooled = torch.mean(gru_out, dim=1)
        return self.classifier(pooled)


# ============================================================================
# Model 3: Dilated Residual Temporal Convolutional Network (Res-TCN)
# ============================================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))
        return F.relu(out + res)


class SignBridgeTCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 258,
        num_classes: int = 262,
        channels: list = [128, 128, 128],
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super(SignBridgeTCN, self).__init__()

        layers = []
        in_c = input_dim
        for i, out_c in enumerate(channels):
            dilation = 2 ** i  # 1, 2, 4
            layers.append(TemporalBlock(in_c, out_c, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            in_c = out_c

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T)
        x_trans = x.transpose(1, 2)
        tcn_out = self.network(x_trans)
        # Global temporal pooling
        pooled = torch.mean(tcn_out, dim=2)
        return self.classifier(pooled)


# ============================================================================
# Model 4: Multi-Head Self-Attention Transformer Encoder
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SignBridgeTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 258,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 262,
        dropout: float = 0.2
    ):
        super(SignBridgeTransformer, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer_encoder(x_pos)

        pooled = torch.mean(encoded, dim=1)
        normed = self.norm(pooled)
        return self.classifier(normed)


def get_model_by_name(name: str, input_dim: int, num_classes: int) -> nn.Module:
    """Factory method for instantiating one of the 4 benchmark architectures."""
    name_clean = name.lower().replace("-", "").replace("_", "")
    if "bilstm" in name_clean or "lstm" in name_clean:
        return SignBridgeBiLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, num_classes=num_classes)
    elif "cnngru" in name_clean or "gru" in name_clean:
        return SignBridgeCNNGRU(input_dim=input_dim, hidden_dim=128, num_classes=num_classes)
    elif "tcn" in name_clean:
        return SignBridgeTCN(input_dim=input_dim, num_classes=num_classes)
    elif "transformer" in name_clean or "attention" in name_clean:
        return SignBridgeTransformer(input_dim=input_dim, d_model=128, nhead=4, num_layers=2, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture '{name}'")

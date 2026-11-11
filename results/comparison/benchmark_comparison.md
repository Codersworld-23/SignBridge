# SignBridge 4-Model Benchmark Comparison (Held-Out Test Set)

| Architecture                 |   Top-1 Acc (%) |   Top-5 Acc (%) |   F1-Score (%) |   WER (%) |   BLEU-1 |   BLEU-4 |   ROUGE-L |   METEOR |   PER (%) |   Latency (ms) |
|:-----------------------------|----------------:|----------------:|---------------:|----------:|---------:|---------:|----------:|---------:|----------:|---------------:|
| Bi-LSTM (Baseline)           |           29.36 |           64.12 |          22.46 |     70.64 |    29.36 |     0.09 |     29.36 |    29.36 |     70.64 |           1.27 |
| CNN-BiGRU (Local-Global)     |           36.39 |           71.52 |          29.76 |     63.61 |    36.39 |     0.09 |     36.39 |    36.39 |     63.61 |           3.14 |
| Res-TCN (Dilated Temporal)   |           31.37 |           66.75 |          24.52 |     68.63 |    31.37 |     0.09 |     31.37 |    31.37 |     68.63 |           0.92 |
| Transformer (Self-Attention) |            0.63 |            2.63 |           0    |     99.37 |     0.63 |     0.03 |      0.63 |     0.63 |     99.37 |           0.59 |

*Evaluated strictly on independent 15% held-out test partition.*
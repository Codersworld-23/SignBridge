# SignBridge: A Hybrid Neuro-Fuzzy Framework for Indian Sign Language Recognition and Translation

[![Course](https://img.shields.io/badge/Coursework-Semester%207%20Fuzzy%20Logic%20%2F%20Soft%20Computing-blue)](#)
[![Dataset](https://img.shields.io/badge/Dataset-AI4Bharat%20INCLUDE%20(262%20Classes)-green)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20MediaPipe%20%7C%20Fuzzy%20Logic-orange)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#)

---

## 📖 Abstract & Executive Summary

**SignBridge** is an intelligent, real-time sign language recognition (SLR) and sign language translation (SLT) system developed for Indian Sign Language (ISL). Recognizing dynamic continuous signs from video streams is inherently challenged by rapid hand kinematics, interpersonal signing variances, and transitional non-sign artifacts between gestures. 

To overcome these challenges, SignBridge implements a **four-stage hybrid neuro-fuzzy pipeline**:
1. **Spatial Representation:** 3D skeletal landmark extraction via Google MediaPipe Holistic (258 spatial features per frame).
2. **Temporal Modeling:** A progressive four-model ablation suite encompassing **Bi-LSTM**, **CNN-BiGRU**, **Dilated Res-TCN**, and **Multi-Head Self-Attention Transformer**.
3. **Soft Computing & Decision Stabilization:** A 3-input **Mamdani Fuzzy Inference System (FIS)** that evaluates neural softmax confidence, temporal prediction agreement, and kinematic hand velocity to eliminate false-positive flicker and transitional errors.
4. **Natural Language Translation Engine:** Sequential phrase assembly with cooldown debouncing and real-time Heads-Up Display (HUD) feedback.

---

## 🏛️ System Architecture

```
                       [ Live Webcam Stream / Video Input ]
                                        │
                                        ▼
                  [ Stage 1: 3D Spatial Feature Extraction ]
                        Google MediaPipe Holistic Tracker
                 ├── 33 Pose Keypoints (x, y, z, visibility) = 132
                 ├── 21 Left Hand Keypoints (x, y, z)        =  63
                 └── 21 Right Hand Keypoints (x, y, z)       =  63
                 Total Feature Vector: D = 258 dimensions per frame
                                        │
                                        ▼
                  [ Stage 2: Temporal Sequence Normalization ]
                 Sliding Window Buffer: T = 30 frames (~1.0 sec)
                                        │
                                        ▼
               [ Stage 3: Multi-Architecture Deep Learning Suite ]
     ┌──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
     │   Model 1 (M1)       │   Model 2 (M2)       │   Model 3 (M3)       │   Model 4 (M4)       │
     │   Bi-LSTM Baseline   │   CNN-BiGRU Hybrid   │   Dilated Res-TCN    │   Transformer Encoder│
     │  (Attention Pooling) │  (Spatial-Temporal)  │ (Non-Recurrent TCN)  │ (Self-Attention)     │
     └──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
                                        │
                         Softmax Probability Vector P(C)
                                        │
                                        ▼
                  [ Stage 4: Mamdani Fuzzy Inference System (FIS) ]
     Crisp Input 1: Neural Confidence P_conf ∈ [0, 1]
     Crisp Input 2: Temporal Frame Stability S_stab ∈ [0, 1]
     Crisp Input 3: Hand Kinematic Velocity V_motion ∈ [0, 1]
                                        │
                       Mamdani Min-Max Rule Composition
                                        │
                       Centroid Defuzzification Engine
                                        │
                      Fuzzy State: [COMMIT | HOLD | REJECT]
                                        │
                                        ▼
                    [ Stage 5: Sentence Construction & HUD ]
                     Real-time Debouncing & Display Output
```

---

## 📂 Dataset Specification: AI4Bharat INCLUDE

SignBridge is evaluated on the full **AI4Bharat INCLUDE** benchmark (*ACM Multimedia 2020*), the premier large-scale corpus for Indian Sign Language recognition.

* **Total Video Takes:** 4,284 dynamic signing recordings.
* **Lexical Coverage:** **262 distinct ISL sign classes** spanning 15 semantic domains (greetings, pronouns, family, jobs, adjectives, electronics, places, time, and society).
* **Signer & Environmental Diversity:** Recorded with deaf and professional signers across diverse lighting, backgrounds, and clothing.
* **Storage & Representation:** Converted from raw multi-gigabyte video into pre-extracted, high-density MediaPipe Holistic arrays. Every recording is resampled to a standardized tensor of shape:
  $$\mathbf{X} \in \mathbb{R}^{30 \times 258}$$
* **Local Location:** `data/processed/<class_name>/sample_xxxx.npy`

---

## 🔬 Experimental Protocol: Strict 3-Way Data Partition

To eliminate data leakage and guarantee empirical reproducibility, SignBridge enforces an immutable **3-way stratified split**:

$$\text{Total Samples } (N = 4,284) \implies \begin{cases} \mathbf{Train\ (70\%):} & \sim 3,000 \text{ samples for gradient updates} \\ \mathbf{Validation\ (15\%):} & \sim 642 \text{ samples for hyperparameter tuning \& early stopping} \\ \mathbf{Test\ (15\%):} & \sim 642 \text{ samples strictly held-out for final evaluation} \end{cases}$$

The exact assignment of each sample is saved in [`results/split_manifest.json`](results/split_manifest.json) with a fixed seed ($S=42$). The test set is evaluated **only once** per architecture after all training has concluded.

---

## 🧠 Neural Architecture Suite (Ablation Study)

To provide an exhaustive comparative analysis for the coursework report, four progressive architectures are benchmarked:

### 1. Model 1: Bidirectional LSTM with Attention-Pooling (Bi-LSTM)
* **Rationale:** Establishes the sequential recurrence baseline.
* **Architecture:** 
  $$\mathbf{x}_t \longrightarrow \text{Linear}(258 \to 128) \longrightarrow \text{BatchNorm1d} \longrightarrow \text{Bi-LSTM}(2 \text{ layers}, h=128) \longrightarrow \text{Attention Weights } \alpha_t \longrightarrow \mathbf{c} = \sum \alpha_t \mathbf{h}_t \longrightarrow \text{Linear}(256 \to 262)$$
* **Strength:** Models forward and backward temporal dependencies with adaptive attention over key movement phases.

### 2. Model 2: Spatial-Temporal Hybrid (CNN-BiGRU)
* **Rationale:** Decouples local intra-sign motion primitives from global sequence context.
* **Architecture:** 
  Two 1D temporal convolution blocks ($\text{kernel}=3, 5$ with $\text{BatchNorm}$ and $\text{ReLU}$) extract short-term finger flicks and wrist trajectories, followed by a 2-layer Bidirectional GRU for global phrase dynamics.

### 3. Model 3: Dilated Residual Temporal Convolutional Network (Res-TCN)
* **Rationale:** Non-recurrent temporal architecture with exponential receptive field expansion without vanishing gradients.
* **Architecture:** 
  Stacked residual blocks with causal dilated 1D convolutions ($d \in \{1, 2, 4\}$), weight normalization, and $1 \times 1$ skip connections:
  $$\mathbf{y} = \text{Activation}(\text{ConvBlock}_d(\mathbf{x}) + \mathbf{W}_{\text{res}}\mathbf{x})$$

### 4. Model 4: Multi-Head Self-Attention Transformer Encoder
* **Rationale:** Represents state-of-the-art sequence modeling through all-to-all pairwise attention.
* **Architecture:**
  Linear token projection ($258 \to 128$) + Fixed Sinusoidal Positional Embeddings $\text{PE}_{(pos, 2i)}$ + 2 Transformer Encoder layers ($d_{\text{model}}=128$, $n_{\text{head}}=4$, $d_{\text{ff}}=256$, $\text{dropout}=0.2$) with LayerNorm and Pre-LN residual connections.

---

## 📐 Soft Computing: Mamdani Fuzzy Inference System (FIS)

Standard argmax classification fails on live camera feeds because signers generate continuous transitional movements between signs. SignBridge resolves this via a mathematically rigorous **Mamdani Fuzzy Logic Controller**.

### 1. Universes of Discourse & Membership Functions

* **Input 1: Neural Softmax Confidence ($P_{\text{conf}} \in [0.0, 1.0]$)**
  * $\mu_{\text{Low}}(x) = \text{trapmf}(x; [0.0, 0.0, 0.35, 0.55])$
  * $\mu_{\text{Medium}}(x) = \text{trimf}(x; [0.40, 0.65, 0.85])$
  * $\mu_{\text{High}}(x) = \text{trapmf}(x; [0.70, 0.85, 1.0, 1.0])$

* **Input 2: Temporal Stability Ratio ($S_{\text{stab}} \in [0.0, 1.0]$)**
  $$\text{Fraction of past } N=15 \text{ frames agreeing on candidate class } C^*$$
  * $\mu_{\text{Flickering}}(x) = \text{trapmf}(x; [0.0, 0.0, 0.25, 0.45])$
  * $\mu_{\text{Moderate}}(x) = \text{trimf}(x; [0.35, 0.60, 0.80])$
  * $\mu_{\text{Stable}}(x) = \text{trapmf}(x; [0.65, 0.80, 1.0, 1.0])$

* **Input 3: Hand Kinematic Velocity ($V_{\text{motion}} \in [0.0, 1.0]$)**
  $$\text{Normalized Euclidean inter-frame landmark displacement: } \|\mathbf{k}_t - \mathbf{k}_{t-1}\|_2$$
  * $\mu_{\text{Still}}(x) = \text{trapmf}(x; [0.0, 0.0, 0.15, 0.35])$
  * $\mu_{\text{Moving}}(x) = \text{trimf}(x; [0.20, 0.45, 0.70])$
  * $\mu_{\text{Rapid}}(x) = \text{trapmf}(x; [0.55, 0.75, 1.0, 1.0])$

* **Output Variable: Decision Commitment Score ($D \in [0.0, 1.0]$)**
  * $\mu_{\text{Reject}}(z) = \text{trapmf}(z; [0.0, 0.0, 0.20, 0.40])$
  * $\mu_{\text{Hold}}(z) = \text{trimf}(z; [0.30, 0.50, 0.70])$
  * $\mu_{\text{Accept}}(z) = \text{trapmf}(z; [0.60, 0.75, 1.0, 1.0])$

### 2. Mamdani Rule Base (Sample)
$$\begin{aligned}
R_1: & \text{ IF } P_{\text{conf}} \text{ is High } \land S_{\text{stab}} \text{ is Stable } \land V_{\text{motion}} \text{ is Still } \implies D \text{ is Accept} \\
R_2: & \text{ IF } P_{\text{conf}} \text{ is High } \land V_{\text{motion}} \text{ is Rapid } \implies D \text{ is Hold (Transition)} \\
R_3: & \text{ IF } P_{\text{conf}} \text{ is Medium } \land S_{\text{stab}} \text{ is Stable } \land V_{\text{motion}} \text{ is Still } \implies D \text{ is Hold} \\
R_4: & \text{ IF } P_{\text{conf}} \text{ is Low } \lor S_{\text{stab}} \text{ is Flickering } \implies D \text{ is Reject (Noise)}
\end{aligned}$$

### 3. Centroid Defuzzification
$$\text{Crisp Decision Score } z^* = \frac{\int_{0}^{1} z \cdot \mu_{\text{aggregated}}(z) \, dz}{\int_{0}^{1} \mu_{\text{aggregated}}(z) \, dz}$$

* **Decision Boundary:**
  * If $z^* \ge 0.65 \implies$ **`COMMIT SIGN`** (Append word to sentence buffer).
  * If $0.35 \le z^* < 0.65 \implies$ **`HOLD / TRANSITION`** (Buffer without committing).
  * If $z^* < 0.35 \implies$ **`REJECT`** (Idle / background noise).

---

## 📈 Evaluation Metrics Suite (SLR & SLT)

All models are benchmarked across both recognition and translation tasks:

| Domain | Metric | Formulation / Significance |
|---|---|---|
| **SLR** | **Top-1 Accuracy** | Exact top prediction match across 262 classes |
| **SLR** | **Top-5 Accuracy** | Top-5 categorical rank accuracy for large-scale vocabulary |
| **SLR** | **Macro / Weighted F1** | Harmonic mean of precision and recall accounting for class balance |
| **SLR** | **WER (Word Error Rate)** | $\frac{S + D + I}{N}$ computed via Levenshtein Dynamic Programming |
| **SLT** | **BLEU (1 to 4)** | Modified $n$-gram precision with brevity penalty (NLTK smooth method 1) |
| **SLT** | **ROUGE-L** | Longest Common Subsequence (LCS) F1 measure |
| **SLT** | **METEOR** | Unigram precision/recall with harmonic weighting |
| **SLT** | **PER** | Position-Independent Word Error Rate (bag-of-words distance) |
| **SLT** | **TER** | Translation Edit Rate (minimum edits to match reference) |

---

## 📁 Project Directory Structure

```
SignBridge/
├── data/
│   ├── raw/                           # Raw video clips & metadata archives
│   └── processed/                     # 262 classes, 4,284 normalized (30, 258) .npy arrays
├── models/                            # Active deployment model checkpoints (.pth)
├── results/                           # Academic Project Report Artifacts
│   ├── 1_bilstm/                      # Model 1 Dedicated Output
│   │   ├── checkpoint_best.pth        # Model weights
│   │   ├── learning_curves.png        # Epoch loss & accuracy
│   │   ├── confusion_matrix.png       # Test confusion matrix
│   │   ├── classification_report.txt  # Full precision/recall/F1 table
│   │   └── test_metrics.json          # SLR & SLT metric values
│   ├── 2_cnngru/                      # Model 2 Dedicated Output
│   ├── 3_tcn/                         # Model 3 Dedicated Output
│   ├── 4_transformer/                 # Model 4 Dedicated Output
│   ├── comparison/                    # Master Comparative Analysis
│   │   ├── benchmark_comparison.md    # Markdown table
│   │   ├── benchmark_comparison.csv   # Spreadsheet table
│   │   ├── benchmark_comparison.tex   # LaTeX table for academic paper
│   │   ├── comparison_learning_curves.png # Combined loss/accuracy curves
│   │   └── comparison_metrics_bar.png # Multi-metric comparative bar chart
│   └── split_manifest.json            # Permanent 70% Train / 15% Val / 15% Test record
├── src/
│   ├── config.py                      # Global configuration & dynamic class discovery
│   ├── keypoints_extractor.py         # MediaPipe Holistic 3D feature extractor
│   ├── fuzzy_engine.py                # Standalone Mamdani Fuzzy Inference System
│   ├── model.py                       # 4 Neural Architectures (BiLSTM, CNN-BiGRU, TCN, Transformer)
│   ├── dataset.py                     # Data augmentation & strict 3-way split loaders
│   ├── metrics.py                     # SLR & SLT evaluation metric suite
│   ├── benchmark.py                   # Automated 4-model ablation study pipeline
│   ├── train.py                       # Single architecture training loop
│   ├── inference.py                   # Real-time inference backend
│   └── convert_all_dataset.py         # Multi-threaded Parquet-to-NumPy converter
├── train_single.py                    # Manual CLI trainer for individual models
├── webcam_live.py                     # Real-time webcam interface with Fuzzy HUD
├── requirements.txt                   # Dependency specifications
└── README.md
```

---

## 🚀 Execution Guide

### 1. Manual Model Training (Individual Execution)

Train any of the 4 models individually. Results automatically route into their respective `results/<model_folder>/`:

```powershell
# Model 1: Bi-LSTM
python train_single.py --model bilstm --epochs 15

# Model 2: CNN-BiGRU
python train_single.py --model cnngru --epochs 15

# Model 3: Res-TCN
python train_single.py --model tcn --epochs 15

# Model 4: Transformer
python train_single.py --model transformer --epochs 15
```

### 2. Generate Master Comparison (Tables & Plots)

Once the models are trained, compile all results into comparative tables and charts:

```powershell
python train_single.py --compare
```

Outputs will be saved in `results/comparison/`:
* `benchmark_comparison.tex` (LaTeX table for your project report)
* `benchmark_comparison.md` & `.csv`
* `comparison_learning_curves.png`
* `comparison_metrics_bar.png`

### 3. Real-Time Webcam Translation Interface

Once training is complete, attach your webcam and launch real-time continuous sign translation:

```powershell
# Run with Bi-LSTM:
python webcam_live.py --model bilstm

# Run with Transformer:
python webcam_live.py --model transformer

# Use secondary USB webcam:
python webcam_live.py --model transformer --camera 1
```

**Interface Controls:**
* **Real-time Skeletal HUD:** Live 3D hand and pose landmarks overlaid with status telemetry.
* **Fuzzy Decision Indicator:** `[COMMIT]` (Green), `[HOLD]` (Yellow), or `[REJECT]` (Gray).
* **Keyboard Hotkeys:**
  * **`c`** : Clear sentence buffer
  * **`Backspace`** : Delete last committed word
  * **`q`** : Quit application

---

## 📚 References & Academic Citations

1. **INCLUDE Dataset:** Sridhar, A., Ganesan, R. G., Kumar, P., & Khapra, M. M. (2020). *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*. Proceedings of the 28th ACM International Conference on Multimedia (MM '20), 1366–1375.
2. **MediaPipe Holistic:** Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv:1906.08172.
3. **Fuzzy Set Theory:** Zadeh, L. A. (1965). *Fuzzy sets*. Information and Control, 8(3), 338–353.
4. **Mamdani Inference:** Mamdani, E. H., & Assilian, S. (1975). *An experiment in linguistic synthesis with a fuzzy logic controller*. International Journal of Man-Machine Studies, 7(1), 1–13.
5. **Sign Language Translation Survey:** Camgoz, N. C., et al. (2018). *Neural Sign Language Translation*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

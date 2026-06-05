# Loss-Aware Metric Learning for Efficient Person Re-Identification

Person re-identification across non-overlapping surveillance cameras using a ResNet50-IBN backbone, Generalized Mean (GeM) pooling, and a Focal Tversky Loss-centered multi-loss strategy. During inference, a distribution-aware optimized k-reciprocal re-ranking refines retrieval rankings without modifying learned features.

Published as: *Loss-Aware Metric Learning with Optimized Re-Ranking for Efficient Person Re-Identification*, in IEEE INTERSYS Conference, 2026.

---

## Results

### Market-1501

| Setting | mAP (%) | Rank-1 (%) |
|---|---|---|
| Without re-ranking | 89.4 | 95.4 |
| Default re-ranking | 94.6 | 96.2 |
| Optimized re-ranking (proposed) | **95.1** | **96.3** |

### DukeMTMC-reID

| Setting | mAP (%) | Rank-1 (%) |
|---|---|---|
| Without re-ranking | 80.8 | 90.2 |
| Default re-ranking | 89.9 | 92.2 |
| Optimized re-ranking (proposed) | **92.05** | **93.3** |

### Comparison with state-of-the-art (Market-1501)

| Method | Backbone | mAP (%) | Rank-1 (%) |
|---|---|---|---|
| PCB | ResNet50 | 77.4 | 92.3 |
| AGW | ResNet50 | 87.8 | 95.1 |
| TransReID | ViT | 89.5 | 95.2 |
| **Proposed (w/ optimized RR)** | **ResNet50-IBN** | **95.1** | **96.3** |

---

## Method

### Architecture

- **Backbone:** ResNet50-IBN — IBN blocks in layers 2 and 3 improve generalization across illumination and camera variation. Stride in layer4 modified from 2 to 1 to preserve spatial resolution.
- **Pooling:** Generalized Mean (GeM) pooling with learnable parameter p=3, balancing between average and max pooling.
- **Neck:** Batch Normalization Neck (BNNeck) — decouples metric learning from classification space. Normalized features used for retrieval; unnormalized features used for metric loss during training.

### Multi-Loss Training

Four losses jointly optimize inter-class separability and intra-class compactness:

```
L_total = L_FTL + L_CE + L_Triplet + 0.0005 * L_Center
```

- **Focal Tversky Loss (core):** Controls asymmetric false positive/negative penalties. Parameters: α=0.7, β=0.3, γ=0.75. Addresses class imbalance and hard identity samples.
- **Label-smoothed cross-entropy:** Stabilizes identity classification.
- **Batch-hard triplet loss:** Enforces inter-class separation in embedding space.
- **Center loss:** Reduces intra-class variance.

### Optimized K-Reciprocal Re-Ranking

Final distance combines Jaccard (neighbourhood structure) and Euclidean (global similarity):

```
d_final = (1 - λ) * d_Jaccard + λ * d_Euclidean
```

Parameters tuned via distribution-aware optimization: **k1=20, k2=6, λ=0.3**.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.00035 |
| Weight decay | 5e-4 |
| Epochs | 200 |
| Warmup | 10 epochs |
| LR decay epochs | 40, 70, 100 |
| Input size | 256 × 128 |
| Batch size | 32 |

Data augmentation: random horizontal flip, random crop with padding, random erasing.  
Test-time augmentation: average of original and horizontally flipped features.

---

## Datasets

| Dataset | Identities | Images | Cameras |
|---|---|---|---|
| [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) | 1,501 | 32,668 | 6 |
| [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation) | 1,404 | 36,411 | 8 |

Download datasets and place under `data/`:
```
data/
  Market-1501-v15.09.15/
    bounding_box_train/
    bounding_box_test/
    query/
  DukeMTMC-reID/
    bounding_box_train/
    bounding_box_test/
    query/
```

---

## Usage

### Installation

```bash
git clone https://github.com/036priyadharshini/hierarchical-fusion-person-re-identification
cd hierarchical-fusion-person-re-identification
pip install -r requirements.txt
```

### Evaluation

```bash
python evaluate.py \
  --data-root data/Market-1501-v15.09.15 \
  --model-path checkpoints/model.pth
```

---

## Ablation Study (Market-1501)

| Configuration | mAP (%) | Rank-1 (%) |
|---|---|---|
| Baseline (CE + Triplet) | 87.3 | 94.8 |
| + Center Loss | 88.5 | 95.1 |
| + Focal Tversky Loss | 89.4 | 95.4 |
| + Optimized Re-ranking | **95.1** | **96.3** |

---

## Stack

`PyTorch` · `ResNet50-IBN` · `GeM Pooling` · `ONNX Runtime` · `OpenCV`

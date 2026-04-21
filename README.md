# hierarchical-fusion-person-re-identification
Hierarchical fusion of gait and visual hotspot features for cross-camera person re-identification using OSNet, Hi-AFA, and multi-modal feature fusion
# Hierarchical Fusion of Gait and Visual Hotspot Feature Aggregation for Cross-Camera Person Re-Identification

**M.E. Research Project | Government College of Technology, Coimbatore | 2025**

Person re-identification across non-overlapping surveillance cameras — combining 
appearance, gait dynamics, biometric features, and attention-driven hotspot detection 
in a unified multi-modal framework.

---

## Results

| Dataset | Rank-1 | Rank-5 | mAP | IDF1 | IDSW/100 |
|---------|--------|--------|-----|------|----------|
| MARS | 91.5% | 96.1% | 88.2% | 91.2% | 1.2 |
| DukeMTMC-VideoReID | 97.2% | 99.1% | 96.8% | 96.5% | 0.6 |

---

## The Problem

Standard Re-ID systems fail when:
- People wear similar or identical clothing
- The person is partially occluded
- Camera angle changes drastically between views
- Lighting conditions vary significantly

This project addresses all four by fusing **what a person looks like** with 
**how a person moves**.

---

## Architecture

![Architecture](docs/architecture.png)

The pipeline consists of five stages:

1. **Initial Feature Extraction** — OSNet backbone processes 384×128 RGB 
   person crops through multi-scale convolution blocks, producing a 
   24×8×512 shared feature map
2. **Human Hotspot Detection** — Saliency-based attention highlights the 
   top-50 discriminative body regions (head, shoulders, torso) that carry 
   identity information even under partial occlusion
3. **Hierarchical Multi-Branch Structure (Hi-AFA)** — Four parallel branches 
   capture fine-grained textures → body parts → structural cues → global 
   appearance, with cross-branch refinement
4. **Gait Feature Extraction** — CNN-RNN model trained on CASIA-B extracts 
   a 512-D temporal motion embedding encoding stride, rhythm, and limb 
   dynamics
5. **Multi-Modal Fusion** — Appearance (8704-D) + biometric (3072-D) + 
   hotspot (512-D) + gait (512-D) → compressed to 2048-D identity embedding

---

## Key Modules

### Feature Suppression Operation (FSO)
Prevents the model from over-relying on dominant visual regions by masking 
highly activated areas during training, forcing discovery of secondary 
discriminative cues.

### Lightweight Dual Attention Module (LDAM)
Combines channel attention (which features matter) and spatial attention 
(where to focus) with minimal parameter overhead.

### Multi-Modal Fusion Weights
| Modality | Weight |
|----------|--------|
| Appearance features | 50% |
| Gait embeddings | 25% |
| Biometric part features | 15% |
| Attention hotspot cues | 10% |

---

## Sample Results

### Hotspot Detection
![Hotspot](results/hotspot_detection.png)

### Multi-Branch Feature Maps
![Multi-Branch](results/multi_branch.png)

### Tracking with Consistent IDs Across Camera Views
Persons correctly re-identified with the same ID after disappearing from 
one camera and reappearing from a different angle.

---

## Datasets

| Dataset | Identities | Sequences | Environment |
|---------|-----------|-----------|-------------|
| [MARS](https://zheng-lab.cecs.anu.edu.au/Project/project_mars.html) | 1,261 | 20,715 tracklets | University campus |
| [DukeMTMC-VideoReID](https://www.kaggle.com/datasets/leonardonaldi/duke-mtmc) | 1,404 | 2,196 tracklets | Outdoor campus |
| [CASIA-B](https://www.kaggle.com/datasets/trnquanghuyn/casia-b) | 124 subjects | 13,640 sequences | Indoor lab |

Download datasets from the links above and place them under `data/` directory.

---

## Installation

```bash
git clone https://github.com/036priyadharshini/hierarchical-fusion-person-reid
cd hierarchical-fusion-person-reid
pip install -r requirements.txt
```

---

## Training

```bash
python src/train.py \
  --dataset mars \
  --data-dir ./data/mars \
  --epochs 120 \
  --batch-size 32 \
  --gpu-id 0
```

---

## Evaluation

```bash
python src/test.py \
  --dataset mars \
  --data-dir ./data/mars \
  --model-path ./checkpoints/best_model.pth
```

---

## Tech Stack

`PyTorch` `YOLOX` `Torchreid` `ONNX Runtime` `OpenCV` `Python 3.8+`

---

## Project Report

Full methodology, literature review, and result analysis available in 
[`docs/project_report.pdf`](docs/project_report.pdf).

---

## Citation

If you find this work useful:

```
@project{priyadharshini2025reid,
  title={Hierarchical Fusion of Gait and Visual Hotspot Feature 
         Aggregation for Cross-Camera Person Re-Identification},
  author={Priyadharshini G},
  institution={Government College of Technology, Coimbatore},
  year={2025}
}
```

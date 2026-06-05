import os
import re
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from scipy.spatial.distance import cdist
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IBN(nn.Module):
    """Instance-Batch Normalization block.

    Splits channels equally between Instance Normalization and Batch
    Normalization to improve generalization across camera domains.
    """

    def __init__(self, planes: int):
        super().__init__()
        half = int(planes / 2)
        self.half = half
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(planes - half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.split(x, self.half, dim=1)
        out = torch.cat((self.IN(split[0].contiguous()),
                         self.BN(split[1].contiguous())), dim=1)
        return out

class GeM(nn.Module):
    """Generalized Mean Pooling.

    Balances between average pooling (p=1) and max pooling (p -> inf)
    via a learnable parameter p, initialized to 3.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

def build_resnet50_ibn() -> nn.Module:
    """Build ResNet-50 backbone with IBN blocks in layers 2 and 3.

    Stride in layer4 is modified from 2 to 1 to preserve spatial resolution.
    """
    resnet = models.resnet50(weights=None)

    # Stride modification in layer4
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)

    # Replace BN with IBN in layers 2 and 3
    for layer in (resnet.layer2, resnet.layer3):
        for block in layer:
            block.bn1 = IBN(block.bn1.num_features)

    return nn.Sequential(*list(resnet.children())[:-2])

class ReIDModel(nn.Module):
    """Person Re-ID model: ResNet50-IBN + GeM pooling + BNNeck.

    During training returns (classification logits, global features).
    During inference returns BN-normalized features used for retrieval.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = build_resnet50_ibn()
        self.pool = GeM(p=3.0)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        feat_map = self.backbone(x)
        global_feat = self.pool(feat_map).view(feat_map.size(0), -1)
        bn_feat = self.bottleneck(global_feat)

        if self.training:
            return self.classifier(bn_feat), global_feat
        return bn_feat

class Market1501(Dataset):
    """Market-1501 dataset loader for query and gallery splits."""

    PATTERN = re.compile(r'([-\d]+)_c(\d)')

    def __init__(self, root: str, split: str, transform):
        assert split in ('query', 'gallery', 'train')
        self.transform = transform
        split_dir = {
            'train':   'bounding_box_train',
            'query':   'query',
            'gallery': 'bounding_box_test',
        }[split]

        data_dir = Path(root) / split_dir
        self.samples = []

        for img_path in sorted(data_dir.glob('*.jpg')):
            m = self.PATTERN.search(img_path.name)
            if not m:
                continue
            pid, cam = int(m.group(1)), int(m.group(2))
            if pid == -1:
                continue
            if split == 'gallery' and pid == 0:
                continue
            self.samples.append((str(img_path), pid, cam))

        if split == 'train':
            pids = sorted({s[1] for s in self.samples})
            self._pid2label = {pid: i for i, pid in enumerate(pids)}
        else:
            self._pid2label = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, pid, cam = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        label = self._pid2label[pid] if self._pid2label else pid
        return img, label, cam

def k_reciprocal_reranking(
    query_feat: np.ndarray,
    gallery_feat: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Distribution-aware k-reciprocal re-ranking.

    Combines Jaccard distance (neighbourhood structure) with Euclidean
    distance (global similarity) as:

        d_final = (1 - lambda) * d_Jaccard + lambda * d_Euclidean

    Parameters from empirical distribution-aware optimization:
        k1=20, k2=6, lambda=0.3

    Args:
        query_feat:   (Q, D) normalized query features.
        gallery_feat: (G, D) normalized gallery features.
        k1:           k-reciprocal neighbourhood size.
        k2:           query expansion neighbourhood size.
        lambda_value: weight for original Euclidean distance.

    Returns:
        dist: (Q, G) final distance matrix.
    """
    query_feat   = F.normalize(torch.from_numpy(query_feat),   p=2, dim=1).numpy()
    gallery_feat = F.normalize(torch.from_numpy(gallery_feat), p=2, dim=1).numpy()

    all_feat = np.concatenate([query_feat, gallery_feat], axis=0)
    num_all  = all_feat.shape[0]
    num_query = query_feat.shape[0]

    original_dist = cdist(all_feat, all_feat, metric='euclidean')
    original_dist = original_dist / original_dist.max()
    original_dist = original_dist.T

    initial_rank = np.argsort(original_dist, axis=1)
    V = np.zeros((num_all, num_all), dtype=np.float32)

    for i in range(num_all):
        fwd = initial_rank[i, :k1 + 1]
        bwd = initial_rank[fwd, :k1 + 1]
        fi  = np.where(bwd == i)[0]
        k_rec = fwd[fi]

        # Expand k-reciprocal set
        expanded = k_rec.copy()
        for j in range(len(k_rec)):
            c = k_rec[j]
            half_k1 = int(np.around(k1 / 2))
            c_fwd  = initial_rank[c, :half_k1 + 1]
            c_bwd  = initial_rank[c_fwd, :half_k1 + 1]
            c_fi   = np.where(c_bwd == c)[0]
            c_krec = c_fwd[c_fi]
            overlap = len(np.intersect1d(c_krec, k_rec))
            if overlap > 2.0 / 3.0 * len(c_krec):
                expanded = np.append(expanded, c_krec)

        expanded = np.unique(expanded)
        weights  = np.exp(-original_dist[i, expanded])
        V[i, expanded] = weights / weights.sum()

    original_dist = original_dist[:num_query]

    # Query expansion
    if k2 > 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(num_all):
            V_qe[i] = V[initial_rank[i, :k2]].mean(axis=0)
        V = V_qe

    # Jaccard distance
    inv_index    = [np.where(V[:, i] != 0)[0] for i in range(num_all)]
    jaccard_dist = np.zeros((num_query, num_all), dtype=np.float32)

    for i in range(num_query):
        nonzero_idx  = np.where(V[i] != 0)[0]
        neighbor_idx = [inv_index[n] for n in nonzero_idx]
        temp_min     = np.zeros(num_all, dtype=np.float32)
        for j, nidx in enumerate(nonzero_idx):
            temp_min[neighbor_idx[j]] += np.minimum(
                V[i, nidx], V[neighbor_idx[j], nidx]
            )
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min)

    final_dist = (
        (1 - lambda_value) * jaccard_dist
        + lambda_value * original_dist
    )
    return final_dist[:num_query, num_query:]

def compute_cmc_map(
    dist: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_cams: np.ndarray,
    gallery_cams: np.ndarray,
) -> tuple:
    """Compute CMC (Rank-1/5/10) and mAP.

    Same-camera, same-identity matches are excluded following the standard
    Market-1501 evaluation protocol.
    """
    aps, cmc = [], np.zeros(len(gallery_pids))

    for i in range(len(query_pids)):
        order   = np.argsort(dist[i])
        junk    = (gallery_pids[order] == query_pids[i]) & \
                  (gallery_cams[order] == query_cams[i])
        keep    = ~junk
        matches = (gallery_pids[order] == query_pids[i])[keep]

        if not matches.any():
            continue

        cmc[np.where(matches)[0][0]:] += 1

        num_rel   = matches.sum()
        precision = matches.cumsum() / (np.arange(len(matches)) + 1)
        aps.append((precision * matches).sum() / num_rel)

    cmc /= len(query_pids)
    return np.mean(aps) * 100, cmc[0] * 100, cmc[4] * 100, cmc[9] * 100

def extract_features(
    model: nn.Module,
    loader: DataLoader,
) -> tuple:
    """Extract features with horizontal flip test-time augmentation."""
    feats, pids, cams = [], [], []

    model.eval()
    with torch.no_grad():
        for imgs, p, c in tqdm(loader, desc='Extracting', ncols=80):
            imgs = imgs.to(device)
            with autocast():
                f  = model(imgs)
                f += model(torch.flip(imgs, dims=[3]))
                f /= 2.0
            feats.append(f.cpu())
            pids.append(p)
            cams.append(c)

    return (
        torch.cat(feats).numpy(),
        torch.cat(pids).numpy(),
        torch.cat(cams).numpy(),
    )

def evaluate(data_root: str, model_path: str, num_classes: int = 751):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    query_loader = DataLoader(
        Market1501(data_root, 'query',   transform),
        batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
    )
    gallery_loader = DataLoader(
        Market1501(data_root, 'gallery', transform),
        batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
    )

    model = ReIDModel(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"Loaded: {model_path}")
    print(f"Device: {device}\n")

    qf, qp, qc = extract_features(model, query_loader)
    gf, gp, gc = extract_features(model, gallery_loader)
    print(f"Features: {len(qp)} query / {len(gp)} gallery\n")

    # Baseline (cosine distance, no re-ranking)
    qf_n = F.normalize(torch.from_numpy(qf), p=2, dim=1)
    gf_n = F.normalize(torch.from_numpy(gf), p=2, dim=1)
    dist_base = torch.cdist(qf_n, gf_n, p=2).numpy()
    map_b, r1_b, r5_b, r10_b = compute_cmc_map(dist_base, qp, gp, qc, gc)

    print("Without re-ranking")
    print(f"  mAP:    {map_b:.2f}%")
    print(f"  Rank-1: {r1_b:.2f}%")
    print(f"  Rank-5: {r5_b:.2f}%")
    print(f"  Rank-10:{r10_b:.2f}%\n")

    # With optimized re-ranking (k1=20, k2=6, lambda=0.3)
    print("Running optimized re-ranking (k1=20, k2=6, lambda=0.3)...")
    dist_rr = k_reciprocal_reranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
    map_r, r1_r, r5_r, r10_r = compute_cmc_map(dist_rr, qp, gp, qc, gc)

    print("\nWith optimized re-ranking")
    print(f"  mAP:    {map_r:.2f}%")
    print(f"  Rank-1: {r1_r:.2f}%")
    print(f"  Rank-5: {r5_r:.2f}%")
    print(f"  Rank-10:{r10_r:.2f}%\n")

    print(f"Gain:  +{map_r - map_b:.2f}% mAP  /  +{r1_r - r1_b:.2f}% Rank-1")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate ReID model on Market-1501'
    )
    parser.add_argument('--data-root',   type=str, required=True,
                        help='Path to Market-1501-v15.09.15 directory')
    parser.add_argument('--model-path',  type=str, required=True,
                        help='Path to saved model checkpoint (.pth)')
    parser.add_argument('--num-classes', type=int, default=751,
                        help='Number of training identities (default: 751)')
    args = parser.parse_args()

    evaluate(args.data_root, args.model_path, args.num_classes)

"""
engine/evaluator.py

Evaluation pipeline: extract features → distance matrix → CMC + mAP.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    """
    Runs the standard re-ID evaluation protocol.

    Args:
        model:   ReIDModel (inference mode returns L2-normalised embeddings).
        device:  torch.device.
        metric:  'cosine' | 'euclidean'.
        ranks:   List of CMC ranks to report.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        metric: str = "cosine",
        ranks: list[int] = None,
    ):
        self.model  = model
        self.device = device
        self.metric = metric
        self.ranks  = ranks or [1, 5, 10, 20]

    @torch.no_grad()
    def evaluate(self, query_loader: DataLoader, gallery_loader: DataLoader) -> dict:
        self.model.eval()
        q_feats, q_pids, q_camids = self._extract(query_loader,   "Query")
        g_feats, g_pids, g_camids = self._extract(gallery_loader, "Gallery")

        dist = self._distance_matrix(q_feats, g_feats)
        cmc, mAP = self._compute_metrics(dist, q_pids, g_pids, q_camids, g_camids)

        results = {f"rank{r}": float(cmc[r - 1]) for r in self.ranks}
        results["mAP"] = float(mAP)
        return results

    # ------------------------------------------------------------------

    def _extract(self, loader: DataLoader, tag: str):
        feats, pids, camids = [], [], []
        for batch in tqdm(loader, desc=f"Extracting {tag}", leave=False):
            imgs = batch["image"].to(self.device)
            f    = self.model(imgs)              # inference → normalised embedding
            feats.append(f.cpu())
            pids.extend(batch["pid"].tolist()   if hasattr(batch["pid"],   "tolist") else batch["pid"])
            camids.extend(batch["camid"].tolist() if hasattr(batch["camid"], "tolist") else batch["camid"])
        return torch.cat(feats, 0), np.array(pids), np.array(camids)

    def _distance_matrix(self, q: torch.Tensor, g: torch.Tensor) -> np.ndarray:
        if self.metric == "cosine":
            return (1.0 - q @ g.T).numpy()
        q_sq = q.pow(2).sum(1, keepdim=True)
        g_sq = g.pow(2).sum(1, keepdim=True)
        return (q_sq + g_sq.T - 2 * (q @ g.T)).clamp(min=0).sqrt().numpy()

    def _compute_metrics(self, dist, q_pids, g_pids, q_camids, g_camids):
        Q, max_rank = dist.shape[0], max(self.ranks)
        all_cmc = np.zeros((Q, max_rank))
        all_ap  = np.zeros(Q)
        valid   = 0

        for i in range(Q):
            order   = np.argsort(dist[i])
            gp      = g_pids[order]
            gc      = g_camids[order]

            # Exclude same-cam same-id (junk probes)
            keep    = ~((gp == q_pids[i]) & (gc == q_camids[i]))
            matches = (gp[keep] == q_pids[i]).astype(np.float32)

            if matches.sum() == 0:
                continue
            valid += 1

            cmc = np.minimum(matches.cumsum(), 1)
            all_cmc[i, :min(max_rank, len(cmc))] = cmc[:max_rank]

            num_rel = matches.sum()
            tmp     = matches.cumsum() / (np.arange(len(matches)) + 1)
            all_ap[i] = (tmp * matches).sum() / num_rel

        cmc = all_cmc[:valid].mean(axis=0)
        mAP = all_ap[:valid].mean()
        return cmc, mAP
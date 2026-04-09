"""
engine/evaluator.py

Standard re-ID evaluation: extract features → distance matrix → CMC + mAP.
Same-camera same-identity matches are excluded (Market-1501 protocol).
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, device: torch.device,
                 metric: str = "cosine", ranks: list[int] = None):
        self.model  = model
        self.device = device
        self.metric = metric
        self.ranks  = ranks or [1, 5, 10, 20]

    @torch.no_grad()
    def evaluate(self, query_loader: DataLoader, gallery_loader: DataLoader) -> dict:
        self.model.eval()
        qf, qp, qc = self._extract(query_loader,   "Query")
        gf, gp, gc = self._extract(gallery_loader, "Gallery")

        dist       = self._dist(qf, gf)
        cmc, mAP   = self._cmc_map(dist, qp, gp, qc, gc)

        results = {f"rank{r}": float(cmc[r - 1]) for r in self.ranks}
        results["mAP"] = float(mAP)
        return results

    # ------------------------------------------------------------------

    def _extract(self, loader, tag):
        feats, pids, camids = [], [], []
        for batch in tqdm(loader, desc=tag, leave=False):
            feats.append(self.model(batch["image"].to(self.device)).cpu())
            pids.extend(batch["pid"].tolist())
            camids.extend(batch["camid"].tolist())
        return torch.cat(feats), np.array(pids), np.array(camids)

    def _dist(self, q, g) -> np.ndarray:
        if self.metric == "cosine":
            return (1.0 - q @ g.T).numpy()
        qs = q.pow(2).sum(1, keepdim=True)
        gs = g.pow(2).sum(1, keepdim=True)
        return (qs + gs.T - 2 * (q @ g.T)).clamp(0).sqrt().numpy()

    def _cmc_map(self, dist, qp, gp, qc, gc):
        Q, R = dist.shape[0], max(self.ranks)
        cmc_all, ap_all, n = np.zeros((Q, R)), np.zeros(Q), 0

        for i in range(Q):
            order   = np.argsort(dist[i])
            gp_s, gc_s = gp[order], gc[order]
            keep    = ~((gp_s == qp[i]) & (gc_s == qc[i]))   # exclude same-cam junk
            hits    = (gp_s[keep] == qp[i]).astype(np.float32)
            if hits.sum() == 0:
                continue
            n += 1
            cmc_all[i, :min(R, len(hits))] = np.minimum(hits.cumsum(), 1)[:R]
            tmp = hits.cumsum() / (np.arange(len(hits)) + 1)
            ap_all[i] = (tmp * hits).sum() / hits.sum()

        return cmc_all[:n].mean(0), ap_all[:n].mean()

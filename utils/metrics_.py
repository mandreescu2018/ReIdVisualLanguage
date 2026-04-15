"""utils/metrics.py — standalone distance + rank helpers (reused by evaluator)."""
import numpy as np
import torch


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """(Q, D) × (G, D) → (Q, G) cosine distance matrix. Assumes L2-normalised inputs."""
    return (1.0 - a @ b.T).numpy()


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a_sq = a.pow(2).sum(1, keepdim=True)
    b_sq = b.pow(2).sum(1, keepdim=True)
    return (a_sq + b_sq.T - 2 * (a @ b.T)).clamp(min=0).sqrt().numpy()

def euclidean_distance2(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    # dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

if __name__ == "__main__":
    # Quick test
    q = torch.randn(3, 3)
    g = torch.randn(3, 3)

    print("Cosine distance:\n", cosine_distance(q, g))
    print("Cosine similarity:\n", cosine_similarity(q, g))    
    print("Euclidean distance:\n", euclidean_distance(q, g))
    print("Euclidean distance 2:\n", euclidean_distance2(q, g))
    print("built-in euclidean distance:\n", torch.cdist(q, g).numpy())
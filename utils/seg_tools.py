import sys

import torch
import open3d as o3d
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from math import gamma, sqrt
import math
from typing import List, Dict, Tuple

def rot_to_angle(R, eps=1e-12):
    trace = R.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta
def normalize(v: torch.Tensor, eps: float = 1e-12):
    n = torch.linalg.norm(v).clamp_min(eps)
    return v / n

def screw_from_Rt(R: torch.Tensor,
                  t: torch.Tensor,
                  eps_theta: float = 0.2,   # about 11.5°（uint：rad）
                  eps_norm: float = 1e-12):
    """
    Input:  R (3,3), t (3,)
    Output:  p, u, theta, phi, motion_type {0:translation, 1: rotation / screw}
    """
    theta = rot_to_angle(R)
    is_rotation = (theta >= eps_theta)

    if not is_rotation:
        # nearly pure translation
        t_norm = torch.linalg.norm(t)
        if t_norm < 10 * eps_norm:
            u = torch.tensor([1., 0., 0.], device=t.device, dtype=t.dtype)
        else:
            u = t / t_norm.clamp_min(eps_norm)
        phi = t_norm  # total translation distance
        motion_type = torch.tensor(0, device=t.device, dtype=torch.long)  # translation
        p = torch.zeros_like(t)  # 纯平移时轴不唯一，这里返回0向量
        theta = torch.tensor(0.0, device=t.device, dtype=t.dtype)
        return p, u, theta, phi, motion_type

    # Rotation branch
    sin_theta = torch.sin(theta).clamp_min(1e-12)
    wx = (R - R.t()) / (2.0 * sin_theta)
    u = torch.stack([wx[2, 1], wx[0, 2], wx[1, 0]])
    u = normalize(u, eps_norm)

    phi = torch.dot(u, t)

    I = torch.eye(3, device=R.device, dtype=R.dtype)
    A = I - R
    b = t - phi * u

    p = torch.linalg.pinv(A, rcond=1e-6) @ b

    motion_type = torch.tensor(1, device=t.device, dtype=torch.long)
    return p, u, theta, phi, motion_type

def skew(u: torch.Tensor) -> torch.Tensor:
    u = u / (u.norm() + 1e-8)
    return torch.tensor([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0],
    ], device=u.device, dtype=u.dtype)

def se3_log(R: torch.Tensor, t: torch.Tensor, eps: float = 1e-6):
    """
    SE(3) (R, t) → twist (w, v)
    """
    device, dtype = R.device, R.dtype
    costheta = ((R.trace() - 1) / 2).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(costheta)
    if theta.abs() < eps:
        w = torch.zeros(3, device=device, dtype=dtype)
        V_inv = torch.eye(3, device=device, dtype=dtype)
    else:
        lnR = (theta / (2 * torch.sin(theta))) * (R - R.T)
        w = torch.tensor([lnR[2, 1], lnR[0, 2], lnR[1, 0]], device=device, dtype=dtype)
        u = w / theta
        K = skew(u)
        c = torch.cos(theta)
        s = torch.sin(theta)
        V = (torch.eye(3, device=device, dtype=dtype)
             + (1 - c) / (theta ** 2) * K
             + (theta - s) / (theta ** 3) * (K @ K))
        V_inv = V.inverse()
    v = V_inv @ t
    return w, v

def se3_exp(xi):
    # xi: [6] -> (R,t)
    w = xi[:3]; v = xi[3:]
    theta = torch.linalg.norm(w)
    if theta < 1e-8:
        R = torch.eye(3, device=xi.device, dtype=xi.dtype)
        t = v
    else:
        wn = w / theta
        W = hat(wn)
        A = torch.sin(theta)
        B = 1 - torch.cos(theta)
        R = torch.eye(3, device=xi.device, dtype=xi.dtype) + A*W + B*(W@W)
        V = torch.eye(3, device=xi.device, dtype=xi.dtype) + (1 - B/theta)*W + ((theta - A)/(theta))* (W@W)
        t = (V @ v)
    return R, t

def screw_from_wv(w: torch.Tensor,
                  v: torch.Tensor,
                  eps: float = 1e-3):
    theta = w.norm()
    if theta < eps:
        phi = v.norm()
        u = v / (phi + 1e-8)
        p = torch.zeros_like(w)
        return p, u, torch.tensor(0., device=w.device, dtype=w.dtype), phi

    u = w / theta
    p = torch.cross(w, v) / (theta * theta)
    phi = torch.dot(u, v)
    return p, u, theta, phi


def fit_rigid_transform(A, B):
    # A, B: [K, 3]
    centroid_A = A.mean(dim=0)
    centroid_B = B.mean(dim=0)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am.t() @ Bm
    U, S, V = torch.svd(H)
    R = V @ U.t()
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.t()
    t = centroid_B - R @ centroid_A
    return R, t

def fit_rigid_transform_weighted(A, B, w):
    # A,B: [N,3], w: [N] in [0,1]
    w = w.clamp_min(1e-6).unsqueeze(1)         # [N,1]
    wsum = float(w.sum().item())
    ca = (w * A).sum(0, keepdim=True) / wsum
    cb = (w * B).sum(0, keepdim=True) / wsum
    A0, B0 = A - ca, B - cb
    H = (w * A0).t() @ B0
    U,S,Vt = torch.linalg.svd(H)
    R = Vt.t() @ U.t()
    if torch.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.t() @ U.t()
    t = cb.squeeze(0) - (R @ ca.squeeze(0))
    return R, t

def ransac_fit(X_0, X_1, num_iter=100, inlier_thresh=0.01, sample_points = 3):
    N = X_0.shape[0]
    best_inlier_mask = None
    best_num_inliers = 0
    best_R, best_t = None, None
    for _ in range(num_iter):
        idx = torch.randperm(N)[:sample_points]
        R, t = fit_rigid_transform(X_0[idx], X_1[idx])
        X0_pred_1 = (R @ X_0.t()).t() + t
        errs = torch.norm(X0_pred_1 - X_1, dim=1)
        inlier_mask = errs < inlier_thresh
        num_inliers = inlier_mask.sum().item()
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_R, best_t = R, t
    return best_inlier_mask, best_R, best_t

def ransac_fit_two_stage(X_0, X_05, X_1, num_iter=100, inlier_thresh=0.01, sample_points=3, knn_idx= None):
    N = X_0.shape[0]
    best_inlier_mask = None
    best_num_inliers = 0
    best_R1, best_t1, best_R2, best_t2 = None, None, None, None

    for _ in range(num_iter):
        if knn_idx is None:
            idx = torch.randperm(N)[:sample_points]
        else:
            key_idx = torch.randperm(N)[:1]
            neigh_idxs = knn_idx[key_idx]
            idx = torch.randperm(neigh_idxs.shape[0])[:sample_points]

        R2, t2 = fit_rigid_transform(X_0[idx], X_1[idx])
        R1, t1 = fit_rigid_transform(X_0[idx], X_05[idx])

        X0_pred_05 = (R1 @ X_0.t()).t() + t1
        X0_pred_1 = (R2 @ X_0.t()).t() + t2
        err_05 = torch.norm(X0_pred_05 - X_05, dim=1)
        err_1 = torch.norm(X0_pred_1 - X_1, dim=1)
        errs = (err_05  + err_1  * 2 ) / 2
        inlier_mask = errs < inlier_thresh
        num_inliers = inlier_mask.sum().item()
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_R1, best_t1 = R1, t1
            best_R2, best_t2 = R2, t2
    return best_inlier_mask, best_R1, best_t1, best_R2, best_t2

@torch.no_grad()
def em_inlier_joint_likelihood(
    errs: torch.Tensor,        # [N]
    d: int = 3,
    iters: int = 50,
    learn_pi: bool = True,
    laplace_out: bool = True,
    odds_log_tau: float = 0.0,
    beta_r: float = 1.0,
    w_thresh: float = 0.5,
):
    device = errs.device
    eps = 1e-12
    r = errs.clamp_min(0.0)

    med = r.median()
    mad = (r - med).abs().median()
    sigma = float(max((mad / 0.6745).item(), 1e-6))
    pi = 0.5

    from math import gamma
    C = 1.0 / ((2.0 ** (d/2 - 1.0)) * gamma(d/2))
    def p_in_r(rv, s):
        s = max(float(s), 1e-6)
        z = (rv / s).clamp_min(1e-12)
        return C * (z ** (d - 1)) * torch.exp(-beta_r * (z ** 2) / 2.0) / s

    if laplace_out:
        b_out = max(0.5 * float(torch.quantile(r, 0.9).item()), 1e-6)
        def p_out_r(rv): return torch.exp(-rv / b_out) / b_out
    else:
        Rmax = float(r.max().item() + 1e-6)
        def p_out_r(rv): return torch.full_like(rv, 1.0 / Rmax)

    w = torch.full_like(r, 0.5)
    for _ in range(iters):
        pin  = p_in_r(r, sigma)
        pout = p_out_r(r)
        mix  = pi * pin + (1 - pi) * pout + eps
        w    = (pi * pin) / mix

        if learn_pi:
            pi = float(min(max(w.mean().item(), 1e-3), 1 - 1e-3))
        sigma2 = float((w * (r**2)).sum().item() / max(d * w.sum().item(), 1e-12))
        sigma  = max(sigma2, 1e-12) ** 0.5

    pin = p_in_r(r, sigma); pout = p_out_r(r)
    log_odds = torch.log(pin + eps) - torch.log(pout + eps) \
             + torch.log(torch.tensor(pi + eps, device=device)) \
             - torch.log(torch.tensor(1 - pi + eps, device=device))
    mask = (log_odds > odds_log_tau)
    if w_thresh > 0:
        mask = mask & (w > w_thresh)

    return mask, w.clamp(0,1), pi, sigma,log_odds


def smooth_labels_by_spatial_voting(points, labels, neighbor_k=15, vote_thr=0.7):
    N, T, _ = points.shape
    points = points.cpu().numpy() if hasattr(points, "cpu") else points

    update_to_1 = []
    update_to_0 = []

    knn_indices_per_frame = []
    for t in range(T):
        pts_t = points[:, t, :]
        nn = NearestNeighbors(n_neighbors=neighbor_k + 1).fit(pts_t)
        knn_idx = nn.kneighbors(pts_t, return_distance=False)[:, 1:]
        knn_indices_per_frame.append(knn_idx)
    knn_indices_per_frame = np.stack(knn_indices_per_frame, axis=1)  # [N, T, neighbor_k]

    for i in range(N):
        if labels[i] == 0:
            neighbor_votes = []
            for t in range(T):
                neighbors = knn_indices_per_frame[i, t]
                neighbor_labels = labels[neighbors]
                frac_1 = np.mean(neighbor_labels == 1)
                neighbor_votes.append(frac_1 >= vote_thr)
            if all(neighbor_votes):
                update_to_1.append(i)

    for i in range(N):
        if labels[i] == 1:
            neighbor_votes = []
            for t in range(T):
                neighbors = knn_indices_per_frame[i, t]
                neighbor_labels = labels[neighbors]
                frac_0 = np.mean(neighbor_labels == 0)
                neighbor_votes.append(frac_0 >= vote_thr)
            if all(neighbor_votes):
                update_to_0.append(i)

    return update_to_1, update_to_0

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return ra
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        return ra

def recompute_model_from_indices(idx: torch.Tensor,
                                 X0: torch.Tensor, X1: torch.Tensor):
    R, t = fit_rigid_transform(X0[idx], X1[idx])
    p, u, theta, phi, motion_type = screw_from_Rt(R, t)
    rad = float(torch.median(torch.norm(X1[idx] - X1[idx].mean(0, keepdim=True), dim=1)).item())
    st  = float(torch.median(torch.norm(X1[idx] - X0[idx], dim=1)).item())
    return dict(R=R, t=t, u=u, theta=theta, phi=phi, p=p,
                rad=rad, s_t=st, idx=idx, motion_type=motion_type)

def cross_median_residual(R: torch.Tensor, t: torch.Tensor,
                          idx: torch.Tensor,
                          X0: torch.Tensor, X1: torch.Tensor) -> float:
    X0s = X0[idx]
    X1s = X1[idx]
    pred = (R @ X0s.t()).t() + t
    return float(torch.median(torch.norm(pred - X1s, dim=1)).item())


def axis_line_distance(p1: torch.Tensor, u1: torch.Tensor,
                       p2: torch.Tensor, u2: torch.Tensor) -> float:
    u1 = u1 / (u1.norm() + 1e-12)
    u2 = u2 / (u2.norm() + 1e-12)
    w0 = p1 - p2
    a = torch.dot(u1,u1); b = torch.dot(u1,u2); c = torch.dot(u2,u2)
    d = torch.dot(u1,w0); e = torch.dot(u2,w0)
    denom = a*c - b*b
    if abs(float(denom)) < 1e-9:
        return float((w0 - torch.dot(w0,u1)*u1).norm().item())
    s = (b*e - c*d) / denom
    t = (a*e - b*d) / denom
    p_cl1 = p1 + s*u1
    p_cl2 = p2 + t*u2
    return float((p_cl1 - p_cl2).norm().item())

def compatible_same_type(mi: Dict, mj: Dict,
                         X0: torch.Tensor, X1: torch.Tensor,
                         cfg: Dict, use_cross_check: bool=True) -> Tuple[bool, float]:
    ci = X1[mi['idx']].mean(0)
    cj = X1[mj['idx']].mean(0)
    print(torch.norm(ci - cj).item(),(mi['rad'] + mj['rad']) )
    if torch.norm(ci - cj).item() > 3 * (mi['rad'] + mj['rad']):
        return False, np.inf

    if mi['motion_type'] != mj['motion_type']:
        return False, np.inf

    if mi['motion_type'] == 1:
        cos_axis = torch.clamp(torch.dot(mi['u'], mj['u']) /
                               ((mi['u'].norm()+1e-12)*(mj['u'].norm()+1e-12)), -1.0, 1.0)
        ang_axis = float(torch.acos(cos_axis).item())

        if ang_axis > cfg['tau_axis']:
            return False, np.inf

        dax = axis_line_distance(mi['p'], mi['u'], mj['p'], mj['u'])

        if dax > cfg['tau_axdist']: return False, np.inf

        if abs(mi['theta'] - mj['theta']) > cfg['tau_theta']:
            return False, np.inf

        if use_cross_check:
            ri = cross_median_residual(mi['R'], mi['t'], mj['idx'], X0, X1)
            rj = cross_median_residual(mj['R'], mj['t'], mi['idx'], X0, X1)
            if not (ri < cfg['tau_res'] and rj < cfg['tau_res']):
                return False, np.inf

        score = (ang_axis/(cfg['tau_axis']+1e-9)
                 + dax/(cfg['tau_axdist']+1e-9)
                 + abs(mi['theta']-mj['theta'])/(cfg['tau_theta']+1e-9))
        return True, float(score)

    else:  # 'trans'
        ti, tj = mi['t'], mj['t']
        ni = ti.norm().item() + 1e-12; nj = tj.norm().item() + 1e-12
        cos_dir = torch.clamp(torch.dot(ti,tj)/(ni*nj), -1.0, 1.0)
        if float(cos_dir.item()) <= cfg['tau_dir']:
            return False, np.inf

        phi_i = float(mi.get('phi', ni))
        phi_j = float(mj.get('phi', nj))
        if abs(phi_i - phi_j) > cfg['tau_phi']:
            return False, np.inf

        if use_cross_check:
            ri = cross_median_residual(mi['R'], mi['t'], mj['idx'], X0, X1)
            rj = cross_median_residual(mj['R'], mj['t'], mi['idx'], X0, X1)
            if not (ri < cfg['tau_res'] and rj < cfg['tau_res']):
                return False, np.inf

        score = ((1 - float(cos_dir.item()))/(1 - cfg['tau_dir'] + 1e-9)
                 + abs(phi_i - phi_j)/(cfg['tau_phi']+1e-9))
        return True, float(score)
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
svd = torch.svd
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lib.pointops.functions import pointops
from pytorch3d.ops import ball_query
from sklearn.neighbors import NearestNeighbors
import numpy as np
from utils.seg_tools import fit_rigid_transform
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def produce_edge_matrix_nfmt(verts: torch.Tensor, edge_shape, ii, jj, nn, device="cuda") -> torch.Tensor:
	"""Given a tensor of verts postion, p (V x 3), produce a tensor E, where, for neighbour list J,
	E_in = p_i - p_(J[n])"""

	E = torch.zeros(edge_shape).to(device)
	E[ii, nn] = verts[ii] - verts[jj]

	return E


def estimate_rotation(source, target, ii, jj, nn, K=10, weight=None, sample_idx=None):
    # input: source, target: [Nv, 3]; ii, jj, nn: [Ne,], weight: [Nv, K]
    # output: rotation: [Nv, 3, 3]
    Nv = len(source)
    source_edge_mat = produce_edge_matrix_nfmt(source, (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    target_edge_mat = produce_edge_matrix_nfmt(target, (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1
        print("!!! Edge weight is None !!!")
    if sample_idx is not None:
        source_edge_mat = source_edge_mat[sample_idx]
        target_edge_mat = target_edge_mat[sample_idx]
    ### Calculate covariance matrix in bulk
    D = torch.diag_embed(weight, dim1=1, dim2=2)  # [Nv, K, K]
    # S = torch.bmm(source_edge_mat.permute(0, 2, 1), target_edge_mat)  # [Nv, 3, 3]
    S = torch.bmm(source_edge_mat.permute(0, 2, 1), torch.bmm(D, target_edge_mat))  # [Nv, 3, 3]
    ## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
    unchanged_verts = torch.unique(
        torch.where((source_edge_mat == target_edge_mat).all(dim=1))[0])  # any verts which are undeformed
    S[unchanged_verts] = 0

    # t2 = time.time()
    U, sig, W = svd(S)
    R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations
    # t3 = time.time()

    # Need to flip the column of U corresponding to smallest singular value
    # for any det(Ri) <= 0
    entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
    if len(entries_to_flip) > 0:
        Umod = U.clone()
        cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
        Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
        R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))
    # t4 = time.time()
    # print(f'0-1: {t1-t0}, 1-2: {t2-t1}, 2-3: {t3-t2}, 3-4: {t4-t3}')
    return R


def cal_connectivity_from_points_v2(points=None, radius=0.1, K=10):
    # input: [T, Nv,3]
    # output: information of edges
    # ii : [Ne,] the i th vert
    # jj: [Ne,] j th vert is connect to i th vert.
    # nn: ,  [Ne,] the n th neighbour of i th vert is j th vert.
    T, Nv = points.shape[0], points.shape[1]
    nn_dist, nn_idx, nns = ball_query(points, points, K= K + 1, radius=radius)
    nn_dist, nn_idx, nns = nn_dist[:, :, 1:], nn_idx[:, :, 1:], nns[:, :, 1:]
    nn_dist_ori, nn_idx_ori = nn_dist, nn_idx
    nn_idx = F.one_hot(nn_idx + 1, num_classes=nn_idx.shape[1] + 1).to(torch.bool)
    nn_idx = nn_idx.any(dim=2).all(dim=0).to(torch.float)
    nn_idx[:, 0] = 0.0
    num_nonzero = nn_idx.sum(dim=1).to(torch.uint8)
    _, nn_idx = torch.topk(nn_idx, k=K, dim=1, largest=True)
    nn_idx = (nn_idx - 1).abs()

    ii = torch.arange(Nv)[:, None].cuda().long().expand(Nv, K)
    jj = nn_idx
    nn = torch.arange(K)[None].cuda().long().expand(Nv, K)
    # mask = jj != -1
    indices = torch.arange(nn_idx.shape[1]).expand_as(nn_idx).to("cuda")
    mask = indices < num_nonzero[:, None]
    ii, jj, nn = ii[mask], jj[mask], nn[mask]

    return ii, jj, nn, _


def cal_arap_error(nodes_sequence, ii, jj, nn, K=10, weight=None, sample_num= 4096):
    # input: nodes_sequence: [Nt, Nv, 3]; ii, jj, nn: [Ne,], weight: [Nv, K]
    # output: arap error: float
    Nt, Nv, _ = nodes_sequence.shape
    # laplacian_mat = cal_laplacian(Nv, ii, jj, nn)  # [Nv, Nv]
    # laplacian_mat_inv = invert_matrix(laplacian_mat)
    arap_error = 0
    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1
    source_edge_mat = produce_edge_matrix_nfmt(nodes_sequence[0], (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    sample_idx = torch.arange(Nv).cuda()
    if Nv > sample_num:
        sample_idx = torch.from_numpy(np.random.choice(Nv, sample_num)).long().cuda()
    else:
        source_edge_mat = source_edge_mat[sample_idx]
    weight = weight[sample_idx]
    for idx in range(1, Nt):
        # t1 = time.time()
        with torch.no_grad():
            rotation = estimate_rotation(nodes_sequence[0], nodes_sequence[idx], ii, jj, nn, K=K, weight=weight, sample_idx=sample_idx)  # [Nv, 3, 3]
        # Compute energy
        target_edge_mat = produce_edge_matrix_nfmt(nodes_sequence[idx], (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
        target_edge_mat = target_edge_mat[sample_idx]
        rot_rigid = torch.bmm(rotation, source_edge_mat[sample_idx].permute(0, 2, 1)).permute(0, 2, 1)  # [Nv, K, 3]
        stretch_vec = target_edge_mat - rot_rigid  # stretch vector
        stretch_norm = (torch.norm(stretch_vec, dim=2) ** 2)  # norm over (x,y,z) space
        arap_error += (weight * stretch_norm).sum()
    return arap_error

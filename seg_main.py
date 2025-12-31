import os
import sys
from marshal import loads
from traceback import print_tb

from PIL.ImageOps import deform
from PIL import Image

from urllib3 import proxy_from_url
import tqdm
import json
import torch
import torch.nn.functional as F
import numpy as np
from random import randint
from posix import device_encoding
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

from argparse import ArgumentParser, Namespace

try:
    from tensorboardX import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.general_utils import safe_state, get_linear_noise_func
from utils.system_utils import save_input_points, save_point_cloud_ply
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel, DeformModel
from utils.loss_utils import l1_loss, ssim, cal_connectivity_from_points_v2, cal_arap_error
from gaussian_renderer import render, render_mix
from train import training_report
from utils.seg_tools import se3_log, screw_from_wv, screw_from_Rt, fit_rigid_transform, ransac_fit, \
    em_inlier_joint_likelihood, fit_rigid_transform_weighted, compatible_same_type, UnionFind, recompute_model_from_indices
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


class AiM:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.tb_writer = prepare_output_and_logger(dataset)

        self.start_static_gaussian = GaussianModel(dataset.sh_degree)
        self.end_static_scene = Scene(dataset, gaussians=self.start_static_gaussian, load_iteration=None,
                                      state="end", init_point_size="large")

        self.start_static_scene = Scene(dataset, gaussians=self.start_static_gaussian, load_iteration=-1,
                                        state="start", init_point_size="large")
        self.start_static_gaussian.training_setup(opt)

        self.moving_gaussian = GaussianModel(dataset.sh_degree)
        self.motion_scene = Scene(dataset, gaussians=self.moving_gaussian, load_iteration=-1, state="motion",
                                  init_point_size="small")
        self.moving_gaussian.training_setup(opt)

        self.deform = DeformModel(is_blender=dataset.is_blender, is_6dof=dataset.is_6dof)
        self.deform.train_setting(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)

        self.total_iteration = 1
        self.start_static_iteration = 1
        self.motion_iteration = 1

        self.start_static_viewpoint_stack = None
        self.motion_viewpoint_stack = None

        self.motion_R_set = []
        self.motion_t_set = []
        self.motion_assignment = None
        self.sub_gaussian_set = []
        self.num_static = None
        self.num_dynamic = None
        self.static_mask = None
        self.dynamic_mask = None

    def init_sub_gs(self, old_gaussian, new_gaussian, sample_idx, id=None, vis=False, state=None):
        new_gaussian.copy_from_sampled_gaussian(old_gaussian, sample_idx)
        if vis:
            if state is None:
                new_gaussian.save_ply(self.args.model_path + f"/sub_{id}_point_cloud.ply")
            else:
                new_gaussian.save_ply(self.args.model_path + f"/sub_{id}_point_cloud_{state}.ply")

    def segment(self):
        self.deform.load_weights(self.dataset.model_path)
        self.deform.deform.eval()

        self.num_static = self.start_static_gaussian.get_xyz.shape[0]
        self.num_dynamic = self.moving_gaussian.get_xyz.shape[0]
        print(self.num_static, self.num_dynamic)
        self.dynamic_mask = torch.zeros(self.num_static + self.num_dynamic, dtype=torch.bool,
                                        device=self.moving_gaussian.get_xyz.device)
        self.dynamic_mask[self.num_static:] = True

        with torch.no_grad():
            end_points, _ = self.load_traj(t=1, vis=False)
            full_labels = np.full((self.num_static + self.num_dynamic), fill_value=0, dtype=int)
            full_labels[self.dynamic_mask.cpu().numpy()] = 1
            save_point_cloud_ply(end_points.cpu().numpy(), full_labels, args.model_path + "/seg_init_dual")
        self.motion_assignment = full_labels

        traj = []

        for time in np.linspace(0, 1, 3):
            traj_t, _ = self.load_traj(t=time, vis=True)
            traj.append(traj_t)
        traj_seq = torch.stack(traj, dim=1)
        mean_deformation = (torch.norm(traj_seq[:, -1, :] - traj_seq[:, 0, :], p=2, dim=-1)).mean()
        print(mean_deformation)

        self.motion_R_set = [torch.eye(3, device=traj_seq.device)]
        self.motion_t_set = [torch.zeros(3, device=traj_seq.device)]

        #threshold = torch.tensor(0.15, device=mean_deformation.device, dtype=mean_deformation.dtype)
        self.motion_segmentation_step_sequential(traj_seq, max_iter=2000, threshold = mean_deformation,
                                                 min_inliers=0.1 * self.num_dynamic)
        valid_mask = self.motion_assignment != -1
        self.motion_assignment = self.motion_assignment[valid_mask]
        self.moving_gaussian.prune_points(~valid_mask[self.num_static:])
        self.num_dynamic = self.moving_gaussian.get_xyz.shape[0]

        with torch.no_grad():
            end_points, combined_gaussian_end = self.load_traj(t=1, vis=False)
            end_points, combined_gaussian_start = self.load_traj(t=0, vis=False)
            save_point_cloud_ply(end_points.cpu().numpy(), self.motion_assignment,args.model_path + "/motion_seg_final")

        self.motion_R_set = torch.stack(self.motion_R_set, dim=0)
        self.motion_t_set = torch.stack(self.motion_t_set, dim=0)

        motions = []
        for c in range(len(self.motion_R_set)):
            p, u, theta, phi, motion_type = screw_from_Rt(self.motion_R_set[c], self.motion_t_set[c])

            motion_info = {
                "motion_type": motion_type.cpu().numpy().tolist(),
                "theta": float(theta),
                "phi": float(phi),
                "axis": u.cpu().numpy().tolist(),
                "center": p.cpu().numpy().tolist(),
                "R": self.motion_R_set[c].cpu().numpy().tolist(),
                "t": self.motion_t_set[c].cpu().numpy().tolist(),
            }
            motions.append(motion_info)

            sub_new_gaussian_start = GaussianModel(self.dataset.sh_degree)
            sample_idx = torch.where(torch.from_numpy(self.motion_assignment) == c)[0]
            print(len(sample_idx))
            self.init_sub_gs(combined_gaussian_start, sub_new_gaussian_start, sample_idx, id=c, vis=True, state="start")

            sub_new_gaussian_end = GaussianModel(self.dataset.sh_degree)
            sample_idx = torch.where(torch.from_numpy(self.motion_assignment) == c)[0]
            self.init_sub_gs(combined_gaussian_end, sub_new_gaussian_end, sample_idx, id=c, vis=True, state="end")
            self.sub_gaussian_set.append(sub_new_gaussian_end)

        with open(args.model_path + "/motion.json", "w") as f:
            json.dump(motions, f, indent=4)

    def load_traj(self, t, vis=False):
        new_gaussian = GaussianModel(self.dataset.sh_degree)
        with torch.no_grad():
            N_d = self.moving_gaussian.get_xyz.shape[0]
            time = torch.ones([N_d, 1], device="cuda") * t
            xyz = self.moving_gaussian.get_xyz.detach() + \
                  self.deform.step(self.moving_gaussian.get_xyz.detach(), time)[0]
            rotation = torch.nn.functional.normalize(self.moving_gaussian._rotation.detach() + \
                                                     self.deform.step(
                                                         self.moving_gaussian.get_xyz.detach(),
                                                         time)[1])
            new_gaussian.copy_from_dual_gaussian(self.start_static_gaussian, self.moving_gaussian, xyz, rotation)
        if vis:
            new_gaussian.save_ply(args.model_path + "/motion_traj_t=" + str(t) + "/color_gs.ply")
            save_input_points(args.model_path + "/motion_traj_t=" + str(t),
                              new_gaussian.get_xyz[:, None, :])

        return new_gaussian.get_xyz.detach(), new_gaussian

    def load_dyn_traj(self, t):
        with torch.no_grad():
            N_d = self.moving_gaussian.get_xyz.shape[0]
            time = torch.ones([N_d, 1], device="cuda") * t
            traj = self.moving_gaussian.get_xyz.detach() + \
                   self.deform.step(self.moving_gaussian.get_xyz.detach(), time)[0]
        return traj

    def motion_segmentation_step_sequential(self, traj_seq,
                                            min_inliers=20, max_iter=100,
                                            threshold = 0.01,
                                            em_iters=50, odds_log_tau=0.0):

        dyn_idx = torch.where(self.dynamic_mask)[0]
        X = traj_seq[self.dynamic_mask]  # [N,T,3]
        N, T, _ = X.shape
        X0, X05, X1 = X[:, 0, :], X[:, 1, :], X[:, 2, :]
        labels = torch.full((N,), -1, dtype=torch.long, device=X.device)
        unassigned = torch.arange(N, device=X.device)

        model_id = 1
        n_after = 0
        n_before = 1
        stagnant = 0
        patience = 10
        while unassigned.numel() >= min_inliers:

            if n_after == n_before:
                stagnant += 1
            else:
                stagnant = 0
            n_before = unassigned.numel()
            if stagnant >= patience:
                print(f"[Seq] no progress for {patience} rounds, stop.")
                break

            X0_cur = X0[unassigned]
            X05_cur = X05[unassigned]
            X1_cur = X1[unassigned]

            inlier_mask, errors, R_half, t_half, R, t, idx_set = self.ransac_propose_two_stage(
                X0_cur, X05_cur, X1_cur, num_iter=max_iter,
                sample_points=3, inlier_thresh= min(threshold, 0.25)#0.05
            )

            # p, u, theta, phi, mtype = screw_from_Rt(R, t, eps_theta=0.2)
            # print(
            #     f"theta={theta.cpu().numpy():.3f} | p={p.cpu().numpy()} |"
            #     f"u={u.cpu().numpy()} |phi={phi.cpu().numpy():.3f} |t={t.cpu().numpy()}")
            final_mask1 = torch.zeros(inlier_mask.shape[0], dtype=torch.bool,
                                        device=X.device)
            em_mask1, w, pi, sigma, log_odds = em_inlier_joint_likelihood(errors[inlier_mask],
                                                                         iters=em_iters,
                                                                         odds_log_tau=odds_log_tau,
                                                                          w_thresh=0.5)
            final_mask1[inlier_mask] = em_mask1
            if final_mask1.sum().item() == 0:
                n_after = unassigned.numel()
                continue
            keep = self.keep_largest_cc(X1_cur[final_mask1])
            final_mask = final_mask1.clone()
            final_mask[final_mask1] = keep
            em_mask = em_mask1.clone()
            em_mask[em_mask1] = keep
            if final_mask.sum().item() < min_inliers:
                n_after = unassigned.numel()
                continue
            #
            w_irls = w[em_mask].clamp(1e-4, 1.0)
            R2_refit, t2_refit = fit_rigid_transform_weighted(X0_cur[final_mask], X1_cur[final_mask], w_irls)
            idx_inlier = unassigned[final_mask]
            labels[idx_inlier] = model_id

            self.motion_R_set.append(R2_refit)
            self.motion_t_set.append(t2_refit)
            p, u, theta, phi, mtype = screw_from_Rt(R2_refit, t2_refit, eps_theta=0.2)
            print(
                f"[Seq] Motion #{model_id} | size={final_mask.sum()} | theta={theta.cpu().numpy():.3f} | p={p.cpu().numpy()} |"
                f"u={u.cpu().numpy()} |phi={phi.cpu().numpy():.3f} |t={t2_refit.cpu().numpy()}")
            self.motion_assignment[dyn_idx.cpu().numpy()] = labels.cpu().numpy()

            unassigned = unassigned[~final_mask]
            n_after = unassigned.numel()
            model_id += 1

        if model_id > 2:
            new_models, new_labels = self.search_and_merge(model_id, X0, X1, labels)
            motion_R_set_new = []
            motion_t_set_new = []
            existing_id = []
            for key in sorted(new_labels.keys()):
                labels[labels == key+1] = new_labels[key]+1
                if new_labels[key] not in existing_id:
                    existing_id.append(new_labels[key])
                    motion_R_set_new.append(new_models[new_labels[key]]['R'])
                    motion_t_set_new.append(new_models[new_labels[key]]['t'])
            for id_ in range(len(motion_R_set_new)):
                p, u, theta, phi, mtype = screw_from_Rt(motion_R_set_new[id_], motion_t_set_new[id_], eps_theta=0.2)
                print(
                    f"After Merging: [Seq] Motion #{id_} | size={len(new_models[id_]['idx'])} | theta={theta.cpu().numpy():.3f} | p={p.cpu().numpy()} |"
                    f"u={u.cpu().numpy()} |phi={phi.cpu().numpy():.3f} |t={motion_t_set_new[id_].cpu().numpy()}")
            self.motion_R_set = [torch.eye(3, device=traj_seq.device)] + motion_R_set_new
            self.motion_t_set = [torch.zeros(3, device=traj_seq.device)] + motion_t_set_new
        model_id = len(self.motion_R_set)

        if unassigned.numel() > 0 :
            if model_id > 2:
                K = model_id - 1
                X0_u, X1_u = X0[unassigned], X1[unassigned]  # [M,3]
                M = X0_u.shape[0]
                Error = torch.zeros(M, K, device=X0_u.device, dtype=torch.float32)
                Distance = torch.zeros(M, K, device=X0_u.device, dtype=torch.float32)
                for k in range(1, K + 1):
                    Rk, tk = self.motion_R_set[k - 1], self.motion_t_set[k - 1]  # 注意 k-1
                    pred = (Rk @ X0_u.t()).t() + tk
                    rk = torch.norm(pred - X1_u, dim=1)  # [M]
                    distance = torch.cdist(X1_u, X1[labels==k])
                    Distance[:,k-1] = distance.min(dim=1)[0]
                    Error[:, k - 1] = rk

                distance_best, k_best = Distance.min(dim=1)
                row = torch.arange(M, device=X0_u.device)
                err_best = Error[row, k_best]
                assign_mask = torch.isfinite(distance_best)
                labels[unassigned[assign_mask]] = (k_best[assign_mask] + 1).to(labels.dtype)

                keep_mask = torch.ones_like(unassigned, dtype=torch.bool)
                keep_mask[assign_mask] = False
                unassigned = unassigned[keep_mask]
            else:
                X0_u, X1_u = X0[unassigned], X1[unassigned]  # [M,3]
                M = X0_u.shape[0]

                Rk, tk = self.motion_R_set[-1], self.motion_t_set[-1]
                pred = (Rk @ X0_u.t()).t() + tk
                rk = torch.norm(pred - X1_u, dim=1)  # [M]
                r_static = torch.norm(X0_u - X1_u, dim=1)

                accept = (rk < r_static) & (rk < 0.05) #(rk < r_static < 0.01).any(dim=1)
                newly_idx_u = torch.where(accept)[0]
                if newly_idx_u.numel() > 0:
                    labels[unassigned[newly_idx_u]] = 1
                keep_mask = torch.ones_like(unassigned, dtype=torch.bool)
                keep_mask[newly_idx_u] = False
                unassigned = unassigned[keep_mask]
        self.motion_assignment[dyn_idx.cpu().numpy()] = labels.cpu().numpy()

        return labels
    
    def ransac_propose_two_stage(self, X_0, X_05, X_1,
                                 num_iter=200, sample_points=3,inlier_thresh=0.01
                                 ):
        N = X_0.shape[0]
        best_inlier_mask = None
        best_errors = None
        best_num_inliers = 0
        best_R1, best_t1, best_R2, best_t2, best_idx_set = None, None, None, None, None

        for _ in range(num_iter):
            idx = torch.randperm(N)[:sample_points]
            R2, t2 = fit_rigid_transform(X_0[idx], X_1[idx])
            R1, t1 = fit_rigid_transform(X_0[idx], X_05[idx])

            X0_pred_05 = (R1 @ X_0.t()).t() + t1
            X0_pred_1 = (R2 @ X_0.t()).t() + t2
            err_05 = torch.norm(X0_pred_05 - X_05, dim=1)
            err_1 = torch.norm(X0_pred_1 - X_1, dim=1)
            errs = err_05 + err_1 / 2
            inlier_mask = errs < inlier_thresh
            num_inliers = inlier_mask.sum().item()
            if num_inliers > best_num_inliers:
                best_errors = errs
                best_num_inliers = num_inliers
                best_inlier_mask = inlier_mask
                best_R1, best_t1 = R1, t1
                best_R2, best_t2 = R2, t2
                best_idx_set = idx
        return best_inlier_mask, best_errors, best_R1, best_t1, best_R2, best_t2,best_idx_set

    def connected_components_from_knn(self, idx_knn: torch.Tensor, min_size: int = 20):
        M, k = idx_knn.shape
        visited = torch.zeros(M, dtype=torch.bool, device=idx_knn.device)
        comps = []

        nbrs = [[] for _ in range(M)]
        for i in range(M):
            for j in idx_knn[i].tolist():
                nbrs[i].append(j)
                nbrs[j].append(i)

        for i in range(M):
            if visited[i]: continue
            stack = [i]
            visited[i] = True
            comp = [i]
            while stack:
                v = stack.pop()
                for u in nbrs[v]:
                    if not visited[u]:
                        visited[u] = True
                        stack.append(u)
                        comp.append(u)
            if len(comp) >= min_size:
                comps.append(torch.tensor(comp, device=idx_knn.device, dtype=torch.long))
        return comps

    def search_and_merge(self, motion_id, X0, X1, labels, use_cross_check=True):
        models = []
        for k in range(1, motion_id):
            R = self.motion_R_set[k]
            t = self.motion_t_set[k]
            inlier_idx = torch.where(labels == k)[0]
            p, u, theta, phi, motion_type = screw_from_Rt(R, t)
            rad = float(torch.median(
                torch.norm(X1[inlier_idx] - X1[inlier_idx].mean(0, keepdim=True), dim=1)).item())
            st = float(torch.median(torch.norm(X1[inlier_idx]
                                               - X0[inlier_idx], dim=1)).item())
            models.append(dict(R=R, t=t, u=u, theta=theta, p=p, rad=rad, s_t=st, motion_type=motion_type,
                               idx=inlier_idx.to(X0.device)))

        tau_axis = 0.2
        tau_axdist = 0.5 * np.median([m['rad'] for m in models])
        tau_theta = 0.2
        tau_dir = 0.98
        tau_phi = 0.04
        st_med = np.median([m['s_t'] for m in models]) if len(models) > 0 else 1.0
        tau_res = 0.75 * st_med
        cfg = dict(tau_axis=tau_axis, tau_axdist=tau_axdist,
                   tau_theta=tau_theta, tau_dir=tau_dir,
                   tau_phi=tau_phi, tau_res=tau_res)

        K = len(models)
        if K <= 1:
            remap = {i: 0 for i in range(K)}
            return models, remap
        pairs = []
        for i in range(K):
            for j in range(i + 1, K):
                ok, score = compatible_same_type(models[i], models[j], X0, X1, cfg, use_cross_check)
                if ok:
                    pairs.append((i, j, score))
        uf = UnionFind(K)
        pairs.sort(key=lambda x: x[2])
        idx_map = {i: models[i]['idx'].clone() for i in range(K)}
        for i, j, _ in pairs:
            a, b = uf.find(i), uf.find(j)
            if a == b:
                continue
            idx_union = torch.unique(torch.cat([idx_map[a], idx_map[b]], dim=0))
            na, nb = idx_map[a].numel(), idx_map[b].numel()
            ratio = max(na, nb) / max(1, min(na, nb))
            dominant = a if na >= nb else b
            if ratio >= 2.0:
                dom_m = models[dominant]
                new_m = {
                    'R': dom_m['R'], 't': dom_m['t'],
                    'u': dom_m['u'], 'theta': dom_m['theta'],
                    'phi': dom_m.get('phi', torch.norm(dom_m['t']).item()),
                    'p': dom_m['p'],
                    'motion_type': dom_m['motion_type'],
                    'idx': idx_union,
                    'rad': float(
                        torch.median(torch.norm(X1[idx_union] - X1[idx_union].mean(0, keepdim=True), dim=1)).item()),
                    's_t': float(torch.median(torch.norm(X1[idx_union] - X0[idx_union], dim=1)).item()),
                }
            else:
                new_m = recompute_model_from_indices(idx_union, X0, X1)
            ok_a, _ = compatible_same_type(new_m, models[a], X0, X1, cfg, use_cross_check)
            ok_b, _ = compatible_same_type(new_m, models[b], X0, X1, cfg, use_cross_check)
            if ok_a and ok_b:
                r = uf.union(a, b)
                idx_map[r] = idx_union
                models[r] = new_m
        reps = {}
        for i in range(K):
            r = uf.find(i)
            if r not in reps:
                reps[r] = True
        new_models = []
        old2new = {}
        rid2new = {rid: new_id for new_id, rid in enumerate(reps.keys())}
        for i in range(K):
            r = uf.find(i)
            old2new[i] = rid2new[r]

        for rid in reps.keys():
            new_models.append(models[rid])
        return new_models, old2new

    def keep_largest_cc(self, X, k=12, min_keep=200):
        N = X.shape[0]
        X_np = X.detach().cpu().numpy()
        nn = NearestNeighbors(n_neighbors=k + 1,
                              algorithm="auto",
                              metric="euclidean")
        nn.fit(X_np)
        dist, idx = nn.kneighbors(X_np, return_distance=True)

        nbr = torch.from_numpy(idx).to(X.device)
        vis = torch.zeros(N, dtype=torch.bool, device=X.device)
        comps = []
        for s in range(N):
            if vis[s]: continue
            stack = [s];
            vis[s] = True;
            comp = [s]
            while stack:
                u = stack.pop()
                for v in nbr[u].tolist():
                    if not vis[v]:
                        vis[v] = True;
                        stack.append(v);
                        comp.append(v)
            comps.append(torch.tensor(comp, device=X.device))
        comps.sort(key=lambda t: t.numel(), reverse=True)
        keep = torch.zeros(N, dtype=torch.bool, device=X.device)
        if len(comps) > 0:
            keep[comps[0]] = True

        if keep.sum() < min_keep:
            keep[:] = True
        return keep




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[] + list(range(0, 100001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[] + list(range(0, 100001, 10000)))
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    model = AiM(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),
                testing_iterations=args.test_iterations, saving_iterations=args.save_iterations)

    model.segment()
    print("\nTraining complete.")

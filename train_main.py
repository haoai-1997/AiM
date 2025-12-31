import os
import sys
from marshal import loads
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
from train_report import training_report
from utils.seg_tools import se3_log, screw_from_wv, screw_from_Rt, fit_rigid_transform, ransac_fit, \
    ransac_fit_two_stage, smooth_labels_by_spatial_voting
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random

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

        self.start_static_scene = Scene(dataset, gaussians=self.start_static_gaussian, load_iteration=None,
                                        state="start", init_point_size="large")
        self.start_static_gaussian.training_setup(opt)

        self.moving_gaussian = GaussianModel(dataset.sh_degree)
        self.motion_scene = Scene(dataset, gaussians=self.moving_gaussian, load_iteration=None, state="motion",
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

        self.ema_start_loss_for_log = 0.0
        self.best_start_psnr = 0.0
        self.best_start_iteration = 0.0

        self.ema_motion_loss_for_log = 0.0
        self.best_motion_psnr = 0.0
        self.best_motion_iteration = 0.0

        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01,
                                                 max_steps=self.args.iterations_motion)
        self.motion_R_set = []
        self.motion_t_set = []
        self.motion_assignment = None
        self.sub_gaussian_set = []
        self.num_static = None
        self.num_dynamic = None
        self.static_mask = None
        self.dynamic_mask = None
        self.later_views = None

    def train(self):
        for _ in tqdm.trange(self.opt.iterations_start):
            self.train_start_step()
        self.later_views = []

        for view in self.motion_scene.getTrainCameras().copy():
            if view.fid > 0.5:
                self.later_views.append(view)

        self.start_static_gaussian.denom = torch.zeros((self.start_static_gaussian.get_xyz.shape[0], 1), device="cuda")
        self.start_static_gaussian.xyz_gradient_accum = torch.zeros((self.start_static_gaussian.get_xyz.shape[0], 1),
                                                                    device="cuda")
        for _ in tqdm.trange(self.opt.iterations_motion):
            self.train_motion_step()

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

    def train_start_step(self):
        self.iter_start.record()
        if self.start_static_iteration % 1000 == 0:
            self.start_static_gaussian.oneupSHdegree()

        if not self.start_static_viewpoint_stack:
            self.start_static_viewpoint_stack = self.start_static_scene.getTrainCameras().copy()

        viewpoint_cam = self.start_static_viewpoint_stack.pop(randint(0, len(self.start_static_viewpoint_stack) - 1))

        d_xyz, d_rotation = 0.0, 0.0

        random_bg_color = ((not self.dataset.white_background and self.opt.random_bg_color)
                           and viewpoint_cam.gt_alpha_mask is not None)

        render_pkg_re = render(viewpoint_cam, self.start_static_gaussian, self.pipe, self.background, d_xyz, d_rotation,
                               random_bg_color=random_bg_color)

        image, viewspace_point_tensor, visibility_filter, radii, depth, alpha = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["depth"], \
            render_pkg_re["alpha"]

        gt_image = viewpoint_cam.original_image.cuda()
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()

        if random_bg_color:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * render_pkg_re['bg_color'][:, None, None]
        elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None and self.opt.gt_alpha_mask_as_scene_mask:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        gt_depth = viewpoint_cam.depth.cuda()
        if gt_depth is not None:
            invalid_mask = (gt_depth < 0.1) & (gt_alpha_mask > 0.5)
        else:
            invalid_mask = (gt_alpha_mask > 0.5)
        valid_mask = ~invalid_mask
        n_valid_pixel = valid_mask.sum()
        if n_valid_pixel > 100 and gt_depth is not None:
            depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * valid_mask).sum() / n_valid_pixel
        else:
            depth_loss = 0

        if self.start_static_iteration > self.opt.start_densify_until_iter:
            opacity_loss = torch.relu(0.005 - self.start_static_gaussian.get_opacity).mean()
        else:
            opacity_loss = 0

        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (
                1.0 - ssim(image, gt_image)) + 0.01 * opacity_loss + depth_loss

        loss.backward()
        self.iter_end.record()

        with torch.no_grad():
            self.ema_start_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_start_loss_for_log
            if self.start_static_iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_start_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.total_iteration == self.opt.iterations:
                self.progress_bar.close()

            cur_psnr = training_report(self.tb_writer, self.total_iteration, Ll1, loss, l1_loss,
                                       self.iter_start.elapsed_time(self.iter_end), self.testing_iterations,
                                       self.start_static_scene,
                                       render, (self.pipe, self.background),
                                       self.dataset.load2gpu_on_the_fly)

            if self.total_iteration in self.testing_iterations or self.start_static_iteration == self.opt.iterations_start:
                if cur_psnr.item() > self.best_start_psnr:
                    self.best_start_psnr = cur_psnr.item()
                    self.best_start_iteration = self.start_static_iteration
            if self.total_iteration in self.saving_iterations or self.total_iteration == self.opt.iterations_start:
                print("\n[ITER {}] Saving Gaussians".format(self.total_iteration))
                self.start_static_scene.save(self.total_iteration)
                save_image(image, self.args.model_path + f"render_start_image_mix_{self.total_iteration // 10000}.png")
                save_image(gt_image, self.args.model_path + f"gt_start_image_mix_{self.total_iteration // 10000}.png")

            if self.start_static_iteration < self.opt.start_densify_until_iter:
                self.start_static_gaussian.max_radii2D[visibility_filter] = torch.max(
                    self.start_static_gaussian.max_radii2D[visibility_filter], radii[visibility_filter])

                self.start_static_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.start_static_iteration > self.opt.start_densify_from_iter \
                        and self.start_static_iteration % self.opt.start_densification_interval == 0:
                    size_threshold = 20 if self.start_static_iteration > self.opt.start_opacity_reset_interval else None
                    self.start_static_gaussian.densify_and_prune(self.opt.start_densify_grad_threshold, 0.005,
                                                                 self.start_static_scene.cameras_extent,
                                                                 size_threshold)
                if self.start_static_iteration % self.opt.start_opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.start_static_iteration == self.opt.start_densify_from_iter):
                    self.start_static_gaussian.reset_opacity()
        if self.start_static_iteration <= self.opt.iterations_start:
            self.start_static_gaussian.optimizer.step()
            self.start_static_gaussian.update_learning_rate(self.start_static_iteration)
            self.start_static_gaussian.optimizer.zero_grad(set_to_none=True)

        if self.start_static_iteration % 2000 == 0 or self.start_static_iteration == self.opt.iterations_start:
            print("Best PSNR = {} with {} Gaussians in Static Iteration {}".format(self.best_start_psnr,
                                                                                   self.start_static_gaussian.get_xyz.shape[
                                                                                       0],
                                                                                   self.best_start_iteration))

        self.total_iteration += 1
        self.start_static_iteration += 1

    def arap_loss_v2(self, t_samp_num=8, neighbor_k=10):
        q_times = torch.rand(t_samp_num).to("cuda")
        if self.moving_gaussian.get_xyz.shape[0] > 5000:
            idx = torch.randperm(self.moving_gaussian.get_xyz.shape[0])[:5000]
            means3D = self.moving_gaussian.get_xyz[idx]
        else:
            means3D = self.moving_gaussian.get_xyz
        means3D_deform_list = []
        for q_time in q_times:
            q_time = q_time[None].repeat(means3D.shape[0], 1).to(means3D.device)
            means3D_deform, _ = self.deform.step(means3D.detach(), q_time)
            means3D_deform_list.append(means3D_deform)
        means3D_deform_list = torch.stack(means3D_deform_list, 0)
        means3D_t = means3D[None].repeat(t_samp_num, 1, 1).detach() + means3D_deform_list
        ii, jj, nn, _ = cal_connectivity_from_points_v2(means3D_t, K=neighbor_k)
        error = cal_arap_error(means3D_t, ii, jj, nn, K=neighbor_k)
        return error, (ii, jj, nn, _)

    def train_motion_step(self):
        if self.motion_iteration % 1000 == 0:
            self.moving_gaussian.oneupSHdegree()

        if self.motion_iteration < self.opt.motion_prune_until_iter:
            if not self.motion_viewpoint_stack:
                self.motion_viewpoint_stack = self.later_views.copy()
        else:
            start_idx = random.sample(range(len(self.start_static_scene.getTrainCameras().copy())), 5)
            end_idx = random.sample(range(len(self.end_static_scene.getTrainCameras().copy())), 10)
            if not self.motion_viewpoint_stack:
                self.motion_viewpoint_stack = self.motion_scene.getTrainCameras().copy() + \
                                              [self.start_static_scene.getTrainCameras().copy()[i] for i in
                                               start_idx] + [self.end_static_scene.getTrainCameras().copy()[i] for i in
                                                             end_idx]

        total_frame = len(self.motion_viewpoint_stack)
        time_interval = 1 / total_frame
        viewpoint_cam = self.motion_viewpoint_stack.pop(randint(0, len(self.motion_viewpoint_stack) - 1))
        fid = viewpoint_cam.fid

        if self.motion_iteration < self.opt.warm_up:
            d_xyz, d_rotation = 0, 0
        else:
            N_s = self.moving_gaussian.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N_s, -1)
            ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N_s,
                                                                                                  -1) * time_interval \
                                                          * self.smooth_term(self.motion_iteration)
            d_xyz, d_rotation = self.deform.step(self.moving_gaussian.get_xyz.detach(),
                                                 time_input + ast_noise)
        random_bg_color = ((not self.dataset.white_background and self.opt.random_bg_color)
                           and viewpoint_cam.gt_alpha_mask is not None)

        render_pkg_re = render_mix(viewpoint_cam, self.moving_gaussian, self.start_static_gaussian, self.pipe,
                                   self.background, d_xyz, d_rotation, random_bg_color=random_bg_color)

        image, viewspace_point_tensor, visibility_filter, radii, depth, alpha = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["depth"], \
            render_pkg_re["alpha"]

        gt_image = viewpoint_cam.original_image.cuda()
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        if random_bg_color:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * render_pkg_re['bg_color'][:, None, None]
        elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None and self.opt.gt_alpha_mask_as_scene_mask:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        gt_depth = viewpoint_cam.depth.cuda()
        if gt_depth is not None:
            invalid_mask = (gt_depth < 0.1) & (gt_alpha_mask > 0.5)
        else:
            invalid_mask = (gt_alpha_mask > 0.5)

        valid_mask = ~invalid_mask
        n_valid_pixel = valid_mask.sum()
        if n_valid_pixel > 100 and gt_depth is not None:
            depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * valid_mask).sum() / n_valid_pixel
        else:
            depth_loss = 0

        Ll1 = l1_loss(image, gt_image)

        if self.opt.warm_up < self.motion_iteration <= self.opt.motion_prune_until_iter + self.opt.motion_assignment_interval:
            deform_loss = 1e-3 * (torch.abs(d_xyz.norm(dim=1)).mean() + torch.abs(d_rotation.norm(dim=1)).mean())
        elif self.motion_iteration > self.opt.motion_prune_until_iter + self.opt.motion_assignment_interval:
            if self.motion_iteration % 10 == 0:
                deform_loss = 0.001 * self.arap_loss_v2(t_samp_num=8, neighbor_k=32)[0]
            else:
                deform_loss = 0
        else:
            deform_loss = 0

        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (
                1.0 - ssim(image, gt_image)) + deform_loss + depth_loss

        if self.motion_iteration <= self.opt.motion_prune_until_iter:
            self.start_static_gaussian._xyz.requires_grad_(False)
            self.start_static_gaussian._rotation.requires_grad_(False)
            self.start_static_gaussian._scaling.requires_grad_(False)
            self.start_static_gaussian._features_dc.requires_grad_(False)
            self.start_static_gaussian._features_rest.requires_grad_(False)
            self.start_static_gaussian._opacity.requires_grad_(True)
        else:
            self.start_static_gaussian._xyz.requires_grad_(True)
            self.start_static_gaussian._rotation.requires_grad_(True)
            self.start_static_gaussian._scaling.requires_grad_(True)
            self.start_static_gaussian._features_dc.requires_grad_(True)
            self.start_static_gaussian._features_rest.requires_grad_(True)
            self.start_static_gaussian._opacity.requires_grad_(True)

        loss.backward()
        self.iter_end.record()

        with torch.no_grad():
            self.ema_motion_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_motion_loss_for_log
            if self.motion_iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_motion_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.total_iteration == self.opt.iterations:
                self.progress_bar.close()

            if self.total_iteration in self.saving_iterations:
                self.start_static_scene.save(self.total_iteration)
                self.motion_scene.save(self.total_iteration)
                self.deform.save_weights(args.model_path, self.total_iteration)
                # for time in np.linspace(0, 1, 3):
                #     traj_t, _ = self.load_traj(t=time, vis=True)
                # save_image(image, self.args.model_path + f"render_image_mix_{self.total_iteration // 10000}.png")
                # save_image(gt_image, self.args.model_path + f"gt_image_mix_{self.total_iteration // 10000}.png")

            if self.motion_iteration < self.opt.motion_densify_until_iter:

                motion_visibility_filter = visibility_filter[self.start_static_gaussian.get_xyz.shape[0]:]
                motion_radii = radii[self.start_static_gaussian.get_xyz.shape[0]:]
                self.moving_gaussian.max_radii2D[motion_visibility_filter] = torch.max(
                    self.moving_gaussian.max_radii2D[motion_visibility_filter],
                    motion_radii[motion_visibility_filter])
                self.moving_gaussian.add_part_densification_stats(viewspace_point_tensor,
                                                                  motion_visibility_filter,
                                                                  start=self.start_static_gaussian.get_xyz.shape[0],
                                                                  end=self.start_static_gaussian.get_xyz.shape[0] \
                                                                      + self.moving_gaussian.get_xyz.shape[0] + 1)

                static_visibility_filter = visibility_filter[:self.start_static_gaussian.get_xyz.shape[0]]
                static_radii = radii[:self.start_static_gaussian.get_xyz.shape[0]]
                self.start_static_gaussian.max_radii2D[static_visibility_filter] = torch.max(
                    self.start_static_gaussian.max_radii2D[static_visibility_filter],
                    static_radii[static_visibility_filter])

                self.start_static_gaussian.add_part_densification_stats(viewspace_point_tensor,
                                                                        static_visibility_filter,
                                                                        start=0,
                                                                        end=
                                                                        self.start_static_gaussian.get_xyz.shape[
                                                                            0])

                if self.motion_iteration > self.opt.motion_prune_until_iter:  # self.opt.warm_up + 6 * self.opt.motion_assignment_interval:
                    if (self.motion_iteration - self.opt.warm_up) % self.opt.motion_assignment_interval == 0:
                        sampling_idx_moving, sampling_idx_to_static = self.detect_static_mask(min_inliers=25,
                                                                                              num_iter=300,
                                                                                              inlier_thresh=0.08,
                                                                                              theta_thr=0.2,
                                                                                              phi_thr=0.05,
                                                                                              n_neighbors=16,
                                                                                              knn=False)

                        sampled_xyz = self.moving_gaussian._xyz[sampling_idx_to_static]
                        sampled_rotation = self.moving_gaussian._rotation[sampling_idx_to_static]
                        sampled_scaling = self.moving_gaussian._scaling[sampling_idx_to_static]
                        sampled_features_dc = self.moving_gaussian._features_dc[sampling_idx_to_static]
                        sampled_features_rest = self.moving_gaussian._features_rest[sampling_idx_to_static]
                        sampled_opacities = self.moving_gaussian._opacity[sampling_idx_to_static]

                        self.moving_gaussian.prune_points(~sampling_idx_moving)

                        self.start_static_gaussian.densification_postfix(sampled_xyz, sampled_features_dc,
                                                                         sampled_features_rest, sampled_opacities,
                                                                         sampled_scaling, sampled_rotation)

                        prune_static_mask, prune_dynamic_mask = self.smooth_by_spatial_voting(neighbor_k=128,
                                                                                              vote_thr=0.7)
                        self.start_static_gaussian.prune_points(prune_static_mask)
                        self.moving_gaussian.prune_points(prune_dynamic_mask)

                if self.motion_iteration > self.opt.motion_densify_from_iter and \
                        self.motion_iteration % self.opt.motion_densification_interval == 0:

                    if self.motion_iteration <= self.opt.motion_prune_until_iter:
                        size_threshold = 20 if self.motion_iteration > self.opt.motion_opacity_reset_interval else None
                        self.moving_gaussian.densify_and_prune(self.opt.motion_densify_grad_threshold, 0.005,
                                                               self.start_static_scene.cameras_extent, size_threshold)

                    else:
                        self.moving_gaussian.densify_and_prune(self.opt.motion_densify_grad_threshold, 0.005,
                                                               self.start_static_scene.cameras_extent, 20)
                        self.start_static_gaussian.densify_and_prune(self.opt.motion_densify_grad_threshold, 0.005,
                                                                     self.start_static_scene.cameras_extent, 20)

                        if self.motion_iteration % self.opt.motion_opacity_reset_interval == 0:
                            self.start_static_gaussian.reset_opacity()
                            self.moving_gaussian.reset_opacity()

        if self.motion_iteration < self.opt.iterations_motion:
            self.start_static_gaussian.optimizer.step()
            self.start_static_gaussian.update_learning_rate(self.motion_iteration)
            self.start_static_gaussian.optimizer.zero_grad(set_to_none=True)

            self.moving_gaussian.optimizer.step()
            self.moving_gaussian.update_learning_rate(self.motion_iteration)
            self.moving_gaussian.optimizer.zero_grad(set_to_none=True)

            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.deform.update_learning_rate(self.motion_iteration)

        if self.total_iteration % 2000 == 0 or self.motion_iteration == self.opt.iterations_motion:
            print(" {} Static Gaussians and {} Moving Gaussians in Iteration {}".format(
                self.start_static_gaussian.get_xyz.shape[0],
                self.moving_gaussian.get_xyz.shape[0],
                self.total_iteration))

        self.total_iteration += 1
        self.motion_iteration += 1

    def smooth_by_spatial_voting(self, neighbor_k=64, vote_thr=0.8):
        traj = []
        with torch.no_grad():
            N = self.moving_gaussian.get_xyz.shape[0]
            for t in [0, 0.5, 1]:
                time = t * torch.ones([N, 1], device=self.moving_gaussian.get_xyz.device)
                xyz = self.moving_gaussian.get_xyz.detach() + \
                      self.deform.step(self.moving_gaussian.get_xyz.detach(), time)[0].detach()
                xyz_total = torch.cat([self.start_static_gaussian.get_xyz, xyz], dim=0)
                traj.append(xyz_total)
        traj_seq = torch.stack(traj, dim=1)
        num_static = self.start_static_gaussian.get_xyz.shape[0]
        num_dynamic = self.moving_gaussian.get_xyz.shape[0]
        labels = np.full((num_static + num_dynamic), fill_value=0, dtype=int)
        labels[num_static:] = 1

        update_to_1, update_to_0 = smooth_labels_by_spatial_voting(traj_seq, labels, neighbor_k=neighbor_k,
                                                                   vote_thr=vote_thr)
        prune_static_mask = torch.zeros(num_static, dtype=torch.bool, device=self.moving_gaussian.get_xyz.device)
        prune_dynamic_mask = torch.zeros(num_dynamic, dtype=torch.bool,
                                         device=self.moving_gaussian.get_xyz.device)
        for idx in update_to_1:
            prune_static_mask[idx] = True
        for idx in update_to_0:
            prune_dynamic_mask[idx - num_static] = True
        print(f"Pruning {len(update_to_1)} static {len(update_to_0)} dynamic.")
        return prune_static_mask, prune_dynamic_mask

    def load_dyn_traj(self, t):
        with torch.no_grad():
            N_d = self.moving_gaussian.get_xyz.shape[0]
            time = torch.ones([N_d, 1], device="cuda") * t
            traj = self.moving_gaussian.get_xyz.detach() + \
                   self.deform.step(self.moving_gaussian.get_xyz.detach(), time)[0]
        return traj

    def detect_static_mask(self, min_inliers=20, num_iter=300, inlier_thresh=0.02, theta_thr=0.2, phi_thr=0.05,
                           knn=True, n_neighbors=64):
        with torch.no_grad():
            start_point = self.load_dyn_traj(t=0.0)
            end_point = self.load_dyn_traj(t=1.0)
            mid_point = self.load_dyn_traj(t=0.5)

        N = mid_point.shape[0]
        labels_remains = torch.zeros(N, dtype=torch.bool, device=start_point.device)
        labels_remove = torch.zeros(N, dtype=torch.bool, device=start_point.device)

        unassigned = torch.arange(N, device=start_point.device)

        while unassigned.numel() >= min_inliers:
            start_point_cur = start_point[unassigned]
            end_point_cur = end_point[unassigned]
            mid_point_cur = mid_point[unassigned]
            if knn:
                k = min(n_neighbors + 1, end_point_cur.shape[0])
                nn = NearestNeighbors(n_neighbors=k).fit(end_point_cur.cpu().numpy())
                knn_idx = nn.kneighbors(end_point_cur.cpu().numpy(), return_distance=False)[:, 1:]
            else:
                knn_idx = None

            inlier_mask, _, _, R, t = ransac_fit_two_stage(
                start_point_cur, mid_point_cur, end_point_cur, num_iter=num_iter, inlier_thresh=inlier_thresh,
                sample_points=3, knn_idx=knn_idx
            )

            if inlier_mask is None or inlier_mask.sum() < min_inliers:
                break
            idx_inlier = unassigned[inlier_mask]
            p, u, theta, phi, mtype = screw_from_Rt(R, t, eps_theta=theta_thr)

            if torch.abs(theta) >= theta_thr or torch.abs(phi) >= phi_thr:
                labels_remains[idx_inlier] = 1
            if torch.abs(theta) < theta_thr and torch.abs(phi) < phi_thr:
                labels_remove[idx_inlier] = 1

            unassigned = unassigned[~inlier_mask]

        print(
            f"Outliner: {unassigned.numel()}, Dynamic points: {labels_remains.sum()}, Static points: {labels_remove.sum()}")
        torch.cuda.empty_cache()
        return labels_remains, labels_remove


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

    model.train()
    # model.test()
    print("\nTraining complete.")

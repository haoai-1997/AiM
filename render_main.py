import os
import sys
from marshal import loads
from traceback import print_tb

from PIL.ImageOps import deform
from PIL import Image

from lib.pointops.functions import pointops
from urllib3 import proxy_from_url
import tqdm
import json
import torch
import open3d as o3d
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
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel, DeformModel
from gaussian_renderer import render, render_mix
from train import training_report
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import open3d as o3d
from utils.mesh_utils import to_cam_open3d, post_process_mesh
from utils.render_utils import save_img_f32, save_img_u8, transform_poses_pca, focus_point_fn

class AiM:
    def __init__(self, args, dataset, opt, pipe):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe

        self.moving_gaussian = GaussianModel(dataset.sh_degree)
        self.motion_scene = Scene(dataset, gaussians=self.moving_gaussian, load_iteration=-1, state="motion",
                                  init_point_size="small")
        self.moving_gaussian.training_setup(opt)

        self.deform = DeformModel(is_blender=dataset.is_blender, is_6dof=dataset.is_6dof)
        self.deform.train_setting(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def eval_mesh_gs(self):
        for self.time in [0.0, 0.5, 1.0]:
            self.mesh_dir = self.args.model_path + f'motion_traj_t={self.time}' +"/"
            self.all_gaussian = GaussianModel(self.dataset.sh_degree)
            self.all_gaussian.load_ply(self.mesh_dir + 'color_gs.ply',og_number_points=-1)
            self.viewpoint_stack =  self.motion_scene.getTrainCameras().copy()

            with torch.no_grad():
                d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
                self.rgbmaps = []
                self.depthmaps = []
                for viewpoint_cam in self.viewpoint_stack:
                    random_bg_color = False

                    # Render
                    render_pkg_re = render(viewpoint_cam, self.all_gaussian, self.pipe, self.background, d_xyz, d_rotation,
                                   random_bg_color=random_bg_color)

                    image, depth = render_pkg_re["render"], render_pkg_re["depth"]  # 1,800,800
                    self.rgbmaps.append(image.cpu())
                    self.depthmaps.append(depth.cpu())
                self.estimate_bounding_sphere()
                name = 'fuse.ply'
                depth_trunc = (self.radius * 2.0)  # if args.depth_trunc < 0  else args.depth_trunc
                voxel_size = 4e-03 #1e-3#5e-04 # (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
                sdf_trunc = 5 * voxel_size #5e-3#2e-3 # 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = self.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                o3d.io.write_triangle_mesh(os.path.join(self.mesh_dir, name), mesh)
                print("mesh saved at {}".format(os.path.join(self.mesh_dir, name)))
                # post-process the mesh and save, saving the largest N clusters
                num_cluster = 1
                mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
                o3d.io.write_triangle_mesh(os.path.join(self.mesh_dir, name.replace('.ply', '_post.ply')), mesh_post)
                print("mesh post processed saved at {}".format(os.path.join(self.mesh_dir, name.replace('.ply', '_post.ply'))))

                plt.imshow(self.rgbmaps[0].permute(1, 2, 0).detach().cpu())
                plt.savefig(os.path.join(self.mesh_dir, f"{self.time}_image.png"))
                plt.close()
                print(f'Final time {self.time}')
        for name_id in range(2):
            for name_state in ["start","end"]:
                name_ = f"sub_{name_id}_point_cloud_{name_state}"
                self.mesh_dir = self.args.model_path + f"/{name_}"
                self.all_gaussian = GaussianModel(self.dataset.sh_degree)
                self.all_gaussian.load_ply(self.mesh_dir + ".ply", og_number_points=-1)
                self.viewpoint_stack = self.motion_scene.getTrainCameras().copy()

                with torch.no_grad():
                    d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
                    self.rgbmaps = []
                    self.depthmaps = []
                    for viewpoint_cam in self.viewpoint_stack:
                        random_bg_color = False

                        # Render
                        render_pkg_re = render(viewpoint_cam, self.all_gaussian, self.pipe, self.background, d_xyz, d_rotation,
                                               random_bg_color=random_bg_color)

                        image, depth = render_pkg_re["render"], render_pkg_re["depth"]  # 1,800,800
                        self.rgbmaps.append(image.cpu())
                        self.depthmaps.append(depth.cpu())
                    self.estimate_bounding_sphere()
                    name = 'fuse.ply'
                    depth_trunc = (self.radius * 2.0)  # if args.depth_trunc < 0  else args.depth_trunc
                    voxel_size = 4e-03 #  1e-3  # 5e-04 # (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
                    sdf_trunc = 5 * voxel_size # 5e-3  # 2e-3 # 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                    mesh = self.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    o3d.io.write_triangle_mesh(self.mesh_dir+name, mesh)
                    print("mesh saved at {}".format(self.mesh_dir+name))
                    # post-process the mesh and save, saving the largest N clusters
                    num_cluster = 1
                    mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
                    o3d.io.write_triangle_mesh(self.mesh_dir+ name.replace('.ply', '_post.ply'), mesh_post)
                    print("mesh post processed saved at {}".format(self.mesh_dir+ name.replace('.ply', '_post.ply')))
                    print(f'Final name {name_}')
    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        for i, cam_o3d in enumerate(to_cam_open3d(self.viewpoint_stack)):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")),
                depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
                depth_scale=1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    model = AiM(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args))

    model.eval_mesh_gs()
    print("\nTraining complete.")
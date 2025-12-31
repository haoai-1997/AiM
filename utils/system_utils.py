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

from errno import EEXIST
from os import makedirs, path
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def save_input_points(model_path, input_xyz_seq):
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    frame_length = input_xyz_seq.shape[1]
    for i in range(frame_length):
        point_cloud_path = os.path.join(model_path, "point_cloud_seq_{}.ply".format(i))
        mkdir_p(os.path.dirname(point_cloud_path))

        xyz = input_xyz_seq[:, i, :].detach().cpu().numpy()
        N = xyz.shape[0]
        header = f"""ply
        format ascii 1.0
        element vertex {N}
        property float x
        property float y
        property float z
        end_header
        """
        with open(point_cloud_path, 'w') as f:
            f.write(header)
            for x, y, z in xyz:
                f.write(f"{x} {y} {z}\n")
            f.close()

def save_point_cloud_ply(points, assignment, output_dir="frames_ply"):
    def write_ply(filename: str, verts: np.ndarray, colors: np.ndarray):
        N = len(verts)
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        with open(filename, "w") as f:
            f.write(header)
            for (x, y, z), (r, g, b) in zip(verts, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    os.makedirs(output_dir, exist_ok=True)

    if torch.is_tensor(points):
        trajs = points.cpu().numpy()
    else:
        trajs = np.array(points)

    labels = np.array(assignment, dtype=int)
    num_labels = assignment.max() + 1

    cmap = cm.get_cmap("tab20", num_labels)
    lut = (cmap(range(num_labels))[:, :3] * 255).astype(np.uint8)
    frame_pts = trajs  # (N,3)
    frame_cols = lut[labels]  # (N,3)
    ply_name = os.path.join(output_dir, f"segmented_point.ply")
    write_ply(ply_name, frame_pts, frame_cols)


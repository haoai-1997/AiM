import os
import json
import numpy as np
import math
import shutil
import random
import OpenEXR, Imath
import cv2
import sys
import matplotlib.pyplot as plt


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def copy_files(src_directory, dest_directory):
    os.makedirs(dest_directory, exist_ok=True)

    for filename in os.listdir(src_directory):
        src_file = os.path.join(src_directory, filename)
        dest_file = os.path.join(dest_directory, filename)

        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")


def process_depth_file(src_file, dest_file):
    assert src_file.endswith(".exr")

    exr = OpenEXR.InputFile(src_file)
    dw = exr.header()['dataWindow']
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr.channel('V', pt)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(h, w).copy()

    valid_mask = depth < 8190.0
    depth[~valid_mask] = 0.0
    depth_mm = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
    cv2.imwrite(dest_file, depth_mm)


def copy_depth_files(src_directory, dest_directory):
    os.makedirs(dest_directory, exist_ok=True)

    for filename in os.listdir(src_directory):
        if filename.endswith(".npy"):
            continue
        elif filename.endswith(".png"):
            continue
        else:
            if filename == "render_meta.json":
                continue
            else:
                assert filename.endswith(".exr")

                src_file = os.path.join(src_directory, filename)

                dest_file = os.path.join(dest_directory, filename.split(".")[0] + ".png")
                exr = OpenEXR.InputFile(src_file)
                dw = exr.header()['dataWindow']
                w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                depth_str = exr.channel('V', pt)
                depth = np.frombuffer(depth_str, dtype=np.float32).reshape(h, w).copy()

                valid_mask = depth < 8190.0
                depth[~valid_mask] = 0.0
                depth_mm = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
                cv2.imwrite(dest_file, depth_mm)

def copy_single_file(src_directory, dest_directory):
    os.makedirs(dest_directory, exist_ok=True)
    for filename in sorted(os.listdir(src_directory)):
        src_file = os.path.join(src_directory, filename)
        dest_file = os.path.join(dest_directory, filename)
        print(filename)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")
        break


dir = "./spaien_data/com_part/render_200"
obj_names = ["table/31249"]
# "storage/45135", "blade1/103706", "fridge/10905", "foldchair/102255", "stapler/103111", "scissor/11100","washer/103776", "laptop/10211", "oven/101917", "USB/100109", "storage/45135"
for obj_name in obj_names:
    old_file = os.path.join(dir, obj_name)
    new_file = "../data/com_part/" + obj_name
    splits = ["train", "test"]  # , "train","test"

    ### start state·
    state_id = 0
    os.makedirs(new_file + '/start', exist_ok=True)
    for split in splits:
        copy_files(old_file + '/' + str(state_id) + '/' + split + '/rgb',
                   new_file + '/start' + '/' + split + '/rgb')
        copy_depth_files(old_file + '/' + str(state_id) + '/' + split + '/depth',
                         new_file + '/start' + '/' + split + '/depth')
        write_json = {}

        with open(old_file + '/' + str(state_id) + '/camera_' + split + '.json', 'r', encoding='utf-8') as f:
            spaien_data = json.load(f)
            f.close()

        K = np.array(spaien_data["K"]).astype(np.float32)
        focal_x = K[0][0]
        pixel_x = 2 * K[0][2]
        focal_y = K[1][1]
        pixel_y = 2 * K[1][2]
        camera_angle_x = focal2fov(focal_x, pixel_x)
        write_json["camera_angle_x"] = camera_angle_x
        write_json["frames"] = []
        spaien_data.pop("K")
        keys = sorted(spaien_data.keys())
        for t, key in enumerate(keys):
            frame_data = {}
            frame_data["file_path"] = './' + split + '/rgb/' + key
            frame_data["depth_file_path"] = './' + split + '/depth/' + key
            frame_data["time"] = 0.0
            frame_data["transform_matrix"] = spaien_data[key]
            write_json["frames"].append(frame_data)
        with open(new_file + '/start/' + 'transforms_' + split + '.json', 'w') as f:
            json.dump(write_json, f, indent=4)

    ### end state
    os.makedirs(new_file + '/end', exist_ok=True)
    for split in splits:
        if split == "test":
            copy_single_file(old_file + '/199/train/rgb',
                              new_file + '/end' + '/' + split + '/rgb')
            copy_depth_files(old_file + '/199/train/depth',
                             new_file + '/end' + '/' + split + '/depth')

            write_json = {}

            with open(old_file + '/199/camera_train.json', 'r', encoding='utf-8') as f:
                spaien_data = json.load(f)
                f.close()

            K = np.array(spaien_data["K"]).astype(np.float32)
            focal_x = K[0][0]
            pixel_x = 2 * K[0][2]
            focal_y = K[1][1]
            pixel_y = 2 * K[1][2]
            camera_angle_x = focal2fov(focal_x, pixel_x)
            write_json["camera_angle_x"] = camera_angle_x
            write_json["frames"] = []
            spaien_data.pop("K")
            keys = sorted(spaien_data.keys())
            for t, key in enumerate(keys):
                frame_data = {}
                frame_data["file_path"] = './' + split + '/rgb/' + key
                frame_data["depth_file_path"] = './' + split + '/depth/' + key
                frame_data["time"] = 1.0
                frame_data["transform_matrix"] = spaien_data[key]
                write_json["frames"].append(frame_data)
                break
            with open(new_file + '/end/' + 'transforms_' + split + '.json', 'w') as f:
                json.dump(write_json, f, indent=4)

        else:
            copy_files(old_file + '/199/train/rgb',
                       new_file + '/end' + '/' + split + '/rgb')
            copy_depth_files(old_file + '/199/train/depth',
                             new_file + '/end' + '/' + split + '/depth')
            write_json = {}

            with open(old_file + '/199/camera_train.json', 'r', encoding='utf-8') as f:
                spaien_data = json.load(f)
                f.close()

            K = np.array(spaien_data["K"]).astype(np.float32)
            focal_x = K[0][0]
            pixel_x = 2 * K[0][2]
            focal_y = K[1][1]
            pixel_y = 2 * K[1][2]
            camera_angle_x = focal2fov(focal_x, pixel_x)

            write_json["camera_angle_x"] = camera_angle_x
            write_json["frames"] = []
            spaien_data.pop("K")
            keys = sorted(spaien_data.keys())
            for t, key in enumerate(keys):
                frame_data = {}
                frame_data["file_path"] = './' + split + '/rgb/' + key
                frame_data["depth_file_path"] = './' + split + '/depth/' + key
                frame_data["time"] = 1.0
                frame_data["transform_matrix"] = spaien_data[key]
                write_json["frames"].append(frame_data)
            with open(new_file + '/end/' + 'transforms_' + split + '.json', 'w') as f:
                json.dump(write_json, f, indent=4)

    ## motion states

    motion_state_ids = [i for i in range(len(os.listdir(old_file)))]
    os.makedirs(new_file + '/motion/train/rgb', exist_ok=True)
    os.makedirs(new_file + '/motion/train/depth', exist_ok=True)
    os.makedirs(new_file + '/motion/test/rgb', exist_ok=True)
    os.makedirs(new_file + '/motion/test/depth', exist_ok=True)

    write_json_motion = {}
    write_json_motion["frames"] = []
    num = 0
    for motion_state_id in motion_state_ids:
        for i in range(1):
            idx_ = 0
            src_file = old_file + '/' + str(motion_state_id) + f'/train/rgb/' + f"{idx_:04d}" + '.png'
            dest_file = new_file + f"/motion/train/rgb/{(num):04d}.png"
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")
            src_file = old_file + '/' + str(motion_state_id) + f'/train/depth/' + f"{idx_:04d}" + '.exr'
            dest_file = new_file + f"/motion/train/depth/{(num):04d}.png"
            process_depth_file(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

            with open(old_file + '/' + str(motion_state_id) + '/camera_train.json', 'r', encoding='utf-8') as f:
                spaien_data = json.load(f)
                f.close()
            K_state = np.array(spaien_data["K"]).astype(np.float32)
            focal_x = K_state[0][0]
            pixel_x = 2 * K_state[0][2]
            focal_y = K_state[1][1]
            pixel_y = 2 * K_state[1][2]
            camera_angle_x = focal2fov(focal_x, pixel_x)
            frame_data = {}
            frame_data["file_path"] = f"./train/rgb/{(num):04d}"
            frame_data["depth_file_path"] = f"./train/depth/{(num):04d}"
            frame_data["rotation"] = (2 * np.pi) / (len(motion_state_ids))
            if 0 < motion_state_id < 199:
                frame_data["time"] = 0.0 + motion_state_id * 1.0 / 199

            elif motion_state_id == 0:
                frame_data["time"] = 0.0
            else:
                frame_data["time"] = 1.0

            frame_data["transform_matrix"] = spaien_data[f"{idx_:04d}"]
            write_json_motion["camera_angle_x"] = camera_angle_x
            write_json_motion["frames"].append(frame_data)
            num += 1
    with open(new_file + "/motion/transforms_train.json", 'w') as f:
        json.dump(write_json_motion, f, indent=4)

    write_test_json_motion = {}
    write_test_json_motion["frames"] = []
    num = 0
    motion_state_ids = [i for i in range(len(os.listdir(old_file)))]
    for motion_state_id in motion_state_ids:
        for i in range(1):
            select_id = random.randint(0, len(os.listdir(old_file)) - 1)
            src_file = old_file + '/' + str(motion_state_id) + f'/train/rgb/0000.png'
            dest_file = new_file + f"/motion/test/rgb/{(num):04d}.png"
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")
            src_file = old_file + '/' + str(motion_state_id) + f'/train/depth/0000.exr'
            dest_file = new_file + f"/motion/test/depth/{(num):04d}.png"
            process_depth_file(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

            with open(old_file + '/' + str(motion_state_id) + '/camera_train.json', 'r', encoding='utf-8') as f:
                spaien_data = json.load(f)
                f.close()
            K_state = np.array(spaien_data["K"]).astype(np.float32)
            focal_x = K_state[0][0]
            pixel_x = 2 * K_state[0][2]
            focal_y = K_state[1][1]
            pixel_y = 2 * K_state[1][2]
            camera_angle_x = focal2fov(focal_x, pixel_x)
            frame_data = {}
            frame_data["file_path"] = f"./test/rgb/{(num):04d}"
            frame_data["depth_file_path"] = f"./test/depth/{(num):04d}"
            frame_data["rotation"] = (2 * np.pi) / (len(motion_state_ids))

            if 0 < motion_state_id < 199:
                frame_data["time"] = 0.0 + motion_state_id * 1.0 / 199
            elif motion_state_id == 0:
                frame_data["time"] = 0.0
            else:
                frame_data["time"] = 1.0

            frame_data["transform_matrix"] = spaien_data[f"0000"]
            write_test_json_motion["camera_angle_x"] = camera_angle_x
            write_test_json_motion["frames"].append(frame_data)
            num += 1
        break
    with open(new_file + "/motion/transforms_test.json", 'w') as f:
        json.dump(write_test_json_motion, f, indent=4)

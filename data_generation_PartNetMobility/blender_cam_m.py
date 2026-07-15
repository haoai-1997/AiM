import random
import os
import sys
import bpy
import mathutils
import json
import argparse
import numpy as np
import math


def set_if_has(obj, attr, value):
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def remove_cube():
    bpy.ops.object.select_all(action='DESELECT')
    if "Cube" in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()


def setup_lighting_and_world(radius):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'

    ee = bpy.context.scene.eevee
    ee.use_raytracing = False
    ee.ray_tracing_method = 'SCREEN'

    # === 清理现有灯光 ===
    for L in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(L, do_unlink=True)

    # === World：均匀的“天空/棚房”环境光 ===
    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    nodes, links = nt.nodes, nt.links

    # 清空并重建世界节点
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    # 纯色偏中性灰，避免过曝/偏色（也可换 HDRI）
    bg.inputs["Color"].default_value = (0.2, 0.2, 0.2, 1.0)
    bg.inputs["Strength"].default_value = 0.9 # 环境强度：1.0~1.5之间调

    links.new(bg.outputs["Background"], out.inputs["Surface"])

    # # === 软箱灯参数（覆盖大、均匀） ===
    ENERGY = 10
    DIST = 2
    SIZE = 1

    def add_area(name, loc, target=(0, 0, 0), energy=ENERGY, size=SIZE):
        bpy.ops.object.light_add(type='AREA', location=loc)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.shape = 'SQUARE'
        light.data.size = size

        # 朝向目标
        direction = mathutils.Vector(target) - mathutils.Vector(loc)
        light.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        return light

    # ---- 16个方向 ----
    dirs = [
        (DIST, 0, 0), (-DIST, 0, 0),
        (0, DIST, 0), (0, -DIST, 0),
        (0, 0, DIST), (0, 0, -DIST),

        # (DIST, DIST, 1), (-DIST, DIST, 1),
        (DIST, -DIST, DIST), (-DIST, -DIST, DIST),
        (DIST, DIST, -DIST), (-DIST, DIST, -DIST),
        (DIST, -DIST, -DIST), (-DIST, -DIST, -DIST)
    ]

    # 归一化并放置灯
    for i, d in enumerate(dirs):
        v = mathutils.Vector(d).normalized() * DIST
        add_area(f"Softbox_{i:02d}", loc=v)


def make_pbr_material(obj,
                      roughness=1.0,
                      specular=0.0):

    for slot in obj.material_slots:
        mat = slot.material
        if not mat:
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        orig_img = None
        for n in nodes:
            if n.type == 'TEX_IMAGE' and n.image:
                orig_img = n.image
                break

        for n in list(nodes):
            nodes.remove(n)

        tex = nodes.new('ShaderNodeTexImage')
        if orig_img: tex.image = orig_img
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        out = nodes.new('ShaderNodeOutputMaterial')

        tex.location = (-300, 0)
        bsdf.location = (0, 0)
        out.location = (200, 0)
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Specular IOR Level'].default_value = specular

        if orig_img:
            links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])


def camera_setting(exp_dir, n_frames=100, stage='train', track_to='start', radius=6.0, frame_id=0, category=None):

    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    remove_cube()

    setup_lighting_and_world(radius=radius)

    cam_obj = bpy.data.objects['Camera']
    cam = bpy.data.objects['Camera'].data
    cam.type = 'PERSP'
    cam.sensor_fit = 'AUTO'
    cam.sensor_width = 36.0
    cam.sensor_height = 36.0

    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = 1.0

    constraint = cam_obj.constraints.new('TRACK_TO')
    constraint.target = bpy.data.objects[track_to]
    bpy.context.scene.frame_end = n_frames - 1
    bpy.context.scene.frame_start = 0
    bpy.context.scene.render.resolution_x = 800
    bpy.context.scene.render.resolution_y = 800
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = f'{exp_dir}/{stage}/rgb/'
    bpy.context.scene.render.film_transparent = True

    if stage == 'test':
        phi = [np.pi / 3. for i in range(n_frames)]
        theta = [i * (2 * np.pi) / n_frames for i in range(n_frames)]
    else:
        if frame_id == 0:
            phi = np.random.random_sample(n_frames) * 0.5 * np.pi
            theta = np.random.random_sample(n_frames) * 2. * np.pi
        elif frame_id == 199:
            phi = np.random.random_sample(n_frames) * 0.5 * np.pi
            theta = np.random.random_sample(n_frames) * 2. * np.pi
        else:
            assert category in ['blade1', 'foldchair', 'fridge', 'laptop', 'oven', 'scissor', 'stapler', 'storage', "table",'USB', 'washer']

            if category in ["fridge", "table", "storage", "oven"]:
                x = np.linspace(0, 10 * np.pi, 198)
                theta =  [-0.5 * np.pi - 0.48 * np.pi * np.sin(x[frame_id-1])]
                phi_min = 0.25 * np.pi
                phi_max = 0.55 * np.pi
                L = 45
                u = ((frame_id-1) % L) / L
                tri = 0.5 * (1 - np.cos(2 * np.pi * u))
                phi = [phi_min + (phi_max - phi_min) * tri]
            if category in ["foldchair", "stapler", "laptop", "washer"]:
                x = np.linspace(0, 10 * np.pi, 198)
                theta =  [-0.5 * np.pi - 0.48 * np.pi * np.sin(x[frame_id-1])]
                phi_min = 0.1 * np.pi
                phi_max = 0.5 * np.pi
                L = 45
                u = ((frame_id-1) % L) / L
                tri = 0.5 * (1 - np.cos(2 * np.pi * u))
                phi = [phi_min + (phi_max - phi_min) * tri]
            if category in ["blade1", "scissor", "USB"]:
                x = np.linspace(0, 10 * np.pi, 198)
                theta = [-0.5 * np.pi - 1 * np.pi * np.sin(x[frame_id - 1])]
                phi_min = 0.0 * np.pi
                phi_max = 0.5 * np.pi
                L = 45
                u = ((frame_id-1) % L) / L
                tri = 0.5 * (1 - np.cos(2 * np.pi * u))
                phi = [phi_min + (phi_max - phi_min) * tri]
            if category in ["blade1"]:
                x = np.linspace(0, 20 * np.pi, 198)
                theta = [-0.5 * np.pi - 1 * np.pi * np.sin(x[frame_id - 1])]
                phi_min = 0.0 * np.pi
                phi_max = 1 * np.pi
                L = 30
                u = ((frame_id-1) % L) / L
                tri = 0.5 * (1 - np.cos(2 * np.pi * u))
                phi = [phi_min + (phi_max - phi_min) * tri]
    for f in range(n_frames):
        x = radius * np.cos(theta[f]) * np.sin(phi[f])
        y = radius * np.sin(theta[f]) * np.sin(phi[f])
        z = radius * np.cos(phi[f])
        cam_obj.location = np.array([x, y, z])
        cam_obj.keyframe_insert(data_path="location", frame=f)
        print("Camera location:", cam_obj.location)
        print("Camera matrix:", cam_obj.matrix_world)

    cam_obj = bpy.data.objects.get("Camera")
    if cam_obj:
        track_to_constraint = None
        for constraint in cam_obj.constraints:
            if constraint.type == 'TRACK_TO':
                track_to_constraint = constraint
                break

        if track_to_constraint:
            print("Camera Tracking Location:", track_to_constraint.target.location)
            if track_to_constraint.target.location == mathutils.Vector((0, 0, 0)):
                print("Camera Tracking Location is at the Origin.")
            else:
                print("Camera Tracking Location is not at the Origin.")
        else:
            print("Track_To_Constraint Not Found")
    else:
        print("Obj Camera Not Found")

    cam_obj = bpy.data.objects['Camera']
    _sanity_check_intrinsics(cam_obj)

def get_K(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100.0
    W = scale * scene.render.resolution_x
    H = scale * scene.render.resolution_y

    fx = 0.5 * W / math.tan(0.5 * camd.angle_x)
    fy = 0.5 * H / math.tan(0.5 * camd.angle_y)

    cx = 0.5 * W - camd.shift_x * W
    cy = 0.5 * H + camd.shift_y * W

    return [[fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]]

def get_camera_dict():
    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end

    cam_dict = {}
    cameras = []
    for obj in scene.objects:
        if obj.type != 'CAMERA':
            continue
        cameras.append((obj, obj.data))
    for obj, obj_data in cameras:
        K = get_K(obj_data)
        cam_dict['K'] = K

    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        frame_name = '{:0>4d}'.format(f)
        for obj, obj_data in cameras:
            mat = obj.matrix_world
            mat_list = [
                [mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
                [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
                [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
                [mat[3][0], mat[3][1], mat[3][2], mat[3][3]],
            ]
            cam_dict[frame_name] = mat_list
    return cam_dict


def camera_export(stage='train', dst_dir=''):
    file_path = f'{dst_dir}/camera_{stage}.json'
    with open(file_path, 'w') as fh:
        cam = get_camera_dict()
        json.dump(cam, fh, indent=4)
        fh.close()

def _sanity_check_intrinsics(cam_obj):
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100.0
    W = scale * scene.render.resolution_x
    H = scale * scene.render.resolution_y
    K = np.array(get_K(cam_obj.data), dtype=np.float64)
    fx, fy = K[0,0], K[1,1]

    anglex_from_K = 2.0 * math.atan(W / (2.0 * fx))
    angley_from_K = 2.0 * math.atan(H / (2.0 * fy))
    print(f"[CHECK] W={W:.0f}, H={H:.0f}, fx={fx:.2f}, fy={fy:.2f}")
    print(f"[CHECK] FOVx_from_K={math.degrees(anglex_from_K):.4f}°, Blender angle_x={math.degrees(cam_obj.data.angle_x):.4f}°")
    print(f"[CHECK] FOVy_from_K={math.degrees(angley_from_K):.4f}°, Blender angle_y={math.degrees(cam_obj.data.angle_y):.4f}°")

if __name__ == "__main__":
    category_name = sys.argv[-3].split("/")[0]
    category_id = sys.argv[-3].split("/")[1]
    if category_id in ["47254","11304"]:
        root_dir = "./spaien_data/two_part"
    else:
        root_dir = "./spaien_data/com_part"

    objs = [f"{root_dir}/obj_file_200/{category_name}/{category_id}"]


    stage = sys.argv[-1]
    for obj in objs:
        entries = os.listdir(obj + "/" + "textured_objs/")

        f_name = obj.split("/")[-2] + "/" + obj.split("/")[-1]

        frame_id = int(sys.argv[-2])
        radius = float(sys.argv[-3].split("/")[2])

        src_dir = obj + "/" + "textured_objs/" + str(frame_id) + "/" + str(frame_id) + ".obj"
        dst_dir = f"{root_dir}/render_200/" + f_name + "/" + str(frame_id)

        assert os.path.exists(src_dir), f'{src_dir} does not exist'
        os.makedirs(dst_dir, exist_ok=True)

        bpy.ops.wm.obj_import(filepath=src_dir, import_vertex_groups=True)
        main_obj = bpy.context.selected_objects[0]
        main_obj.location = (0.0, 0.0, 0.0)
        main_obj.rotation_euler = (
            main_obj.rotation_euler.x,
            main_obj.rotation_euler.y,
            main_obj.rotation_euler.z
        )
        main_obj.scale = (1.0, 1.0, 1.0)

        for mat in bpy.data.materials:
            mat.use_backface_culling = True

        centroid = main_obj.matrix_world.to_translation()
        empty = bpy.data.objects.new("Centroid", None)
        bpy.context.collection.objects.link(empty)
        empty.location = centroid

        if stage == "train":
            if frame_id == 0:
                n_frames = 100
            elif frame_id == 199:
                n_frames = 100
            else:
                n_frames = 1
        else:
            if frame_id == 0:
                n_frames = 5
            else:
                n_frames = 0

        if n_frames != 0:
            camera_setting(exp_dir=dst_dir, n_frames=n_frames, stage=stage, track_to=empty.name, radius=radius,
                           frame_id=frame_id, category= category_name)

            camera_export(stage, dst_dir)

            bpy.ops.render.render(animation=True)

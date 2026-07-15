import bpy
import sys, json, shutil
import numpy as np
from pathlib import Path
from mathutils import Matrix
import math

def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def set_camera_from_K(cam, K, resx, resy):
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    cam.type = 'PERSP'
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width  = 36.0
    cam.sensor_height = 36.0
    cam.lens = fx * cam.sensor_width / resx
    cam.sensor_height = cam.lens * (resy / fy)
    cam.shift_x = (0.5 * resx - cx) / resx
    cam.shift_y = (cy - 0.5*resy) / resx  # 注意除以 W

def import_obj(obj_path: Path):
    bpy.ops.wm.obj_import(
        filepath=str(obj_path),
        use_split_objects=True,
        use_split_groups=True
    )

# ---------------- Compositor: Depth with NaN ----------------
def setup_scene_and_compositor(dst_dir: Path, res_x=800, res_y=800, engine='BLENDER_EEVEE_NEXT', clip_end_val=10000.0):
    scene = bpy.context.scene

    bpy.ops.object.select_all(action='SELECT')
    if bpy.ops.object.delete.poll():
        bpy.ops.object.delete(use_global=False)

    scene.render.engine = 'BLENDER_EEVEE_NEXT' if engine not in ('BLENDER_EEVEE_NEXT', 'CYCLES') else engine
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = True


    vl = scene.view_layers[0]
    vl.use_pass_z = True

    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'

    scene.use_nodes = True
    nt = scene.node_tree
    while nt.nodes:
        nt.nodes.remove(nt.nodes[0])

    n_rl = nt.nodes.new("CompositorNodeRLayers")
    n_rl.layer = vl.name

    n_gt0 = nt.nodes.new("CompositorNodeMath"); n_gt0.operation = 'GREATER_THAN'
    n_gt0.inputs[1].default_value = 0.0
    nt.links.new(n_rl.outputs["Depth"], n_gt0.inputs[0])

    n_clip = nt.nodes.new("CompositorNodeValue"); n_clip.outputs[0].default_value = clip_end_val
    n_ltfar = nt.nodes.new("CompositorNodeMath"); n_ltfar.operation = 'LESS_THAN'
    nt.links.new(n_rl.outputs["Depth"], n_ltfar.inputs[0])
    nt.links.new(n_clip.outputs[0], n_ltfar.inputs[1])

    n_mask = nt.nodes.new("CompositorNodeMath"); n_mask.operation = 'MULTIPLY'
    nt.links.new(n_gt0.outputs[0], n_mask.inputs[0])
    nt.links.new(n_ltfar.outputs[0], n_mask.inputs[1])

    n_zeroA = nt.nodes.new("CompositorNodeValue"); n_zeroA.outputs[0].default_value = 0.0
    n_zeroB = nt.nodes.new("CompositorNodeValue"); n_zeroB.outputs[0].default_value = 0.0
    n_nan = nt.nodes.new("CompositorNodeMath"); n_nan.operation = 'DIVIDE'
    nt.links.new(n_zeroA.outputs[0], n_nan.inputs[0])
    nt.links.new(n_zeroB.outputs[0], n_nan.inputs[1])

    n_one = nt.nodes.new("CompositorNodeValue"); n_one.outputs[0].default_value = 1.0
    n_inv = nt.nodes.new("CompositorNodeMath"); n_inv.operation = 'SUBTRACT'
    nt.links.new(n_one.outputs[0], n_inv.inputs[0])
    nt.links.new(n_mask.outputs[0], n_inv.inputs[1])

    n_term_valid = nt.nodes.new("CompositorNodeMath"); n_term_valid.operation = 'MULTIPLY'
    nt.links.new(n_mask.outputs[0], n_term_valid.inputs[0])
    nt.links.new(n_rl.outputs["Depth"], n_term_valid.inputs[1])

    n_term_nan = nt.nodes.new("CompositorNodeMath"); n_term_nan.operation = 'MULTIPLY'
    nt.links.new(n_inv.outputs[0], n_term_nan.inputs[0])
    nt.links.new(n_nan.outputs[0], n_term_nan.inputs[1])

    n_out = nt.nodes.new("CompositorNodeMath"); n_out.operation = 'ADD'
    nt.links.new(n_term_valid.outputs[0], n_out.inputs[0])
    nt.links.new(n_term_nan.outputs[0], n_out.inputs[1])

    n_file_exr = nt.nodes.new("CompositorNodeOutputFile")
    n_file_exr.base_path = str(dst_dir)
    n_file_exr.file_slots[0].path = "####"  # 0000.exr

    fmt = n_file_exr.format
    fmt.file_format = 'OPEN_EXR'
    fmt.color_mode = 'BW'  # 单通道
    fmt.color_depth = '32'  # float32
    fmt.exr_codec = 'ZIP'  # 无损压缩

    try:
        n_file_exr.file_slots[0].use_node_path = False  # 禁止自动建 "Image/" 子目录
    except AttributeError:
        pass

    nt.links.new(n_out.outputs[0], n_file_exr.inputs[0])

    return scene, clip_end_val

# ---------------- Main ----------------
def decide_length(stage: str, frame_id: int):
    if stage == "train":
        return 100 if frame_id in (0,199) else 1
    else:
        return 5 if frame_id == 0 else 0

def main():
    # ---------------- Paths ----------------
    OBJ_ROOT = Path("./spaien_data/com_part/obj_file_200/")
    IMG_ROOT = Path("./spaien_data/com_part/render_200/")

    name = sys.argv[-3]
    frame_id = int(sys.argv[-2])
    stage = sys.argv[-1]

    obj_dir = OBJ_ROOT / name / "textured_objs"
    assert obj_dir.exists(), f"{obj_dir} not found"

    obj_idx = frame_id
    src_obj = obj_dir / str(obj_idx) / f"{obj_idx}.obj"
    assert src_obj.exists(), f"{src_obj} not found"

    dst_dir = IMG_ROOT / name / str(frame_id) / stage / "depth"
    ensure_empty_dir(dst_dir)

    cam_json = IMG_ROOT / name / str(frame_id) / f"camera_{stage}.json"
    assert cam_json.exists(), f"{cam_json} not found"
    contents = load_json(cam_json)
    K = np.array(contents["K"], dtype=np.float64)

    scene, clip_end_val = setup_scene_and_compositor(dst_dir, res_x=800, res_y=800, engine='BLENDER_EEVEE_NEXT', clip_end_val=10000.0)

    import_obj(src_obj)

    cam_data = bpy.data.cameras.new("Cam")
    set_camera_from_K(cam_data, K, scene.render.resolution_x, scene.render.resolution_y)
    cam_data.clip_start = 0.001
    cam_data.clip_end   = clip_end_val
    cam_obj = bpy.data.objects.new("CamObj", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    length = decide_length(stage, frame_id)

    for j in range(length):
        key = f"{j:04d}"
        if key not in contents:
            print(f"[WARN] pose {key} missing; skip")
            continue
        pose = np.array(contents[key], dtype=np.float64)
        cam_obj.matrix_world = Matrix(pose.tolist())
        scene.frame_current = j
        bpy.ops.render.render(write_still=True, use_viewport=False)

    print("Done.")

if __name__ == "__main__":
    main()

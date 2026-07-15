import sys
import numpy as np
import json
import os
import trimesh
import shutil
from lxml import etree as ET


def save_axis_mesh(k, center, filepath):
    """
    利用 trimesh 创建一个箭头（由圆柱和圆锥组成），表示关节旋转轴，
    并将其保存到文件中。
    """
    # 参数设置
    shaft_length = 1.0
    cone_height = 0.08
    cone_radius = 0.04
    cylinder_radius = 0.02

    shaft = trimesh.creation.cylinder(radius=cylinder_radius, height=shaft_length, sections=32)
    cone = trimesh.creation.cone(radius=cone_radius, height=cone_height, sections=32)
    cone.apply_translation([0, 0, shaft_length])
    arrow = trimesh.util.concatenate([shaft, cone])
    arrow_direction = np.array([0, 0, 1], dtype=np.float32)
    k = np.array(k, dtype=np.float32)
    if not np.allclose(arrow_direction, k):
        R, _ = trimesh.geometry.align_vectors(arrow_direction, k)
    else:
        R = np.eye(4)
    arrow.apply_transform(R)
    arrow.apply_translation(center[:3])
    arrow.export(filepath)


def normalize(v):
    return v / np.linalg.norm(v)


def rewrite_json_from_urdf(src_root):
    urdf_file = os.path.join(src_root, 'mobility.urdf')
    tree = ET.parse(urdf_file)
    visuals_dict = {}
    for link in tree.getroot().iter('link'):
        meshes = []
        for mesh in link.findall('.//mesh'):
            fn = mesh.attrib.get('filename')
            if fn:
                meshes.append(fn)
        visuals_dict[link.attrib['name']] = meshes

    meta_path = os.path.join(src_root, 'mobility_v2.json')
    meta = json.load(open(meta_path, 'r'))
    for entry in meta:
        link_name = f"link_{entry['id']}"
        entry['visuals'] = visuals_dict.get(link_name, [])

    out_path = os.path.join(src_root, 'mobility_v2_self.json')
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=2)


def get_arti_info(entry, motion):
    info = {'axis': {'o': entry['jointData']['axis']['origin'],
                     'd': entry['jointData']['axis']['direction']}}
    if entry['joint'] == 'hinge':
        info['type'] = 'rotate'
        info['l'], info['r'] = motion['rotate']
    elif entry['joint'] == 'slider':
        info['type'] = 'translate'
        info['l'], info['r'] = motion['translate']
    else:
        raise NotImplementedError(f"Joint {entry['joint']} not supported.")
    return info


def load_articulations(src_root, motions_list):
    """返回多关节信息列表和整个 meta"""
    meta = json.load(open(os.path.join(src_root, 'mobility_v2_self.json'), 'r'))
    os.remove(os.path.join(src_root, 'mobility_v2_self.json'))

    arti_list = []
    # 对 motions_list 中每个 motion 提取信息
    for m in motions_list:

        entry = next(e for e in meta if e['id'] == m['joint_id'])
        arti = get_arti_info(entry, m['motion'])
        arti_list.append((m['joint_id'], arti))
    return arti_list, meta


def generate_state(arti_list, meta, src_root, exp_dir, state, total_steps):
    # 清空并准备输出路径
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    dynamic_meshes = {jid: [] for jid, _ in arti_list}
    static_meshes = []

    # 加载网格
    for entry in meta:
        # 判断 entry 是否属于动态
        is_dynamic = False
        for jid, _ in arti_list:
            if entry['id'] == jid or entry.get('parent') == jid:
                for fn in entry['visuals']:
                    p = os.path.join(src_root, fn)
                    if os.path.exists(p):
                        mesh = trimesh.load(p, force='mesh')
                        dynamic_meshes[jid].append(mesh)
                is_dynamic = True
                break
        if not is_dynamic:
            for fn in entry['visuals']:
                p = os.path.join(src_root, fn)
                if os.path.exists(p):
                    mesh = trimesh.load(p, force='mesh')
                    static_meshes.append(mesh)

    # 变换并导出
    t = state / float(total_steps - 1)
    for jid, info in arti_list:
        meshes = dynamic_meshes[jid]
        if info['type'] == 'rotate':
            angle = np.deg2rad(info['l'] + (info['r'] - info['l']) * t)
            R = trimesh.transformations.rotation_matrix(angle,
                                                       info['axis']['d'],
                                                       point=info['axis']['o'])
            for m in meshes:
                m.apply_transform(R)
        else:
            dist = info['l'] + (info['r'] - info['l']) * t
            T = trimesh.transformations.translation_matrix(
                np.array(info['axis']['d']) * dist)
            for m in meshes:
                m.apply_transform(T)

    # 合并所有并导出 obj
    all_meshes = list(dynamic_meshes.values()) + static_meshes
    # 展平动态列表
    flat = []
    for group in dynamic_meshes.values(): flat.extend(group)
    all_meshes = flat + static_meshes

    if all_meshes:
        combined = trimesh.util.concatenate(all_meshes)
        out_path = os.path.join(exp_dir, f"{state}.obj")
        combined.export(out_path)

def save_motions_list(dst_root, motions_list):
    """把 motions_list 保存到 dst_root/motion_list.json"""
    os.makedirs(dst_root, exist_ok=True)
    out_f = os.path.join( "/".join(dst_root.split('/')[:-1]) , "motion_list.json")
    with open(out_f, "w", encoding="utf-8") as f:
        json.dump(motions_list, f, indent=2, ensure_ascii=False)
    print(f"Saved motions_list to {out_f}")

def main(model_id, motions_list, src_root, dst_root, steps):
    # 生成各状态编号
    states = [i for i in range(0, steps)]
    # 根据 URDF 重写 json 文件
    rewrite_json_from_urdf(src_root)
    arti_list, meta = load_articulations(src_root, motions_list)

    save_motions_list(dst_root, motions_list)
    # 针对每个状态生成新的网格
    for state in states:
        exp_dir = os.path.join(dst_root, str(state))
        os.makedirs(exp_dir, exist_ok=True)
        generate_state(arti_list, meta, src_root, exp_dir, state, len(states))
        print(f'State {state} done')

if __name__ == '__main__':
    category = ['fridge', 'storage','table', 'storage']
    model_id = ['11304', '47254', '31249', '47648']
    motions_list= [
        [
            {'joint_id': 0, 'motion': {'type': 'rotate', 'rotate': [0, -180]}},
            {'joint_id': 1, 'motion': {'type': 'rotate', 'rotate': [0, -90]}},
        ],
        [
            {'joint_id': 1, 'motion': {'type': 'translate', 'translate': [0, 0.55]}},
            {'joint_id': 0, 'motion': {'type': 'rotate', 'rotate': [0, 90]}},

        ],
        [
            {'joint_id': 0, 'motion': {'type': 'translate', 'translate': [0, 0.38]}},
            {'joint_id': 1, 'motion': {'type': 'translate', 'translate': [0.35, 0]}},
            {'joint_id': 3, 'motion': {'type': 'rotate', 'rotate': [0, -90]}},
            {'joint_id': 4, 'motion': {'type': 'rotate', 'rotate': [0, 90]}},


        ],
        [
            {'joint_id': 0, 'motion': {'type': 'rotate', 'rotate': [0, 120]}},
            {'joint_id': 1, 'motion': {'type': 'rotate', 'rotate': [0, -120]}},
            {'joint_id': 2, 'motion': {'type': 'rotate', 'rotate': [0, -60]}},
            {'joint_id': 3, 'motion': {'type': 'rotate', 'rotate': [0, 60]}},
            {'joint_id': 4, 'motion': {'type': 'translate', 'translate': [0, 0.1]}},
            {'joint_id': 5, 'motion': {'type': 'translate', 'translate': [0, 0.16]}},
        ]
    ]
    root_dir = './spaien_data/'
    for idx in range(2,3):
        steps = int(sys.argv[1])
        if model_id[idx] in ["11304", "47254"]:
            src_root = os.path.join(root_dir + 'two_part/original_data', category[idx], model_id[idx])
            dst_root = os.path.join(root_dir + f'two_part/obj_file_{steps}/',category[idx], model_id[idx], 'textured_objs')
        else:
            src_root = os.path.join(root_dir + 'com_part/original_data', category[idx], model_id[idx])
            dst_root = os.path.join(root_dir + f'com_part/obj_file_{steps}/',category[idx], model_id[idx], 'textured_objs')
        main(model_id, motions_list[idx], src_root, dst_root, steps)

import sys
import numpy as np
import json
import os
import trimesh
import shutil

from lxml import etree as ET


def save_axis_mesh(k, center, filepath):
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

def rewrite_json_from_urdf(src_root):
    urdf_file = os.path.join(src_root, 'mobility.urdf')
    tree = ET.parse(urdf_file)
    root_elem = tree.getroot()
    visuals_dict = {}
    for link in root_elem.iter('link'):
        meshes = []
        for visual in link.iter('visual'):
            try:
                filename = visual[1][0].attrib['filename']
                meshes.append(filename)
            except Exception as e:
                pass
        visuals_dict[link.attrib['name']] = meshes

    with open(os.path.join(src_root, 'mobility_v2.json'), 'r') as f:
        meta = json.load(f)
    for entry in meta:
        link_name = 'link_{}'.format(entry['id'])
        entry['visuals'] = visuals_dict.get(link_name, [])

    with open(os.path.join(src_root, 'mobility_v2_self.json'), 'w') as json_out_file:
        json.dump(meta, json_out_file)


def get_rotation_axis_angle(k, theta):

    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx ** 2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky ** 2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz ** 2) * (1 - cos)
    return R


def get_arti_info(entry, motion):
    res = {
        'axis': {
            'o': entry['jointData']['axis']['origin'],
            'd': entry['jointData']['axis']['direction']
        }
    }
    if entry['joint'] == 'hinge':
        assert motion['type'] == 'rotate'
        R_limit_l, R_limit_r = motion['rotate'][0], motion['rotate'][1]
        res.update({
            'rotate': {
                'l': R_limit_l,
                'r': R_limit_r
            },
        })
    elif entry['joint'] == 'slider':
        assert motion['type'] == 'translate'
        T_limit_l, T_limit_r = motion['translate'][0], motion['translate'][1]
        res.update({
            'translate': {
                'l': T_limit_l,
                'r': T_limit_r
            }
        })
    else:
        raise NotImplementedError('{} joint is not implemented'.format(entry['joint']))
    return res


def load_articulation(src_root, joint_id, motions):
    with open(os.path.join(src_root, 'mobility_v2_self.json'), 'r') as f:
        meta = json.load(f)
    os.remove(os.path.join(src_root, 'mobility_v2_self.json'))
    arti_info = None
    for entry in meta:
        if entry['id'] == joint_id:
            arti_info = get_arti_info(entry, motions['motion'])
            break
    return arti_info, meta


def export_axis_mesh(arti, exp_dir):
    center = np.array(arti['axis']['o'], dtype=np.float32)
    k = np.array(arti['axis']['d'], dtype=np.float32)
    save_axis_mesh(k, center, os.path.join(exp_dir, 'axis_rotate.ply'))
    save_axis_mesh(-k, center, os.path.join(exp_dir, 'axis_rotate_oppo.ply'))

def save_motions_list(dst_root, motions_list,idx):
    os.makedirs(dst_root, exist_ok=True)
    out_f = os.path.join(dst_root.split("textured_objs")[0], "motion_list.json")
    with open(out_f, "w", encoding="utf-8") as f:
        json.dump(motions_list[idx], f, indent=2, ensure_ascii=False)
    print(f"Saved motions_list to {out_f}")


def generate_state(arti_info, meta, src_root, exp_dir, state, total_len, motions):
    joint_id = motions['joint_id']
    motion_type = motions['motion']['type']

    dynamic_meshes = []
    static_meshes = []

    for entry in meta:
        if entry['id'] == joint_id or entry['parent'] == joint_id:
            for mesh_fname in entry['visuals']:
                mesh_path = os.path.join(src_root, mesh_fname)
                if os.path.exists(mesh_path):
                    mesh = trimesh.load(mesh_path, force='mesh')
                    dynamic_meshes.append(mesh)

    if motion_type == 'rotate':
        if state == 'start':
            degree = arti_info['rotate']['l']
        elif state == 'end':
            degree = arti_info['rotate']['r']
        elif state == 'canonical':
            degree = 0.5 * (arti_info['rotate']['r'] + arti_info['rotate']['l'])
        else:
            degree = arti_info['rotate']['l'] + (arti_info['rotate']['r'] - arti_info['rotate']['l']) * state / (total_len -1)
        theta = np.deg2rad(degree)
        axis = np.array(arti_info['axis']['d'])
        center = np.array(arti_info['axis']['o'])
        R = trimesh.transformations.rotation_matrix(theta, axis, point=center)
        for mesh in dynamic_meshes:
            mesh.apply_transform(R)

    elif motion_type == 'translate':
        if state == 'start':
            dist = arti_info['translate']['l']
        elif state == 'end':
            dist = arti_info['translate']['r']
        elif state == 'canonical':
            dist = 0.5 * (arti_info['translate']['r'] + arti_info['translate']['l'])
        else:
            dist = arti_info['translate']['l']+  state * (arti_info['translate']['r'] - arti_info['translate']['l']) / (total_len-1)
        translation = np.array(arti_info['axis']['d']) * dist
        T = trimesh.transformations.translation_matrix(translation)
        for mesh in dynamic_meshes:
            mesh.apply_transform(T)
    else:
        raise NotImplementedError

    for entry in meta:
        if entry['id'] != joint_id and entry['parent'] != joint_id:
            for mesh_fname in entry['visuals']:
                mesh_path = os.path.join(src_root, mesh_fname)
                if os.path.exists(mesh_path):
                    mesh = trimesh.load(mesh_path, force='mesh')
                    static_meshes.append(mesh)

    all_meshes = dynamic_meshes + static_meshes
    if len(all_meshes) > 0:
        combined = trimesh.util.concatenate(all_meshes)
    else:
        combined = None
    output_path = os.path.join(exp_dir, f'{state}.obj')
    if combined is not None:
        combined.export(output_path)


def record_motion_json(motions, arti_info, dst_root):
    axis_o = np.array(arti_info['axis']['o'])
    axis_d = np.array(arti_info['axis']['d'])
    R_coord = np.array([[1., 0., 0.],
                        [0., 0., -1.],
                        [0., 1., 0.]])
    axis_o = np.matmul(R_coord, axis_o).tolist()
    axis_d = np.matmul(R_coord, axis_d).tolist()
    arti_info['axis']['o'] = axis_o
    arti_info['axis']['d'] = axis_d
    arti_info['type'] = motions['motion']['type']

    with open(os.path.join(dst_root, 'trans.json'), 'w') as f:
        conf = {
            'input': motions,
            'trans_info': arti_info
        }
        json.dump(conf, f)

    return arti_info


def main(motions, src_root, dst_root, steps, idx):

    states = [i for i in range(0, steps)]
    rewrite_json_from_urdf(src_root)

    arti_info, meta = load_articulation(src_root, motions['joint_id'], motions)
    save_motions_list(dst_root, motions_list, idx)

    for state in states:
        exp_dir = os.path.join(dst_root, str(state))
        os.makedirs(exp_dir, exist_ok=True)
        generate_state(arti_info, meta, src_root, exp_dir, state, len(states), motions)
        print(f'State {state} done')



if __name__ == '__main__':
    category = ['blade1', 'foldchair', 'fridge', 'laptop', 'oven', 'scissor', 'stapler', 'storage','USB', 'washer']
    model_id = ['103706', '102255', '10905', '10211','101917' , '11100', '103111', '45135', '100109', '103776']
    motions_list= [{
        'joint_id': 0,
        'motion': {
            'type': 'translate',  # "rotate" / "translate"
            'rotate': [0, 0],
            'translate': [0.0, 0.8],
        },
    },
        {
            'joint_id': 1,
            'motion': {
                'type': 'rotate',
                'rotate': [0, -90],
                'translate': [0.0, 0.0],
            },
        },
        {
            'joint_id': 0,
            'motion': {
                'type': 'rotate',
                'rotate': [-110, 20],
                'translate': [0.0, 0.0],
            },
        },
        {
            'joint_id': 1,
            'motion': {
                'type': 'rotate',
                'rotate': [105, -15],
                'translate': [0.0, 0.0],
            },
        },
        {
            'joint_id': 0,
            'motion': {
                'type': 'rotate',
                'rotate': [0, 90],
                'translate': [0.0, 0.0],
            },
        },
        {
            'joint_id': 0,
            'motion': {
                'type': 'rotate',
                'rotate': [-45, 45],
                'translate': [0.0, 0.0],
            },
        },
        {
            'joint_id': 1,
            'motion': {
                'type': 'rotate',
                'rotate': [5, -85],
                'translate': [0.0, 0.0],
            },
        },
        {
            'joint_id': 0,
            'motion': {
                'type': 'translate',
                'rotate': [0, 0],
                'translate': [0.0, 0.56],
            },
        },
        {
            'joint_id': 0,
            'motion': {
                'type': 'rotate',
                'rotate': [0, 90],
                'translate': [0.0, 0.0],
            },
        },
        {
            'joint_id': 0,
            'motion': {
                'type': 'rotate',
                'rotate': [0, -80],
                'translate': [0.0, 0.0],
            },
        },

    ]
    root_dir = './spaien_data/'
    for idx in range(len(motions_list)):
        if idx == 9 or idx == 4:
            continue
        steps = int(sys.argv[1])
        src_root = os.path.join(root_dir + 'single_part/original_data', category[idx], model_id[idx])
        dst_root = os.path.join(root_dir + f'single_part/obj_file_{steps}/',category[idx], model_id[idx], 'textured_objs')

        main(motions_list[idx], src_root, dst_root, steps, idx)

# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# AMASS: Archive of Motion Capture as Surface Shapes <https://arxiv.org/abs/1904.03278>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2019.08.09
import torch

from human_body_prior.tools.omni_tools import makepath
from human_body_prior.tools.omni_tools import copy2cpu as c2c

import json
import numpy as np
import glob
import os
import pickle

def compute_vertex_normal(vertices, indices):
    # code obtained from https://github.com/BachiLi/redner
    # redner/pyredner/shape.py
    def dot(v1, v2):
        # v1 := 13776 x 3
        # v1 := 13776 x 3
        # return := 13776

        return torch.sum(v1 * v2, dim=1)

    def squared_length(v):
        # v = 13776 x 3
        return torch.sum(v * v, dim=1)

    def length(v):
        # v = 13776 x 3
        # 13776
        return torch.sqrt(squared_length(v))

    # Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
    # vertices := 6890 x 3
    # indices := 13776 x 3
    normals = torch.zeros(vertices.shape, dtype=torch.float32, device=vertices.device)
    v = [vertices[indices[:, 0].long(), :],
         vertices[indices[:, 1].long(), :],
         vertices[indices[:, 2].long(), :]]

    for i in range(3):
        v0 = v[i]
        v1 = v[(i + 1) % 3]
        v2 = v[(i + 2) % 3]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = length(e1)
        e2_len = length(e2)
        side_a = e1 / torch.reshape(e1_len, [-1, 1])  # 13776, 3
        side_b = e2 / torch.reshape(e2_len, [-1, 1])  # 13776, 3
        if i == 0:
            n = torch.cross(side_a, side_b)  # 13776, 3
            n = n / torch.reshape(length(n), [-1, 1])
        angle = torch.where(dot(side_a, side_b) < 0,
                            np.pi - 2.0 * torch.asin(0.5 * length(side_a + side_b)),
                            2.0 * torch.asin(0.5 * length(side_b - side_a)))
        sin_angle = torch.sin(angle)  # 13776

        # XXX: Inefficient but it's PyTorch's limitation
        contrib = n * (sin_angle / (e1_len * e2_len)).reshape(-1, 1).expand(-1, 3)  # 13776, 3
        index = indices[:, i].long().reshape(-1, 1).expand([-1, 3])  # torch.Size([13776, 3])
        normals.scatter_add_(0, index, contrib)

    normals = normals / torch.reshape(length(normals), [-1, 1])
    return normals.contiguous()

def rotate_mesh(mesh_v, angle):

    angle = np.radians(angle)
    # rz = np.array([
    #     [np.cos(angle), -np.sin(angle), 0. ],
    #     [np.sin(angle), np.cos(angle), 0. ],
    #     [0., 0., 1. ]
    # ])
    rx = np.array([
        [1., 0., 0.],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle), ],
    ])
    return rx.dot(mesh_v.T).T

def registration2markers(registration_dir, out_marker_dir):
    np.random.seed(100)
    m2b_distance = 0.0095

    genders = {'50002': 'male', '50004': 'female', '50007': 'male', '50009': 'male', '50020': 'female',
               '50021': 'female', '50022': 'female', '50025': 'female', '50026': 'male', '50027': 'male'}

    with open('ssm_all_marker_placements.json') as f:
        all_marker_placements = json.load(f)
    all_mrks_keys = list(all_marker_placements.keys())

    for dfaust_subject in genders.keys():
        subject_reg_pkls = glob.glob(os.path.join(registration_dir, dfaust_subject, '*.pkl'))

        chosen_k = all_mrks_keys[np.random.choice(len(all_marker_placements))]
        chosen_marker_set = all_marker_placements[chosen_k]
        print('chose %s markerset for dfaust subject %s'%(chosen_k, dfaust_subject))
        for reg_pkl in subject_reg_pkls:
            with open(reg_pkl, 'rb') as f: data = pickle.load(f, encoding='latin-1')

            marker_data = np.zeros([len(data['v']), len(chosen_marker_set), 3])

            cur_m2b_distance = m2b_distance + abs(np.random.normal(0, m2b_distance / 3., size=[3]))  # Noise in 3D

            for fIdx in range(0, len(data['v'])):
                vertices = rotate_mesh(data['v'][fIdx].copy(), 90)
                vn = c2c(compute_vertex_normal(torch.Tensor(vertices), torch.Tensor(data['f'])))

                for mrk_id, vid in enumerate(chosen_marker_set.values()):
                    marker_data[fIdx, mrk_id] = vertices[vid] + cur_m2b_distance * vn[vid]

            outpath = makepath(os.path.join(out_marker_dir, dfaust_subject, os.path.basename(reg_pkl)), isfile=True)
            np.savez(outpath, **{
                'markers':marker_data,
                'labels': list(chosen_marker_set.keys()),
                'frame_rate':60,
                'gender': genders[dfaust_subject]
            })


if __name__ == '__main__':
    registration_dir = 'PATH_TO_DFAUT/REGISTRATION_PKLS' # download from http://dfaust.is.tue.mpg.de/downloads
    out_marker_dir = 'OUTPUT_FOR_SYNTHETIC_MOCAP/*.npz'
    registration2markers(registration_dir, out_marker_dir)


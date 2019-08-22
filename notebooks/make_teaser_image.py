# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
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
import numpy as np

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

bm_path = '../body_models/smplh/male/model.npz' # obtain from http://mano.is.tue.mpg.de/downloads

comp_device = torch.device('cuda')
bm = BodyModel(bm_path=bm_path, batch_size=1, num_betas=10).to(comp_device)

npz_data_path = '../github_data/amass_sample.npz'
bdata = np.load(npz_data_path)
print(list(bdata.keys()))

root_orient = torch.Tensor(bdata['poses'][:, :3]).to(comp_device)
pose_body = torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device)
pose_hand = torch.Tensor(bdata['poses'][:, 66:]).to(comp_device)
betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device)

faces = c2c(bm.f)

from human_body_prior.mesh import MeshViewer
from human_body_prior.mesh.sphere import points_to_spheres
import trimesh
from human_body_prior.tools.omni_tools import colors
from human_body_prior.tools.visualization_tools import imagearray2file
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from tqdm import tqdm

imw, imh=1600, 1800
step = 10
T = bdata['poses'].shape[0]//step

mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
images = np.zeros([2, 3, T, imh, imw, 3], dtype=np.float32)

count = 0
for fId in tqdm(range(1, bdata['poses'].shape[0], step)):
    if count > T: break
    body = bm(pose_body=pose_body[fId:fId+1], pose_hand=pose_hand[fId:fId+1], betas=betas, root_orient=root_orient[fId:fId+1])

    body_mesh = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    joints_mesh = points_to_spheres(c2c(body.Jtr[0]), vc=colors['red'])
    mrks = bdata['marker_data'][fId] - bdata['trans'][fId]
    mrks_mesh = points_to_spheres(mrks, vc=colors['blue'])

    all_meshes = [body_mesh] + joints_mesh + mrks_mesh
    apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))
    apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-90), (0, 1, 0)))
    mv.set_static_meshes(mrks_mesh)
    images[0, 0, count] = mv.render()
    mv.set_static_meshes([body_mesh])
    images[0, 1, count] = mv.render()
    mv.set_static_meshes([body_mesh]+joints_mesh)
    images[0, 2, count] = mv.render(render_wireframe=True)

    apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(90), (0, 1, 0)))
    mv.set_static_meshes(mrks_mesh)
    images[1, 0, count] = mv.render()
    mv.set_static_meshes([body_mesh])
    images[1, 1, count] = mv.render()
    mv.set_static_meshes([body_mesh]+joints_mesh)
    images[1, 2, count] = mv.render(render_wireframe=True)
    count += 1


imagearray2file(images, './teaser.gif')

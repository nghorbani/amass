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
import numpy as np

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from os import path as osp
support_dir = '../../../support_data/'

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amass_npz_fname = osp.join(support_dir, 'github_data/amass_sample.npz') # the path to body data
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = bdata['gender']

print('Data keys available:%s'%list(bdata.keys()))

print('The subject of the mocap sequence is  {}.'.format(subject_gender))


from human_body_prior.body_model.body_model import BodyModel

bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))

num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
faces = c2c(bm.f)

time_length = len(bdata['trans'])

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
    'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
}

print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
print('time_length = {}'.format(time_length))


from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.tools.vis_tools import imagearray2file
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from tqdm import tqdm

imw, imh=1600, 1800
step = 10
T = time_length//step

mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
images = np.zeros([2, 3, T, imh, imw, 3], dtype=np.float32)

body= bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'pose_hand', 'betas', 'root_orient']})

count = 0
for fId in tqdm(range(1, time_length, step)):
    if count > T: break
    # body = bm(pose_body=pose_body[fId:fId+1], pose_hand=pose_hand[fId:fId+1], betas=betas, root_orient=root_orient[fId:fId+1])

    body_mesh = trimesh.Trimesh(vertices=c2c(body.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    joints_mesh = points_to_spheres(c2c(body.Jtr[fId]), vc=colors['red'])
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

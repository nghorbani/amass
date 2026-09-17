# -*- coding: utf-8 -*-
"""
The notebooks construct BodyModel with the keyword arguments of the pinned human_body_prior
(bm_fname, num_betas, num_dmpls, dmpl_fname) and run it on AMASS parameter vectors. This test
builds a tiny synthetic body model with the arrays the loader reads, instantiates BodyModel exactly
as the notebooks do, and runs one forward pass, so the calls are checked without the licence-gated
SMPL+H and SMPL-X model files.
"""
import numpy as np
import pytest
import torch

from human_body_prior.body_model.body_model import BodyModel

NUM_BETAS = 16
NUM_DMPLS = 8
TIME_LENGTH = 3


def write_synthetic_model(path, num_joints, num_verts=12, num_faces=4, num_betas=NUM_BETAS, seed=0):
    """Write a model.npz with the keys BodyModel reads; shapes follow the real SMPL family files."""
    rng = np.random.default_rng(seed)
    parents = np.arange(num_joints) - 1  # a kinematic chain, root parent is -1 like the real files
    kintree_table = np.stack([parents, np.arange(num_joints)]).astype(np.int64)
    weights = rng.random((num_verts, num_joints))
    weights /= weights.sum(axis=1, keepdims=True)
    np.savez(
        path,
        v_template=rng.standard_normal((num_verts, 3)),
        f=rng.integers(0, num_verts, size=(num_faces, 3)),
        shapedirs=rng.standard_normal((num_verts, 3, num_betas)) * 1e-2,
        posedirs=rng.standard_normal((num_verts, 3, 9 * (num_joints - 1))) * 1e-2,
        J_regressor=np.eye(num_joints, num_verts),
        kintree_table=kintree_table,
        weights=weights,
    )
    return str(path)


def write_synthetic_dmpls(path, num_verts=12, num_dmpls=NUM_DMPLS, seed=1):
    rng = np.random.default_rng(seed)
    np.savez(path, eigvec=rng.standard_normal((num_verts, 3, num_dmpls)) * 1e-2)
    return str(path)


@pytest.fixture
def smplh_files(tmp_path):
    # SMPL+H: 52 joints; posedirs holds 9 rotation-matrix entries per non-root joint, 9 * 51 = 459,
    # and the loader keys the model type on that width divided by three (153)
    bm_fname = write_synthetic_model(tmp_path / 'smplh_model.npz', num_joints=52)
    dmpl_fname = write_synthetic_dmpls(tmp_path / 'dmpl_model.npz')
    return bm_fname, dmpl_fname


def test_notebook_01_and_04_smplh_with_dmpls(smplh_files):
    bm_fname, dmpl_fname = smplh_files
    comp_device = torch.device('cpu')

    # notebooks/01-AMASS_Visualization.ipynb and 04-AMASS_DMPL.ipynb, verbatim call
    bm = BodyModel(bm_fname=bm_fname, num_betas=NUM_BETAS, num_dmpls=NUM_DMPLS, dmpl_fname=dmpl_fname).to(comp_device)
    faces = bm.f.detach().cpu().numpy()
    assert faces.shape[1] == 3

    poses = torch.zeros(TIME_LENGTH, 156)  # root (3) + body (63) + hands (90), as in an AMASS npz
    body_parms = {
        'root_orient': poses[:, :3],
        'pose_body': poses[:, 3:66],
        'pose_hand': poses[:, 66:],
        'trans': torch.zeros(TIME_LENGTH, 3),
        'betas': torch.zeros(TIME_LENGTH, NUM_BETAS),
        'dmpls': torch.zeros(TIME_LENGTH, NUM_DMPLS),
    }
    body = bm(**body_parms)
    assert body.v.shape == (TIME_LENGTH, 12, 3)
    assert body.Jtr.shape == (TIME_LENGTH, 52, 3)

    # the pose-and-shape-only call of the first visualisation cell
    body_pose_beta = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas']})
    assert body_pose_beta.v.shape == (TIME_LENGTH, 12, 3)


def test_notebook_02_smplh_without_dmpls(smplh_files):
    bm_fname, _ = smplh_files
    # notebooks/02-AMASS_DNN.ipynb
    bm = BodyModel(bm_fname=bm_fname, num_betas=NUM_BETAS)
    body = bm(pose_body=torch.zeros(TIME_LENGTH, 63), betas=torch.zeros(TIME_LENGTH, NUM_BETAS))
    assert body.v.shape == (TIME_LENGTH, 12, 3)


def test_notebook_03_smplx(tmp_path):
    # SMPL-X: 55 joints, 9 * 54 = 486 posedirs entries, model-type key 162
    bm_smplx_fname = write_synthetic_model(tmp_path / 'smplx_model.npz', num_joints=55)
    # notebooks/03-AMASS_Visualization_Advanced.ipynb
    bm = BodyModel(bm_fname=bm_smplx_fname, num_betas=NUM_BETAS)
    body = bm(pose_body=torch.zeros(TIME_LENGTH, 63), betas=torch.zeros(TIME_LENGTH, NUM_BETAS))
    assert body.v.shape == (TIME_LENGTH, 12, 3)
    assert body.Jtr.shape == (TIME_LENGTH, 55, 3)

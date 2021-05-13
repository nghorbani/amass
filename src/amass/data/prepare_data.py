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

import os
import numpy as np
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.tools.rotation_tools import euler2em, em2euler, aa2matrot
from human_body_prior.tools.omni_tools import copy2cpu as c2c

import shutil, sys
from torch.utils.data import Dataset
import glob
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import tables as pytables
from tqdm import tqdm

gdr2num = {'male':-1, 'neutral':0, 'female':1}
gdr2num_rev = {v:k for k,v in gdr2num.items()}

def remove_Zrot(pose):
    noZ = em2euler(pose[:3].copy())
    noZ[2] = 0
    pose[:3] = euler2em(noZ).copy()
    return pose

def dump_amass2pytroch(datasets, amass_dir, out_posepath, logger = None, rnd_seed = 100, keep_rate = 0.01):
    '''
    Select random number of frames from central 80 percent of each mocap sequence
    Save individual data features like pose and shape per frame in pytorch pt files
    test set will have the extra field for original markers

    :param datasets: the name of the dataset
    :param amass_dir: directory of downloaded amass npz files. should be in this structure: path/datasets/subjects/*_poses.npz
    :param out_posepath: the path for final pose.pt file
    :param logger: an instance of human_body_prior.tools.omni_tools.log2file
    :param rnd_seed:
    :return: Number of datapoints dumped using out_poseth address pattern
    '''
    import glob

    np.random.seed(rnd_seed)

    makepath(out_posepath, isfile=True)

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(out_posepath.replace('pose.pt', '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % out_posepath)

    data_pose = []
    data_dmpl = []
    data_betas = []
    data_gender = []
    data_trans = []

    for ds_name in datasets:
        npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/*_poses.npz'))
        logger('randomly selecting data points from %s.' % (ds_name))
        for npz_fname in tqdm(npz_fnames):
            try:
                cdata = np.load(npz_fname)
            except:
                logger('Could not read %s! skipping..'%npz_fname)
                continue
            N = len(cdata['poses'])

            cdata_ids = np.random.choice(list(range(int(0.1*N), int(0.9*N),1)), int(keep_rate*0.8*N), replace=False)#removing first and last 10% of the data to avoid repetitive initial poses
            if len(cdata_ids)<1: continue

            data_pose.extend(cdata['poses'][cdata_ids].astype(np.float32))
            data_dmpl.extend(cdata['dmpls'][cdata_ids].astype(np.float32))
            data_trans.extend(cdata['trans'][cdata_ids].astype(np.float32))
            data_betas.extend(np.repeat(cdata['betas'][np.newaxis].astype(np.float32), repeats=len(cdata_ids), axis=0))
            data_gender.extend([gdr2num[str(cdata['gender'].astype(np.str))] for _ in cdata_ids])

    assert len(data_pose) != 0

    torch.save(torch.tensor(np.asarray(data_pose, np.float32)), out_posepath)
    torch.save(torch.tensor(np.asarray(data_dmpl, np.float32)), out_posepath.replace('pose.pt', 'dmpl.pt'))
    torch.save(torch.tensor(np.asarray(data_betas, np.float32)), out_posepath.replace('pose.pt', 'betas.pt'))
    torch.save(torch.tensor(np.asarray(data_trans, np.float32)), out_posepath.replace('pose.pt', 'trans.pt'))
    torch.save(torch.tensor(np.asarray(data_gender, np.int32)), out_posepath.replace('pose.pt', 'gender.pt'))

    return len(data_pose)

class AMASS_Augment(Dataset):
    """Use this dataloader to do any augmentation task in parallel"""

    def __init__(self, dataset_dir, dtype=torch.float32):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)

        self.dtype = dtype

    def __len__(self):
       return len(self.ds['trans'])

    def __getitem__(self, idx):
        return self.fetch_data(idx)

    def fetch_data(self, idx):
        '''
        This an exampl of augmenting the data fields. Furthermore, one can add random noise to data fields here as well.
        There should be a match between returning dictionary field names and the one in AMASS_ROW.
        :param idx:
        :return:
        '''
        sample = {k: self.ds[k][idx] for k in self.ds.keys()}

        sample['pose_matrot'] = aa2matrot(sample['pose'].view([-1,3])).view(1,-1)

        return sample

def prepare_amass(amass_splits, amass_dir, work_dir, logger=None):

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(os.path.join(work_dir, '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % work_dir)

    stageI_outdir = os.path.join(work_dir, 'stage_I')

    shutil.copy2(sys.argv[0], os.path.join(work_dir, os.path.basename(sys.argv[0])))

    logger('Stage I: Fetch data from AMASS npz files')

    for split_name, datasets in amass_splits.items():
        outpath = makepath(os.path.join(stageI_outdir, split_name, 'pose.pt'), isfile=True)
        if os.path.exists(outpath): continue
        dump_amass2pytroch(datasets, amass_dir, outpath, logger=logger)

    logger('Stage II: augment the data and save into h5 files to be used in a cross framework scenario.')


    class AMASS_ROW(pytables.IsDescription):
        gender = pytables.Int16Col(1)  # 1-character String
        pose = pytables.Float32Col(52*3)  # float  (single-precision)
        dmpl = pytables.Float32Col(8)  # float  (single-precision)
        pose_matrot = pytables.Float32Col(52*9)  # float  (single-precision)
        betas = pytables.Float32Col(16)  # float  (single-precision)
        trans = pytables.Float32Col(3)  # float  (single-precision)

    stageII_outdir = makepath(os.path.join(work_dir, 'stage_II'))

    batch_size = 256
    max_num_epochs = 1  # how much augmentation we would get

    for split_name in amass_splits.keys():
        h5_outpath = os.path.join(stageII_outdir, '%s.h5' % split_name)
        if os.path.exists(h5_outpath): continue

        ds = AMASS_Augment(dataset_dir=os.path.join(stageI_outdir, split_name))
        logger('%s has %d data points!' % (split_name, len(ds)))
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=32, drop_last=False)
        with pytables.open_file(h5_outpath, mode="w") as h5file:
            table = h5file.create_table('/', 'data', AMASS_ROW)

            for epoch_num in range(max_num_epochs):
                for bId, bData in tqdm(enumerate(dataloader)):
                    for i in range(len(bData['trans'])):
                        for k in bData.keys():
                            table.row[k] = c2c(bData[k][i])
                        table.row.append()
                    table.flush()

    logger('\nStage III: dump every data field for all the splits as final pytorch pt files')
    # we would like to use pt files because their interface could run in multiple threads
    stageIII_outdir = makepath(os.path.join(work_dir, 'stage_III'))

    for split_name in amass_splits.keys():
        h5_filepath = os.path.join(stageII_outdir, '%s.h5' % split_name)
        if not os.path.exists(h5_filepath) : continue

        with pytables.open_file(h5_filepath, mode="r") as h5file:
            data = h5file.get_node('/data')
            data_dict = {k:[] for k in data.colnames}
            for id in range(len(data)):
                cdata = data[id]
                for k in data_dict.keys():
                    data_dict[k].append(cdata[k])

        for k,v in data_dict.items():
            outfname = makepath(os.path.join(stageIII_outdir, split_name, '%s.pt' % k), isfile=True)
            if os.path.exists(outfname): continue
            torch.save(torch.from_numpy(np.asarray(v)), outfname)

    logger('Dumped final pytorch dataset at %s' % stageIII_outdir)

if __name__ == '__main__':
    # ['CMU', 'Transitions_mocap', 'MPI_Limits', 'SSM_synced', 'TotalCapture', 'Eyes_Japan_Dataset', 'MPI_mosh', 'MPI_HDM05', 'HumanEva', 'ACCAD', 'EKUT', 'SFU', 'KIT', 'H36M', 'TCD_handMocap', 'BML']

    msg = ''' Using standard AMASS dataset preparation pipeline: 
    0) Donwload all npz files from https://amass.is.tue.mpg.de/ 
    1) Convert npz files to pytorch readable pt files. 
    2) Either use these files directly or augment them in parallel and write into h5 files
    3)[optional] If you have augmented your data, dump augmented results into final pt files and use with your dataloader'''

    expr_code = 'VXX_SVXX_TXX' #VERSION_SUBVERSION_TRY

    amass_dir = 'PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz'

    work_dir = makepath('WHERE_YOU_WANT_YOUR_FILE_TO_BE_DUMPED/%s' % (expr_code))

    logger = log2file(os.path.join(work_dir, '%s.log' % (expr_code)))
    logger('[%s] AMASS Data Preparation Began.'%expr_code)
    logger(msg)

    amass_splits = {
        'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap']#ACCAD
    }
    amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))

    prepare_amass(amass_splits, amass_dir, work_dir, logger=logger)
# AMASS Data Processing Tools

Here we provide basic tools to turn compressed numpy arrays that are in *.npz* format holding 
 [SMPLH](http://mano.is.tue.mpg.de/) body parameters obtained from [AMASS](https://amass.is.tue.mpg.de/dataset),
into other suitable formats for deep learning frameworks. The final outcomes are PyTorch readable *.pt* files, as well as *.h5* files
that are commonly usable in machine learning fields. 
The provided data preparation code has three stages that could be flexibly modified for your own specific needs.

**Stage I** goes over the previously downloaded  numpy npz files, subsamples them and saves them all into one place as PyTorch pt files.

**Stage II** makes use of PyTorch to apply all sorts of data augmentations 
in parallel on the original data and saves them as final HDF5 files.
HDF5 makes it possible to write on the file in chunks so that you wouldn't run out of memory
if you do huge number of data augmentation steps. PyTorch also handles data augmentation in parallel so 
you should achieve your augmentation goal as fast as possible. 

**Stage III** simply turns the h5 files into an alternative pt files to be readaly usable by PyTorch.

The progress at all stages is logged and could be inspected at any time during the process or later.
We also use an experiment ID that helps refering to a specific data preparation run that can be traced back.
We follow the recommended data splits by AMASS for train/validation/test. That is for current time:
```python
amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap', 'ACCAD']
}
```
Please note that AMASS dataset is planned to grow, therefore, the training split is expected to increase in size. 
Furthermore, we only include the original mocap marker data only in the test split. 

# Visualizing the Body Parameters in AMASS
A single data file in amass has the parameters to control gender, pose, shape, global translation and soft tissue dynamics 
in correspondence with the original motion capture sequence.
```python
import torch
from human_body_prior.body_model.body_model import BodyModel
import numpy as np

bm_path = 'PATH_TO_SMPLH_model.pkl' # obtain from http://mano.is.tue.mpg.de/downloads

comp_device = torch.device('cuda')
bm = BodyModel(bm_path=bm_path, batch_size=1, model_type='smplH', num_betas=10).to(comp_device)

npz_data = 'github_data/amass_sample.npz'
fId = 0 # frame id of the mocap sequence
root_orient = torch.Tensor(npz_data['pose'][fId:fId+1, :3], dtype=torch.float32).to(comp_device)
body_pose = torch.Tensor(npz_data['pose'][fId:fId+1, 3:66], dtype=torch.float32).to(comp_device)
hands_pose = torch.Tensor(npz_data['pose'][fId:fId+1, 3:66], dtype=torch.float32).to(comp_device)
betas = torch.Tensor(npz_data['betas'][:10][np.newaxis], dtype=torch.float32).to(comp_device)

body = bm(root_orient=root_orient, body_pose=body_pose, betas=betas)

from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c

mv = MeshViewer(width=800, height=800, use_offscreen=True)
mv.render_wireframe = True #
```



# AMASS for Training Neural Networks on Human Body Data 
Similar to [ImageNet](http://www.image-net.org/), AMASS is a large dataset suitable for deep learning purposes. 
The only difference is that instead of images it holds human body parameters,
controlling the surface mesh of the [SMPLH body model](http://mano.is.tue.mpg.de/).

Here we provide basic tools for loading them
for training a deep neural network that has to do with human body and motion. 

For this tutorial you need PyTorch>=1.10.0.


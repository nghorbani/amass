# AMASS: Archive of Motion Capture as Surface Shapes

![alt text](support_data/github_data/datasets_preview.png "Samples of bodies in AMASS recovered from Motion Capture sequences")

[AMASS](http://amass.is.tue.mpg.de) is a large database of human motion unifying different optical marker-based motion capture datasets by representing them within a common framework and parameterization. 
 AMASS is readily useful for animation, visualization, and generating training data for deep learning.

Here we provide tools and tutorials to use AMASS in your research projects. More specifically:
- Following the recommended splits of data by AMASS, we provide three non-overlapping train/validation/test splits.
- AMASS uses an extended version of [SMPL+H](http://mano.is.tue.mpg.de/) with [DMPLs](https://smpl.is.tue.mpg.de/). 
Here we show how to load different components and visualize a body model with AMASS data.
- AMASS is also compatible with [SMPL](http://smpl.is.tue.mpg.de) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) body models. 
We show how to use the body data from AMASS to animate these models.
## Table of Contents
  * [Installation](#installation)
  * [Body Models](#body-models)
  * [Tutorials](#tutorials)
  * [Citation](#citation)
  * [License](#license)
  * [Contact](#contact)

## Installation
**Requirements**
- Python 3.7
- [PyTorch 1.7.1](https://pytorch.org/get-started)
- [Human Body Prior](https://github.com/nghorbani/human_body_prior)
- [Pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html#osmesa) for visualizations

Clone this repo and run the following from the root folder:
```bash
python install -r requirements.txt
python setup.py develop
```

## Body Models
AMASS uses [MoSh++](https://amass.is.tue.mpg.de) pipeline to fit [SMPL+H body model](https://mano.is.tue.mpg.de/)
to human optical marker based motion capture (mocap) data.
In the paper we use SMPL+H with extended shape space, i.e. 16 betas, and 8 [DMPLs](https://smpl.is.tue.mpg.de/). 
Please download models and place them them in body_models folder of this repository after you obtained the code from GitHub.

## Tutorials
We release tools and Jupyter notebooks to demonstrate how to use AMASS to animate SMPL+H body model.

Furthermore, as promised in the supplementary material of the paper, we release code to produce synthetic mocap using 
[DFaust](http://dfaust.is.tue.mpg.de) registrations.

Please refer to [tutorials](/notebooks) for further details.

## Citation
Please cite the following paper if you use this code directly or indirectly in your research/projects:
```
@inproceedings{AMASS:2019,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and F. Troje, Nikolaus and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year={2019},
  month = {Oct},
  url = {https://amass.is.tue.mpg.de},
  month_numeric = {10}
}
```
## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](./LICENSE) 
and any accompanying documentation before you download and/or use the AMASS dataset, and software, (the "Model & Software"). 
 By downloading and/or using the Model & Software 
 (including downloading, cloning, installing, and any other use of this GitHub repository), 
 you acknowledge that you have read these terms and conditions, understand them, 
 and agree to be bound by them. If you do not agree with these terms and conditions, 
 you must not download and/or use the Model & Software.
  Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).
 
 ## Contact
The code in this repository is developed by [Nima Ghorbani](https://nghorbani.github.io/).

If you have any questions you can contact us at [amass@tuebingen.mpg.de](mailto:amass@tuebingen.mpg.de).

For commercial licensing, please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de)

To find out about the latest developments stay tuned to [AMASS twitter](https://twitter.com/mocap_amass).

## Contribute to AMASS
The research community needs more human motion data. 
If you have interesting marker based motion capture data, and you are willing to share it for research purposes, 
then we will label and clean your mocap and MoSh it for you and add it to the AMASS dataset, 
naturally citing you as the original owner of the marker data.
For this purposes feel free to contact [amass@tuebingen.mpg.de](mailto:amass@tuebingen.mpg.de).

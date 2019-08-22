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


from setuptools import setup, find_packages

setup(name='amass',
      version='0.1.0',
      packages=find_packages(),
      include_package_data=True,
      author='Nima Ghorbani',
      author_email='nima.gbani@gmail.com',
      maintainer='Nima Ghorbani',
      maintainer_email='nima.gbani@gmail.com',
      url='https://github.com/nghorbani/amass',
      description='AMASS: Archive of Motion Capture as Surface Shapes',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      install_requires=['torch==1.1.0', 'human_body_prior',  'trimesh', 'pyrender', 'numpy'],
      dependency_links=[
          "https://github.com/nghorbani/configer/tarball/master#egg=configer"
          "https://github.com/nghorbani/human_body_prior/tarball/master#egg=human_body_prior"
      ],
      classifiers=[
          "Intended Audience :: Developers",
          "Intended Audience :: Researchers",
          "Natural Language :: English",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: POSIX",
          "Operating System :: POSIX :: BSD",
          "Operating System :: POSIX :: Linux",
          "Operating System :: Microsoft :: Windows",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7", ],
      )

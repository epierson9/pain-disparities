#!/bin/bash

conda create -y -n knee_localizer python=3.6
conda install -y -n knee_localizer numpy opencv scipy
source activate knee_localizer

pip install pip -U
pip install pydicom
pip install tqdm

pip install -e .

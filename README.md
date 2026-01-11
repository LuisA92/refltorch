# Installation

## macOS

If using macOS, install DIALS and PyTorch using micromamba or conda. 

```bash

# create new micromamba environment and install DIALS and PyTorch
micromamba create -n refltorch \
  -c conda-forge \
  python=3.11 \
  dials \
  pytorch
micromamba activate refltorch

# clone repo and go into project directory
git clone https://github.com/LuisA92/refltorch.git
cd refltorch

# install project
pip install uv
uv pip install -e .

```

## Linux

```bash

# create new micromamba environment
micromamba create -n refltorch python=3.11
micromamba activate refltorch

# install dials
micromamba install -c conda-forge dials

# clone repo and go into project directory
git clone https://github.com/LuisA92/refltorch.git
cd refltorch

# install project
pip install uv
uv pip install -e .

# install PyTorch

```

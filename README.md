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

# CLone the repository
git clone https://github.com/LuisA92/refltorch.git
cd refltorch

# Install uv and install project
pip install uv
uv pip install -e .

```

## Linux

```bash

# create new micromamba environment
micromamba create -n refltorch python=3.11
micromamba activate refltorch

# Install DIALS
micromamba install -c conda-forge dials

# Clone the repository
git clone https://github.com/LuisA92/refltorch.git
cd refltorch

# Install uv and install project
pip install uv
uv pip install -e .

```



#

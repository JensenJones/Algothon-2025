# SIG Algothon – Python environment guide

## ➊ One-time bootstrap

```bash
git clone git@github.com:JensenJones/Algothon-2025.git
cd Algothon-2025
conda env create -n sigalgo -f environment.yml
cp .condarc "$(conda info --envs | awk '/sigalgo/ {print \$2}')/"
conda activate sigalgo
## ➊ One-time bootstrap (new machine / fresh clone)

```bash
# clone the repo
git clone git@github.com:JensenJones/Algothon-2025.git
cd Algothon-2025

# create Conda env from the locked manifest
conda env create -n sigalgo -f environment.yml

# copy project .condarc into the new env’s root so channels are identical
cp .condarc "$(conda info --envs | awk '/sigalgo/ {print $2}')/"

# activate it
conda activate sigalgo

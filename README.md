## `AnanseScanpy` package: implementation of scANANSE for Scanpy objects in Python
[![Anaconda-Server Badge](https://anaconda.org/bioconda/anansescanpy/badges/version.svg)](https://anaconda.org/bioconda/anansescanpy)
[![PyPI version](https://badge.fury.io/py/anansescanpy.svg)](https://badge.fury.io/py/anansescanpy)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/anansescanpy/README.html)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/anansescanpy/badges/downloads.svg)](https://anaconda.org/bioconda/anansescanpy)
[![Maintainability](https://api.codeclimate.com/v1/badges/04272eaade7b247b4af2/maintainability)](https://codeclimate.com/github/Arts-of-coding/AnanseScanpy/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/04272eaade7b247b4af2/test_coverage)](https://codeclimate.com/github/Arts-of-coding/AnanseScanpy/test_coverage)


## Installation

The most straightforward way to install the most recent version of AnanseScanpy is via conda using PyPI.

### Install package through Conda
If you have not used Bioconda before, first set up the necessary channels (in this order!). 
You only have to do this once.
```
$ conda config --add channels defaults
$ conda config --add channels bioconda
$ conda config --add channels conda-forge
```

Then install AnanseScanpy with:
```
$ conda install anansescanpy
```

### Install package through PyPI
```
$ pip install anansescanpy
```

### Install package through GitHub
```
$ git clone https://github.com/Arts-of-coding/AnanseScanpy.git
$ cd AnanseScanpy
$ conda env create -f requirements.yaml
$ conda activate AnanseScanpy
$ pip install -e .
```

### Install Jupyter Notebook
```
$ pip install jupyter
```

## Start using the package

### Run the package either in the console
```
$ python3
```

### Or run the package in jupyter notebook
```
$ jupyter notebook
```

## For extended documentation see our ipynb vignette with PBMC sample data
### Of which the sample data can be downloaded
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7575107.svg)](https://zenodo.org/records/7575107)
```
$ wget https://zenodo.org/records/7575107/files/rna_PBMC.h5ad?download=1 -O scANANSE/rna_PBMC.h5ad
$ wget https://zenodo.org/records/7575107/files/atac_PBMC.h5ad?download=1 -O scANANSE/atac_PBMC.h5ad
```

### installing and running anansnake 

Follow the instructions its respective github page, https://github.com/vanheeringen-lab/anansnake
Next automatically use the generated files to run GRN analysis using your single cell cluster data:


```{bash eval=FALSE}
snakemake --use-conda --conda-frontend mamba \
--configfile scANANSE/analysis/config.yaml \
--snakefile scANANSE/anansnake/Snakefile \
--resources mem_mb=48_000 --cores 12
```

### Thanks to:

* Jos Smits and his Seurat equivalent of this package https://github.com/JGASmits/AnanseSeurat
* Siebren Frohlich and his anansnake implementation https://github.com/vanheeringen-lab/anansnake

### How to cite this software:
Smits JGA, Arts JA, Frölich S et al. scANANSE gene regulatory network and motif analysis of single-cell clusters [version 1; peer review: awaiting peer review]. F1000Research 2023, 12:243 (https://doi.org/10.12688/f1000research.130530.1)

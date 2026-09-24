# Variational-Inference-based Background Elimination in Spectroscopy (VIBES)

This repository contains code for removing interferences from substrates, matrices, and instrumental artifacts in mid-infrared (IR)
spectra using a probabilistic approach. In this approach background contributions are modeled using a PCA analysis of interference 
examples while analyte signals are represented by a Gibbs distribution associated with a specified loss function. The parameters 
of the model is automatically calibrated to a spectrum using approximate maximum likelihood estimation enabled by an
variational approximation. Consequentially, we refer to this approach as Variational-Inference-based Background Elimination in Spectroscopy (VIBES).
The repository also contains code for alternative correction procedures.

## Installation

Install pinned development dependencies using:

```
pip install -r requirements.txt
```

If you are using Conda to manage your Python environments:

```
conda env create -f environment.yml
```

## Repository structure
    * /data - the python scripts target data inside this folder. 
    * /notebooks - folder containing notebooks. Currently only contains illustration.ipynb.
    * /scripts - Python scripts for generating the results of the paper.
    * /src - Python module.

## Getting started

### Quickstart 

After installation we recommend working through the notebook illustration.ipynb. This notebook illustrates the core functionalities of 
the repository with synthetic data examples.

### Using the python scripts

To correct a batch of spectra one needs to create a subfolder inside /data containing two files namely spectra.parquet and blanks.parquet. 
The former contains the spectra which are to be corrected as rows while the latter contains interference examples. For the former, the 
index should contain identifyers for the rows (spectra). The Python scripts can than be called targeting this folder. The ouput of the scripts 
are dictionaries containing information about the corrections performed and is stored inside the same folder containing the data. 

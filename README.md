# UQ_AE-SINDy

This repository contains the scripts, data, results, and plots used to reproduce the findings of the paper "Uncertainty Quantification for Reduced-Order Surrogate Models Applied to Cloud Microphysics," submitted to the Machine Learning and the Physical Sciences workshop at the 39th Conference on Neural Information Processing Systems (NeurIPS).

These files are part of a larger repository that will be released publicly as part of an extended study. Note that this larger study includes two additional latent dynamical models besides SINDy; these components have been disabled here and are not functional in this smaller repository.

## Commands for Figures and Tables from Paper

The following can be run from the command line to generate Figures 2 and 3 in the paper. The Python scripts have been written such that, after adjusting for the Python file location, they can be run from anywhere within the project.

* Figure 2: `python3 plotting_paper.py congestus_coal_200m_9600 -a SINDy -t '0 60' -m cv+20 -s full -g 7 289 404 -title n`
  * -> output saved as `results/UQ/conformal/ae_SINDy/congestus_coal_200m_9600_cv+20_full_7_289_404.pdf`
* Figure 3: `python3 errors_all.py congestus_coal_200m_9600 -a SINDy -s nomass`
  * -> output saved as `results/UQ/conformal/ae_SINDy/errors_congestus_coal_200m_9600_all_nomass.pdf`

As explained below (under Directories -> UQ), the results in Tables 1 and 2 are located in `ae_SINDy.out`. These results are outputs from the Slurm script `UQ/conformal/ae_SINDy_test.sh`.

## Requirements

The following Python packages are required to run the scripts in this repository:

```
matplotlib==3.10.3
numpy==2.2.5
plotly==6.0.1
PySDM==2.118
PySDM_examples==2.118
pysindy==1.7.5
scipy==1.15.3
torch==2.4.1
xarray==2023.6.0
dask==2024.1.1
netCDF4==1.6.2
```

To install dependencies at once, create a `requirements.txt` file with the above contents and run: 

```
pip install -r requirements.txt
```

## Directories 

Here, we describe the contents of each directory and the files therein.

### UQ

This subdirectory contains `errors_all.py`, `plotting_paper.py`, and the folder `conformal.` Here, we list information on the Python scripts:
* `errors_all.py` is a script for plotting predictive errors (the areas/averages between conformal prediction bands) across all subsets of the network, with or without the total mass (which turns out to be exactly conserved in this case). 
* `plotting_paper.py` allows one to plot the conformal prediction bands, model predictions, and exact/true results for different model outputs at specified timesteps and sample numbers. As described in the paper, one has four choices of outputs corresponding to the decoding only (`decoder`), latent space predictions only (`latent`), entire end-to-end outputs (decoder + latent, `full`), and mass evolution only (`mass`).
* `conformal` contains the following:
  * `ae_SINDy.py` is a script that runs vanilla or split conformal predictions on the dataset specified as a file path relative to the project root. The script allows one to specify miscoverage rates, train-test split, the conformal predictions method used, and training hyperparameters.
  * `ae_SINDy_cv.py` is the same as `ae_SINDy.py` except for specifically running CV+ conformal predictions *in parallel.*
  * `cp_test.py` calculates empirical coverage levels for different model outputs.
  * `extract_test_idx.py` returns the indices of samples (relative to the full dataset) that lie in the testing data. These are especially useful for parsing through example DSD predictions to plot.

The Slurm script `UQ/conformal/ae_SINDy.sh` generates the conformal prediction intervals used in the paper. These are returned as the .pkl files stored in `UQ/conformal/results/ae_SINDy`. In turn, the Slurm script `UQ/conformal/ae_SINDy_test.sh` tests the empirical coverage imparted by these bands across different subsets of the model architecture. Results for the latter are contained in `ae_SINDy.out` and comprise those in Tables 1 and 2 of the paper.

`conformal/test_idx.txt` contains all test indices used for the split indicated in the paper.

### data

Contains the .nc file for the dataset used in the paper: `congestus_coal_200m_9600.nc`.

### results

`Optuna/ERF Dataset` contains results from the hyperparameter study, as refined using [Optuna](https://optuna.org/) and described in Appendix C. 

`UQ/conformal/ae_SINDy` contains the precise figures in the study. All figures outputted from  `errors_all.py` and `plotting_paper.py` are saved here; hence, the ones included here are Figures 2 and 3 in the paper.

### src

This directory contains utilities for training the model architecture (`training.py`); plotting results (`plotting.py`; not used in this paper); setting thresholds for training (`thresholding.py`); evaluating performance metrics (`diagnostics.py`); and utilities for opening, processing, and splitting the dataset (`data_utils.py`). 

### training_scripts

This directory contains only one file: `train_ae_sindy.py`. This script is used to train and evaluate the full AE-SINDy model architecture.

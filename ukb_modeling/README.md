# Python Modeling Pipeline

# RFC Model Template - Quick Start Guide

This repository contains a Random Forest Classifier (RFC) model template for use in the Schneiderlab.


## Getting Started 1.0
1. Clone this repository to your local Github folder
2. Create a virtual environment (e.g. with Anaconda Navigator if you want a graphical user interface (GUI), otherwise your method of choice)
3. Activate the environment and open it in a terminal (careful, you MUST BE in the environment that you will use for working with the repo, otherwise the next steps do not work)
4. Maneuver to the proper folder, therefore type in terminal: "cd "C:\Users\YOURUSERNAME\YOURPATH\GitHub\modeling_pipeline\src"
5. Now you should be in the src folder of the repository, with the following files present (ls for Linux)
6. Install requirements.txt with "pip install -r requirements.txt" or "conda install -r requirements.txt"
   (works only if 4. was successful)
7a. Run "pip install -e ." in the same folder in your command line. This makes the modeling_pipeline into a python package and configures some necessities.
7b Alternatively, you can do this in a ipynb but make sure to have the proper path: %pip install -e "C:\***Path-to-your-GitHub\modeling_pipeline\src"

## Getting Started 2.0
8. Copy the folder 'Project_Template' and start your own project.
  8a. Fill in necessary variables in your own copy of the user_input.yaml (from time to time you might want to look at new functionalities in the main user_input.yaml)
  8b. Adapt your copy! of the Single Model Template.ipynb as you wish
  8c. The notebook works best with the R-Preprocessing pipeline from "ukb_scripts"
9. If not using the R-Preprocessing pipeline, you'll need specific dataframes as structured e.g. at the top of "Single Model Template"

## Key Components and overview
The `Single_Model_Template.ipynb` notebook serves as a template that new users can copy and adapt for their own projects. It provides a comprehensive framework for training, evaluating, calibrating and visualizing machine learning models, particularly Random Forest Classifiers.
The notebook links to several important functions and classes:

- **Pipeline**: Core class that handles data loading, preprocessing, and model training
- **trained_model**: Manages model training with cross-validation
- **eval**: Provides comprehensive evaluation metrics and visualization tools
- **plot**: Contains various visualization functions for model analysis
- **ablation**: For performing ablation studies to understand feature importance

## Overview graphic:

![image](https://github.com/user-attachments/assets/2ae019bd-6729-4553-8f3a-e3cadc678de6)

## Workflow

1. Initialize user input parameters
2. Create a pipeline object
3. Train models (RFC, XGB, etc.)
4. Build master RFC model
5. Evaluate performance
6. Generate visualizations (ROC curves, violin plots, confusion matrices)

## Requirements

See the `requirements.txt` file for dependencies. Key downstream packages include:
- sklearn
- pandas
- numpy
- matplotlib
- seaborn
- joblib


## Data Structure

Expected input formats include:
- Training/validation data split into X (features) and y (labels)
- Column mappings for one-hot encoding
- Optional metadata for extended analysis

For help or questions, refer to existing lab documentation or contact senior members of the Schneiderlab.




## Inference / External validation

There are several steps to ease external validation.
1. We provide synthetic data for 10 representative patients together with prediction values we derived from the TOP15 model (All and PAR, respectively).
2. We provide a function, "validate_model_setup" that loads the synthetic data and uses the prediction model in your setup as a positive control that you are using the proper model and python setup

3. We provide a py file that loads your data (given it is in the appropriate format/same as our synthetic data), runs the pipeline and gives out several outputs
- A full prediction_values.xlsx and joblib file with all your input data and prediction values
- A reduced prediction_values file that is anonymized, to pass to your collaborators
- TPRS output to plot AUROCs
- Summary statistics based on the co-provided ground truth (column "target")

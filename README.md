# PRE_SCREEN_HCC

Preprocessing and Modelling Pipelines for the research project "Machine learning predicts hepatocellular carcinoma risk from routine clinical data: a large population-based multicentric study"

## This repository is a collection of three separate repositories used to create the project "Machine learning predicts hepatocellular carcinoma risk from routine clinical data: a large population-based multicentric study"

The subfolders ext_val_aou ; ukb_modeling ; ukb_preprocessing each contain separate Readme files to guide you through the respective repository.

Please consult the respective readme files in the subfolders

The order of processing is

1. ukb_preprocessing
- Includes all preprocessing steps done for UKB data to get from raw dataframes to the X_train, y_train, X_test and y_test dataframes used in ukb_modeling
- Most relevant scripts are numbered 01 to 06, other scripts are helper functions or intended for separate visualizations not central for data processing


2. ext_val_aou
- Preprocessing of external data (branched from ukb_preprocessing with some adjustments)
(Adjustments mostly due to the fact that AOU data has to be processed in the AOU-specific jupyter environment)


3. modeling
- Model development, validation and first testing, separated in two project folders:
-     - HCC (for training, and internal testing on UKB)
-     - ext_val_hcc (adapted project files for external validation in the all of us jupyter environment)




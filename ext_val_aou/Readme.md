# AOU Data Pre‑processing Pipeline

Welcome to the **All of Us (AOU) Data Pre‑processing** repository.\
This repo contains the notebooks and helper scripts that transform raw AOU data into cleaned, analysis‑ready feature tables stored in a Google Cloud Storage (GCS) bucket.\
After completing the steps here you can proceed to the **modelling** sub‑repository to build and externally validate prediction models.

---

## Repository layout

```text
├── 01_Extract_demographics_lifestyle.ipynb
├── 02_Extract_EHR_diagnosis.ipynb
├── 02a_Ethnicity_Processing.ipynb
├── 03_Merge_EHR_Diagnosis.ipynb
├── 03b_Process_EHR_diagnosis.ipynb
├── 03c_Process_DOI_y_Target.ipynb
├── 04_Extract_Blood.ipynb
├── 05_Process_Blood.ipynb
├── 06_Table X and y Processing.ipynb
├── preprocessing_functions.R
├── preprocessing_visualizations.R
├── config.R
└── modelling/          ⬅ external‑validation notebooks & scripts (separate repo)
```

- **Extract notebooks (01, 02, 02a, 04)** – read raw AOU tables, filter/reshape records, and write intermediate Parquet/CSV files to your GCS bucket.
- **Process notebooks (03, 03b, 03c, 05, 06)** – join, de‑duplicate and engineer features to create the final `` (predictors) and `` (targets) tables.
- **R helper scripts**
  - `preprocessing_functions.R` – utility functions shared by R notebooks.
  - `preprocessing_visualizations.R` – quick EDA & QC plots of processed tables.
  - `config.R` – central place to set project‑specific parameters (variable names, loading of global configs etc.).

---

## Prerequisites

| Component                    | Version / Notes                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| **AOU Researcher Workbench** | JupyterLab environment with both Python 3 and R kernels.                                 |
| Python packages              | `pandas`, `numpy`, `pyarrow`, `google‑cloud‑storage`, `fsspec`, `gcsfs`                  |
| R packages                   | `tidyverse`, `bigrquery`, `arrow`, `googleCloudStorageR`                                 |
| GCP access                   | Service account or Workbench‑managed credentials with write access to the target bucket. |
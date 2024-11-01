install.packages(c("stats", "base", "broom", "tidyr", "tidyverse", "readr", "dplyr", "gtools", "data.table", "ggplot2", "rJava", "xlsx", "readxl", "hrbrthemes", "kableExtra", "ggthemes", "ggrepel", "extrafont", "remotes", "writexl", "table1", "jsonlite", "gtsummary", "gt", "circlize", "openxlsx", "lubridate", "maps", "magrittr", "scales", "purrr"))
library(stats)
library(base)
library(broom)
library(tidyr)
library(tidyverse)
library(readr)
library(dplyr)
library(gtools)
library(data.table)
#library(tableone)
library(ggplot2)
library(rJava)
library(xlsx)
library(readxl)
library(hrbrthemes)
library(kableExtra)
library(ggplot2)
library(ggthemes)
library(ggrepel)
library(extrafont)
library(remotes)
library(writexl)
library(table1)
library(jsonlite)
library(gtsummary)
library(gt)
library(gtools)
library(data.table)
library(circlize)
#library(PheWAS)
library(openxlsx)
library(lubridate)
library(maps)
library(ggrepel)
library(magrittr)
library(scales)
library(purrr)
library(flextable)
library(officer)
library(webshot2)
options(knitr.table.format = "html")
options(java.parameters = "-Xmx8000m")



#nextrafont::font_import()
#loadfonts(dev="win")
#windowsFonts()
#fonts()



# Load functions and user-specific variables
#1. Define your own paths and variables in the R_user_configs.json (Github ukb_scripts)
#2. R Detects your Microsoft / Apple/ Linux username (could result in conflicts with same name)
#3a. R assigns your user variables to environment
#3b. Assigning your project-specific variables
#4. Sourcing Global functions
#5. Creation of paths in project directory
#6. Imputation choice

user_configs <- fromJSON("../R_user_configs.json") #add/change your user_config in this JSON file
project_configs <- fromJSON("../R_project_configs.json") #add/change project configs here

if (.Platform$OS.type == "windows") { # Detect username
  user <- Sys.getenv("USERNAME")
} else{
  user <- Sys.getenv("USER")
}  

#user <- janni (Choose one of the users in the user_configs if your OS username does not appear in the JSOn file, or add your own data to the JSON)
if (user %in% names(user_configs)) {    # Set user-specific variables after over
  home <- user_configs[[user]]$home
  ukb <- user_configs[[user]]$ukb
  sharepoint <- user_configs[[user]]$sharepoint
  sharepoint_ukb <- file.path(sharepoint, "ukb")
  project_key <- user_configs[[user]]$project_key
  biobank_key <- user_configs[[user]]$biobank_key
  master_table <- user_configs[[user]]$master_table 
  na_mode <- user_configs[[user]]$na_mode               #change this in R_user_configs to either impute or remove
} else {
  print("Please add your system information in the JSON file referenced in the directory")
}



#project_key <- hcc #hcc or cca (choose one of the present project_keys) or write a new key in the command line
project_path <- file.path(sharepoint, "projects", project_key)
IOI <- project_configs[[project_key]]$IOI
IOIs <- project_configs[[project_key]]$IOIs
DOI <- project_configs[[project_key]]$DOI
icd_dict <- project_configs[[project_key]]$icd_dict
risk_constellation <- project_configs[[project_key]]$risk_constellation
risk_constellation_codes <- project_configs[[project_key]]$risk_constellation_codes
icd_dict_path <- file.path(sharepoint_ukb, "meta", icd_dict)
par_icd_codes <- project_configs[[project_key]]$par_icd_codes
diag_codes <- project_configs[[project_key]]$diag_codes
timeframe <- as.numeric(project_configs[[project_key]]$timeframe)
reduce_model <- as.logical(project_configs[[project_key]]$reduce_model)
vec_remove_columns <- project_configs[[project_key]]$remove_columns


dir.create(file.path(project_path, "/data/dataframes"), showWarnings = FALSE) #Create folders 
dir.create(file.path(project_path, "/supplement"), showWarnings = FALSE)
dir.create(file.path(project_path, "supplement_visuals"), showWarnings = FALSE)
supplement_visuals_dir <- file.path(project_path, "supplement_visuals")


check_for_NAs <- "no" # "yes" or "no"



######## Dynamic file paths ##########
hesin_path <- "raw/hesin.txt"
hesin_diag_path <- "raw/hesin_diag_Jan2024.txt"
death_date_path <- "raw/death_Jan2024.txt"
death_cause_path <- "raw/death_cause_Jan2024.txt"
withdrawals_path <- "raw/withdrawals.txt"

source("preprocessing_functions.R") #Make sure to run this before changing the working directory to the sharepoint




# Project specific patients-at-risk subsettings
par_index <- read_excel(icd_dict_path, sheet= "Patients at risk") #Load table with diagnosis for subsetting
par_groups <- unique(par_index$Group) #Store unique groups (e.g. Cirrhosis, Viral hepatitis)
par_subset <- project_configs[[project_key]]$par_subset #load project-specific subsets of patients-at-risk
vec_risk_constellation <- par_index$Diagnosis[par_index$Group %in% par_subset] #subset index for project-specific requirements
par_icd_codes <- par_index$ICD10[par_index$Group %in% par_subset]


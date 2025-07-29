# List ofrequired packages
required_pkgs <- c(
  "tidyverse",
  "bigrquery", 
  "writexl",
  "readxl",
  "ggplot2",
  "scales",
  "data.table",
  "lubridate",
  "svglite",
  "showtext",
  "sysfonts",
  "systemfonts",
  "jsonlite",
  "purrr",
   "glue",
    "fs",
    "mice",
    "knitr",
    "stringr",
    "ranger",
    "tibble"
    
)



# Install any packages that aren't already installed
new_pkgs <- setdiff(required_pkgs, rownames(installed.packages()))
if (length(new_pkgs)) {
  install.packages(new_pkgs, repos = "https://cloud.r-project.org")
}

# Load them all
lapply(required_pkgs, library, character.only = TRUE)




source("preprocessing_functions.r") #Make sure to run this before changing the working directory to the sharepoint
source("preprocessing_visualizations.r")
#Settings
today <- format(Sys.Date(), "%d_%m_%Y")

# Set options to display full numbers in console output
options(scipen = 999)  




user_configs <- fromJSON("R_user_configs.json") #add/change your user_config in this JSON file
project_configs <- fromJSON("R_project_configs.json") #add/change project configs here



## This is a relic from local development. For now, user is defined manually in ipynb files.
# if (.Platform$OS.type == "windows") { # Detect username
#   user <- Sys.getenv("USERNAME")
# } else{
#   user <- Sys.getenv("USER")
# }  

#user <- janni (Choose one of the users in the user_configs if your OS username does not appear in the JSOn file, or add your own data to the JSON)
if (user %in% names(user_configs)) {    # Set user-specific variables after over
  project_key <- user_configs[[user]]$project_key
  biobank_key <- user_configs[[user]]$biobank_key
  master_table <- user_configs[[user]]$master_table 
  na_mode <- user_configs[[user]]$na_mode               #change this in R_user_configs to either impute or remove
} else {
  print("Please add your system information in the JSON file referenced in the directory")
}



#project_key <- hcc #hcc or cca (choose one of the present project_keys) or write a new key in the command line

IOI <- project_configs[[project_key]]$IOI    #IOI = ICD code of interest
IOIs <- project_configs[[project_key]]$IOIs # multiple ICD codes of interest ;)
DOI <- project_configs[[project_key]]$DOI   # Diagnosis of interest 

risk_constellation <- project_configs[[project_key]]$risk_constellation
risk_constellation_codes <- project_configs[[project_key]]$risk_constellation_codes

par_icd_codes <- project_configs[[project_key]]$par_icd_codes
diag_codes <- project_configs[[project_key]]$diag_codes
timeframe <- as.numeric(project_configs[[project_key]]$timeframe)
reduce_model <- as.logical(project_configs[[project_key]]$reduce_model)
vec_remove_columns <- project_configs[[project_key]]$remove_columns
oper_codes <- project_configs[[project_key]]$oper_codes

project_path <- "../.." #Move two steps up from aou_scripts/src for all folders to be created
dir.create(file.path(project_path, "/tables"), showWarnings = FALSE) #Create folders 
dir.create(file.path(project_path, "/visuals"), showWarnings = FALSE) #Create folders 
dir.create(file.path(project_path, "/data"), showWarnings = FALSE) #Create folders 
dir.create(file.path(project_path, "/data/dataframes"), showWarnings = FALSE) #Create folders 
dir.create(file.path(project_path, "/supplement"), showWarnings = FALSE)
dir.create(file.path(project_path, "/supplement_visuals"), showWarnings = FALSE)

data_path <- file.path(project_path, "data")
suppl_path <- file.path(project_path, "supplement")

check_for_NAs <- "no" # "yes" or "no"





# Project specific patients-at-risk subsettings
par_index <- read_excel(file.path(project_path, master_table), sheet= "Patients at risk") #Load table with diagnosis for subsetting, PAR = Patients at risk
par_groups <- unique(par_index$Group) #Store unique groups (e.g. Cirrhosis, Viral hepatitis)
par_subset <- project_configs[[project_key]]$par_subset #load project-specific subsets of patients-at-risk
vec_risk_constellation <- par_index$Diagnosis[par_index$Group %in% par_subset] #subset index for project-specific requirements
par_icd_codes <- par_index$ICD10[par_index$Group %in% par_subset]


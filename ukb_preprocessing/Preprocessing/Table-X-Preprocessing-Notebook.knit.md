---
title: "Preprocessing UKB Multi-omic data"
output:
  word_document: default
  html_notebook: default
editor_options:
  markdown:
    wrap: 72
---


```r
source("../config.R") #Loads libraries, variables and global function
```

```
## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
## ✔ dplyr     1.1.4     ✔ purrr     1.0.2
## ✔ forcats   1.0.0     ✔ readr     2.1.5
## ✔ ggplot2   3.5.0     ✔ stringr   1.5.1
## ✔ lubridate 1.9.3     ✔ tibble    3.2.1
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ dplyr::filter() masks stats::filter()
## ✖ dplyr::lag()    masks stats::lag()
## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
## 
## Attache Paket: 'data.table'
## 
## 
## Die folgenden Objekte sind maskiert von 'package:lubridate':
## 
##     hour, isoweek, mday, minute, month, quarter, second, wday, week,
##     yday, year
## 
## 
## Die folgenden Objekte sind maskiert von 'package:dplyr':
## 
##     between, first, last
## 
## 
## Das folgende Objekt ist maskiert 'package:purrr':
## 
##     transpose
## 
## 
## NOTE: Either Arial Narrow or Roboto Condensed fonts are required to use these themes.
## 
##       Please use hrbrthemes::import_roboto_condensed() to install Roboto Condensed and
## 
##       if Arial Narrow is not on your system, please see https://bit.ly/arialnarrow
## 
## 
## Attache Paket: 'kableExtra'
## 
## 
## Das folgende Objekt ist maskiert 'package:dplyr':
## 
##     group_rows
## 
## 
## Registering fonts with R
## 
## 
## Attache Paket: 'table1'
## 
## 
## Die folgenden Objekte sind maskiert von 'package:base':
## 
##     units, units<-
## 
## 
## 
## Attache Paket: 'jsonlite'
## 
## 
## Das folgende Objekt ist maskiert 'package:purrr':
## 
##     flatten
```

```
## Warning: Paket 'circlize' wurde unter R Version 4.3.3 erstellt
```

```
## ========================================
## circlize version 0.4.16
## CRAN page: https://cran.r-project.org/package=circlize
## Github page: https://github.com/jokergoo/circlize
## Documentation: https://jokergoo.github.io/circlize_book/book/
## 
## If you use it in published research, please cite:
## Gu, Z. circlize implements and enhances circular visualization
##   in R. Bioinformatics 2014.
## 
## This message can be suppressed by:
##   suppressPackageStartupMessages(library(circlize))
## ========================================
```

```
## Warning: Paket 'openxlsx' wurde unter R Version 4.3.3 erstellt
```

```
## 
## Attache Paket: 'openxlsx'
## 
## Die folgenden Objekte sind maskiert von 'package:xlsx':
## 
##     createWorkbook, loadWorkbook, read.xlsx, saveWorkbook, write.xlsx
```

```
## Warning: Paket 'maps' wurde unter R Version 4.3.3 erstellt
```

```
## 
## Attache Paket: 'maps'
## 
## Das folgende Objekt ist maskiert 'package:purrr':
## 
##     map
## 
## 
## Attache Paket: 'magrittr'
## 
## Das folgende Objekt ist maskiert 'package:purrr':
## 
##     set_names
## 
## Das folgende Objekt ist maskiert 'package:tidyr':
## 
##     extract
## 
## 
## Attache Paket: 'scales'
## 
## Das folgende Objekt ist maskiert 'package:purrr':
## 
##     discard
## 
## Das folgende Objekt ist maskiert 'package:readr':
## 
##     col_factor
```

# Preparing import commands


```r
#### Prepare vector for reading out correct metabolite columns (once done and added to fread function in import section, this has no longer to be run until new metabolites come in)
#NMR_Metabolomics_Index \<- read_excel("\~/PostDoc/Results/Metabolomics/NMR MetabolomicsIndex.xlsx")
#NMR_Metabolomics_Index$datafield <- paste0('"', NMR_Metabolomics_Index$datafield,'-0.0",') #nmrstring \<- cat(NMR_Metabolomics_Index\$datafield, sep="", file="metabolomics_index.txt")

#### Prepare vector for reading out serum parameter columns (once done and added to fread function in import section, this has no longer to be run until new metabolites come in)

#Blood_Marker_Index \<- read_excel("C:/Users/Jan/OneDrive - rwth-aachen.de/Dokumente/PostDoc/Results/Serum Marker/Blood parameters.xlsx")
#Blood_Marker_Index$datafield <- paste0('"', Blood_Marker_Index$datafield, '-0.0",') 
#setwd("\~/PostDoc/Results/Serum Marker") 
#Blood_Marker_String \<- cat( Blood_Marker_Index\$datafield, sep="", file="blood_marker_index.txt") #when pasting to fread function, delete last comma, and add )) at end of command for syntax correction
```


# Create par vectors (for "patients at risk" of certain diseases)
1. Grouped diagnosis are being extracted from meta table
2. Vectors for diagnosis are being created
3. Vectors can be combined as required for diagnosis



```r
setwd(sharepoint_ukb)
btd <- read_excel(icd_dict_path, sheet= "ICD_Singles") #biliary tract disease
group_btd <- "Biliary Tract diseases"
btd <- btd$Diagnosis[btd$Group==group_btd]

cld  <- read_excel(icd_dict_path, sheet= "ICD_Liver") ##### change this to import all from ICD_Liver 
group_cld <- "Chronic Liver disease" 
cld_diags <- c("CLD", "Cirrhosis", "Viral Hepatitis")
cld <- cld[cld$Group %in% cld_diags, ]
cld <- cld$Diagnosis

cca <- read_excel(icd_dict_path, sheet= "ICD_Singles")
group_cca <- "Hepatobiliary cancer"
cca <- cca$Diagnosis[cca$Group == group_cca]

hcc <- c("Liver cancer, HCC", "Liver cancer, unspecified")

cholelithiasis <- c("Cholelithiasis")

if (DOI == "CCa") {
  vec_risk_constellation <- c(cld, btd, cholelithiasis)
} else if (DOI == "HCC") {
  vec_risk_constellation <- cld
}
```

# Loading Data:

Importing and preprocessing of multimodal data

| Nr  | Description      | Dataset         |        n |
|-----|------------------|:----------------|---------:|
| 1a  | Patient IDs      | df_eid          |   502411 |
| 1b  | Covariates       | df_covariates   |  Varying |
| 1c  | ICD Diagnosis    | df_diagnosis    |   442277 |
| 2   | Blood parameters | df_blood        |   456855 |
| 3   | Metabolomics     | df_metabolomics | 248286   |
| 4   | Genomics         | df_snp          | 488149   |
| 5   | Radiomics        | df_mri          | \~50.000 | (Pending)



#### 1. EID, ICD, Covariates, PAR(Importing tables previously processed)

```r
#setwd(drive)
#df_eid <- fread("ukb52200.csv", select=c("eid")) %>%    #all eids in UKB
#write.csv(df_eid, file="C:/Users/Jan/OneDrive/Dokumente/PostDoc/Patient_tables/UKB_Patients_eids.csv", row.names=FALSE)    

#a) df_eid 
setwd(sharepoint_ukb)
df_withdrawals <- read.delim("raw/withdrawals.txt")
df_eid <- fread("extracted/UKB_Patients_eids.csv") %>% check_and_remove_withdrawals(df_withdrawals)
```

```
## No patients with withdrawn consent in df. You may pass!
```

```r
#b) Covariate data for all 502411 patients: df_covariates 502411
#Importing df_covariates: Preprocessing performed in Script "Extract and process covariates"
setwd(project_path)
load("data/dataframes/df_covariates.RData")
df_covariates <- check_and_remove_withdrawals(df_covariates, df_withdrawals)
```

```
## No patients with withdrawn consent in df. You may pass!
```

```r
#c) Diagnosis data for selected  diagnosis (import diagnosis data for selected diagnosis in Script "Extract_multiple_Diagnosis"))
setwd(project_path)
load("data/dataframes/df_diagnosis.RData")
df_diagnosis <- df_diagnosis %>% check_and_remove_withdrawals(df_withdrawals)
```

```
## No patients with withdrawn consent in df. You may pass!
```

```r
#sanity(df_diagnosis)
```

```r
# pre-check filtering of the dataframe for loss of DOI cases

filter_rows_with_pos_entries(df_diagnosis)
```

```
## [1] "Current patients-at-risk-setting includes the following groups:"
## [1] "BTD"              "CLD"              "Cirrhosis"        "Viral Hepatitis" 
## [5] "Cholelithiasis"   "Blood_Parameters"
```

```
## df_diagnosis contains 143927 Patients at risk.
```

```
## Key: <eid>
##             eid Acute renal failure Alcoholic cirrhosis Alcoholic fatty liver
##           <int>              <fctr>              <fctr>                <fctr>
##      1: 1000043                   0                   0                     0
##      2: 1000066                   1                   0                     0
##      3: 1000092                   0                   0                     0
##      4: 1000107                   0                   0                     0
##      5: 1000128                   0                   0                     0
##     ---                                                                      
## 143923: 6024470                   0                   0                     0
## 143924: 6024518                   0                   0                     0
## 143925: 6024543                   0                   0                     0
## 143926: 6024551                   0                   0                     0
## 143927: 6024566                   0                   0                     0
##         Alcoholic fibrosis Alcoholic hepatic failure Alcoholic hepatitis
##                     <fctr>                    <fctr>              <fctr>
##      1:                  0                         0                   0
##      2:                  0                         0                   0
##      3:                  0                         0                   0
##      4:                  0                         0                   0
##      5:                  0                         0                   0
##     ---                                                                 
## 143923:                  0                         0                   0
## 143924:                  0                         0                   0
## 143925:                  0                         0                   0
## 143926:                  0                         0                   0
## 143927:                  0                         0                   0
##         Alcoholic liver disease, unspecified Ascites Autoimmune hepatitis
##                                       <fctr>  <fctr>               <fctr>
##      1:                                    0       0                    0
##      2:                                    0       0                    0
##      3:                                    0       0                    0
##      4:                                    0       0                    0
##      5:                                    0       0                    0
##     ---                                                                  
## 143923:                                    0       0                    0
## 143924:                                    0       0                    0
## 143925:                                    0       0                    0
## 143926:                                    0       0                    0
## 143927:                                    0       0                    0
##         Biliary cirrhosis, unspecified Biliary Cyst
##                                 <fctr>       <fctr>
##      1:                              0            0
##      2:                              0            0
##      3:                              0            0
##      4:                              0            0
##      5:                              0            0
##     ---                                            
## 143923:                              0            0
## 143924:                              0            0
## 143925:                              0            0
## 143926:                              0            0
## 143927:                              0            0
##         Biliary tract cancer, Ampulla vateri Biliary tract cancer, extrahepatic
##                                       <fctr>                             <fctr>
##      1:                                    0                                  0
##      2:                                    0                                  0
##      3:                                    0                                  0
##      4:                                    0                                  0
##      5:                                    0                                  0
##     ---                                                                        
## 143923:                                    0                                  0
## 143924:                                    0                                  0
## 143925:                                    0                                  0
## 143926:                                    0                                  0
## 143927:                                    0                                  0
##         Biliary tract cancer, overlapping lesion
##                                           <fctr>
##      1:                                        0
##      2:                                        0
##      3:                                        0
##      4:                                        0
##      5:                                        0
##     ---                                         
## 143923:                                        0
## 143924:                                        0
## 143925:                                        0
## 143926:                                        0
## 143927:                                        0
##         Biliary tract cancer, unspecified Cholangitis Chronic Hepatitis B
##                                    <fctr>      <fctr>              <fctr>
##      1:                                 0           0                   0
##      2:                                 0           0                   0
##      3:                                 0           0                   0
##      4:                                 0           0                   0
##      5:                                 0           0                   0
##     ---                                                                  
## 143923:                                 0           0                   0
## 143924:                                 0           0                   0
## 143925:                                 0           0                   0
## 143926:                                 0           0                   0
## 143927:                                 0           0                   0
##         Chronic Hepatitis C Disease of biliary tract, unspecified
##                      <fctr>                                <fctr>
##      1:                   0                                     0
##      2:                   0                                     0
##      3:                   0                                     0
##      4:                   0                                     0
##      5:                   0                                     0
##     ---                                                          
## 143923:                   0                                     0
## 143924:                   0                                     0
## 143925:                   0                                     0
## 143926:                   0                                     0
## 143927:                   0                                     0
##         Fistula of bile duct Granulomatous hepatitis, not elsewhere classified
##                       <fctr>                                            <fctr>
##      1:                    0                                                 0
##      2:                    0                                                 0
##      3:                    0                                                 0
##      4:                    0                                                 0
##      5:                    0                                                 0
##     ---                                                                       
## 143923:                    0                                                 0
## 143924:                    0                                                 0
## 143925:                    0                                                 0
## 143926:                    0                                                 0
## 143927:                    0                                                 0
##         Hepatic fibrosis Hepatic fibrosis and sclerosis Hepatic sclerosis
##                   <fctr>                         <fctr>            <fctr>
##      1:                0                              0                 0
##      2:                0                              0                 0
##      3:                0                              0                 0
##      4:                0                              0                 0
##      5:                0                              0                 0
##     ---                                                                  
## 143923:                0                              0                 0
## 143924:                0                              0                 0
## 143925:                0                              0                 0
## 143926:                0                              0                 0
## 143927:                0                              0                 0
##         Hepatomegaly with splenomegaly, not elsewhere classified
##                                                           <fctr>
##      1:                                                        0
##      2:                                                        0
##      3:                                                        0
##      4:                                                        0
##      5:                                                        0
##     ---                                                         
## 143923:                                                        0
## 143924:                                                        0
## 143925:                                                        0
## 143926:                                                        0
## 143927:                                                        0
##         Hepatomegaly, not elsewhere classified Hepatorenal Syndrome
##                                         <fctr>               <fctr>
##      1:                                      0                    0
##      2:                                      0                    0
##      3:                                      0                    0
##      4:                                      0                    0
##      5:                                      0                    0
##     ---                                                            
## 143923:                                      0                    0
## 143924:                                      0                    0
## 143925:                                      0                    0
## 143926:                                      0                    0
## 143927:                                      0                    0
##         Inflammatory liver disease, unspecified
##                                          <fctr>
##      1:                                       0
##      2:                                       0
##      3:                                       0
##      4:                                       0
##      5:                                       0
##     ---                                        
## 143923:                                       0
## 143924:                                       0
## 143925:                                       0
## 143926:                                       0
## 143927:                                       0
##         Intrahepatic bile duct carcinoma Jaundice  NAFLD   NASH
##                                   <fctr>   <fctr> <fctr> <fctr>
##      1:                                0        0      0      0
##      2:                                0        0      0      0
##      3:                                0        0      0      0
##      4:                                0        0      0      0
##      5:                                0        0      0      0
##     ---                                                        
## 143923:                                0        0      0      0
## 143924:                                0        0      0      0
## 143925:                                0        0      0      0
## 143926:                                0        0      0      0
## 143927:                                0        0      0      0
##         Nonspecific reactive hepatitis Obstruction of bile duct
##                                 <fctr>                   <fctr>
##      1:                              0                        0
##      2:                              0                        0
##      3:                              0                        0
##      4:                              0                        0
##      5:                              0                        0
##     ---                                                        
## 143923:                              0                        0
## 143924:                              0                        0
## 143925:                              0                        0
## 143926:                              0                        0
## 143927:                              0                        0
##         Oesophageal varices w bleeding Oesophageal varices w_o bleeding
##                                 <fctr>                           <fctr>
##      1:                              0                                0
##      2:                              0                                0
##      3:                              0                                0
##      4:                              0                                0
##      5:                              0                                0
##     ---                                                                
## 143923:                              0                                0
## 143924:                              0                                0
## 143925:                              0                                0
## 143926:                              0                                0
## 143927:                              0                                0
##         Other and unspecified cirrhosis of liver
##                                           <fctr>
##      1:                                        0
##      2:                                        0
##      3:                                        0
##      4:                                        0
##      5:                                        0
##     ---                                         
## 143923:                                        0
## 143924:                                        0
## 143925:                                        0
## 143926:                                        0
## 143927:                                        0
##         Other specified diseases of biliary tract Perforation of bile duct
##                                            <fctr>                   <fctr>
##      1:                                         0                        0
##      2:                                         0                        0
##      3:                                         0                        0
##      4:                                         0                        0
##      5:                                         0                        0
##     ---                                                                   
## 143923:                                         0                        0
## 143924:                                         0                        0
## 143925:                                         0                        0
## 143926:                                         0                        0
## 143927:                                         0                        0
##         Primary biliary cirrhosis Secondary biliary cirrhosis
##                            <fctr>                      <fctr>
##      1:                         0                           0
##      2:                         0                           0
##      3:                         0                           0
##      4:                         0                           0
##      5:                         0                           0
##     ---                                                      
## 143923:                         0                           0
## 143924:                         0                           0
## 143925:                         0                           0
## 143926:                         0                           0
## 143927:                         0                           0
##         Spasm of sphincter of Oddi Splenomegaly, not elsewhere classified
##                             <fctr>                                 <fctr>
##      1:                          0                                      0
##      2:                          0                                      0
##      3:                          0                                      0
##      4:                          0                                      0
##      5:                          0                                      0
##     ---                                                                  
## 143923:                          0                                      0
## 143924:                          0                                      0
## 143925:                          0                                      0
## 143926:                          0                                      0
## 143927:                          0                                      0
##         Alcoholic liver disease Arterial hypertension Cholelithiasis
##                          <fctr>                <fctr>         <fctr>
##      1:                       0                     1              0
##      2:                       0                     1              0
##      3:                       0                     0              0
##      4:                       0                     0              0
##      5:                       0                     0              0
##     ---                                                             
## 143923:                       0                     0              1
## 143924:                       0                     0              0
## 143925:                       0                     1              1
## 143926:                       0                     1              0
## 143927:                       0                     1              0
##         Chronic kidney disease Colorectal cancer Complications of cirrhosis
##                         <fctr>            <fctr>                     <fctr>
##      1:                      0                 0                          0
##      2:                      0                 0                          0
##      3:                      0                 0                          0
##      4:                      0                 0                          0
##      5:                      0                 0                          0
##     ---                                                                    
## 143923:                      0                 0                          0
## 143924:                      0                 0                          0
## 143925:                      0                 0                          0
## 143926:                      0                 0                          0
## 143927:                      0                 0                          0
##         Esophageal cancer Fibrosis and cirrhosis Gastric cancer
##                    <fctr>                 <fctr>         <fctr>
##      1:                 0                      0              0
##      2:                 0                      0              0
##      3:                 0                      0              0
##      4:                 0                      0              0
##      5:                 0                      0              0
##     ---                                                        
## 143923:                 0                      0              0
## 143924:                 0                      0              0
## 143925:                 0                      0              0
## 143926:                 0                      0              0
## 143927:                 0                      0              0
##         Hepatobiliary cancer    HSM    IBD Inflammatory liver disease
##                       <fctr> <fctr> <fctr>                     <fctr>
##      1:                    0      0      0                          0
##      2:                    0      0      0                          0
##      3:                    0      0      0                          0
##      4:                    0      0      0                          0
##      5:                    0      0      0                          0
##     ---                                                              
## 143923:                    0      0      0                          0
## 143924:                    0      0      0                          0
## 143925:                    0      0      0                          0
## 143926:                    0      0      0                          0
## 143927:                    0      0      0                          0
##         Pancreatic cancer Sepsis Small intestine cancer Stomach disorders
##                    <fctr> <fctr>                 <fctr>            <fctr>
##      1:                 0      0                      0                 0
##      2:                 0      0                      0                 0
##      3:                 0      0                      0                 0
##      4:                 0      0                      0                 0
##      5:                 0      0                      0                 0
##     ---                                                                  
## 143923:                 0      0                      0                 0
## 143924:                 0      0                      0                 0
## 143925:                 0      0                      0                 0
## 143926:                 0      0                      0                 0
## 143927:                 0      0                      0                 0
##             DM Elevated_AST Elevated_ALT Elevated_GGT Elevated_AP
##         <fctr>       <fctr>       <fctr>       <fctr>      <fctr>
##      1:      0            0            0            1           0
##      2:      0            0            0            1           0
##      3:      0            0            0            1           1
##      4:      0            0            0            1           0
##      5:      0            0            1            1           1
##     ---                                                          
## 143923:      0            0            0            0           0
## 143924:      0            0            0            0           1
## 143925:      0            0            0            0           1
## 143926:      0            0            0            1           0
## 143927:      0            0            0            0           1
##         Elevated_Liver_Enzymes
##                         <fctr>
##      1:                      1
##      2:                      1
##      3:                      1
##      4:                      1
##      5:                      1
##     ---                       
## 143923:                      0
## 143924:                      0
## 143925:                      0
## 143926:                      1
## 143927:                      0
```

```r
innerjoin_df_y(filter_rows_with_pos_entries(df_diagnosis))
```

```
## [1] "Current patients-at-risk-setting includes the following groups:"
## [1] "BTD"              "CLD"              "Cirrhosis"        "Viral Hepatitis" 
## [5] "Cholelithiasis"   "Blood_Parameters"
```

```
## df_diagnosis contains 143927 Patients at risk.
```

```
## filter_rows_with_pos_entries(df_diagnosis) contains 437CCa cases.
```

```
## Key: <eid>
##          eid Acute renal failure Alcoholic cirrhosis Alcoholic fatty liver
##        <int>              <fctr>              <fctr>                <fctr>
##   1: 1012437                   0                   0                     0
##   2: 1025442                   0                   0                     0
##   3: 1045246                   0                   0                     0
##   4: 1046570                   0                   0                     0
##   5: 1053057                   0                   0                     0
##  ---                                                                      
## 433: 5935906                   0                   0                     0
## 434: 5956307                   0                   0                     0
## 435: 6001377                   0                   0                     0
## 436: 6018798                   0                   0                     0
## 437: 6021117                   0                   0                     0
##      Alcoholic fibrosis Alcoholic hepatic failure Alcoholic hepatitis
##                  <fctr>                    <fctr>              <fctr>
##   1:                  0                         0                   0
##   2:                  0                         0                   0
##   3:                  0                         0                   0
##   4:                  0                         0                   0
##   5:                  0                         0                   0
##  ---                                                                 
## 433:                  0                         0                   0
## 434:                  0                         0                   0
## 435:                  0                         0                   0
## 436:                  0                         0                   0
## 437:                  0                         0                   0
##      Alcoholic liver disease, unspecified Ascites Autoimmune hepatitis
##                                    <fctr>  <fctr>               <fctr>
##   1:                                    0       0                    0
##   2:                                    0       0                    0
##   3:                                    0       0                    0
##   4:                                    0       0                    0
##   5:                                    0       0                    0
##  ---                                                                  
## 433:                                    0       0                    0
## 434:                                    0       0                    0
## 435:                                    0       0                    0
## 436:                                    0       0                    0
## 437:                                    0       0                    0
##      Biliary cirrhosis, unspecified Biliary Cyst
##                              <fctr>       <fctr>
##   1:                              0            0
##   2:                              0            0
##   3:                              0            0
##   4:                              0            0
##   5:                              0            0
##  ---                                            
## 433:                              0            0
## 434:                              0            0
## 435:                              0            0
## 436:                              0            0
## 437:                              0            0
##      Biliary tract cancer, Ampulla vateri Biliary tract cancer, extrahepatic
##                                    <fctr>                             <fctr>
##   1:                                    1                                  0
##   2:                                    0                                  0
##   3:                                    0                                  1
##   4:                                    0                                  0
##   5:                                    0                                  0
##  ---                                                                        
## 433:                                    0                                  0
## 434:                                    0                                  0
## 435:                                    0                                  0
## 436:                                    0                                  0
## 437:                                    0                                  0
##      Biliary tract cancer, overlapping lesion Biliary tract cancer, unspecified
##                                        <fctr>                            <fctr>
##   1:                                        0                                 0
##   2:                                        0                                 0
##   3:                                        0                                 0
##   4:                                        0                                 0
##   5:                                        0                                 0
##  ---                                                                           
## 433:                                        0                                 0
## 434:                                        0                                 0
## 435:                                        0                                 0
## 436:                                        0                                 0
## 437:                                        0                                 0
##      Cholangitis Chronic Hepatitis B Chronic Hepatitis C
##           <fctr>              <fctr>              <fctr>
##   1:           0                   0                   0
##   2:           1                   0                   0
##   3:           0                   0                   0
##   4:           0                   0                   0
##   5:           0                   0                   0
##  ---                                                    
## 433:           0                   0                   0
## 434:           0                   0                   0
## 435:           0                   0                   0
## 436:           0                   0                   0
## 437:           0                   0                   0
##      Disease of biliary tract, unspecified Fistula of bile duct
##                                     <fctr>               <fctr>
##   1:                                     1                    0
##   2:                                     0                    0
##   3:                                     0                    0
##   4:                                     0                    0
##   5:                                     0                    0
##  ---                                                           
## 433:                                     0                    0
## 434:                                     0                    0
## 435:                                     0                    0
## 436:                                     0                    0
## 437:                                     0                    0
##      Granulomatous hepatitis, not elsewhere classified Hepatic fibrosis
##                                                 <fctr>           <fctr>
##   1:                                                 0                0
##   2:                                                 0                0
##   3:                                                 0                0
##   4:                                                 0                0
##   5:                                                 0                0
##  ---                                                                   
## 433:                                                 0                0
## 434:                                                 0                0
## 435:                                                 0                0
## 436:                                                 0                0
## 437:                                                 0                0
##      Hepatic fibrosis and sclerosis Hepatic sclerosis
##                              <fctr>            <fctr>
##   1:                              0                 0
##   2:                              0                 0
##   3:                              0                 0
##   4:                              0                 0
##   5:                              0                 0
##  ---                                                 
## 433:                              0                 0
## 434:                              0                 0
## 435:                              0                 0
## 436:                              0                 0
## 437:                              0                 0
##      Hepatomegaly with splenomegaly, not elsewhere classified
##                                                        <fctr>
##   1:                                                        0
##   2:                                                        0
##   3:                                                        0
##   4:                                                        0
##   5:                                                        0
##  ---                                                         
## 433:                                                        0
## 434:                                                        0
## 435:                                                        0
## 436:                                                        0
## 437:                                                        0
##      Hepatomegaly, not elsewhere classified Hepatorenal Syndrome
##                                      <fctr>               <fctr>
##   1:                                      0                    0
##   2:                                      0                    0
##   3:                                      0                    0
##   4:                                      0                    0
##   5:                                      0                    0
##  ---                                                            
## 433:                                      0                    0
## 434:                                      0                    0
## 435:                                      0                    0
## 436:                                      0                    0
## 437:                                      0                    0
##      Inflammatory liver disease, unspecified Intrahepatic bile duct carcinoma
##                                       <fctr>                           <fctr>
##   1:                                       0                                0
##   2:                                       0                                0
##   3:                                       0                                1
##   4:                                       0                                0
##   5:                                       0                                0
##  ---                                                                         
## 433:                                       0                                0
## 434:                                       0                                0
## 435:                                       0                                0
## 436:                                       0                                0
## 437:                                       0                                0
##      Jaundice  NAFLD   NASH Nonspecific reactive hepatitis
##        <fctr> <fctr> <fctr>                         <fctr>
##   1:        1      0      0                              0
##   2:        0      0      0                              0
##   3:        0      0      0                              0
##   4:        0      0      0                              0
##   5:        0      0      0                              0
##  ---                                                      
## 433:        0      0      0                              0
## 434:        0      0      0                              0
## 435:        0      0      0                              0
## 436:        0      0      0                              0
## 437:        0      0      0                              0
##      Obstruction of bile duct Oesophageal varices w bleeding
##                        <fctr>                         <fctr>
##   1:                        1                              0
##   2:                        0                              0
##   3:                        0                              0
##   4:                        0                              0
##   5:                        0                              0
##  ---                                                        
## 433:                        0                              0
## 434:                        0                              0
## 435:                        0                              0
## 436:                        0                              0
## 437:                        0                              0
##      Oesophageal varices w_o bleeding Other and unspecified cirrhosis of liver
##                                <fctr>                                   <fctr>
##   1:                                0                                        0
##   2:                                1                                        0
##   3:                                0                                        0
##   4:                                0                                        0
##   5:                                0                                        0
##  ---                                                                          
## 433:                                0                                        0
## 434:                                0                                        0
## 435:                                0                                        0
## 436:                                0                                        0
## 437:                                0                                        0
##      Other specified diseases of biliary tract Perforation of bile duct
##                                         <fctr>                   <fctr>
##   1:                                         0                        0
##   2:                                         0                        0
##   3:                                         1                        0
##   4:                                         0                        0
##   5:                                         0                        0
##  ---                                                                   
## 433:                                         0                        0
## 434:                                         0                        0
## 435:                                         0                        0
## 436:                                         0                        0
## 437:                                         0                        0
##      Primary biliary cirrhosis Secondary biliary cirrhosis
##                         <fctr>                      <fctr>
##   1:                         0                           0
##   2:                         0                           0
##   3:                         0                           0
##   4:                         0                           0
##   5:                         0                           0
##  ---                                                      
## 433:                         0                           0
## 434:                         0                           0
## 435:                         0                           0
## 436:                         0                           0
## 437:                         0                           0
##      Spasm of sphincter of Oddi Splenomegaly, not elsewhere classified
##                          <fctr>                                 <fctr>
##   1:                          0                                      0
##   2:                          0                                      0
##   3:                          0                                      0
##   4:                          0                                      0
##   5:                          0                                      0
##  ---                                                                  
## 433:                          0                                      0
## 434:                          0                                      0
## 435:                          0                                      0
## 436:                          0                                      0
## 437:                          0                                      0
##      Alcoholic liver disease Arterial hypertension Cholelithiasis
##                       <fctr>                <fctr>         <fctr>
##   1:                       0                     1              0
##   2:                       0                     0              0
##   3:                       0                     0              1
##   4:                       0                     1              1
##   5:                       0                     0              0
##  ---                                                             
## 433:                       0                     0              0
## 434:                       0                     1              0
## 435:                       0                     1              0
## 436:                       0                     0              0
## 437:                       0                     1              0
##      Chronic kidney disease Colorectal cancer Complications of cirrhosis
##                      <fctr>            <fctr>                     <fctr>
##   1:                      0                 0                          1
##   2:                      0                 0                          1
##   3:                      0                 0                          0
##   4:                      0                 0                          0
##   5:                      0                 0                          0
##  ---                                                                    
## 433:                      0                 0                          0
## 434:                      0                 0                          0
## 435:                      0                 0                          0
## 436:                      0                 0                          0
## 437:                      0                 0                          0
##      Esophageal cancer Fibrosis and cirrhosis Gastric cancer
##                 <fctr>                 <fctr>         <fctr>
##   1:                 0                      0              0
##   2:                 0                      0              0
##   3:                 0                      0              0
##   4:                 0                      0              0
##   5:                 0                      0              0
##  ---                                                        
## 433:                 0                      0              0
## 434:                 0                      0              0
## 435:                 0                      0              0
## 436:                 0                      0              0
## 437:                 0                      0              0
##      Hepatobiliary cancer    HSM    IBD Inflammatory liver disease
##                    <fctr> <fctr> <fctr>                     <fctr>
##   1:                    1      0      0                          0
##   2:                    0      0      1                          0
##   3:                    1      0      0                          0
##   4:                    0      0      0                          0
##   5:                    0      0      0                          0
##  ---                                                              
## 433:                    0      0      0                          0
## 434:                    0      0      0                          0
## 435:                    0      0      0                          0
## 436:                    0      0      0                          0
## 437:                    0      0      0                          0
##      Pancreatic cancer Sepsis Small intestine cancer Stomach disorders     DM
##                 <fctr> <fctr>                 <fctr>            <fctr> <fctr>
##   1:                 1      0                      0                 1      0
##   2:                 0      0                      0                 0      0
##   3:                 0      1                      0                 0      0
##   4:                 0      0                      0                 1      0
##   5:                 0      0                      0                 1      0
##  ---                                                                         
## 433:                 0      0                      0                 0      0
## 434:                 0      0                      0                 0      0
## 435:                 0      0                      0                 1      1
## 436:                 0      0                      0                 0      0
## 437:                 0      0                      0                 0      0
##      Elevated_AST Elevated_ALT Elevated_GGT Elevated_AP Elevated_Liver_Enzymes
##            <fctr>       <fctr>       <fctr>      <fctr>                 <fctr>
##   1:            0            0            0           0                      0
##   2:            0            0            0           0                      0
##   3:            0            0            0           0                      0
##   4:            0            1            1           0                      1
##   5:            0            0            1           0                      1
##  ---                                                                          
## 433:            0            0            0           1                      0
## 434:            1            1            1           1                      1
## 435:            1            1            1           0                      1
## 436:            0            0            1           0                      1
## 437:            0            0            1           0                      1
##      status
##       <int>
##   1:      1
##   2:      1
##   3:      1
##   4:      1
##   5:      1
##  ---       
## 433:      1
## 434:      1
## 435:      1
## 436:      1
## 437:      1
```


#### 2. Blood sample data for all 502411 patients df_blood 502411


```r
#setwd(drive)
#import blood sample data from baseline and change names to measured parameters
#df_blood <- fread("ukb52200.csv", select=c("eid", "30160-0.0","30220-0.0","30150-0.0","30210-0.0","30030-0.0","30020-0.0","30300-0.0","30290-0.0","30280-0.0","30120-0.0","30180-0.0","30050-0.0","30060-0.0","30040-0.0","30100-0.0","30260-0.0","30270-0.0","30130-0.0","30190-0.0","30140-0.0","30200-0.0","30170-0.0","30230-0.0","30080-0.0","30090-0.0","30110-0.0","30010-0.0","30070-0.0","30250-0.0","30240-0.0","30000-0.0","30620-0.0","30600-0.0","30610-0.0","30630-0.0","30640-0.0","30650-0.0","30710-0.0","30680-0.0","30690-0.0","30700-0.0","30720-0.0","30660-0.0","30730-0.0","30740-0.0","30750-0.0","30760-0.0","30770-0.0","30780-0.0","30790-0.0","30800-0.0","30810-0.0","30820-0.0","30830-0.0","30850-0.0","30840-0.0","30860-0.0","30870-0.0","30880-0.0","30670-0.0","30890-0.0"))
#setwd(sharepoint_ukb)
#write.csv(df_blood, file="extracted/UKB_Patients_blood.csv, row.names=FALSE)    

setwd(sharepoint_ukb)
df_blood <- fread("extracted/UKB_Patients_blood.csv") %>% check_and_remove_withdrawals(df_withdrawals)
```

```
## Warning in check_and_remove_withdrawals(., df_withdrawals): Patients with
## withdrawn consent have been removed from the dataframe. Rows before: 502411,
## Rows after: 502309, Rows removed: 102
```

```r
# changing names to measured parameters
#Blood_Marker_Index <- read_excel("meta/Master_Table_JC.xlsx", sheet="Blood count and biochemistry")           #Import Index dataframe
Blood_Marker_Index <- read_excel(master_table, sheet="Blood count and biochemistry")    
Blood_Marker_Index$datafield <- paste0(Blood_Marker_Index$datafield, '-0.0')    #adapt syntax
blood_names <- Blood_Marker_Index$Description                                   #Create names vector
blood_names <- c("eid", blood_names)                                            #add eid in front of name vector
df_blood <- setNames(df_blood, blood_names)                                     #Change names from codes to actual measurements

#Overview and handling of NAs by imputing the most common value or removing
na_blood_col <- data.frame(na_columnwise(df_blood))
```

```
## [1] "eid : 0"
## [1] "Basophill (%) : 25210"
## [1] "Basophill count : 25204"
## [1] "Eosinophill (%) : 25210"
## [1] "Eosinophill count : 25204"
## [1] "Erythrocytes : 24322"
## [1] "Erythrocyte distribution width : 24322"
## [1] "Haematocrit : 32633"
## [1] "Haemoglobin : 32632"
## [1] "High light scatter reticulocyte (%) : 32634"
## [1] "High light scatter reticulocyte count : 25210"
## [1] "Immature reticulocyte fraction : 25204"
## [1] "Leukocyte count : 24325"
## [1] "Lymphocyte (%) : 24329"
## [1] "Lymphocyte count : 24324"
## [1] "MCH : 24330"
## [1] "MCHC : 32634"
## [1] "MCV : 32632"
## [1] "Mean platelet volume : 25210"
## [1] "Mean reticulocyte volume : 25204"
## [1] "Mean sphered cell volume : 25210"
## [1] "Monocyte count : 25204"
## [1] "Monocyte percentage : 25221"
## [1] "Neutrophill count : 25225"
## [1] "Neutrophill percentage : 24325"
## [1] "Nucleated red blood cell (%) : 24329"
## [1] "Nucleated red blood cell count : 24330"
## [1] "Platelet count : 24322"
## [1] "Platelet crit : 24324"
## [1] "Platelet distribution width : 32633"
## [1] "Reticulocyte (%) : 32633"
## [1] "Reticulocyte count : 24327"
## [1] "Alanine aminotransferase : 33087"
## [1] "Albumin : 72397"
## [1] "Alkaline phosphatase : 32886"
## [1] "Apolipoprotein A : 74962"
## [1] "Apolipoprotein B : 35280"
## [1] "Aspartate aminotransferase : 34691"
## [1] "Calcium : 33920"
## [1] "Cholesterol : 72541"
## [1] "C-reactive protein : 32899"
## [1] "Creatinine : 33131"
## [1] "Cystatin C : 32930"
## [1] "Direct bilirubin : 103854"
## [1] "Gamma glutamyltransferase : 33146"
## [1] "Glucose : 72906"
## [1] "HbA1c : 35983"
## [1] "HDL cholesterol : 72602"
## [1] "IGF-1 : 35445"
## [1] "LDL direct : 33782"
## [1] "Lipoprotein A : 126822"
## [1] "Oestradiol : 425688"
## [1] "Phosphate : 73222"
## [1] "Rheumatoid factor : 461012"
## [1] "SHBG : 76626"
## [1] "Testosterone : 77269"
## [1] "Total bilirubin : 34927"
## [1] "Total protein : 72868"
## [1] "Triglycerides : 33274"
## [1] "Urate : 33463"
## [1] "Urea : 33217"
## [1] "Vitamin D : 54127"
```

```r
na_blood_row <- rowSums(is.na(df_blood))
hist(na_blood_row)
```

![](Table-X-Preprocessing-Notebook_files/figure-docx/unnamed-chunk-6-1.png)<!-- -->

```r
#write.xlsx(na_blood_col, file=paste(project, "/supplement/NA_count_df_blood.xlsx", sep=''))

if (na_mode == "impute") {
  summary(df_blood[,1:5])
  df_blood <- df_blood %>% 
    mutate_at(vars(-1), mean.impute) %>% # Apply mean.impute to all columns except the first one
    select(-c("Rheumatoid factor", "Oestradiol")) #delete columns with mostly NA (Oestradiol 402.879 NAs; Rheumatoid factor 437.202 NAs)
  summary(df_blood[,1:5])
  
} else if (na_mode == "remove") {     # Your removal code here
  df_blood <- omit.NA(df_blood, 20)   # Delete rows with >= 20 NAs per Row
  df_blood <- df_blood %>% #delete columns with mostly NA values (Lipoprotein A 111.652 NAs; Oestradiol 402.879 NAs; Rheumatoid factor 437.202 NAs)
  select(-c("Rheumatoid factor", "Oestradiol"))     
}
```

```
##       eid          Basophill (%)     Basophill count   Eosinophill (%) 
##  Min.   :1000018   Min.   :0.00000   Min.   : 0.0000   Min.   :0.0000  
##  1st Qu.:2256279   1st Qu.:0.00000   1st Qu.: 0.3000   1st Qu.:0.1000  
##  Median :3512393   Median :0.02000   Median : 0.4500   Median :0.1500  
##  Mean   :3512381   Mean   :0.03406   Mean   : 0.5697   Mean   :0.1749  
##  3rd Qu.:4768523   3rd Qu.:0.04000   3rd Qu.: 0.6400   3rd Qu.:0.2000  
##  Max.   :6024636   Max.   :3.03000   Max.   :33.8000   Max.   :9.6000  
##  Eosinophill count
##  Min.   :  0.000  
##  1st Qu.:  1.400  
##  Median :  2.200  
##  Mean   :  2.572  
##  3rd Qu.:  3.200  
##  Max.   :100.000
```

```r
#save(df_blood, file=paste(project_path, "/data/dataframes/df_blood.RData", sep='')) # Save df_blood as .RData (preserves column classes)

vec_blood <- setdiff(colnames(df_blood), "eid") #this gets important for minmax normalization later on

#Sanity check
#sanity(df_blood)


# Select all patients with elevated liver enzymes at baseline
if (project_key == "hcc"){
    df_blood_risk <- df_blood %>%
  select(c("eid", "Aspartate aminotransferase", "Alanine aminotransferase", "Gamma glutamyltransferase" )) %>%
  inner_join(select(df_covariates, eid, SEX), by = "eid") %>%
  mutate(
    Elevated_AST = if_else(SEX == "Female" & `Aspartate aminotransferase` > 35 | SEX == "Male" & `Aspartate aminotransferase` > 50, 1, 0),
    Elevated_ALT = if_else(SEX == "Female" & `Alanine aminotransferase` > 35 | SEX == "Male" & `Alanine aminotransferase` > 50, 1, 0),
    Elevated_GGT = if_else(SEX == "Female" & `Gamma glutamyltransferase` > 40 | SEX == "Male" & `Gamma glutamyltransferase` > 60, 1, 0)
  ) %>%
  mutate(Elevated_Liver_Enzymes = if_else(Elevated_AST == 1 | Elevated_ALT == 1 | Elevated_GGT == 1, 1, 0)) %>%
  select(-c("Aspartate aminotransferase", "Alanine aminotransferase", "Gamma glutamyltransferase", "SEX"))
}
if (project_key == "cca") {   
  df_blood_risk <- df_blood %>%
  select(c("eid", "Aspartate aminotransferase", "Alanine aminotransferase", "Gamma glutamyltransferase", "Alkaline phosphatase" )) %>%
  inner_join(select(df_covariates, eid, SEX), by = "eid") %>%
  mutate(
    Elevated_AST = if_else(SEX == "Female" & `Aspartate aminotransferase` > 35 | SEX == "Male" & `Aspartate aminotransferase` > 50, 1, 0),
    Elevated_ALT = if_else(SEX == "Female" & `Alanine aminotransferase` > 35 | SEX == "Male" & `Alanine aminotransferase` > 50, 1, 0),
    Elevated_GGT = if_else(SEX == "Female" & `Gamma glutamyltransferase` > 40 | SEX == "Male" & `Gamma glutamyltransferase` > 60, 1, 0),
    Elevated_AP = if_else(SEX == "Female" & `Alkaline phosphatase` > 105 | SEX == "Male" & `Alkaline phosphatase` > 130, 1, 0)
  ) %>%
  mutate(Elevated_Liver_Enzymes = if_else(Elevated_AST == 1 | Elevated_ALT == 1 | Elevated_GGT == 1, 1, 0)) %>%
  select(-c("Aspartate aminotransferase", "Alanine aminotransferase", "Gamma glutamyltransferase", "SEX", "Alkaline phosphatase"))
  
}



df_diagnosis <- inner_join(df_blood_risk, df_diagnosis, by="eid")
vec_blood_risk <- colnames(select(df_blood_risk, -eid))


df_par <- filter_rows_with_pos_entries(df_diagnosis)
```

```
## [1] "Current patients-at-risk-setting includes the following groups:"
## [1] "BTD"              "CLD"              "Cirrhosis"        "Viral Hepatitis" 
## [5] "Cholelithiasis"   "Blood_Parameters"
```

```
## df_diagnosis contains 26334 Patients at risk.
```

```r
#innerjoin_df_y(df_test_par)
```


#### 3. Metabolomics data for 250k patients df_metabolomics


```r
#Import metabolomics (data for n=106804 patients, obtained from EDTA plasma from baseline recruitment)
#df_metabolomics<- fread("ukb674682.csv", select=c("eid", "23474-0.0","23475-0.0","23476-0.0","23477-0.0","23460-0.0","23479-0.0","23440-0.0","23439-0.0","23441-0.0","23433-0.0","23432-0.0","23431-0.0","23484-0.0","23526-0.0","23561-0.0","23533-0.0","23498-0.0","23568-0.0","23540-0.0","23505-0.0","23575-0.0","23547-0.0","23512-0.0","23554-0.0","23491-0.0","23519-0.0","23580-0.0","23610-0.0","23635-0.0","23615-0.0","23590-0.0","23640-0.0","23620-0.0","23595-0.0","23645-0.0","23625-0.0","23600-0.0","23630-0.0","23585-0.0","23605-0.0","23485-0.0","23418-0.0","23527-0.0","23417-0.0","23562-0.0","23534-0.0","23499-0.0","23569-0.0","23541-0.0","23506-0.0","23576-0.0","23548-0.0","23513-0.0","23416-0.0","23555-0.0","23492-0.0","23520-0.0","23581-0.0","23611-0.0","23636-0.0","23616-0.0","23591-0.0","23641-0.0","23621-0.0","23596-0.0","23646-0.0","23626-0.0","23601-0.0","23631-0.0","23586-0.0","23606-0.0","23473-0.0","23404-0.0","23481-0.0","23430-0.0","23523-0.0","23429-0.0","23558-0.0","23530-0.0","23495-0.0","23565-0.0","23537-0.0","23502-0.0","23572-0.0","23544-0.0","23509-0.0","23428-0.0","23551-0.0","23488-0.0","23516-0.0","23478-0.0","23443-0.0","23450-0.0","23457-0.0","23486-0.0","23422-0.0","23528-0.0","23421-0.0","23563-0.0","23535-0.0","23500-0.0","23570-0.0","23542-0.0","23507-0.0","23577-0.0","23549-0.0","23514-0.0","23420-0.0","23556-0.0","23493-0.0","23521-0.0","23582-0.0","23612-0.0","23637-0.0","23617-0.0","23592-0.0","23642-0.0","23622-0.0","23597-0.0","23647-0.0","23627-0.0","23602-0.0","23632-0.0","23587-0.0","23607-0.0","23470-0.0","23461-0.0","23462-0.0","23480-0.0","23406-0.0","23463-0.0","23465-0.0","23405-0.0","23471-0.0","23466-0.0","23449-0.0","23456-0.0","23447-0.0","23454-0.0","23444-0.0","23451-0.0","23445-0.0","23459-0.0","23452-0.0","23468-0.0","23437-0.0","23434-0.0","23483-0.0","23414-0.0","23525-0.0","23413-0.0","23560-0.0","23532-0.0","23497-0.0","23567-0.0","23539-0.0","23504-0.0","23574-0.0","23546-0.0","23511-0.0","23412-0.0","23553-0.0","23490-0.0","23518-0.0","23579-0.0","23609-0.0","23634-0.0","23614-0.0","23589-0.0","23639-0.0","23619-0.0","23594-0.0","23644-0.0","23624-0.0","23599-0.0","23629-0.0","23584-0.0","23604-0.0","23446-0.0","23458-0.0","23453-0.0","23472-0.0","23402-0.0","23448-0.0","23455-0.0","23438-0.0","23400-0.0","23401-0.0","23436-0.0","23464-0.0","23427-0.0","23415-0.0","23442-0.0","23419-0.0","23482-0.0","23426-0.0","23524-0.0","23425-0.0","23559-0.0","23531-0.0","23496-0.0","23423-0.0","23566-0.0","23538-0.0","23503-0.0","23573-0.0","23545-0.0","23510-0.0","23424-0.0","23552-0.0","23489-0.0","23517-0.0","23411-0.0","23407-0.0","23487-0.0","23410-0.0","23529-0.0","23409-0.0","23564-0.0","23536-0.0","23501-0.0","23571-0.0","23543-0.0","23508-0.0","23578-0.0","23550-0.0","23515-0.0","23408-0.0","23557-0.0","23494-0.0","23522-0.0","23435-0.0","23583-0.0","23613-0.0","23638-0.0","23618-0.0","23593-0.0","23643-0.0","23623-0.0","23598-0.0","23648-0.0","23628-0.0","23603-0.0","23633-0.0","23588-0.0","23608-0.0","23469-0.0","23403-0.0","23467-0.0"))
#write.csv(df_metabolomics, file="C:/Users/Jan/OneDrive/Dokumente/PostDoc/Patient_tables/UKB_Patients_metabolomics.csv", row.names=FALSE)    

setwd(sharepoint_ukb)
df_metabolomics <- fread("extracted/metabolomics250k.csv") %>% check_and_remove_withdrawals(df_withdrawals)
```

```
## Warning in check_and_remove_withdrawals(., df_withdrawals): Patients with
## withdrawn consent have been removed from the dataframe. Rows before: 502359,
## Rows after: 502309, Rows removed: 50
```

```r
metabolomics_index <- read_excel(master_table, sheet="NMR_Metabolomics")

# Select non-Na participants only  
df_metabolomics <- na.omit(df_metabolomics)
df_metabolomics$V1 <- NULL #Remove unnecessary v1 column
nrow(df_metabolomics) #correlates to number of participants still included (should be 248286, 248266 after withdrawal removal)
```

```
## [1] 248266
```

```r
#Subset for directly measured metabolites
df_metabolomics <-  setnames(df_metabolomics, old = metabolomics_index$datafield1, new = metabolomics_index$name_processing, skip_absent=TRUE)
summary(df_metabolomics[,1:5])
```

```
##       eid          3-Hydroxybutyrate    Acetate         Acetoacetate     
##  Min.   :1000051   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:2259570   1st Qu.:0.02953   1st Qu.:0.01155   1st Qu.:0.006334  
##  Median :3512646   Median :0.04330   Median :0.01495   Median :0.009952  
##  Mean   :3512846   Mean   :0.06073   Mean   :0.01601   Mean   :0.013107  
##  3rd Qu.:4767172   3rd Qu.:0.06846   3rd Qu.:0.01889   3rd Qu.:0.015802  
##  Max.   :6024584   Max.   :2.33070   Max.   :1.14020   Max.   :0.662590  
##     Acetone        
##  Min.   :0.003475  
##  1st Qu.:0.011106  
##  Median :0.012910  
##  Mean   :0.014196  
##  3rd Qu.:0.015629  
##  Max.   :0.268630
```

```r
#Set names of metabolites
#Order the columns
col_order <- match(metabolomics_index$name_processing, colnames(df_metabolomics))
col_order <- c(1, col_order)
df_metabolomics <- as.data.frame(df_metabolomics)
df_metabolomics <- df_metabolomics[,col_order]

df_metabolomics <- subset(df_metabolomics[1:144]) 

vec_metabolomics <- setdiff(colnames(df_metabolomics), "eid") #this gets important for minmax normalization later on

#sanity(df_metabolomics)
```

#### 4. Genomics


```r
# Set up vector with all chromosomes where there are RAW files
# create "empty" dataframe with eids to later store SNPs
# Collect info from RAW files in for loop and build df with all snps
# duplicate df with merging on "eid" (This creates a _x and _y column for every SNP)
# rename _x and _y in _hom and _het and change scale to 0-1



#Set up a folder inside your project folder called genetics. Inside this folder you store the RAW files for the chromosomes. The following lines of code should put the numberof the chromosomes for which there are Raw files into the vector vec

setwd(project_path)
df_snp <- df_eid
files <- list.files("genetics")
vec <- sort(
  unlist(
    lapply(
      strsplit(files, "_"),
      function(x) as.numeric(gsub("[^0-9]", "", x)))))
df_snp <- df_eid  #create "empty" dataframe with eids to later store SNPs

setwd(sharepoint_ukb)
for (i in vec) {  #Loop through each SNP
  print(i)  #control if loop is working
  snp <- data.frame(read.csv(paste(project_path, "/genetics/", DOI, "_snps_chr",i, ".raw", sep=""), sep=""))  %>% #import snp, change this name according to your file names!
    rename(c("eid"="FID")) %>%  #eid should always be called eid for merging etc...
    dplyr::select(-c("IID", "PAT", "MAT", "SEX", "PHENOTYPE"))  # only eid and rsXYZ are relevant later on.
  df_snp <- merge(df_snp, snp, by.x="eid", by.y="eid", all.x=TRUE)  #append snps to df
}
```

```
## [1] 1
## [1] 3
## [1] 6
## [1] 7
## [1] 10
## [1] 12
## [1] 14
## [1] 19
```

```r
# Convert the numeric values to factors with desired levels
df_snp[,2:ncol(df_snp)] <- lapply(df_snp[,2:ncol(df_snp)], function(column) {
  factor(column, levels=c(0, 1, 2), labels=c("wt", "het", "hom"))
})

print("NAs counted per column before imputation:")
```

```
## [1] "NAs counted per column before imputation:"
```

```r
na_snp_col <- data.frame(na_columnwise(df_snp))
```

```
## [1] "eid : 0"
## [1] "rs1801131_G : 15297"
## [1] "rs3197999_A : 18541"
## [1] "rs1800629_A : 14882"
## [1] "rs1800795_C : 15109"
## [1] "rs3740066_T : 15331"
## [1] "rs4925_A : 14813"
## [1] "rs2617167_A : 14615"
## [1] "rs28929474_T : 14675"
## [1] "rs3212986_A : 14610"
```

```r
# Handle NAs by imputing the most common value or removing
if (na_mode == "impute") {
  df_snp <- impute_snp(df_snp)
  
} else if (na_mode == "remove") {     # Your removal code here
  df_snp <- omit.NA(df_snp, 10)       # Delete rows with too many NAs
}

print("NAs counted per column after imputation:")
```

```
## [1] "NAs counted per column after imputation:"
```

```r
na_columnwise(df_snp)
```

```
## [1] "eid : 0"
## [1] "rs1801131_G : 0"
## [1] "rs3197999_A : 0"
## [1] "rs1800629_A : 0"
## [1] "rs1800795_C : 0"
## [1] "rs3740066_T : 0"
## [1] "rs4925_A : 0"
## [1] "rs2617167_A : 0"
## [1] "rs28929474_T : 0"
## [1] "rs3212986_A : 0"
```

```
##          eid  rs1801131_G  rs3197999_A  rs1800629_A  rs1800795_C  rs3740066_T 
##            0            0            0            0            0            0 
##     rs4925_A  rs2617167_A rs28929474_T  rs3212986_A 
##            0            0            0            0
```

```r
#df_snp <- df_snp %>%
 #   dplyr::select(-c("rs429358_C"))   #delete columns with mostly NA values (rs429358_C)

#For linear instead of categorical model: change range from 0,1,2 to 0, 0.5, 1
# cols <- colnames(df_snp)[colnames(df_snp) != "eid"]
# df_snp[, (cols) := lapply(.SD, function(x) {
#   levels(x) <- c('0' = '0', '1' = '0.5', '2' = '1')
#   return(x)
# }), .SDcols = cols]

#sanity(df_snp)

#export df_snp
#write.csv(df_snp, file=paste(project_path, "/data/dataframes/df_snp.xlsx")  
write.xlsx(na_snp_col, file=paste(project_path, "/supplement/NA_count_df_snp.xlsx", sep=''))
```

#### 5. Radiomics


```r
# 7. MRI data df_mri \~50000

#mri_bulk <- fread("ukb52200.csv", select=c("eid", '20204-2.0','20254-2.0','20203-2.0'))
#write.csv(mri_bulk, file="C:/Users/Jan/OneDrive/Dokumente/PostDoc/Patient_tables/mri_bulk.csv", row.names=FALSE)
# mri_bulk <- fread("Patient_tables/mri_bulk.csv")
# mri_index <- read_excel("UKB MRI/MRI_Abdomen.xlsx")
# setnames(mri_bulk, old = mri_index$datafield1, new = mri_index$Description, skip_absent=TRUE)
# mri_bulk[mri_bulk == ""] <- NA                                #substitutes empty cells for NA
# mri_bulk <- mri_bulk[rowSums(is.na(mri_bulk)) < 3,]         #deletes all rows where all MRI columns are empty
# df_mri <- mri_bulk
```

# Fusion:

#### Merging (and potential normalization) function


```r
# #This code will apply the merge() function to each pair of data frames in the list list(df_eid, df_covariates, df_diagnosis, df_blood, df_snp), merging them based on the "eid" column.

today <- format(Sys.Date(), "%d_%m_%Y")

# Define overarching function
merge_dataframes <- function(include_metabolomics = FALSE, filter_par = FALSE, normalize = FALSE) {
        
        convert_eid_to_integer <- function(df) { # Function to ensure eid is an integer
          df$eid <- as.integer(df$eid)
          return(df)
        }
      
        # Initial list of dataframes to merge
        dfs_to_merge <- list(df_eid, df_covariates, df_diagnosis, df_blood, df_blood_risk, df_snp)
      
        
        dfs_to_merge <- lapply(dfs_to_merge, convert_eid_to_integer) # Convert 'eid' in each dataframe to integer
      
        
        if (include_metabolomics) { # Conditionally add df_metabolomics and/or else
          dfs_to_merge <- c(dfs_to_merge, list(df_metabolomics))
        }
        df_merged <- Reduce(function(x, y) merge(x, y, by = "eid", all = FALSE), dfs_to_merge)  # Merging the dataframes
        
        # Conditionally apply normalization function
        if (normalize==TRUE) {
          df_merged <- normalize_data(df_merged)
          if (!is.integer(df_merged$eid)) {
          stop("Error: 'eid' column is no longer an integer after normalization.")
          }
        }
        df_merged <- df_merged %>% select(-all_of(diag_codes)) # Removing the column of interest
          
        # Filter the "population at risk (par) by a prespecified if required
        if (filter_par) {
          df_merged <- filter_rows_with_pos_entries(df_merged)
        }
        df_merged <- df_merged %>% select(-all_of(vec_blood_risk))
        
        # Determine the group status
        if (include_metabolomics) {
            col_subset <<- "met"
        } else {
            col_subset <<- "basic"
        }
        if (filter_par) {
            row_subset <<- "par"
        } else {
            row_subset <<- "all"
        }
        
        if (normalize) {
          assign("raw", "", envir = .GlobalEnv)
        } else {
          assign("raw", "_raw", envir = .GlobalEnv)
        }

        # Remove NAs and return the dataframe
        return(na.omit(df_merged))
      }
```

# Process and save-function: Defining inner/outer layer, merging with y, creating summary report

```r
#Initialize the summary report
global_summary <- data.frame(
  Subset = character(),
  Layer = character(),
  Patients = numeric(),
  DOI = numeric(),
  stringsAsFactors = FALSE
)

setwd(project_path)

save_dir <- file.path(project_path, "data", today)
  
  # Check if the directory exists; if not, create it
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }

#Define process and save function
process_and_save <- function(df_x, subset_name) {
  # Generate the desired directory path
  

  # if (row_subset == "par") {
  #         df_x <- filter_rows_with_pos_entries(df_x, vec_risk_constellation)
  # }
          
  for (layer in c("INNER", "OUTER")) {
    df_x_temp <- df_x  # Work with a copy of df_x to avoid modifying the original df_x

    if (layer == "INNER") {
      df_y <- read.csv(paste(project_path, "/data/dataframes/df_y.csv", sep='')) %>%
        subset(split_ext == 0) %>%
        select(c("eid", "split_int", "status", "date_of_diag", "assessment", "difftime"))
      #df_y$eid <- as.integer(df$eid)
      print("Inner layer:")
    } 
    
    else {
      df_y <- read.csv(paste(project_path, "/data/dataframes/df_y.csv", sep='')) %>%
        select(c("eid", "split_ext", "status", "date_of_diag", "assessment", "difftime")) %>%
        subset(split_ext == 1)
      #df_y$eid <- as.integer(df$eid)
      cat("\n")
      print("Outer layer:")
    }
    
    df_temp <- df_x_temp %>%
        inner_join(df_y, by = "eid")
    nr_doi <- sum(df_temp$status)             #Amount of positive cases
    nr_all <- length(unique(df_temp$eid))       #All rows  
    
    # Update the global dataframe
    new_entries <- data.frame(
      Subset = subset_name,
      Layer = layer,
      Patients = nr_all,
      Patients_with_DOI = nr_doi,
      stringsAsFactors = FALSE
    )
    global_summary <<- rbind(global_summary, new_entries)
    print(paste("Patients in", row_subset, "subset:", nr_all))
    print(paste("Thereof patients with ", DOI, ": ", nr_doi))
    df_x_temp <- df_temp %>%                  # Select all BUT the status (= info to be predicted)
      select(-(c("status")))      #select(-(c("status", "Date of assessment")))
    df_y_temp <- df_temp %>%                  # Select JUST the status
      select(c("eid", "status", "date_of_diag", "assessment", "difftime"))  
    
    write.csv(df_x_temp, file=paste(project_path, "/data/", today, "/X_", tolower(layer), "_", col_subset, "_", subset_name, raw, ".csv", sep=""), row.names=FALSE)
    write.csv(df_y_temp, file=paste(project_path, "/data/", today, "/y_", tolower(layer), "_", col_subset, "_", subset_name, raw, ".csv", sep=""), row.names=FALSE)
    
    
    
    ### To be implemented: Adapt export to RDtata format: (Works, but implementation in Python not yet)
    
    # save(df_x_temp, file=paste0(project_path, "/data/", today, "/X_", tolower(layer), "_", col_subset, "_", subset_name, "_", raw, ".RData"))
    # save(df_y_temp, file=paste0(project_path, "/data/", today, "/y_", tolower(layer), "_", col_subset, "_", subset_name, "_", raw, ".RData"))
    
    # Assign the dataframes to global variables for manual inspection
    #assign(paste0("df_X_check", row_subset, col_subset, layer), df_x_temp, envir = .GlobalEnv)
    #assign(paste0("df_y_check", row_subset, col_subset, layer), df_y_temp, envir = .GlobalEnv)
  }
}
```


Execution of normalized merge/process_and save (always execute both rows! If not, the global variable "row_subset" will not be set correctly)

### Here, we have to merge X and y for once, to make sure that 
a) for X, all patients with DOI before ukb visit are excluded
b) for y, that all patients with missing data are excluded from y as well
c) in the end, both tables have the same length


### Normalization function

```r
vec_covariates <- c("MultipleDeprivationIndex", "Pack years", "Waist circumference", # Columns for "simple" Minmax in covariates
                  "Weight", "Standing height", "Alk_g_d", "Bloodpressure", "BMI", "AGE", "Handgripstrength")
vec_pc <- c("PC1", "PC2", "PC3", "PC4", "PC5")
#vec_blood / vec_metabolomics are already defined above
vec_all <- unique(c(vec_covariates, vec_pc, vec_blood, vec_metabolomics)) 
  

# Function to normalize data (deployed later inside merge function)
normalize_data <- function(df) {
  for (i in vec_pc) { 
    df[[i]] <- df[[i]] + abs(min(df[[i]], na.rm = TRUE)) # Adjust PC columns to get positive values
  }
  for (i in vec_all) {
    if (i %in% names(df) && is.numeric(df[[i]])) {  # Check if the column exists and is numeric
      df[[i]] <- minmax(df[[i]]) # Apply minmax normalization
    }
  }
  return(df)
}
```

# Export normalized data

```r
export_normalized <- TRUE

par_subset <- project_configs[[project_key]]$par_subset #load project-specific subsets of patients-at-risk

if (export_normalized) {
  
  # For all data (without filtering by chronic liver disease and without metabolomics)
  df_x_all <- merge_dataframes(normalize=TRUE)
  process_and_save(df_x_all, "all")
  
  # For data including metabolomics
  df_x_all_met <- merge_dataframes(include_metabolomics = TRUE, normalize=TRUE)
  process_and_save(df_x_all_met, "all")
  
  # For data filtered by chronic liver disease
  df_x_par <- merge_dataframes(filter_par = TRUE, normalize=TRUE)
  process_and_save(df_x_par, "par")
  write.csv(df_x_par$eid, file = file.path(project_path, "data", today, "par_eids.csv"), row.names = FALSE)


  
  # For data filtered by chronic liver  + metabolomics
  df_x_par_met <- merge_dataframes(filter_par = TRUE, include_metabolomics=TRUE, normalize=TRUE)
  process_and_save(df_x_par_met, "par")
  
  par_subset <- "CLD"
  df_x_par <- merge_dataframes(filter_par = TRUE, normalize=TRUE)
  process_and_save(df_x_par, "par_CLD")
  write.csv(df_x_par$eid, file = file.path(project_path, "data", today, "par_CLD_eids.csv"), row.names = FALSE)

}
```

```
## [1] "Inner layer:"
## [1] "Patients in all subset: 408614"
## [1] "Thereof patients with  CCa :  679"
## 
## [1] "Outer layer:"
## [1] "Patients in all subset: 93612"
## [1] "Thereof patients with  CCa :  169"
## [1] "Inner layer:"
## [1] "Patients in all subset: 211231"
## [1] "Thereof patients with  CCa :  363"
## 
## [1] "Outer layer:"
## [1] "Patients in all subset: 37000"
## [1] "Thereof patients with  CCa :  72"
## [1] "Current patients-at-risk-setting includes the following groups:"
## [1] "BTD"              "CLD"              "Cirrhosis"        "Viral Hepatitis" 
## [5] "Cholelithiasis"   "Blood_Parameters"
```

```
## df_merged contains 143927 Patients at risk.
```

```
## [1] "Inner layer:"
## [1] "Patients in par subset: 115017"
## [1] "Thereof patients with  CCa :  344"
## 
## [1] "Outer layer:"
## [1] "Patients in par subset: 28837"
## [1] "Thereof patients with  CCa :  93"
## [1] "Current patients-at-risk-setting includes the following groups:"
## [1] "BTD"              "CLD"              "Cirrhosis"        "Viral Hepatitis" 
## [5] "Cholelithiasis"   "Blood_Parameters"
```

```
## df_merged contains 73064 Patients at risk.
```

```
## [1] "Inner layer:"
## [1] "Patients in par subset: 61362"
## [1] "Thereof patients with  CCa :  189"
## 
## [1] "Outer layer:"
## [1] "Patients in par subset: 11672"
## [1] "Thereof patients with  CCa :  39"
```

```
## Warning in check_current_par(): Provided subset does not match the expected
## subset for the project key. Proceeding with provided subset.
```

```
## [1] "Current patients-at-risk-setting includes the following groups:"
## [1] "CLD"
```

```
## df_merged contains 4519 Patients at risk.
```

```
## [1] "Inner layer:"
## [1] "Patients in par subset: 3668"
## [1] "Thereof patients with  CCa :  55"
## 
## [1] "Outer layer:"
## [1] "Patients in par subset: 823"
## [1] "Thereof patients with  CCa :  9"
```

```r
par_subset
```

```
## [1] "CLD"
```
# Export absolute data / Create original dataframe X for Table 1 of Publication (Overview of patient data)

```r
export_absolute <- TRUE

if (export_absolute) {

# For all data (without filtering by chronic liver disease and without metabolomics)
df_x_all_raw <- merge_dataframes(normalize=FALSE)
process_and_save(df_x_all_raw, "all")

# For data including metabolomics
df_x_all_met_raw <- merge_dataframes(include_metabolomics = TRUE, normalize=FALSE)
process_and_save(df_x_all_met_raw, "all_met")

# For data filtered by chronic liver disease
df_x_par_raw <- merge_dataframes(filter_par = TRUE, normalize=FALSE)
process_and_save(df_x_par_raw, "par")

# For data filtered by chronic liver  + metabolomics
df_x_par_met_raw <- merge_dataframes(filter_par = TRUE, include_metabolomics=TRUE, normalize=FALSE)
process_and_save(df_x_par_met_raw, "par_met")
}
```

```
## [1] "Inner layer:"
## [1] "Patients in all subset: 408614"
## [1] "Thereof patients with  CCa :  679"
## 
## [1] "Outer layer:"
## [1] "Patients in all subset: 93612"
## [1] "Thereof patients with  CCa :  169"
## [1] "Inner layer:"
## [1] "Patients in all subset: 211231"
## [1] "Thereof patients with  CCa :  363"
## 
## [1] "Outer layer:"
## [1] "Patients in all subset: 37000"
## [1] "Thereof patients with  CCa :  72"
```

```
## Warning in check_current_par(): Provided subset does not match the expected
## subset for the project key. Proceeding with provided subset.
```

```
## [1] "Current patients-at-risk-setting includes the following groups:"
## [1] "CLD"
```

```
## df_merged contains 4519 Patients at risk.
```

```
## [1] "Inner layer:"
## [1] "Patients in par subset: 3668"
## [1] "Thereof patients with  CCa :  55"
## 
## [1] "Outer layer:"
## [1] "Patients in par subset: 823"
## [1] "Thereof patients with  CCa :  9"
```

```
## Warning in check_current_par(): Provided subset does not match the expected
## subset for the project key. Proceeding with provided subset.
```

```
## [1] "Current patients-at-risk-setting includes the following groups:"
## [1] "CLD"
```

```
## df_merged contains 2141 Patients at risk.
```

```
## [1] "Inner layer:"
## [1] "Patients in par subset: 1804"
## [1] "Thereof patients with  CCa :  28"
## 
## [1] "Outer layer:"
## [1] "Patients in par subset: 329"
## [1] "Thereof patients with  CCa :  6"
```


# Summary report (Excel file with count per layer/status, and metadata)

```r
#Summary report
grouped_summary <- global_summary %>%
  group_by(Subset) %>%
  summarise(
    Total_Patients = sum(Patients),
    Total_Patients_with_DOI = sum(Patients_with_DOI)
  )
df_risk_constellation <- data.frame(Risk_constellation = vec_risk_constellation)
df_diags <- data.frame(Integrated_ICD = IOIs)

sheets_list <- list(
  "Per_Layer_Summary" = global_summary,
  "All_Summary" = grouped_summary,
  "Risk Constellation" = df_risk_constellation,
  "Integrated Diagnosis" = df_diags
)
write_xlsx(sheets_list, path = (paste(project_path, "/data/", today, "/summary_reports.xlsx", sep = "")))
```

# Create columngroups table (creates table with column names and to which initial df they belong (necessary for feature importance later))

```r
# Create a list of the dataframes and their names
dfs = list(df_eid=df_eid, df_covariates=df_covariates, df_diagnosis=df_diagnosis, df_blood=df_blood, df_snp=df_snp, df_metabolomics=df_metabolomics)

# Use lapply to loop over the dataframes and create the temporary dataframes
temp_dfs = lapply(names(dfs), function(df_name) {
  data.frame(column_name = colnames(dfs[[df_name]]), source_df = df_name)
})

# Combine the temporary dataframes into one dataframe
df_columngroups = do.call(rbind, temp_dfs) %>%
  subset(column_name != "eid") %>%
  rbind(data.frame(column_name = "split_int", source_df = "df_metadata")) %>%
  rbind(data.frame(column_name = "split_ext", source_df = "df_metadata")) %>%
  mutate(source_df = ifelse(startsWith(column_name, "PC"), "df_snp", source_df)) #Manually change columngroups relationship as needed

today <- format(Sys.Date(), "%d_%m_%Y")
write.csv(df_columngroups, paste(project_path, "/data/", today, "/columngroups.csv", sep=""), row.names=FALSE)

rm(dfs) #Inefficient way for creating columngroups, uses lots of memory
```




# Table Export Preparation

Prepare and modify the df_x_all_raw df to be more visually appealing
when creating the table 1 for the paper


```r
setwd(project_path)
load("data/dataframes/df_y.RData") 
df_y <- df_y %>% select(c("eid", "status"))

covariates_list <- df_columngroups[df_columngroups$source_df == "df_covariates", "column_name"]
icd_list <- df_columngroups[df_columngroups$source_df == "df_diagnosis", "column_name"]
metabolomics_list <- df_columngroups[df_columngroups$source_df == "df_metabolomics", "column_name"]
blood_list <- df_columngroups[df_columngroups$source_df == "df_blood", "column_name"]
snp_list <- df_columngroups[df_columngroups$source_df == "df_snp", "column_name"]

label_list <- list(
  location_name = "Location",
  location_country = "Country",
  Family_diabetes = "Family Diabetes",
  Alk_g_d = "Alcohol (gram per day)",
  High_Alk = "High Alcohol Consumption",
  Path_Alk = "Pathological Alcohol Consumption",
  BMI_cat = "BMI Categories",
  Bloodpressure = "Bloodpressure sys. [mmHg]",
  Weight = "Weight [kg]"
)

table1_order <- c(
  "AGE", "SEX", "Ethnicity", "BMI", "BMI_cat", "Waist circumference", 
  "Weight", "Standing height", "Handgripstrength", 
  "MultipleDeprivationIndex", "Bloodpressure", "Medication", 
  "Family_diabetes", "Smoking status", "Ever smoked", "Pack years" # Continue as necessary
)



table1b_order <- c(
  "AGE", "SEX", "Ethnicity", "BMI", "Waist circumference", 
  "Weight", "Standing height", "Handgripstrength", 
  "MultipleDeprivationIndex", "Bloodpressure", "Medication", "Pack years" # Continue as necessary
)
```

# Table Creation All


```r
head_only <- FALSE
# Change "all" to "par" if you want to create tables of just the "patient at risk subset
df_all <- df_x_all_raw %>% inner_join(df_y, by = "eid")
df_all$status <- ifelse(df_all$status == 0, paste("No ", DOI), 
                                  ifelse(df_all$status == 1, DOI, df_all$status))

df_tbl_1 <- df_all %>%
  select(!any_of(c(blood_list, metabolomics_list, icd_list, "eid", "Date of assessment", "UKB assessment centre", snp_list)))

create_table(df_tbl_1, "Table 1", export_RDS=TRUE, head_only=head_only, remove_SEX=FALSE,  enforced_order=table1_order)
```

```
## Warning for variable 'Ethnicity':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 2L, 2L, 2L, 2L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'BMI_cat':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 2L, 3L, 3L, 3L, 4L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'Smoking status':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 1L, 2L, 2L, 1L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'Ever smoked':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 1L, 2L, 2L, 1L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="avebfioppy" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#avebfioppy table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #avebfioppy thead, #avebfioppy tbody, #avebfioppy tfoot, #avebfioppy tr, #avebfioppy td, #avebfioppy th {
##   border-style: none;
## }
## 
## #avebfioppy p {
##   margin: 0;
##   padding: 0;
## }
## 
## #avebfioppy .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #avebfioppy .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #avebfioppy .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #avebfioppy .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #avebfioppy .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #avebfioppy .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #avebfioppy .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #avebfioppy .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #avebfioppy .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #avebfioppy .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #avebfioppy .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #avebfioppy .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #avebfioppy .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #avebfioppy .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #avebfioppy .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #avebfioppy .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #avebfioppy .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #avebfioppy .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #avebfioppy .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #avebfioppy .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #avebfioppy .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #avebfioppy .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #avebfioppy .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #avebfioppy .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #avebfioppy .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #avebfioppy .gt_left {
##   text-align: left;
## }
## 
## #avebfioppy .gt_center {
##   text-align: center;
## }
## 
## #avebfioppy .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #avebfioppy .gt_font_normal {
##   font-weight: normal;
## }
## 
## #avebfioppy .gt_font_bold {
##   font-weight: bold;
## }
## 
## #avebfioppy .gt_font_italic {
##   font-style: italic;
## }
## 
## #avebfioppy .gt_super {
##   font-size: 65%;
## }
## 
## #avebfioppy .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #avebfioppy .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #avebfioppy .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #avebfioppy .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #avebfioppy .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #avebfioppy .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #avebfioppy .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 502226&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 502226<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 848&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 848<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 501378&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 501378<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">AGE</td>
## <td headers="stat_0" class="gt_row gt_center">56.5 (Â±8.1)</td>
## <td headers="stat_1" class="gt_row gt_center">61.1 (Â±6.2)</td>
## <td headers="stat_2" class="gt_row gt_center">56.5 (Â±8.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">SEX</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.009</td>
## <td headers="q.value" class="gt_row gt_center">0.14</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Female</td>
## <td headers="stat_0" class="gt_row gt_center">273,225 (54%)</td>
## <td headers="stat_1" class="gt_row gt_center">423 (50%)</td>
## <td headers="stat_2" class="gt_row gt_center">272,802 (54%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Male</td>
## <td headers="stat_0" class="gt_row gt_center">229,001 (46%)</td>
## <td headers="stat_1" class="gt_row gt_center">425 (50%)</td>
## <td headers="stat_2" class="gt_row gt_center">228,576 (46%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ethnicity</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">2,777 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">7 (0.8%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,770 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Caucasian</td>
## <td headers="stat_0" class="gt_row gt_center">472,441 (94%)</td>
## <td headers="stat_1" class="gt_row gt_center">806 (95%)</td>
## <td headers="stat_2" class="gt_row gt_center">471,635 (94%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Mixed</td>
## <td headers="stat_0" class="gt_row gt_center">2,953 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,951 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Asian or Asian british</td>
## <td headers="stat_0" class="gt_row gt_center">9,873 (2.0%)</td>
## <td headers="stat_1" class="gt_row gt_center">11 (1.3%)</td>
## <td headers="stat_2" class="gt_row gt_center">9,862 (2.0%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Black or Black british</td>
## <td headers="stat_0" class="gt_row gt_center">8,055 (1.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">10 (1.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">8,045 (1.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Chinese</td>
## <td headers="stat_0" class="gt_row gt_center">1,572 (0.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,569 (0.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Other</td>
## <td headers="stat_0" class="gt_row gt_center">4,555 (0.9%)</td>
## <td headers="stat_1" class="gt_row gt_center">9 (1.1%)</td>
## <td headers="stat_2" class="gt_row gt_center">4,546 (0.9%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI</td>
## <td headers="stat_0" class="gt_row gt_center">27.4 (Â±4.8)</td>
## <td headers="stat_1" class="gt_row gt_center">28.3 (Â±5.2)</td>
## <td headers="stat_2" class="gt_row gt_center">27.4 (Â±4.8)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI Categories</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Underweight</td>
## <td headers="stat_0" class="gt_row gt_center">2,624 (0.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">6 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,618 (0.5%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Normal weight</td>
## <td headers="stat_0" class="gt_row gt_center">157,321 (31%)</td>
## <td headers="stat_1" class="gt_row gt_center">210 (25%)</td>
## <td headers="stat_2" class="gt_row gt_center">157,111 (31%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Overweight</td>
## <td headers="stat_0" class="gt_row gt_center">214,104 (43%)</td>
## <td headers="stat_1" class="gt_row gt_center">363 (43%)</td>
## <td headers="stat_2" class="gt_row gt_center">213,741 (43%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Obese</td>
## <td headers="stat_0" class="gt_row gt_center">125,070 (25%)</td>
## <td headers="stat_1" class="gt_row gt_center">266 (31%)</td>
## <td headers="stat_2" class="gt_row gt_center">124,804 (25%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">3,107 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">3,104 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Waist circumference</td>
## <td headers="stat_0" class="gt_row gt_center">90.3 (Â±13.5)</td>
## <td headers="stat_1" class="gt_row gt_center">93.8 (Â±14.2)</td>
## <td headers="stat_2" class="gt_row gt_center">90.3 (Â±13.5)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Weight [kg]</td>
## <td headers="stat_0" class="gt_row gt_center">78.0 (Â±15.9)</td>
## <td headers="stat_1" class="gt_row gt_center">80.8 (Â±17.1)</td>
## <td headers="stat_2" class="gt_row gt_center">78.0 (Â±15.9)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Standing height</td>
## <td headers="stat_0" class="gt_row gt_center">168.4 (Â±9.3)</td>
## <td headers="stat_1" class="gt_row gt_center">168.7 (Â±9.4)</td>
## <td headers="stat_2" class="gt_row gt_center">168.4 (Â±9.3)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Handgripstrength</td>
## <td headers="stat_0" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="stat_1" class="gt_row gt_center">30.3 (Â±10.9)</td>
## <td headers="stat_2" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MultipleDeprivationIndex</td>
## <td headers="stat_0" class="gt_row gt_center">17.2 (Â±13.9)</td>
## <td headers="stat_1" class="gt_row gt_center">18.6 (Â±14.7)</td>
## <td headers="stat_2" class="gt_row gt_center">17.2 (Â±13.9)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.006</td>
## <td headers="q.value" class="gt_row gt_center">0.092</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Bloodpressure sys. [mmHg]</td>
## <td headers="stat_0" class="gt_row gt_center">137.7 (Â±18.2)</td>
## <td headers="stat_1" class="gt_row gt_center">141.8 (Â±18.5)</td>
## <td headers="stat_2" class="gt_row gt_center">137.7 (Â±18.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Medication</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    No Medication</td>
## <td headers="stat_0" class="gt_row gt_center">342,470 (68%)</td>
## <td headers="stat_1" class="gt_row gt_center">493 (58%)</td>
## <td headers="stat_2" class="gt_row gt_center">341,977 (68%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Metabolic</td>
## <td headers="stat_0" class="gt_row gt_center">138,177 (28%)</td>
## <td headers="stat_1" class="gt_row gt_center">328 (39%)</td>
## <td headers="stat_2" class="gt_row gt_center">137,849 (27%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Hormones</td>
## <td headers="stat_0" class="gt_row gt_center">21,579 (4.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">27 (3.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">21,552 (4.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Family Diabetes</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0" class="gt_row gt_center">407,764 (81%)</td>
## <td headers="stat_1" class="gt_row gt_center">689 (81%)</td>
## <td headers="stat_2" class="gt_row gt_center">407,075 (81%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0" class="gt_row gt_center">94,462 (19%)</td>
## <td headers="stat_1" class="gt_row gt_center">159 (19%)</td>
## <td headers="stat_2" class="gt_row gt_center">94,303 (19%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Smoking status</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Never</td>
## <td headers="stat_0" class="gt_row gt_center">273,386 (54%)</td>
## <td headers="stat_1" class="gt_row gt_center">373 (44%)</td>
## <td headers="stat_2" class="gt_row gt_center">273,013 (54%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Previous</td>
## <td headers="stat_0" class="gt_row gt_center">214,831 (43%)</td>
## <td headers="stat_1" class="gt_row gt_center">449 (53%)</td>
## <td headers="stat_2" class="gt_row gt_center">214,382 (43%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Current</td>
## <td headers="stat_0" class="gt_row gt_center">11,161 (2.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">21 (2.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">11,140 (2.2%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">2,848 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">5 (0.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,843 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ever smoked</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.003</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0" class="gt_row gt_center">200,749 (40%)</td>
## <td headers="stat_1" class="gt_row gt_center">280 (33%)</td>
## <td headers="stat_2" class="gt_row gt_center">200,469 (40%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0" class="gt_row gt_center">298,592 (59%)</td>
## <td headers="stat_1" class="gt_row gt_center">563 (66%)</td>
## <td headers="stat_2" class="gt_row gt_center">298,029 (59%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">2,885 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">5 (0.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,880 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Pack years</td>
## <td headers="stat_0" class="gt_row gt_center">8.3 (Â±14.6)</td>
## <td headers="stat_1" class="gt_row gt_center">12.4 (Â±18.3)</td>
## <td headers="stat_2" class="gt_row gt_center">8.3 (Â±14.6)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD); n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test; Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```r
df_tbl_1b <- df_all %>%
  select(!any_of(c(blood_list, metabolomics_list, icd_list, "eid", "Date of assessment", "UKB assessment centre", snp_list)))

create_table(df_tbl_1b, "Table 1b", export_RDS=TRUE, head_only=head_only, remove_SEX=FALSE,  enforced_order=table1b_order)
```

```
## Warning for variable 'Ethnicity':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 2L, 2L, 2L, 2L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="sidsvzqukv" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#sidsvzqukv table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #sidsvzqukv thead, #sidsvzqukv tbody, #sidsvzqukv tfoot, #sidsvzqukv tr, #sidsvzqukv td, #sidsvzqukv th {
##   border-style: none;
## }
## 
## #sidsvzqukv p {
##   margin: 0;
##   padding: 0;
## }
## 
## #sidsvzqukv .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #sidsvzqukv .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #sidsvzqukv .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #sidsvzqukv .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #sidsvzqukv .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #sidsvzqukv .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #sidsvzqukv .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #sidsvzqukv .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #sidsvzqukv .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #sidsvzqukv .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #sidsvzqukv .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #sidsvzqukv .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #sidsvzqukv .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #sidsvzqukv .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #sidsvzqukv .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #sidsvzqukv .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #sidsvzqukv .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #sidsvzqukv .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #sidsvzqukv .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #sidsvzqukv .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #sidsvzqukv .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #sidsvzqukv .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #sidsvzqukv .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #sidsvzqukv .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #sidsvzqukv .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #sidsvzqukv .gt_left {
##   text-align: left;
## }
## 
## #sidsvzqukv .gt_center {
##   text-align: center;
## }
## 
## #sidsvzqukv .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #sidsvzqukv .gt_font_normal {
##   font-weight: normal;
## }
## 
## #sidsvzqukv .gt_font_bold {
##   font-weight: bold;
## }
## 
## #sidsvzqukv .gt_font_italic {
##   font-style: italic;
## }
## 
## #sidsvzqukv .gt_super {
##   font-size: 65%;
## }
## 
## #sidsvzqukv .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #sidsvzqukv .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #sidsvzqukv .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #sidsvzqukv .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #sidsvzqukv .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #sidsvzqukv .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #sidsvzqukv .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 502226&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 502226<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 848&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 848<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 501378&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 501378<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">AGE</td>
## <td headers="stat_0" class="gt_row gt_center">56.5 (Â±8.1)</td>
## <td headers="stat_1" class="gt_row gt_center">61.1 (Â±6.2)</td>
## <td headers="stat_2" class="gt_row gt_center">56.5 (Â±8.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">SEX</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.009</td>
## <td headers="q.value" class="gt_row gt_center">0.11</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Female</td>
## <td headers="stat_0" class="gt_row gt_center">273,225 (54%)</td>
## <td headers="stat_1" class="gt_row gt_center">423 (50%)</td>
## <td headers="stat_2" class="gt_row gt_center">272,802 (54%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Male</td>
## <td headers="stat_0" class="gt_row gt_center">229,001 (46%)</td>
## <td headers="stat_1" class="gt_row gt_center">425 (50%)</td>
## <td headers="stat_2" class="gt_row gt_center">228,576 (46%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ethnicity</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">2,777 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">7 (0.8%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,770 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Caucasian</td>
## <td headers="stat_0" class="gt_row gt_center">472,441 (94%)</td>
## <td headers="stat_1" class="gt_row gt_center">806 (95%)</td>
## <td headers="stat_2" class="gt_row gt_center">471,635 (94%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Mixed</td>
## <td headers="stat_0" class="gt_row gt_center">2,953 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,951 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Asian or Asian british</td>
## <td headers="stat_0" class="gt_row gt_center">9,873 (2.0%)</td>
## <td headers="stat_1" class="gt_row gt_center">11 (1.3%)</td>
## <td headers="stat_2" class="gt_row gt_center">9,862 (2.0%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Black or Black british</td>
## <td headers="stat_0" class="gt_row gt_center">8,055 (1.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">10 (1.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">8,045 (1.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Chinese</td>
## <td headers="stat_0" class="gt_row gt_center">1,572 (0.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,569 (0.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Other</td>
## <td headers="stat_0" class="gt_row gt_center">4,555 (0.9%)</td>
## <td headers="stat_1" class="gt_row gt_center">9 (1.1%)</td>
## <td headers="stat_2" class="gt_row gt_center">4,546 (0.9%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI</td>
## <td headers="stat_0" class="gt_row gt_center">27.4 (Â±4.8)</td>
## <td headers="stat_1" class="gt_row gt_center">28.3 (Â±5.2)</td>
## <td headers="stat_2" class="gt_row gt_center">27.4 (Â±4.8)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Waist circumference</td>
## <td headers="stat_0" class="gt_row gt_center">90.3 (Â±13.5)</td>
## <td headers="stat_1" class="gt_row gt_center">93.8 (Â±14.2)</td>
## <td headers="stat_2" class="gt_row gt_center">90.3 (Â±13.5)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Weight [kg]</td>
## <td headers="stat_0" class="gt_row gt_center">78.0 (Â±15.9)</td>
## <td headers="stat_1" class="gt_row gt_center">80.8 (Â±17.1)</td>
## <td headers="stat_2" class="gt_row gt_center">78.0 (Â±15.9)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Standing height</td>
## <td headers="stat_0" class="gt_row gt_center">168.4 (Â±9.3)</td>
## <td headers="stat_1" class="gt_row gt_center">168.7 (Â±9.4)</td>
## <td headers="stat_2" class="gt_row gt_center">168.4 (Â±9.3)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Handgripstrength</td>
## <td headers="stat_0" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="stat_1" class="gt_row gt_center">30.3 (Â±10.9)</td>
## <td headers="stat_2" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MultipleDeprivationIndex</td>
## <td headers="stat_0" class="gt_row gt_center">17.2 (Â±13.9)</td>
## <td headers="stat_1" class="gt_row gt_center">18.6 (Â±14.7)</td>
## <td headers="stat_2" class="gt_row gt_center">17.2 (Â±13.9)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.006</td>
## <td headers="q.value" class="gt_row gt_center">0.069</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Bloodpressure sys. [mmHg]</td>
## <td headers="stat_0" class="gt_row gt_center">137.7 (Â±18.2)</td>
## <td headers="stat_1" class="gt_row gt_center">141.8 (Â±18.5)</td>
## <td headers="stat_2" class="gt_row gt_center">137.7 (Â±18.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Medication</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    No Medication</td>
## <td headers="stat_0" class="gt_row gt_center">342,470 (68%)</td>
## <td headers="stat_1" class="gt_row gt_center">493 (58%)</td>
## <td headers="stat_2" class="gt_row gt_center">341,977 (68%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Metabolic</td>
## <td headers="stat_0" class="gt_row gt_center">138,177 (28%)</td>
## <td headers="stat_1" class="gt_row gt_center">328 (39%)</td>
## <td headers="stat_2" class="gt_row gt_center">137,849 (27%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Hormones</td>
## <td headers="stat_0" class="gt_row gt_center">21,579 (4.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">27 (3.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">21,552 (4.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Pack years</td>
## <td headers="stat_0" class="gt_row gt_center">8.3 (Â±14.6)</td>
## <td headers="stat_1" class="gt_row gt_center">12.4 (Â±18.3)</td>
## <td headers="stat_2" class="gt_row gt_center">8.3 (Â±14.6)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD); n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test; Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

# Table Creation PAR

```r
head_only <- FALSE
df_all <- df_x_par_raw %>% inner_join(df_y, by = "eid")
df_all$status <- ifelse(df_all$status == 0, paste("No", DOI), 
                                  ifelse(df_all$status == 1, DOI, df_all$status))

df_tbl_1 <- df_all %>%
  select(!any_of(c(blood_list, metabolomics_list, icd_list, "eid", "Date of assessment", "UKB assessment centre", snp_list)))

create_table(df_tbl_1, "Table 1_par", export_RDS=TRUE, head_only=head_only, remove_SEX=FALSE,  enforced_order=table1_order)
```

```
## Warning for variable 'Ethnicity':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 2L, 2L, 2L, 2L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'BMI_cat':
## simpleWarning in stats::chisq.test(x = structure(c(3L, 4L, 4L, 2L, 3L, 4L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'Medication':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 2L, 2L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'Smoking status':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 1L, 1L, 1L, 1L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'Ever smoked':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 2L, 1L, 2L, 1L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="dhvkqztcej" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#dhvkqztcej table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #dhvkqztcej thead, #dhvkqztcej tbody, #dhvkqztcej tfoot, #dhvkqztcej tr, #dhvkqztcej td, #dhvkqztcej th {
##   border-style: none;
## }
## 
## #dhvkqztcej p {
##   margin: 0;
##   padding: 0;
## }
## 
## #dhvkqztcej .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #dhvkqztcej .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #dhvkqztcej .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #dhvkqztcej .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #dhvkqztcej .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #dhvkqztcej .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #dhvkqztcej .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #dhvkqztcej .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #dhvkqztcej .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #dhvkqztcej .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #dhvkqztcej .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #dhvkqztcej .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #dhvkqztcej .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #dhvkqztcej .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #dhvkqztcej .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #dhvkqztcej .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #dhvkqztcej .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #dhvkqztcej .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #dhvkqztcej .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #dhvkqztcej .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #dhvkqztcej .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #dhvkqztcej .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #dhvkqztcej .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #dhvkqztcej .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #dhvkqztcej .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #dhvkqztcej .gt_left {
##   text-align: left;
## }
## 
## #dhvkqztcej .gt_center {
##   text-align: center;
## }
## 
## #dhvkqztcej .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #dhvkqztcej .gt_font_normal {
##   font-weight: normal;
## }
## 
## #dhvkqztcej .gt_font_bold {
##   font-weight: bold;
## }
## 
## #dhvkqztcej .gt_font_italic {
##   font-style: italic;
## }
## 
## #dhvkqztcej .gt_super {
##   font-size: 65%;
## }
## 
## #dhvkqztcej .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #dhvkqztcej .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #dhvkqztcej .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #dhvkqztcej .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #dhvkqztcej .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #dhvkqztcej .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #dhvkqztcej .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 4491&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 4491<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 64&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 64<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 4427&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No CCa</strong></p>
## <p>n = 4427<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">AGE</td>
## <td headers="stat_0" class="gt_row gt_center">58.0 (Â±7.4)</td>
## <td headers="stat_1" class="gt_row gt_center">61.4 (Â±6.2)</td>
## <td headers="stat_2" class="gt_row gt_center">58.0 (Â±7.4)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">SEX</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.8</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Female</td>
## <td headers="stat_0" class="gt_row gt_center">1,941 (43%)</td>
## <td headers="stat_1" class="gt_row gt_center">29 (45%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,912 (43%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Male</td>
## <td headers="stat_0" class="gt_row gt_center">2,550 (57%)</td>
## <td headers="stat_1" class="gt_row gt_center">35 (55%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,515 (57%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ethnicity</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">31 (0.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">31 (0.7%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Caucasian</td>
## <td headers="stat_0" class="gt_row gt_center">4,190 (93%)</td>
## <td headers="stat_1" class="gt_row gt_center">60 (94%)</td>
## <td headers="stat_2" class="gt_row gt_center">4,130 (93%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Mixed</td>
## <td headers="stat_0" class="gt_row gt_center">25 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">25 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Asian or Asian british</td>
## <td headers="stat_0" class="gt_row gt_center">106 (2.4%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (1.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">105 (2.4%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Black or Black british</td>
## <td headers="stat_0" class="gt_row gt_center">59 (1.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (3.1%)</td>
## <td headers="stat_2" class="gt_row gt_center">57 (1.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Chinese</td>
## <td headers="stat_0" class="gt_row gt_center">15 (0.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (1.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">14 (0.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Other</td>
## <td headers="stat_0" class="gt_row gt_center">65 (1.4%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">65 (1.5%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI</td>
## <td headers="stat_0" class="gt_row gt_center">29.4 (Â±5.7)</td>
## <td headers="stat_1" class="gt_row gt_center">29.2 (Â±5.4)</td>
## <td headers="stat_2" class="gt_row gt_center">29.4 (Â±5.7)</td>
## <td headers="p.value" class="gt_row gt_center">0.7</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI Categories</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.067</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Underweight</td>
## <td headers="stat_0" class="gt_row gt_center">28 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">28 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Normal weight</td>
## <td headers="stat_0" class="gt_row gt_center">842 (19%)</td>
## <td headers="stat_1" class="gt_row gt_center">13 (20%)</td>
## <td headers="stat_2" class="gt_row gt_center">829 (19%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Overweight</td>
## <td headers="stat_0" class="gt_row gt_center">1,751 (39%)</td>
## <td headers="stat_1" class="gt_row gt_center">18 (28%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,733 (39%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Obese</td>
## <td headers="stat_0" class="gt_row gt_center">1,811 (40%)</td>
## <td headers="stat_1" class="gt_row gt_center">30 (47%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,781 (40%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">59 (1.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (4.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">56 (1.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Waist circumference</td>
## <td headers="stat_0" class="gt_row gt_center">97.4 (Â±14.2)</td>
## <td headers="stat_1" class="gt_row gt_center">96.9 (Â±14.3)</td>
## <td headers="stat_2" class="gt_row gt_center">97.4 (Â±14.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.8</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Weight [kg]</td>
## <td headers="stat_0" class="gt_row gt_center">84.0 (Â±17.8)</td>
## <td headers="stat_1" class="gt_row gt_center">83.6 (Â±16.5)</td>
## <td headers="stat_2" class="gt_row gt_center">84.0 (Â±17.9)</td>
## <td headers="p.value" class="gt_row gt_center">0.8</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Standing height</td>
## <td headers="stat_0" class="gt_row gt_center">168.9 (Â±9.1)</td>
## <td headers="stat_1" class="gt_row gt_center">169.0 (Â±9.0)</td>
## <td headers="stat_2" class="gt_row gt_center">168.9 (Â±9.1)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Handgripstrength</td>
## <td headers="stat_0" class="gt_row gt_center">30.5 (Â±11.0)</td>
## <td headers="stat_1" class="gt_row gt_center">30.1 (Â±10.6)</td>
## <td headers="stat_2" class="gt_row gt_center">30.5 (Â±11.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.8</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MultipleDeprivationIndex</td>
## <td headers="stat_0" class="gt_row gt_center">21.8 (Â±16.1)</td>
## <td headers="stat_1" class="gt_row gt_center">20.7 (Â±15.7)</td>
## <td headers="stat_2" class="gt_row gt_center">21.8 (Â±16.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Bloodpressure sys. [mmHg]</td>
## <td headers="stat_0" class="gt_row gt_center">139.4 (Â±18.2)</td>
## <td headers="stat_1" class="gt_row gt_center">143.9 (Â±19.2)</td>
## <td headers="stat_2" class="gt_row gt_center">139.3 (Â±18.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.061</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Medication</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    No Medication</td>
## <td headers="stat_0" class="gt_row gt_center">2,342 (52%)</td>
## <td headers="stat_1" class="gt_row gt_center">34 (53%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,308 (52%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Metabolic</td>
## <td headers="stat_0" class="gt_row gt_center">2,023 (45%)</td>
## <td headers="stat_1" class="gt_row gt_center">27 (42%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,996 (45%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Hormones</td>
## <td headers="stat_0" class="gt_row gt_center">126 (2.8%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (4.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">123 (2.8%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Family Diabetes</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0" class="gt_row gt_center">3,454 (77%)</td>
## <td headers="stat_1" class="gt_row gt_center">50 (78%)</td>
## <td headers="stat_2" class="gt_row gt_center">3,404 (77%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0" class="gt_row gt_center">1,037 (23%)</td>
## <td headers="stat_1" class="gt_row gt_center">14 (22%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,023 (23%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Smoking status</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.7</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Never</td>
## <td headers="stat_0" class="gt_row gt_center">1,956 (44%)</td>
## <td headers="stat_1" class="gt_row gt_center">31 (48%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,925 (43%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Previous</td>
## <td headers="stat_0" class="gt_row gt_center">2,357 (52%)</td>
## <td headers="stat_1" class="gt_row gt_center">32 (50%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,325 (53%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Current</td>
## <td headers="stat_0" class="gt_row gt_center">148 (3.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (1.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">147 (3.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">30 (0.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">30 (0.7%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ever smoked</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.7</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0" class="gt_row gt_center">1,508 (34%)</td>
## <td headers="stat_1" class="gt_row gt_center">24 (38%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,484 (34%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0" class="gt_row gt_center">2,953 (66%)</td>
## <td headers="stat_1" class="gt_row gt_center">40 (63%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,913 (66%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">30 (0.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">30 (0.7%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Pack years</td>
## <td headers="stat_0" class="gt_row gt_center">14.2 (Â±21.2)</td>
## <td headers="stat_1" class="gt_row gt_center">17.6 (Â±28.2)</td>
## <td headers="stat_2" class="gt_row gt_center">14.2 (Â±21.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD); n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test; Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

# Tables ICD/Blood

```r
head_only <- FALSE
df_all <- df_x_all_raw %>% inner_join(df_y, by = "eid")
df_all$status <- ifelse(df_all$status == 0, paste("No ", DOI), 
                                  ifelse(df_all$status == 1, DOI, df_all$status))


df_tbl_icd <- df_all %>%
  select(any_of(c("status", "SEX", icd_list)))

df_tbl_blood <- df_all %>%
  select(any_of(c("status", "SEX", blood_list)))



df_tbl_snp <- df_all %>%
  select(any_of(c("status", "SEX", snp_list)))


df_tbl_metabolomics <- df_x_all_met_raw %>% inner_join(df_y, by = "eid") %>%
  mutate(status = if_else(status == 1, DOI, paste0("No ", DOI))) %>%
  select(any_of(c("status", "SEX", metabolomics_list)))

  
head_only <- TRUE

create_table(df_tbl_blood, "Table Blood", export_RDS=TRUE, head_only=head_only)
```

```
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="nqvrdiqotu" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#nqvrdiqotu table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #nqvrdiqotu thead, #nqvrdiqotu tbody, #nqvrdiqotu tfoot, #nqvrdiqotu tr, #nqvrdiqotu td, #nqvrdiqotu th {
##   border-style: none;
## }
## 
## #nqvrdiqotu p {
##   margin: 0;
##   padding: 0;
## }
## 
## #nqvrdiqotu .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #nqvrdiqotu .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #nqvrdiqotu .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #nqvrdiqotu .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #nqvrdiqotu .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #nqvrdiqotu .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #nqvrdiqotu .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #nqvrdiqotu .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #nqvrdiqotu .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #nqvrdiqotu .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #nqvrdiqotu .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #nqvrdiqotu .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #nqvrdiqotu .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #nqvrdiqotu .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #nqvrdiqotu .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #nqvrdiqotu .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #nqvrdiqotu .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #nqvrdiqotu .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #nqvrdiqotu .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #nqvrdiqotu .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #nqvrdiqotu .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #nqvrdiqotu .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #nqvrdiqotu .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #nqvrdiqotu .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #nqvrdiqotu .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #nqvrdiqotu .gt_left {
##   text-align: left;
## }
## 
## #nqvrdiqotu .gt_center {
##   text-align: center;
## }
## 
## #nqvrdiqotu .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #nqvrdiqotu .gt_font_normal {
##   font-weight: normal;
## }
## 
## #nqvrdiqotu .gt_font_bold {
##   font-weight: bold;
## }
## 
## #nqvrdiqotu .gt_font_italic {
##   font-style: italic;
## }
## 
## #nqvrdiqotu .gt_super {
##   font-size: 65%;
## }
## 
## #nqvrdiqotu .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #nqvrdiqotu .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #nqvrdiqotu .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #nqvrdiqotu .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #nqvrdiqotu .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #nqvrdiqotu .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #nqvrdiqotu .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 502226&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 502226<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 848&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 848<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 501378&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 501378<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alanine aminotransferase</td>
## <td headers="stat_0" class="gt_row gt_center">23.5 (Â±13.7)</td>
## <td headers="stat_1" class="gt_row gt_center">26.5 (Â±19.7)</td>
## <td headers="stat_2" class="gt_row gt_center">23.5 (Â±13.7)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Albumin</td>
## <td headers="stat_0" class="gt_row gt_center">45.2 (Â±2.4)</td>
## <td headers="stat_1" class="gt_row gt_center">44.9 (Â±2.5)</td>
## <td headers="stat_2" class="gt_row gt_center">45.2 (Â±2.4)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alkaline phosphatase</td>
## <td headers="stat_0" class="gt_row gt_center">83.7 (Â±25.5)</td>
## <td headers="stat_1" class="gt_row gt_center">95.0 (Â±66.7)</td>
## <td headers="stat_2" class="gt_row gt_center">83.6 (Â±25.4)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Apolipoprotein A</td>
## <td headers="stat_0" class="gt_row gt_center">1.5 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">1.5 (Â±0.3)</td>
## <td headers="stat_2" class="gt_row gt_center">1.5 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Apolipoprotein B</td>
## <td headers="stat_0" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.14</td>
## <td headers="q.value" class="gt_row gt_center">0.7</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```r
create_table(df_tbl_icd, "Table ICD", export_RDS=TRUE, head_only=head_only, create_binary_table = TRUE)
```

```
## Warning for variable 'Alcoholic cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic fatty liver':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic fibrosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic hepatic failure':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="svisykbgky" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#svisykbgky table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #svisykbgky thead, #svisykbgky tbody, #svisykbgky tfoot, #svisykbgky tr, #svisykbgky td, #svisykbgky th {
##   border-style: none;
## }
## 
## #svisykbgky p {
##   margin: 0;
##   padding: 0;
## }
## 
## #svisykbgky .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #svisykbgky .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #svisykbgky .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #svisykbgky .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #svisykbgky .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #svisykbgky .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #svisykbgky .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #svisykbgky .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #svisykbgky .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #svisykbgky .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #svisykbgky .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #svisykbgky .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #svisykbgky .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #svisykbgky .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #svisykbgky .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #svisykbgky .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #svisykbgky .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #svisykbgky .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #svisykbgky .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #svisykbgky .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #svisykbgky .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #svisykbgky .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #svisykbgky .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #svisykbgky .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #svisykbgky .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #svisykbgky .gt_left {
##   text-align: left;
## }
## 
## #svisykbgky .gt_center {
##   text-align: center;
## }
## 
## #svisykbgky .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #svisykbgky .gt_font_normal {
##   font-weight: normal;
## }
## 
## #svisykbgky .gt_font_bold {
##   font-weight: bold;
## }
## 
## #svisykbgky .gt_font_italic {
##   font-style: italic;
## }
## 
## #svisykbgky .gt_super {
##   font-size: 65%;
## }
## 
## #svisykbgky .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #svisykbgky .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #svisykbgky .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #svisykbgky .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #svisykbgky .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #svisykbgky .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #svisykbgky .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 502226&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 502226<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 848&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 848<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 501378&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 501378<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Acute renal failure</td>
## <td headers="stat_0" class="gt_row gt_center">3,983 (0.8%)</td>
## <td headers="stat_1" class="gt_row gt_center">21 (2.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">3,962 (0.8%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">364 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">361 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.016</td>
## <td headers="q.value" class="gt_row gt_center">0.080</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic fatty liver</td>
## <td headers="stat_0" class="gt_row gt_center">88 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">88 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic fibrosis</td>
## <td headers="stat_0" class="gt_row gt_center">23 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">23 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic hepatic failure</td>
## <td headers="stat_0" class="gt_row gt_center">104 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.1%)</td>
## <td headers="stat_2" class="gt_row gt_center">103 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```r
create_table(df_tbl_metabolomics, "Table Metabolomics", export_RDS=FALSE, head_only=head_only)
```

```
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="hypihtopni" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#hypihtopni table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #hypihtopni thead, #hypihtopni tbody, #hypihtopni tfoot, #hypihtopni tr, #hypihtopni td, #hypihtopni th {
##   border-style: none;
## }
## 
## #hypihtopni p {
##   margin: 0;
##   padding: 0;
## }
## 
## #hypihtopni .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #hypihtopni .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #hypihtopni .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #hypihtopni .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #hypihtopni .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #hypihtopni .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #hypihtopni .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #hypihtopni .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #hypihtopni .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #hypihtopni .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #hypihtopni .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #hypihtopni .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #hypihtopni .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #hypihtopni .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #hypihtopni .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hypihtopni .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #hypihtopni .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #hypihtopni .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #hypihtopni .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hypihtopni .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #hypihtopni .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hypihtopni .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #hypihtopni .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hypihtopni .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #hypihtopni .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hypihtopni .gt_left {
##   text-align: left;
## }
## 
## #hypihtopni .gt_center {
##   text-align: center;
## }
## 
## #hypihtopni .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #hypihtopni .gt_font_normal {
##   font-weight: normal;
## }
## 
## #hypihtopni .gt_font_bold {
##   font-weight: bold;
## }
## 
## #hypihtopni .gt_font_italic {
##   font-style: italic;
## }
## 
## #hypihtopni .gt_super {
##   font-size: 65%;
## }
## 
## #hypihtopni .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #hypihtopni .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #hypihtopni .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #hypihtopni .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #hypihtopni .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #hypihtopni .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #hypihtopni .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 248231&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 248231<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 435&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 435<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 247796&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No CCa</strong></p>
## <p>n = 247796<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">3-Hydroxybutyrate</td>
## <td headers="stat_0" class="gt_row gt_center">0.1 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">0.1 (Â±0.1)</td>
## <td headers="stat_2" class="gt_row gt_center">0.1 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Acetate</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value" class="gt_row gt_center">0.026</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Acetoacetate</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Acetone</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alanine</td>
## <td headers="stat_0" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_2" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.7</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```r
create_table(df_tbl_snp, "Table SNP", export_RDS=TRUE, head_only=head_only)
```

```
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="hkzdqecyjn" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#hkzdqecyjn table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #hkzdqecyjn thead, #hkzdqecyjn tbody, #hkzdqecyjn tfoot, #hkzdqecyjn tr, #hkzdqecyjn td, #hkzdqecyjn th {
##   border-style: none;
## }
## 
## #hkzdqecyjn p {
##   margin: 0;
##   padding: 0;
## }
## 
## #hkzdqecyjn .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #hkzdqecyjn .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #hkzdqecyjn .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #hkzdqecyjn .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #hkzdqecyjn .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #hkzdqecyjn .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #hkzdqecyjn .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #hkzdqecyjn .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #hkzdqecyjn .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #hkzdqecyjn .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #hkzdqecyjn .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #hkzdqecyjn .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #hkzdqecyjn .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #hkzdqecyjn .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #hkzdqecyjn .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hkzdqecyjn .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #hkzdqecyjn .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #hkzdqecyjn .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #hkzdqecyjn .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hkzdqecyjn .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #hkzdqecyjn .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hkzdqecyjn .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #hkzdqecyjn .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hkzdqecyjn .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #hkzdqecyjn .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hkzdqecyjn .gt_left {
##   text-align: left;
## }
## 
## #hkzdqecyjn .gt_center {
##   text-align: center;
## }
## 
## #hkzdqecyjn .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #hkzdqecyjn .gt_font_normal {
##   font-weight: normal;
## }
## 
## #hkzdqecyjn .gt_font_bold {
##   font-weight: bold;
## }
## 
## #hkzdqecyjn .gt_font_italic {
##   font-style: italic;
## }
## 
## #hkzdqecyjn .gt_super {
##   font-size: 65%;
## }
## 
## #hkzdqecyjn .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #hkzdqecyjn .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #hkzdqecyjn .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #hkzdqecyjn .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #hkzdqecyjn .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #hkzdqecyjn .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #hkzdqecyjn .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 502226&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 502226<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 848&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 848<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 501378&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 501378<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">PC1</td>
## <td headers="stat_0" class="gt_row gt_center">-1.4 (Â±53.6)</td>
## <td headers="stat_1" class="gt_row gt_center">-2.7 (Â±50.1)</td>
## <td headers="stat_2" class="gt_row gt_center">-1.4 (Â±53.6)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">PC2</td>
## <td headers="stat_0" class="gt_row gt_center">0.5 (Â±27.5)</td>
## <td headers="stat_1" class="gt_row gt_center">0.5 (Â±28.6)</td>
## <td headers="stat_2" class="gt_row gt_center">0.5 (Â±27.5)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">PC3</td>
## <td headers="stat_0" class="gt_row gt_center">-0.1 (Â±14.7)</td>
## <td headers="stat_1" class="gt_row gt_center">-0.8 (Â±14.7)</td>
## <td headers="stat_2" class="gt_row gt_center">-0.1 (Â±14.7)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">PC4</td>
## <td headers="stat_0" class="gt_row gt_center">0.2 (Â±10.3)</td>
## <td headers="stat_1" class="gt_row gt_center">0.4 (Â±8.8)</td>
## <td headers="stat_2" class="gt_row gt_center">0.2 (Â±10.3)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">PC5</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±7.5)</td>
## <td headers="stat_1" class="gt_row gt_center">0.1 (Â±7.4)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±7.5)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```



# Male/Female stratified table

```r
head_only <- FALSE
# Change "all" to "par" if you want to create tables of just the "patient at risk subset
df_all <- df_x_all_raw %>% inner_join(df_y, by = "eid")
df_all$status <- ifelse(df_all$status == 0, paste("No ", DOI), 
                                  ifelse(df_all$status == 1, DOI, df_all$status))

df_tbl_1 <- df_all %>%
  select(!any_of(c(blood_list, metabolomics_list, icd_list, "eid", "Date of assessment", "UKB assessment centre", snp_list)))


split_create_merge_tables(df_tbl_1, table_name="Table1", feature="SEX", enforced_order=table1_order, remove_SEX=TRUE, export_RDS=TRUE)
```

```
## Warning for variable 'Ethnicity':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 2L, 2L, 2L, 2L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'BMI_cat':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 2L, 3L, 2L, 2L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'Smoking status':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 1L, 2L, 2L, 1L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## Warning for variable 'Ever smoked':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 1L, 2L, 2L, 1L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
```

```
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="ybamglpcpe" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#ybamglpcpe table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #ybamglpcpe thead, #ybamglpcpe tbody, #ybamglpcpe tfoot, #ybamglpcpe tr, #ybamglpcpe td, #ybamglpcpe th {
##   border-style: none;
## }
## 
## #ybamglpcpe p {
##   margin: 0;
##   padding: 0;
## }
## 
## #ybamglpcpe .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #ybamglpcpe .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #ybamglpcpe .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #ybamglpcpe .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #ybamglpcpe .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #ybamglpcpe .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #ybamglpcpe .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #ybamglpcpe .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #ybamglpcpe .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #ybamglpcpe .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #ybamglpcpe .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #ybamglpcpe .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #ybamglpcpe .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #ybamglpcpe .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #ybamglpcpe .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #ybamglpcpe .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #ybamglpcpe .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #ybamglpcpe .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #ybamglpcpe .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #ybamglpcpe .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #ybamglpcpe .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #ybamglpcpe .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #ybamglpcpe .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #ybamglpcpe .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #ybamglpcpe .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #ybamglpcpe .gt_left {
##   text-align: left;
## }
## 
## #ybamglpcpe .gt_center {
##   text-align: center;
## }
## 
## #ybamglpcpe .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #ybamglpcpe .gt_font_normal {
##   font-weight: normal;
## }
## 
## #ybamglpcpe .gt_font_bold {
##   font-weight: bold;
## }
## 
## #ybamglpcpe .gt_font_italic {
##   font-style: italic;
## }
## 
## #ybamglpcpe .gt_super {
##   font-size: 65%;
## }
## 
## #ybamglpcpe .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #ybamglpcpe .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #ybamglpcpe .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #ybamglpcpe .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #ybamglpcpe .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #ybamglpcpe .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #ybamglpcpe .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 273225&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 273225<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 423&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 423<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 272802&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 272802<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">AGE</td>
## <td headers="stat_0" class="gt_row gt_center">56.3 (Â±8.0)</td>
## <td headers="stat_1" class="gt_row gt_center">60.8 (Â±6.3)</td>
## <td headers="stat_2" class="gt_row gt_center">56.3 (Â±8.0)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ethnicity</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">1,266 (0.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,263 (0.5%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Caucasian</td>
## <td headers="stat_0" class="gt_row gt_center">257,298 (94%)</td>
## <td headers="stat_1" class="gt_row gt_center">405 (96%)</td>
## <td headers="stat_2" class="gt_row gt_center">256,893 (94%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Mixed</td>
## <td headers="stat_0" class="gt_row gt_center">1,849 (0.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,847 (0.7%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Asian or Asian british</td>
## <td headers="stat_0" class="gt_row gt_center">4,580 (1.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">4,576 (1.7%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Black or Black british</td>
## <td headers="stat_0" class="gt_row gt_center">4,649 (1.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">4,646 (1.7%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Chinese</td>
## <td headers="stat_0" class="gt_row gt_center">989 (0.4%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">988 (0.4%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Other</td>
## <td headers="stat_0" class="gt_row gt_center">2,594 (0.9%)</td>
## <td headers="stat_1" class="gt_row gt_center">5 (1.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,589 (0.9%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI</td>
## <td headers="stat_0" class="gt_row gt_center">27.1 (Â±5.2)</td>
## <td headers="stat_1" class="gt_row gt_center">28.2 (Â±5.8)</td>
## <td headers="stat_2" class="gt_row gt_center">27.1 (Â±5.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.002</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI Categories</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.008</td>
## <td headers="q.value" class="gt_row gt_center">0.12</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Underweight</td>
## <td headers="stat_0" class="gt_row gt_center">2,077 (0.8%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,073 (0.8%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Normal weight</td>
## <td headers="stat_0" class="gt_row gt_center">102,937 (38%)</td>
## <td headers="stat_1" class="gt_row gt_center">129 (30%)</td>
## <td headers="stat_2" class="gt_row gt_center">102,808 (38%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Overweight</td>
## <td headers="stat_0" class="gt_row gt_center">101,172 (37%)</td>
## <td headers="stat_1" class="gt_row gt_center">161 (38%)</td>
## <td headers="stat_2" class="gt_row gt_center">101,011 (37%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Obese</td>
## <td headers="stat_0" class="gt_row gt_center">65,580 (24%)</td>
## <td headers="stat_1" class="gt_row gt_center">128 (30%)</td>
## <td headers="stat_2" class="gt_row gt_center">65,452 (24%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">1,459 (0.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,458 (0.5%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Waist circumference</td>
## <td headers="stat_0" class="gt_row gt_center">84.8 (Â±12.5)</td>
## <td headers="stat_1" class="gt_row gt_center">88.0 (Â±13.4)</td>
## <td headers="stat_2" class="gt_row gt_center">84.7 (Â±12.5)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Weight [kg]</td>
## <td headers="stat_0" class="gt_row gt_center">71.5 (Â±14.1)</td>
## <td headers="stat_1" class="gt_row gt_center">74.2 (Â±16.2)</td>
## <td headers="stat_2" class="gt_row gt_center">71.5 (Â±14.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.009</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Standing height</td>
## <td headers="stat_0" class="gt_row gt_center">162.5 (Â±6.3)</td>
## <td headers="stat_1" class="gt_row gt_center">162.2 (Â±6.7)</td>
## <td headers="stat_2" class="gt_row gt_center">162.5 (Â±6.3)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Handgripstrength</td>
## <td headers="stat_0" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="stat_1" class="gt_row gt_center">30.8 (Â±11.2)</td>
## <td headers="stat_2" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MultipleDeprivationIndex</td>
## <td headers="stat_0" class="gt_row gt_center">17.0 (Â±13.7)</td>
## <td headers="stat_1" class="gt_row gt_center">17.2 (Â±13.8)</td>
## <td headers="stat_2" class="gt_row gt_center">17.0 (Â±13.7)</td>
## <td headers="p.value" class="gt_row gt_center">0.7</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Bloodpressure sys. [mmHg]</td>
## <td headers="stat_0" class="gt_row gt_center">135.3 (Â±18.7)</td>
## <td headers="stat_1" class="gt_row gt_center">139.1 (Â±18.9)</td>
## <td headers="stat_2" class="gt_row gt_center">135.3 (Â±18.7)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Medication</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    No Medication</td>
## <td headers="stat_0" class="gt_row gt_center">188,787 (69%)</td>
## <td headers="stat_1" class="gt_row gt_center">259 (61%)</td>
## <td headers="stat_2" class="gt_row gt_center">188,528 (69%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Metabolic</td>
## <td headers="stat_0" class="gt_row gt_center">62,859 (23%)</td>
## <td headers="stat_1" class="gt_row gt_center">137 (32%)</td>
## <td headers="stat_2" class="gt_row gt_center">62,722 (23%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Hormones</td>
## <td headers="stat_0" class="gt_row gt_center">21,579 (7.9%)</td>
## <td headers="stat_1" class="gt_row gt_center">27 (6.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">21,552 (7.9%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Family Diabetes</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0" class="gt_row gt_center">220,370 (81%)</td>
## <td headers="stat_1" class="gt_row gt_center">341 (81%)</td>
## <td headers="stat_2" class="gt_row gt_center">220,029 (81%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0" class="gt_row gt_center">52,855 (19%)</td>
## <td headers="stat_1" class="gt_row gt_center">82 (19%)</td>
## <td headers="stat_2" class="gt_row gt_center">52,773 (19%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Smoking status</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.010</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Never</td>
## <td headers="stat_0" class="gt_row gt_center">161,970 (59%)</td>
## <td headers="stat_1" class="gt_row gt_center">216 (51%)</td>
## <td headers="stat_2" class="gt_row gt_center">161,754 (59%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Previous</td>
## <td headers="stat_0" class="gt_row gt_center">105,683 (39%)</td>
## <td headers="stat_1" class="gt_row gt_center">202 (48%)</td>
## <td headers="stat_2" class="gt_row gt_center">105,481 (39%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Current</td>
## <td headers="stat_0" class="gt_row gt_center">4,114 (1.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">4,112 (1.5%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">1,458 (0.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,455 (0.5%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ever smoked</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.026</td>
## <td headers="q.value" class="gt_row gt_center">0.4</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0" class="gt_row gt_center">121,757 (45%)</td>
## <td headers="stat_1" class="gt_row gt_center">161 (38%)</td>
## <td headers="stat_2" class="gt_row gt_center">121,596 (45%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0" class="gt_row gt_center">149,982 (55%)</td>
## <td headers="stat_1" class="gt_row gt_center">259 (61%)</td>
## <td headers="stat_2" class="gt_row gt_center">149,723 (55%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">1,486 (0.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,483 (0.5%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Pack years</td>
## <td headers="stat_0" class="gt_row gt_center">6.5 (Â±11.8)</td>
## <td headers="stat_1" class="gt_row gt_center">10.3 (Â±15.7)</td>
## <td headers="stat_2" class="gt_row gt_center">6.5 (Â±11.8)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD); n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test; Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```
## Warning for variable 'Ethnicity':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 2L, 2L, 2L, 2L, 2L, 2L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'BMI_cat':
## simpleWarning in stats::chisq.test(x = structure(c(3L, 3L, 4L, 3L, 3L, 3L, 3L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Smoking status':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 1L, 2L, 1L, 3L, 2L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Ever smoked':
## simpleWarning in stats::chisq.test(x = structure(c(2L, 1L, 2L, 2L, 2L, 2L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="agtmiwejao" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#agtmiwejao table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #agtmiwejao thead, #agtmiwejao tbody, #agtmiwejao tfoot, #agtmiwejao tr, #agtmiwejao td, #agtmiwejao th {
##   border-style: none;
## }
## 
## #agtmiwejao p {
##   margin: 0;
##   padding: 0;
## }
## 
## #agtmiwejao .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #agtmiwejao .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #agtmiwejao .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #agtmiwejao .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #agtmiwejao .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #agtmiwejao .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #agtmiwejao .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #agtmiwejao .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #agtmiwejao .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #agtmiwejao .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #agtmiwejao .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #agtmiwejao .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #agtmiwejao .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #agtmiwejao .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #agtmiwejao .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #agtmiwejao .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #agtmiwejao .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #agtmiwejao .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #agtmiwejao .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #agtmiwejao .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #agtmiwejao .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #agtmiwejao .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #agtmiwejao .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #agtmiwejao .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #agtmiwejao .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #agtmiwejao .gt_left {
##   text-align: left;
## }
## 
## #agtmiwejao .gt_center {
##   text-align: center;
## }
## 
## #agtmiwejao .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #agtmiwejao .gt_font_normal {
##   font-weight: normal;
## }
## 
## #agtmiwejao .gt_font_bold {
##   font-weight: bold;
## }
## 
## #agtmiwejao .gt_font_italic {
##   font-style: italic;
## }
## 
## #agtmiwejao .gt_super {
##   font-size: 65%;
## }
## 
## #agtmiwejao .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #agtmiwejao .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #agtmiwejao .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #agtmiwejao .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #agtmiwejao .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #agtmiwejao .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #agtmiwejao .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 229001&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 229001<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 425&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 425<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 228576&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 228576<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">AGE</td>
## <td headers="stat_0" class="gt_row gt_center">56.7 (Â±8.2)</td>
## <td headers="stat_1" class="gt_row gt_center">61.3 (Â±6.0)</td>
## <td headers="stat_2" class="gt_row gt_center">56.7 (Â±8.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ethnicity</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">1,511 (0.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,507 (0.7%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Caucasian</td>
## <td headers="stat_0" class="gt_row gt_center">215,143 (94%)</td>
## <td headers="stat_1" class="gt_row gt_center">401 (94%)</td>
## <td headers="stat_2" class="gt_row gt_center">214,742 (94%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Mixed</td>
## <td headers="stat_0" class="gt_row gt_center">1,104 (0.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,104 (0.5%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Asian or Asian british</td>
## <td headers="stat_0" class="gt_row gt_center">5,293 (2.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">5,286 (2.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Black or Black british</td>
## <td headers="stat_0" class="gt_row gt_center">3,406 (1.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">3,399 (1.5%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Chinese</td>
## <td headers="stat_0" class="gt_row gt_center">583 (0.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">581 (0.3%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Other</td>
## <td headers="stat_0" class="gt_row gt_center">1,961 (0.9%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,957 (0.9%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI</td>
## <td headers="stat_0" class="gt_row gt_center">27.8 (Â±4.2)</td>
## <td headers="stat_1" class="gt_row gt_center">28.5 (Â±4.4)</td>
## <td headers="stat_2" class="gt_row gt_center">27.8 (Â±4.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.004</td>
## <td headers="q.value" class="gt_row gt_center">0.062</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI Categories</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.014</td>
## <td headers="q.value" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Underweight</td>
## <td headers="stat_0" class="gt_row gt_center">547 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">545 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Normal weight</td>
## <td headers="stat_0" class="gt_row gt_center">54,384 (24%)</td>
## <td headers="stat_1" class="gt_row gt_center">81 (19%)</td>
## <td headers="stat_2" class="gt_row gt_center">54,303 (24%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Overweight</td>
## <td headers="stat_0" class="gt_row gt_center">112,932 (49%)</td>
## <td headers="stat_1" class="gt_row gt_center">202 (48%)</td>
## <td headers="stat_2" class="gt_row gt_center">112,730 (49%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Obese</td>
## <td headers="stat_0" class="gt_row gt_center">59,490 (26%)</td>
## <td headers="stat_1" class="gt_row gt_center">138 (32%)</td>
## <td headers="stat_2" class="gt_row gt_center">59,352 (26%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">1,648 (0.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,646 (0.7%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Waist circumference</td>
## <td headers="stat_0" class="gt_row gt_center">96.9 (Â±11.3)</td>
## <td headers="stat_1" class="gt_row gt_center">99.6 (Â±12.5)</td>
## <td headers="stat_2" class="gt_row gt_center">96.9 (Â±11.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Weight [kg]</td>
## <td headers="stat_0" class="gt_row gt_center">85.9 (Â±14.3)</td>
## <td headers="stat_1" class="gt_row gt_center">87.5 (Â±15.4)</td>
## <td headers="stat_2" class="gt_row gt_center">85.9 (Â±14.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.034</td>
## <td headers="q.value" class="gt_row gt_center">0.5</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Standing height</td>
## <td headers="stat_0" class="gt_row gt_center">175.6 (Â±6.9)</td>
## <td headers="stat_1" class="gt_row gt_center">175.1 (Â±7.0)</td>
## <td headers="stat_2" class="gt_row gt_center">175.6 (Â±6.9)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Handgripstrength</td>
## <td headers="stat_0" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="stat_1" class="gt_row gt_center">29.7 (Â±10.6)</td>
## <td headers="stat_2" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.071</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MultipleDeprivationIndex</td>
## <td headers="stat_0" class="gt_row gt_center">17.5 (Â±14.2)</td>
## <td headers="stat_1" class="gt_row gt_center">20.0 (Â±15.4)</td>
## <td headers="stat_2" class="gt_row gt_center">17.5 (Â±14.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.016</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Bloodpressure sys. [mmHg]</td>
## <td headers="stat_0" class="gt_row gt_center">140.6 (Â±17.1)</td>
## <td headers="stat_1" class="gt_row gt_center">144.6 (Â±17.7)</td>
## <td headers="stat_2" class="gt_row gt_center">140.6 (Â±17.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Medication</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    No Medication</td>
## <td headers="stat_0" class="gt_row gt_center">153,683 (67%)</td>
## <td headers="stat_1" class="gt_row gt_center">234 (55%)</td>
## <td headers="stat_2" class="gt_row gt_center">153,449 (67%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Metabolic</td>
## <td headers="stat_0" class="gt_row gt_center">75,318 (33%)</td>
## <td headers="stat_1" class="gt_row gt_center">191 (45%)</td>
## <td headers="stat_2" class="gt_row gt_center">75,127 (33%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Hormones</td>
## <td headers="stat_0" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Family Diabetes</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0" class="gt_row gt_center">187,394 (82%)</td>
## <td headers="stat_1" class="gt_row gt_center">348 (82%)</td>
## <td headers="stat_2" class="gt_row gt_center">187,046 (82%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0" class="gt_row gt_center">41,607 (18%)</td>
## <td headers="stat_1" class="gt_row gt_center">77 (18%)</td>
## <td headers="stat_2" class="gt_row gt_center">41,530 (18%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Smoking status</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Never</td>
## <td headers="stat_0" class="gt_row gt_center">111,416 (49%)</td>
## <td headers="stat_1" class="gt_row gt_center">157 (37%)</td>
## <td headers="stat_2" class="gt_row gt_center">111,259 (49%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Previous</td>
## <td headers="stat_0" class="gt_row gt_center">109,148 (48%)</td>
## <td headers="stat_1" class="gt_row gt_center">247 (58%)</td>
## <td headers="stat_2" class="gt_row gt_center">108,901 (48%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Current</td>
## <td headers="stat_0" class="gt_row gt_center">7,047 (3.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">19 (4.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">7,028 (3.1%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">1,390 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,388 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ever smoked</td>
## <td headers="stat_0" class="gt_row gt_center"><br /></td>
## <td headers="stat_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.016</td>
## <td headers="q.value" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0" class="gt_row gt_center">78,992 (34%)</td>
## <td headers="stat_1" class="gt_row gt_center">119 (28%)</td>
## <td headers="stat_2" class="gt_row gt_center">78,873 (35%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0" class="gt_row gt_center">148,610 (65%)</td>
## <td headers="stat_1" class="gt_row gt_center">304 (72%)</td>
## <td headers="stat_2" class="gt_row gt_center">148,306 (65%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0" class="gt_row gt_center">1,399 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,397 (0.6%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Pack years</td>
## <td headers="stat_0" class="gt_row gt_center">10.4 (Â±17.0)</td>
## <td headers="stat_1" class="gt_row gt_center">14.5 (Â±20.3)</td>
## <td headers="stat_2" class="gt_row gt_center">10.4 (Â±17.0)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD); n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test; Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```r
Table_1_stratified <- import_merge_tables(table_name= "Table1", feature="SEX", levels = c("Female", "Male"))
```

```
## [1] "C:/Users/felix/OneDrive - Uniklinik RWTH Aachen/public/projects/cca/tables/Table1_SEX_Female_ukb.RDS"
## [1] "C:/Users/felix/OneDrive - Uniklinik RWTH Aachen/public/projects/cca/tables/Table1_SEX_Male_ukb.RDS"
## <div id="uznfvpcwnp" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#uznfvpcwnp table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #uznfvpcwnp thead, #uznfvpcwnp tbody, #uznfvpcwnp tfoot, #uznfvpcwnp tr, #uznfvpcwnp td, #uznfvpcwnp th {
##   border-style: none;
## }
## 
## #uznfvpcwnp p {
##   margin: 0;
##   padding: 0;
## }
## 
## #uznfvpcwnp .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #uznfvpcwnp .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #uznfvpcwnp .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #uznfvpcwnp .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #uznfvpcwnp .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #uznfvpcwnp .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #uznfvpcwnp .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #uznfvpcwnp .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #uznfvpcwnp .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #uznfvpcwnp .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #uznfvpcwnp .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #uznfvpcwnp .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #uznfvpcwnp .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #uznfvpcwnp .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #uznfvpcwnp .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #uznfvpcwnp .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #uznfvpcwnp .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #uznfvpcwnp .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #uznfvpcwnp .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #uznfvpcwnp .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #uznfvpcwnp .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #uznfvpcwnp .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #uznfvpcwnp .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #uznfvpcwnp .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #uznfvpcwnp .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #uznfvpcwnp .gt_left {
##   text-align: left;
## }
## 
## #uznfvpcwnp .gt_center {
##   text-align: center;
## }
## 
## #uznfvpcwnp .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #uznfvpcwnp .gt_font_normal {
##   font-weight: normal;
## }
## 
## #uznfvpcwnp .gt_font_bold {
##   font-weight: bold;
## }
## 
## #uznfvpcwnp .gt_font_italic {
##   font-style: italic;
## }
## 
## #uznfvpcwnp .gt_super {
##   font-size: 65%;
## }
## 
## #uznfvpcwnp .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #uznfvpcwnp .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #uznfvpcwnp .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #uznfvpcwnp .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #uznfvpcwnp .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #uznfvpcwnp .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #uznfvpcwnp .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings gt_spanner_row">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="2" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_center gt_columns_top_border gt_column_spanner_outer" rowspan="1" colspan="5" scope="colgroup" id="Female">
##         <span class="gt_column_spanner">Female</span>
##       </th>
##       <th class="gt_center gt_columns_top_border gt_column_spanner_outer" rowspan="1" colspan="5" scope="colgroup" id="Male">
##         <span class="gt_column_spanner">Male</span>
##       </th>
##     </tr>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 273225&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 273225<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 423&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 423<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 272802&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 272802<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 229001&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 229001<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 425&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 425<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 228576&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 228576<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">AGE</td>
## <td headers="stat_0_1" class="gt_row gt_center">56.3 (Â±8.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">60.8 (Â±6.3)</td>
## <td headers="stat_2_1" class="gt_row gt_center">56.3 (Â±8.0)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">56.7 (Â±8.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">61.3 (Â±6.0)</td>
## <td headers="stat_2_2" class="gt_row gt_center">56.7 (Â±8.2)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ethnicity</td>
## <td headers="stat_0_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_1" class="gt_row gt_center"><br /></td>
## <td headers="p.value_1" class="gt_row gt_center">0.5</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value_2" class="gt_row gt_center">0.6</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,266 (0.5%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,263 (0.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">1,511 (0.7%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1,507 (0.7%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Caucasian</td>
## <td headers="stat_0_1" class="gt_row gt_center">257,298 (94%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">405 (96%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">256,893 (94%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">215,143 (94%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">401 (94%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">214,742 (94%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Mixed</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,849 (0.7%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,847 (0.7%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">1,104 (0.5%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1,104 (0.5%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Asian or Asian british</td>
## <td headers="stat_0_1" class="gt_row gt_center">4,580 (1.7%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">4,576 (1.7%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">5,293 (2.3%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">5,286 (2.3%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Black or Black british</td>
## <td headers="stat_0_1" class="gt_row gt_center">4,649 (1.7%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">4,646 (1.7%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">3,406 (1.5%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">3,399 (1.5%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Chinese</td>
## <td headers="stat_0_1" class="gt_row gt_center">989 (0.4%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">988 (0.4%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">583 (0.3%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">581 (0.3%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Other</td>
## <td headers="stat_0_1" class="gt_row gt_center">2,594 (0.9%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">5 (1.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">2,589 (0.9%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">1,961 (0.9%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1,957 (0.9%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI</td>
## <td headers="stat_0_1" class="gt_row gt_center">27.1 (Â±5.2)</td>
## <td headers="stat_1_1" class="gt_row gt_center">28.2 (Â±5.8)</td>
## <td headers="stat_2_1" class="gt_row gt_center">27.1 (Â±5.2)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.002</td>
## <td headers="stat_0_2" class="gt_row gt_center">27.8 (Â±4.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">28.5 (Â±4.4)</td>
## <td headers="stat_2_2" class="gt_row gt_center">27.8 (Â±4.2)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.004</td>
## <td headers="q.value_2" class="gt_row gt_center">0.062</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">BMI Categories</td>
## <td headers="stat_0_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_1" class="gt_row gt_center"><br /></td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.008</td>
## <td headers="q.value_1" class="gt_row gt_center">0.12</td>
## <td headers="stat_0_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.014</td>
## <td headers="q.value_2" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Underweight</td>
## <td headers="stat_0_1" class="gt_row gt_center">2,077 (0.8%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">2,073 (0.8%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">547 (0.2%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">545 (0.2%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Normal weight</td>
## <td headers="stat_0_1" class="gt_row gt_center">102,937 (38%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">129 (30%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">102,808 (38%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">54,384 (24%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">81 (19%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">54,303 (24%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Overweight</td>
## <td headers="stat_0_1" class="gt_row gt_center">101,172 (37%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">161 (38%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">101,011 (37%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">112,932 (49%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">202 (48%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">112,730 (49%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Obese</td>
## <td headers="stat_0_1" class="gt_row gt_center">65,580 (24%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">128 (30%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">65,452 (24%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">59,490 (26%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">138 (32%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">59,352 (26%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,459 (0.5%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,458 (0.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">1,648 (0.7%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1,646 (0.7%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Waist circumference</td>
## <td headers="stat_0_1" class="gt_row gt_center">84.8 (Â±12.5)</td>
## <td headers="stat_1_1" class="gt_row gt_center">88.0 (Â±13.4)</td>
## <td headers="stat_2_1" class="gt_row gt_center">84.7 (Â±12.5)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">96.9 (Â±11.3)</td>
## <td headers="stat_1_2" class="gt_row gt_center">99.6 (Â±12.5)</td>
## <td headers="stat_2_2" class="gt_row gt_center">96.9 (Â±11.3)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Weight [kg]</td>
## <td headers="stat_0_1" class="gt_row gt_center">71.5 (Â±14.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">74.2 (Â±16.2)</td>
## <td headers="stat_2_1" class="gt_row gt_center">71.5 (Â±14.1)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.009</td>
## <td headers="stat_0_2" class="gt_row gt_center">85.9 (Â±14.3)</td>
## <td headers="stat_1_2" class="gt_row gt_center">87.5 (Â±15.4)</td>
## <td headers="stat_2_2" class="gt_row gt_center">85.9 (Â±14.3)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.034</td>
## <td headers="q.value_2" class="gt_row gt_center">0.5</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Standing height</td>
## <td headers="stat_0_1" class="gt_row gt_center">162.5 (Â±6.3)</td>
## <td headers="stat_1_1" class="gt_row gt_center">162.2 (Â±6.7)</td>
## <td headers="stat_2_1" class="gt_row gt_center">162.5 (Â±6.3)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.4</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">175.6 (Â±6.9)</td>
## <td headers="stat_1_2" class="gt_row gt_center">175.1 (Â±7.0)</td>
## <td headers="stat_2_2" class="gt_row gt_center">175.6 (Â±6.9)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.2</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Handgripstrength</td>
## <td headers="stat_0_1" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">30.8 (Â±11.2)</td>
## <td headers="stat_2_1" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.6</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">29.7 (Â±10.6)</td>
## <td headers="stat_2_2" class="gt_row gt_center">30.6 (Â±11.0)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.071</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MultipleDeprivationIndex</td>
## <td headers="stat_0_1" class="gt_row gt_center">17.0 (Â±13.7)</td>
## <td headers="stat_1_1" class="gt_row gt_center">17.2 (Â±13.8)</td>
## <td headers="stat_2_1" class="gt_row gt_center">17.0 (Â±13.7)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.7</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">17.5 (Â±14.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">20.0 (Â±15.4)</td>
## <td headers="stat_2_2" class="gt_row gt_center">17.5 (Â±14.2)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.016</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Bloodpressure sys. [mmHg]</td>
## <td headers="stat_0_1" class="gt_row gt_center">135.3 (Â±18.7)</td>
## <td headers="stat_1_1" class="gt_row gt_center">139.1 (Â±18.9)</td>
## <td headers="stat_2_1" class="gt_row gt_center">135.3 (Â±18.7)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">140.6 (Â±17.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">144.6 (Â±17.7)</td>
## <td headers="stat_2_2" class="gt_row gt_center">140.6 (Â±17.1)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Medication</td>
## <td headers="stat_0_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_1" class="gt_row gt_center"><br /></td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    No Medication</td>
## <td headers="stat_0_1" class="gt_row gt_center">188,787 (69%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">259 (61%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">188,528 (69%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">153,683 (67%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">234 (55%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">153,449 (67%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Metabolic</td>
## <td headers="stat_0_1" class="gt_row gt_center">62,859 (23%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">137 (32%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">62,722 (23%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">75,318 (33%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">191 (45%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">75,127 (33%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Hormones</td>
## <td headers="stat_0_1" class="gt_row gt_center">21,579 (7.9%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">27 (6.4%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">21,552 (7.9%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Family Diabetes</td>
## <td headers="stat_0_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_1" class="gt_row gt_center"><br /></td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0_1" class="gt_row gt_center">220,370 (81%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">341 (81%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">220,029 (81%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">187,394 (82%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">348 (82%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">187,046 (82%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0_1" class="gt_row gt_center">52,855 (19%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">82 (19%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">52,773 (19%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">41,607 (18%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">77 (18%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">41,530 (18%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Smoking status</td>
## <td headers="stat_0_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_1" class="gt_row gt_center"><br /></td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.010</td>
## <td headers="stat_0_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Never</td>
## <td headers="stat_0_1" class="gt_row gt_center">161,970 (59%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">216 (51%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">161,754 (59%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">111,416 (49%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">157 (37%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">111,259 (49%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Previous</td>
## <td headers="stat_0_1" class="gt_row gt_center">105,683 (39%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">202 (48%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">105,481 (39%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">109,148 (48%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">247 (58%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">108,901 (48%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Current</td>
## <td headers="stat_0_1" class="gt_row gt_center">4,114 (1.5%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">4,112 (1.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">7,047 (3.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">19 (4.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">7,028 (3.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,458 (0.5%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,455 (0.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">1,390 (0.6%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1,388 (0.6%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ever smoked</td>
## <td headers="stat_0_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_1" class="gt_row gt_center"><br /></td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.026</td>
## <td headers="q.value_1" class="gt_row gt_center">0.4</td>
## <td headers="stat_0_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_1_2" class="gt_row gt_center"><br /></td>
## <td headers="stat_2_2" class="gt_row gt_center"><br /></td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.016</td>
## <td headers="q.value_2" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    0</td>
## <td headers="stat_0_1" class="gt_row gt_center">121,757 (45%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">161 (38%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">121,596 (45%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">78,992 (34%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">119 (28%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">78,873 (35%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    1</td>
## <td headers="stat_0_1" class="gt_row gt_center">149,982 (55%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">259 (61%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">149,723 (55%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">148,610 (65%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">304 (72%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">148,306 (65%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-style: italic;">    Unknown</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,486 (0.5%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,483 (0.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center"><br /></td>
## <td headers="q.value_1" class="gt_row gt_center"><br /></td>
## <td headers="stat_0_2" class="gt_row gt_center">1,399 (0.6%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1,397 (0.6%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Pack years</td>
## <td headers="stat_0_1" class="gt_row gt_center">6.5 (Â±11.8)</td>
## <td headers="stat_1_1" class="gt_row gt_center">10.3 (Â±15.7)</td>
## <td headers="stat_2_1" class="gt_row gt_center">6.5 (Â±11.8)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">10.4 (Â±17.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">14.5 (Â±20.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">10.4 (Â±17.0)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="11"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD); n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="11"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test; Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="11"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```r
#Shorter Table 1


# df_tbl_1b <- df_all %>%
#   select(!any_of(c(blood_list, metabolomics_list, icd_list, "eid", "Date of assessment", "UKB assessment centre", snp_list)))
# 
# 
# split_create_merge_tables(df_tbl_1b, feature="SEX", table_name_prefix="SEX", enforced_order=table1b_order, remove_SEX=FALSE, export_RDS=TRUE)
# 
# Table_1b_stratified <- import_merge_tables(table_name= "Table_1b", feature="SEX", levels = c("Female", "Male"))



# Stratified "Blood Table"
split_create_merge_tables(df_tbl_blood, table_name= "Table_Blood", feature="SEX", enforced_order=FALSE, remove_SEX=TRUE, export_RDS=TRUE)
```

```
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="eaxmwekzql" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#eaxmwekzql table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #eaxmwekzql thead, #eaxmwekzql tbody, #eaxmwekzql tfoot, #eaxmwekzql tr, #eaxmwekzql td, #eaxmwekzql th {
##   border-style: none;
## }
## 
## #eaxmwekzql p {
##   margin: 0;
##   padding: 0;
## }
## 
## #eaxmwekzql .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #eaxmwekzql .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #eaxmwekzql .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #eaxmwekzql .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #eaxmwekzql .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #eaxmwekzql .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #eaxmwekzql .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #eaxmwekzql .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #eaxmwekzql .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #eaxmwekzql .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #eaxmwekzql .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #eaxmwekzql .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #eaxmwekzql .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #eaxmwekzql .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #eaxmwekzql .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #eaxmwekzql .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #eaxmwekzql .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #eaxmwekzql .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #eaxmwekzql .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #eaxmwekzql .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #eaxmwekzql .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #eaxmwekzql .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #eaxmwekzql .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #eaxmwekzql .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #eaxmwekzql .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #eaxmwekzql .gt_left {
##   text-align: left;
## }
## 
## #eaxmwekzql .gt_center {
##   text-align: center;
## }
## 
## #eaxmwekzql .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #eaxmwekzql .gt_font_normal {
##   font-weight: normal;
## }
## 
## #eaxmwekzql .gt_font_bold {
##   font-weight: bold;
## }
## 
## #eaxmwekzql .gt_font_italic {
##   font-style: italic;
## }
## 
## #eaxmwekzql .gt_super {
##   font-size: 65%;
## }
## 
## #eaxmwekzql .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #eaxmwekzql .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #eaxmwekzql .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #eaxmwekzql .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #eaxmwekzql .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #eaxmwekzql .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #eaxmwekzql .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 273225&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 273225<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 423&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 423<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 272802&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 272802<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alanine aminotransferase</td>
## <td headers="stat_0" class="gt_row gt_center">20.5 (Â±11.8)</td>
## <td headers="stat_1" class="gt_row gt_center">22.7 (Â±14.0)</td>
## <td headers="stat_2" class="gt_row gt_center">20.5 (Â±11.8)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.062</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Albumin</td>
## <td headers="stat_0" class="gt_row gt_center">45.0 (Â±2.4)</td>
## <td headers="stat_1" class="gt_row gt_center">44.9 (Â±2.5)</td>
## <td headers="stat_2" class="gt_row gt_center">45.0 (Â±2.4)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alkaline phosphatase</td>
## <td headers="stat_0" class="gt_row gt_center">84.8 (Â±26.5)</td>
## <td headers="stat_1" class="gt_row gt_center">94.5 (Â±38.0)</td>
## <td headers="stat_2" class="gt_row gt_center">84.8 (Â±26.5)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Apolipoprotein A</td>
## <td headers="stat_0" class="gt_row gt_center">1.6 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">1.6 (Â±0.3)</td>
## <td headers="stat_2" class="gt_row gt_center">1.6 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Apolipoprotein B</td>
## <td headers="stat_0" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">1.1 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.084</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Aspartate aminotransferase</td>
## <td headers="stat_0" class="gt_row gt_center">24.6 (Â±9.3)</td>
## <td headers="stat_1" class="gt_row gt_center">26.7 (Â±14.3)</td>
## <td headers="stat_2" class="gt_row gt_center">24.6 (Â±9.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Basophill (%)</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Basophill count</td>
## <td headers="stat_0" class="gt_row gt_center">0.6 (Â±0.7)</td>
## <td headers="stat_1" class="gt_row gt_center">0.6 (Â±0.4)</td>
## <td headers="stat_2" class="gt_row gt_center">0.6 (Â±0.7)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">C-reactive protein</td>
## <td headers="stat_0" class="gt_row gt_center">5.9 (Â±1.1)</td>
## <td headers="stat_1" class="gt_row gt_center">6.0 (Â±1.1)</td>
## <td headers="stat_2" class="gt_row gt_center">5.9 (Â±1.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.053</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Calcium</td>
## <td headers="stat_0" class="gt_row gt_center">2.7 (Â±4.2)</td>
## <td headers="stat_1" class="gt_row gt_center">3.5 (Â±4.5)</td>
## <td headers="stat_2" class="gt_row gt_center">2.7 (Â±4.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.026</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cholesterol</td>
## <td headers="stat_0" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="stat_2" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.050</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Creatinine</td>
## <td headers="stat_0" class="gt_row gt_center">64.9 (Â±13.4)</td>
## <td headers="stat_1" class="gt_row gt_center">65.2 (Â±11.6)</td>
## <td headers="stat_2" class="gt_row gt_center">64.9 (Â±13.4)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cystatin C</td>
## <td headers="stat_0" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Direct bilirubin</td>
## <td headers="stat_0" class="gt_row gt_center">1.7 (Â±0.6)</td>
## <td headers="stat_1" class="gt_row gt_center">1.7 (Â±0.7)</td>
## <td headers="stat_2" class="gt_row gt_center">1.7 (Â±0.6)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Eosinophill (%)</td>
## <td headers="stat_0" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="stat_2" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Eosinophill count</td>
## <td headers="stat_0" class="gt_row gt_center">2.4 (Â±1.7)</td>
## <td headers="stat_1" class="gt_row gt_center">2.4 (Â±1.4)</td>
## <td headers="stat_2" class="gt_row gt_center">2.4 (Â±1.7)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Erythrocyte distribution width</td>
## <td headers="stat_0" class="gt_row gt_center">13.5 (Â±1.0)</td>
## <td headers="stat_1" class="gt_row gt_center">13.6 (Â±1.0)</td>
## <td headers="stat_2" class="gt_row gt_center">13.5 (Â±1.0)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.011</td>
## <td headers="q.value" class="gt_row gt_center">0.7</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Erythrocytes</td>
## <td headers="stat_0" class="gt_row gt_center">39.3 (Â±2.8)</td>
## <td headers="stat_1" class="gt_row gt_center">39.7 (Â±2.8)</td>
## <td headers="stat_2" class="gt_row gt_center">39.3 (Â±2.8)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Gamma glutamyltransferase</td>
## <td headers="stat_0" class="gt_row gt_center">30.8 (Â±32.5)</td>
## <td headers="stat_1" class="gt_row gt_center">41.5 (Â±54.2)</td>
## <td headers="stat_2" class="gt_row gt_center">30.8 (Â±32.5)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.003</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Glucose</td>
## <td headers="stat_0" class="gt_row gt_center">5.1 (Â±1.0)</td>
## <td headers="stat_1" class="gt_row gt_center">5.2 (Â±1.0)</td>
## <td headers="stat_2" class="gt_row gt_center">5.1 (Â±1.0)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.026</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Haematocrit</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.020</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Haemoglobin</td>
## <td headers="stat_0" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="stat_1" class="gt_row gt_center">0.4 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.041</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">HbA1c</td>
## <td headers="stat_0" class="gt_row gt_center">35.8 (Â±5.8)</td>
## <td headers="stat_1" class="gt_row gt_center">37.4 (Â±6.3)</td>
## <td headers="stat_2" class="gt_row gt_center">35.8 (Â±5.7)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">HDL cholesterol</td>
## <td headers="stat_0" class="gt_row gt_center">1.6 (Â±0.4)</td>
## <td headers="stat_1" class="gt_row gt_center">1.5 (Â±0.4)</td>
## <td headers="stat_2" class="gt_row gt_center">1.6 (Â±0.4)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">High light scatter reticulocyte (%)</td>
## <td headers="stat_0" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_2" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.012</td>
## <td headers="q.value" class="gt_row gt_center">0.7</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">High light scatter reticulocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">2.0 (Â±1.0)</td>
## <td headers="stat_1" class="gt_row gt_center">2.1 (Â±0.6)</td>
## <td headers="stat_2" class="gt_row gt_center">2.0 (Â±1.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.10</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">IGF-1</td>
## <td headers="stat_0" class="gt_row gt_center">21.0 (Â±5.6)</td>
## <td headers="stat_1" class="gt_row gt_center">19.4 (Â±5.7)</td>
## <td headers="stat_2" class="gt_row gt_center">21.0 (Â±5.6)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Immature reticulocyte fraction</td>
## <td headers="stat_0" class="gt_row gt_center">29.7 (Â±7.2)</td>
## <td headers="stat_1" class="gt_row gt_center">29.4 (Â±6.7)</td>
## <td headers="stat_2" class="gt_row gt_center">29.7 (Â±7.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">LDL direct</td>
## <td headers="stat_0" class="gt_row gt_center">3.6 (Â±0.8)</td>
## <td headers="stat_1" class="gt_row gt_center">3.7 (Â±0.9)</td>
## <td headers="stat_2" class="gt_row gt_center">3.6 (Â±0.8)</td>
## <td headers="p.value" class="gt_row gt_center">0.11</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Leukocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">31.3 (Â±1.9)</td>
## <td headers="stat_1" class="gt_row gt_center">31.4 (Â±1.7)</td>
## <td headers="stat_2" class="gt_row gt_center">31.3 (Â±1.9)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Lipoprotein A</td>
## <td headers="stat_0" class="gt_row gt_center">45.1 (Â±42.7)</td>
## <td headers="stat_1" class="gt_row gt_center">46.8 (Â±45.7)</td>
## <td headers="stat_2" class="gt_row gt_center">45.1 (Â±42.7)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Lymphocyte (%)</td>
## <td headers="stat_0" class="gt_row gt_center">34.4 (Â±1.1)</td>
## <td headers="stat_1" class="gt_row gt_center">34.3 (Â±0.9)</td>
## <td headers="stat_2" class="gt_row gt_center">34.4 (Â±1.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.11</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Lymphocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">90.9 (Â±4.6)</td>
## <td headers="stat_1" class="gt_row gt_center">91.3 (Â±4.3)</td>
## <td headers="stat_2" class="gt_row gt_center">90.9 (Â±4.6)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.048</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MCH</td>
## <td headers="stat_0" class="gt_row gt_center">9.4 (Â±1.1)</td>
## <td headers="stat_1" class="gt_row gt_center">9.4 (Â±1.0)</td>
## <td headers="stat_2" class="gt_row gt_center">9.4 (Â±1.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.8</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MCHC</td>
## <td headers="stat_0" class="gt_row gt_center">105.6 (Â±7.6)</td>
## <td headers="stat_1" class="gt_row gt_center">105.8 (Â±8.3)</td>
## <td headers="stat_2" class="gt_row gt_center">105.6 (Â±7.6)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MCV</td>
## <td headers="stat_0" class="gt_row gt_center">83.0 (Â±5.1)</td>
## <td headers="stat_1" class="gt_row gt_center">83.1 (Â±5.4)</td>
## <td headers="stat_2" class="gt_row gt_center">83.0 (Â±5.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Mean platelet volume</td>
## <td headers="stat_0" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="stat_1" class="gt_row gt_center">0.5 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.040</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Mean reticulocyte volume</td>
## <td headers="stat_0" class="gt_row gt_center">6.6 (Â±2.5)</td>
## <td headers="stat_1" class="gt_row gt_center">6.6 (Â±1.9)</td>
## <td headers="stat_2" class="gt_row gt_center">6.6 (Â±2.5)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Mean sphered cell volume</td>
## <td headers="stat_0" class="gt_row gt_center">4.2 (Â±1.4)</td>
## <td headers="stat_1" class="gt_row gt_center">4.4 (Â±1.5)</td>
## <td headers="stat_2" class="gt_row gt_center">4.2 (Â±1.4)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Monocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">60.6 (Â±8.2)</td>
## <td headers="stat_1" class="gt_row gt_center">61.1 (Â±7.5)</td>
## <td headers="stat_2" class="gt_row gt_center">60.6 (Â±8.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Monocyte percentage</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Neutrophill count</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.4)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.3)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.4)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Neutrophill percentage</td>
## <td headers="stat_0" class="gt_row gt_center">265.1 (Â±58.8)</td>
## <td headers="stat_1" class="gt_row gt_center">266.9 (Â±61.6)</td>
## <td headers="stat_2" class="gt_row gt_center">265.1 (Â±58.8)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Nucleated red blood cell (%)</td>
## <td headers="stat_0" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Nucleated red blood cell count</td>
## <td headers="stat_0" class="gt_row gt_center">16.4 (Â±0.5)</td>
## <td headers="stat_1" class="gt_row gt_center">16.5 (Â±0.5)</td>
## <td headers="stat_2" class="gt_row gt_center">16.4 (Â±0.5)</td>
## <td headers="p.value" class="gt_row gt_center">0.058</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Phosphate</td>
## <td headers="stat_0" class="gt_row gt_center">1.2 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">1.2 (Â±0.1)</td>
## <td headers="stat_2" class="gt_row gt_center">1.2 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Platelet count</td>
## <td headers="stat_0" class="gt_row gt_center">4.3 (Â±0.3)</td>
## <td headers="stat_1" class="gt_row gt_center">4.4 (Â±0.3)</td>
## <td headers="stat_2" class="gt_row gt_center">4.3 (Â±0.3)</td>
## <td headers="p.value" class="gt_row gt_center">0.12</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Platelet crit</td>
## <td headers="stat_0" class="gt_row gt_center">13.5 (Â±1.0)</td>
## <td headers="stat_1" class="gt_row gt_center">13.6 (Â±1.0)</td>
## <td headers="stat_2" class="gt_row gt_center">13.5 (Â±1.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.14</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Platelet distribution width</td>
## <td headers="stat_0" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Reticulocyte (%)</td>
## <td headers="stat_0" class="gt_row gt_center">1.3 (Â±0.9)</td>
## <td headers="stat_1" class="gt_row gt_center">1.4 (Â±0.5)</td>
## <td headers="stat_2" class="gt_row gt_center">1.3 (Â±0.9)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Reticulocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">6.9 (Â±2.0)</td>
## <td headers="stat_1" class="gt_row gt_center">7.1 (Â±1.9)</td>
## <td headers="stat_2" class="gt_row gt_center">6.9 (Â±2.0)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.004</td>
## <td headers="q.value" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">SHBG</td>
## <td headers="stat_0" class="gt_row gt_center">60.3 (Â±28.6)</td>
## <td headers="stat_1" class="gt_row gt_center">58.3 (Â±27.0)</td>
## <td headers="stat_2" class="gt_row gt_center">60.3 (Â±28.6)</td>
## <td headers="p.value" class="gt_row gt_center">0.13</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Testosterone</td>
## <td headers="stat_0" class="gt_row gt_center">2.3 (Â±2.3)</td>
## <td headers="stat_1" class="gt_row gt_center">2.6 (Â±2.5)</td>
## <td headers="stat_2" class="gt_row gt_center">2.3 (Â±2.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.017</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Total bilirubin</td>
## <td headers="stat_0" class="gt_row gt_center">8.2 (Â±3.6)</td>
## <td headers="stat_1" class="gt_row gt_center">8.1 (Â±3.7)</td>
## <td headers="stat_2" class="gt_row gt_center">8.2 (Â±3.6)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Total protein</td>
## <td headers="stat_0" class="gt_row gt_center">72.4 (Â±3.8)</td>
## <td headers="stat_1" class="gt_row gt_center">72.6 (Â±3.9)</td>
## <td headers="stat_2" class="gt_row gt_center">72.4 (Â±3.8)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Triglycerides</td>
## <td headers="stat_0" class="gt_row gt_center">1.6 (Â±0.8)</td>
## <td headers="stat_1" class="gt_row gt_center">1.8 (Â±1.1)</td>
## <td headers="stat_2" class="gt_row gt_center">1.6 (Â±0.8)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Urate</td>
## <td headers="stat_0" class="gt_row gt_center">273.7 (Â±64.6)</td>
## <td headers="stat_1" class="gt_row gt_center">292.2 (Â±77.4)</td>
## <td headers="stat_2" class="gt_row gt_center">273.6 (Â±64.6)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Urea</td>
## <td headers="stat_0" class="gt_row gt_center">5.2 (Â±1.3)</td>
## <td headers="stat_1" class="gt_row gt_center">5.4 (Â±1.3)</td>
## <td headers="stat_2" class="gt_row gt_center">5.2 (Â±1.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Vitamin D</td>
## <td headers="stat_0" class="gt_row gt_center">48.7 (Â±19.7)</td>
## <td headers="stat_1" class="gt_row gt_center">48.7 (Â±19.4)</td>
## <td headers="stat_2" class="gt_row gt_center">48.7 (Â±19.7)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="izgtwmmzvh" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#izgtwmmzvh table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #izgtwmmzvh thead, #izgtwmmzvh tbody, #izgtwmmzvh tfoot, #izgtwmmzvh tr, #izgtwmmzvh td, #izgtwmmzvh th {
##   border-style: none;
## }
## 
## #izgtwmmzvh p {
##   margin: 0;
##   padding: 0;
## }
## 
## #izgtwmmzvh .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #izgtwmmzvh .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #izgtwmmzvh .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #izgtwmmzvh .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #izgtwmmzvh .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #izgtwmmzvh .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #izgtwmmzvh .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #izgtwmmzvh .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #izgtwmmzvh .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #izgtwmmzvh .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #izgtwmmzvh .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #izgtwmmzvh .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #izgtwmmzvh .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #izgtwmmzvh .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #izgtwmmzvh .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #izgtwmmzvh .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #izgtwmmzvh .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #izgtwmmzvh .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #izgtwmmzvh .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #izgtwmmzvh .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #izgtwmmzvh .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #izgtwmmzvh .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #izgtwmmzvh .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #izgtwmmzvh .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #izgtwmmzvh .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #izgtwmmzvh .gt_left {
##   text-align: left;
## }
## 
## #izgtwmmzvh .gt_center {
##   text-align: center;
## }
## 
## #izgtwmmzvh .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #izgtwmmzvh .gt_font_normal {
##   font-weight: normal;
## }
## 
## #izgtwmmzvh .gt_font_bold {
##   font-weight: bold;
## }
## 
## #izgtwmmzvh .gt_font_italic {
##   font-style: italic;
## }
## 
## #izgtwmmzvh .gt_super {
##   font-size: 65%;
## }
## 
## #izgtwmmzvh .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #izgtwmmzvh .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #izgtwmmzvh .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #izgtwmmzvh .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #izgtwmmzvh .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #izgtwmmzvh .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #izgtwmmzvh .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 229001&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 229001<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 425&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 425<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 228576&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 228576<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alanine aminotransferase</td>
## <td headers="stat_0" class="gt_row gt_center">27.2 (Â±14.8)</td>
## <td headers="stat_1" class="gt_row gt_center">30.3 (Â±23.4)</td>
## <td headers="stat_2" class="gt_row gt_center">27.2 (Â±14.8)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.008</td>
## <td headers="q.value" class="gt_row gt_center">0.4</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Albumin</td>
## <td headers="stat_0" class="gt_row gt_center">45.5 (Â±2.4)</td>
## <td headers="stat_1" class="gt_row gt_center">44.8 (Â±2.5)</td>
## <td headers="stat_2" class="gt_row gt_center">45.5 (Â±2.4)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alkaline phosphatase</td>
## <td headers="stat_0" class="gt_row gt_center">82.3 (Â±24.2)</td>
## <td headers="stat_1" class="gt_row gt_center">95.5 (Â±86.3)</td>
## <td headers="stat_2" class="gt_row gt_center">82.3 (Â±23.9)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.002</td>
## <td headers="q.value" class="gt_row gt_center">0.10</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Apolipoprotein A</td>
## <td headers="stat_0" class="gt_row gt_center">1.4 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">1.4 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">1.4 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Apolipoprotein B</td>
## <td headers="stat_0" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.016</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Aspartate aminotransferase</td>
## <td headers="stat_0" class="gt_row gt_center">28.1 (Â±11.1)</td>
## <td headers="stat_1" class="gt_row gt_center">31.3 (Â±20.3)</td>
## <td headers="stat_2" class="gt_row gt_center">28.1 (Â±11.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.066</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Basophill (%)</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Basophill count</td>
## <td headers="stat_0" class="gt_row gt_center">0.5 (Â±0.5)</td>
## <td headers="stat_1" class="gt_row gt_center">0.5 (Â±0.4)</td>
## <td headers="stat_2" class="gt_row gt_center">0.5 (Â±0.5)</td>
## <td headers="p.value" class="gt_row gt_center">0.8</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">C-reactive protein</td>
## <td headers="stat_0" class="gt_row gt_center">5.5 (Â±1.1)</td>
## <td headers="stat_1" class="gt_row gt_center">5.3 (Â±1.2)</td>
## <td headers="stat_2" class="gt_row gt_center">5.5 (Â±1.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.042</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Calcium</td>
## <td headers="stat_0" class="gt_row gt_center">2.5 (Â±4.2)</td>
## <td headers="stat_1" class="gt_row gt_center">2.9 (Â±3.8)</td>
## <td headers="stat_2" class="gt_row gt_center">2.5 (Â±4.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.023</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cholesterol</td>
## <td headers="stat_0" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="stat_2" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Creatinine</td>
## <td headers="stat_0" class="gt_row gt_center">81.1 (Â±18.6)</td>
## <td headers="stat_1" class="gt_row gt_center">80.8 (Â±15.3)</td>
## <td headers="stat_2" class="gt_row gt_center">81.1 (Â±18.7)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cystatin C</td>
## <td headers="stat_0" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.005</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Direct bilirubin</td>
## <td headers="stat_0" class="gt_row gt_center">2.0 (Â±0.9)</td>
## <td headers="stat_1" class="gt_row gt_center">2.1 (Â±1.6)</td>
## <td headers="stat_2" class="gt_row gt_center">2.0 (Â±0.9)</td>
## <td headers="p.value" class="gt_row gt_center">0.088</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Eosinophill (%)</td>
## <td headers="stat_0" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="stat_2" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.062</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Eosinophill count</td>
## <td headers="stat_0" class="gt_row gt_center">2.8 (Â±1.9)</td>
## <td headers="stat_1" class="gt_row gt_center">2.8 (Â±2.1)</td>
## <td headers="stat_2" class="gt_row gt_center">2.8 (Â±1.9)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Erythrocyte distribution width</td>
## <td headers="stat_0" class="gt_row gt_center">15.0 (Â±1.0)</td>
## <td headers="stat_1" class="gt_row gt_center">14.9 (Â±1.1)</td>
## <td headers="stat_2" class="gt_row gt_center">15.0 (Â±1.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Erythrocytes</td>
## <td headers="stat_0" class="gt_row gt_center">43.2 (Â±3.0)</td>
## <td headers="stat_1" class="gt_row gt_center">43.1 (Â±3.2)</td>
## <td headers="stat_2" class="gt_row gt_center">43.2 (Â±3.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Gamma glutamyltransferase</td>
## <td headers="stat_0" class="gt_row gt_center">45.2 (Â±47.3)</td>
## <td headers="stat_1" class="gt_row gt_center">77.0 (Â±125.0)</td>
## <td headers="stat_2" class="gt_row gt_center">45.2 (Â±47.0)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Glucose</td>
## <td headers="stat_0" class="gt_row gt_center">5.2 (Â±1.3)</td>
## <td headers="stat_1" class="gt_row gt_center">5.5 (Â±1.9)</td>
## <td headers="stat_2" class="gt_row gt_center">5.2 (Â±1.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.003</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Haematocrit</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Haemoglobin</td>
## <td headers="stat_0" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="stat_1" class="gt_row gt_center">0.4 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="p.value" class="gt_row gt_center">0.069</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">HbA1c</td>
## <td headers="stat_0" class="gt_row gt_center">36.5 (Â±7.3)</td>
## <td headers="stat_1" class="gt_row gt_center">38.5 (Â±9.1)</td>
## <td headers="stat_2" class="gt_row gt_center">36.5 (Â±7.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">HDL cholesterol</td>
## <td headers="stat_0" class="gt_row gt_center">1.3 (Â±0.3)</td>
## <td headers="stat_1" class="gt_row gt_center">1.3 (Â±0.3)</td>
## <td headers="stat_2" class="gt_row gt_center">1.3 (Â±0.3)</td>
## <td headers="p.value" class="gt_row gt_center">0.11</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">High light scatter reticulocyte (%)</td>
## <td headers="stat_0" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_1" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_2" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">High light scatter reticulocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">1.9 (Â±1.3)</td>
## <td headers="stat_1" class="gt_row gt_center">1.9 (Â±0.9)</td>
## <td headers="stat_2" class="gt_row gt_center">1.9 (Â±1.3)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">IGF-1</td>
## <td headers="stat_0" class="gt_row gt_center">21.9 (Â±5.4)</td>
## <td headers="stat_1" class="gt_row gt_center">20.1 (Â±5.5)</td>
## <td headers="stat_2" class="gt_row gt_center">21.9 (Â±5.4)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Immature reticulocyte fraction</td>
## <td headers="stat_0" class="gt_row gt_center">27.9 (Â±7.3)</td>
## <td headers="stat_1" class="gt_row gt_center">27.1 (Â±7.3)</td>
## <td headers="stat_2" class="gt_row gt_center">27.9 (Â±7.3)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.023</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">LDL direct</td>
## <td headers="stat_0" class="gt_row gt_center">3.5 (Â±0.8)</td>
## <td headers="stat_1" class="gt_row gt_center">3.3 (Â±0.9)</td>
## <td headers="stat_2" class="gt_row gt_center">3.5 (Â±0.8)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.007</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Leukocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">31.7 (Â±1.8)</td>
## <td headers="stat_1" class="gt_row gt_center">31.9 (Â±1.7)</td>
## <td headers="stat_2" class="gt_row gt_center">31.7 (Â±1.8)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Lipoprotein A</td>
## <td headers="stat_0" class="gt_row gt_center">44.1 (Â±42.4)</td>
## <td headers="stat_1" class="gt_row gt_center">41.4 (Â±39.5)</td>
## <td headers="stat_2" class="gt_row gt_center">44.1 (Â±42.4)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Lymphocyte (%)</td>
## <td headers="stat_0" class="gt_row gt_center">34.6 (Â±1.0)</td>
## <td headers="stat_1" class="gt_row gt_center">34.6 (Â±0.9)</td>
## <td headers="stat_2" class="gt_row gt_center">34.6 (Â±1.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Lymphocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">91.4 (Â±4.4)</td>
## <td headers="stat_1" class="gt_row gt_center">92.2 (Â±4.4)</td>
## <td headers="stat_2" class="gt_row gt_center">91.4 (Â±4.4)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.026</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MCH</td>
## <td headers="stat_0" class="gt_row gt_center">9.3 (Â±1.1)</td>
## <td headers="stat_1" class="gt_row gt_center">9.4 (Â±1.1)</td>
## <td headers="stat_2" class="gt_row gt_center">9.3 (Â±1.1)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MCHC</td>
## <td headers="stat_0" class="gt_row gt_center">106.3 (Â±7.6)</td>
## <td headers="stat_1" class="gt_row gt_center">107.5 (Â±7.3)</td>
## <td headers="stat_2" class="gt_row gt_center">106.3 (Â±7.6)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.046</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MCV</td>
## <td headers="stat_0" class="gt_row gt_center">82.7 (Â±5.2)</td>
## <td headers="stat_1" class="gt_row gt_center">83.8 (Â±5.3)</td>
## <td headers="stat_2" class="gt_row gt_center">82.7 (Â±5.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.006</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Mean platelet volume</td>
## <td headers="stat_0" class="gt_row gt_center">0.5 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">0.5 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">0.5 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.011</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Mean reticulocyte volume</td>
## <td headers="stat_0" class="gt_row gt_center">7.6 (Â±2.7)</td>
## <td headers="stat_1" class="gt_row gt_center">7.7 (Â±2.2)</td>
## <td headers="stat_2" class="gt_row gt_center">7.6 (Â±2.7)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Mean sphered cell volume</td>
## <td headers="stat_0" class="gt_row gt_center">4.3 (Â±1.4)</td>
## <td headers="stat_1" class="gt_row gt_center">4.5 (Â±1.4)</td>
## <td headers="stat_2" class="gt_row gt_center">4.3 (Â±1.4)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.035</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Monocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">61.2 (Â±8.5)</td>
## <td headers="stat_1" class="gt_row gt_center">61.8 (Â±8.5)</td>
## <td headers="stat_2" class="gt_row gt_center">61.2 (Â±8.5)</td>
## <td headers="p.value" class="gt_row gt_center">0.14</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Monocyte percentage</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Neutrophill count</td>
## <td headers="stat_0" class="gt_row gt_center">0.0 (Â±0.4)</td>
## <td headers="stat_1" class="gt_row gt_center">0.0 (Â±0.3)</td>
## <td headers="stat_2" class="gt_row gt_center">0.0 (Â±0.4)</td>
## <td headers="p.value" class="gt_row gt_center">0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Neutrophill percentage</td>
## <td headers="stat_0" class="gt_row gt_center">238.6 (Â±54.9)</td>
## <td headers="stat_1" class="gt_row gt_center">234.0 (Â±55.6)</td>
## <td headers="stat_2" class="gt_row gt_center">238.6 (Â±54.9)</td>
## <td headers="p.value" class="gt_row gt_center">0.091</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Nucleated red blood cell (%)</td>
## <td headers="stat_0" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.14</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Nucleated red blood cell count</td>
## <td headers="stat_0" class="gt_row gt_center">16.6 (Â±0.5)</td>
## <td headers="stat_1" class="gt_row gt_center">16.6 (Â±0.5)</td>
## <td headers="stat_2" class="gt_row gt_center">16.6 (Â±0.5)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Phosphate</td>
## <td headers="stat_0" class="gt_row gt_center">1.1 (Â±0.2)</td>
## <td headers="stat_1" class="gt_row gt_center">1.1 (Â±0.2)</td>
## <td headers="stat_2" class="gt_row gt_center">1.1 (Â±0.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.070</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Platelet count</td>
## <td headers="stat_0" class="gt_row gt_center">4.7 (Â±0.4)</td>
## <td headers="stat_1" class="gt_row gt_center">4.7 (Â±0.4)</td>
## <td headers="stat_2" class="gt_row gt_center">4.7 (Â±0.4)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.006</td>
## <td headers="q.value" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Platelet crit</td>
## <td headers="stat_0" class="gt_row gt_center">13.5 (Â±0.9)</td>
## <td headers="stat_1" class="gt_row gt_center">13.6 (Â±1.1)</td>
## <td headers="stat_2" class="gt_row gt_center">13.5 (Â±0.9)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.062</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Platelet distribution width</td>
## <td headers="stat_0" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="stat_1" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="stat_2" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="p.value" class="gt_row gt_center">0.8</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Reticulocyte (%)</td>
## <td headers="stat_0" class="gt_row gt_center">1.4 (Â±0.9)</td>
## <td headers="stat_1" class="gt_row gt_center">1.4 (Â±0.5)</td>
## <td headers="stat_2" class="gt_row gt_center">1.4 (Â±0.9)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Reticulocyte count</td>
## <td headers="stat_0" class="gt_row gt_center">6.9 (Â±2.1)</td>
## <td headers="stat_1" class="gt_row gt_center">7.2 (Â±1.9)</td>
## <td headers="stat_2" class="gt_row gt_center">6.9 (Â±2.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.038</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">SHBG</td>
## <td headers="stat_0" class="gt_row gt_center">41.3 (Â±16.1)</td>
## <td headers="stat_1" class="gt_row gt_center">43.6 (Â±16.1)</td>
## <td headers="stat_2" class="gt_row gt_center">41.3 (Â±16.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.004</td>
## <td headers="q.value" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Testosterone</td>
## <td headers="stat_0" class="gt_row gt_center">11.6 (Â±3.8)</td>
## <td headers="stat_1" class="gt_row gt_center">11.4 (Â±4.1)</td>
## <td headers="stat_2" class="gt_row gt_center">11.6 (Â±3.8)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Total bilirubin</td>
## <td headers="stat_0" class="gt_row gt_center">10.2 (Â±4.8)</td>
## <td headers="stat_1" class="gt_row gt_center">10.0 (Â±4.6)</td>
## <td headers="stat_2" class="gt_row gt_center">10.2 (Â±4.8)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Total protein</td>
## <td headers="stat_0" class="gt_row gt_center">72.6 (Â±3.8)</td>
## <td headers="stat_1" class="gt_row gt_center">72.5 (Â±3.9)</td>
## <td headers="stat_2" class="gt_row gt_center">72.6 (Â±3.8)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Triglycerides</td>
## <td headers="stat_0" class="gt_row gt_center">2.0 (Â±1.1)</td>
## <td headers="stat_1" class="gt_row gt_center">2.2 (Â±1.3)</td>
## <td headers="stat_2" class="gt_row gt_center">2.0 (Â±1.1)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.002</td>
## <td headers="q.value" class="gt_row gt_center">0.11</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Urate</td>
## <td headers="stat_0" class="gt_row gt_center">351.6 (Â±70.4)</td>
## <td headers="stat_1" class="gt_row gt_center">354.3 (Â±76.5)</td>
## <td headers="stat_2" class="gt_row gt_center">351.6 (Â±70.4)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Urea</td>
## <td headers="stat_0" class="gt_row gt_center">5.6 (Â±1.4)</td>
## <td headers="stat_1" class="gt_row gt_center">5.6 (Â±1.4)</td>
## <td headers="stat_2" class="gt_row gt_center">5.6 (Â±1.4)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Vitamin D</td>
## <td headers="stat_0" class="gt_row gt_center">48.5 (Â±20.2)</td>
## <td headers="stat_1" class="gt_row gt_center">47.4 (Â±21.0)</td>
## <td headers="stat_2" class="gt_row gt_center">48.5 (Â±20.2)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```r
Table_Blood_stratified <- import_merge_tables(table_name= "Table_Blood", feature="SEX", levels = c("Female", "Male"))
```

```
## [1] "C:/Users/felix/OneDrive - Uniklinik RWTH Aachen/public/projects/cca/tables/Table_Blood_SEX_Female_ukb.RDS"
## [1] "C:/Users/felix/OneDrive - Uniklinik RWTH Aachen/public/projects/cca/tables/Table_Blood_SEX_Male_ukb.RDS"
## <div id="zcdwgcgjiy" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#zcdwgcgjiy table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #zcdwgcgjiy thead, #zcdwgcgjiy tbody, #zcdwgcgjiy tfoot, #zcdwgcgjiy tr, #zcdwgcgjiy td, #zcdwgcgjiy th {
##   border-style: none;
## }
## 
## #zcdwgcgjiy p {
##   margin: 0;
##   padding: 0;
## }
## 
## #zcdwgcgjiy .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #zcdwgcgjiy .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #zcdwgcgjiy .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #zcdwgcgjiy .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #zcdwgcgjiy .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #zcdwgcgjiy .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #zcdwgcgjiy .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #zcdwgcgjiy .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #zcdwgcgjiy .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #zcdwgcgjiy .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #zcdwgcgjiy .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #zcdwgcgjiy .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #zcdwgcgjiy .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #zcdwgcgjiy .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #zcdwgcgjiy .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #zcdwgcgjiy .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #zcdwgcgjiy .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #zcdwgcgjiy .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #zcdwgcgjiy .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #zcdwgcgjiy .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #zcdwgcgjiy .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #zcdwgcgjiy .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #zcdwgcgjiy .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #zcdwgcgjiy .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #zcdwgcgjiy .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #zcdwgcgjiy .gt_left {
##   text-align: left;
## }
## 
## #zcdwgcgjiy .gt_center {
##   text-align: center;
## }
## 
## #zcdwgcgjiy .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #zcdwgcgjiy .gt_font_normal {
##   font-weight: normal;
## }
## 
## #zcdwgcgjiy .gt_font_bold {
##   font-weight: bold;
## }
## 
## #zcdwgcgjiy .gt_font_italic {
##   font-style: italic;
## }
## 
## #zcdwgcgjiy .gt_super {
##   font-size: 65%;
## }
## 
## #zcdwgcgjiy .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #zcdwgcgjiy .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #zcdwgcgjiy .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #zcdwgcgjiy .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #zcdwgcgjiy .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #zcdwgcgjiy .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #zcdwgcgjiy .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings gt_spanner_row">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="2" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_center gt_columns_top_border gt_column_spanner_outer" rowspan="1" colspan="5" scope="colgroup" id="Female">
##         <span class="gt_column_spanner">Female</span>
##       </th>
##       <th class="gt_center gt_columns_top_border gt_column_spanner_outer" rowspan="1" colspan="5" scope="colgroup" id="Male">
##         <span class="gt_column_spanner">Male</span>
##       </th>
##     </tr>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 273225&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 273225<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 423&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 423<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 272802&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 272802<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 229001&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 229001<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 425&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 425<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 228576&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 228576<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alanine aminotransferase</td>
## <td headers="stat_0_1" class="gt_row gt_center">20.5 (Â±11.8)</td>
## <td headers="stat_1_1" class="gt_row gt_center">22.7 (Â±14.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">20.5 (Â±11.8)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.062</td>
## <td headers="stat_0_2" class="gt_row gt_center">27.2 (Â±14.8)</td>
## <td headers="stat_1_2" class="gt_row gt_center">30.3 (Â±23.4)</td>
## <td headers="stat_2_2" class="gt_row gt_center">27.2 (Â±14.8)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.008</td>
## <td headers="q.value_2" class="gt_row gt_center">0.4</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Albumin</td>
## <td headers="stat_0_1" class="gt_row gt_center">45.0 (Â±2.4)</td>
## <td headers="stat_1_1" class="gt_row gt_center">44.9 (Â±2.5)</td>
## <td headers="stat_2_1" class="gt_row gt_center">45.0 (Â±2.4)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.6</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">45.5 (Â±2.4)</td>
## <td headers="stat_1_2" class="gt_row gt_center">44.8 (Â±2.5)</td>
## <td headers="stat_2_2" class="gt_row gt_center">45.5 (Â±2.4)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alkaline phosphatase</td>
## <td headers="stat_0_1" class="gt_row gt_center">84.8 (Â±26.5)</td>
## <td headers="stat_1_1" class="gt_row gt_center">94.5 (Â±38.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">84.8 (Â±26.5)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">82.3 (Â±24.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">95.5 (Â±86.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">82.3 (Â±23.9)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.002</td>
## <td headers="q.value_2" class="gt_row gt_center">0.10</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Apolipoprotein A</td>
## <td headers="stat_0_1" class="gt_row gt_center">1.6 (Â±0.2)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1.6 (Â±0.3)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1.6 (Â±0.2)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.4</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">1.4 (Â±0.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1.4 (Â±0.2)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1.4 (Â±0.2)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.5</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Apolipoprotein B</td>
## <td headers="stat_0_1" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1.1 (Â±0.2)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.084</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.016</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Aspartate aminotransferase</td>
## <td headers="stat_0_1" class="gt_row gt_center">24.6 (Â±9.3)</td>
## <td headers="stat_1_1" class="gt_row gt_center">26.7 (Â±14.3)</td>
## <td headers="stat_2_1" class="gt_row gt_center">24.6 (Â±9.3)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value_1" class="gt_row gt_center">0.2</td>
## <td headers="stat_0_2" class="gt_row gt_center">28.1 (Â±11.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">31.3 (Â±20.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">28.1 (Â±11.1)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.066</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Basophill (%)</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.0 (Â±0.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.0 (Â±0.1)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.3</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Basophill count</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.6 (Â±0.7)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.6 (Â±0.4)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.6 (Â±0.7)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.2</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.5 (Â±0.5)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.5 (Â±0.4)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.5 (Â±0.5)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.8</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">C-reactive protein</td>
## <td headers="stat_0_1" class="gt_row gt_center">5.9 (Â±1.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">6.0 (Â±1.1)</td>
## <td headers="stat_2_1" class="gt_row gt_center">5.9 (Â±1.1)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.053</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">5.5 (Â±1.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">5.3 (Â±1.2)</td>
## <td headers="stat_2_2" class="gt_row gt_center">5.5 (Â±1.1)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.042</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Calcium</td>
## <td headers="stat_0_1" class="gt_row gt_center">2.7 (Â±4.2)</td>
## <td headers="stat_1_1" class="gt_row gt_center">3.5 (Â±4.5)</td>
## <td headers="stat_2_1" class="gt_row gt_center">2.7 (Â±4.2)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.026</td>
## <td headers="stat_0_2" class="gt_row gt_center">2.5 (Â±4.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2.9 (Â±3.8)</td>
## <td headers="stat_2_2" class="gt_row gt_center">2.5 (Â±4.2)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.023</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cholesterol</td>
## <td headers="stat_0_1" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="stat_2_1" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.050</td>
## <td headers="stat_0_2" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">2.4 (Â±0.1)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value_2" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Creatinine</td>
## <td headers="stat_0_1" class="gt_row gt_center">64.9 (Â±13.4)</td>
## <td headers="stat_1_1" class="gt_row gt_center">65.2 (Â±11.6)</td>
## <td headers="stat_2_1" class="gt_row gt_center">64.9 (Â±13.4)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.6</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">81.1 (Â±18.6)</td>
## <td headers="stat_1_2" class="gt_row gt_center">80.8 (Â±15.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">81.1 (Â±18.7)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.6</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cystatin C</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1.0 (Â±0.2)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.9 (Â±0.2)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.005</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Direct bilirubin</td>
## <td headers="stat_0_1" class="gt_row gt_center">1.7 (Â±0.6)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1.7 (Â±0.7)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1.7 (Â±0.6)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">2.0 (Â±0.9)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2.1 (Â±1.6)</td>
## <td headers="stat_2_2" class="gt_row gt_center">2.0 (Â±0.9)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.088</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Eosinophill (%)</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.4</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.2 (Â±0.1)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.062</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Eosinophill count</td>
## <td headers="stat_0_1" class="gt_row gt_center">2.4 (Â±1.7)</td>
## <td headers="stat_1_1" class="gt_row gt_center">2.4 (Â±1.4)</td>
## <td headers="stat_2_1" class="gt_row gt_center">2.4 (Â±1.7)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.6</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">2.8 (Â±1.9)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2.8 (Â±2.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">2.8 (Â±1.9)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.5</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Erythrocyte distribution width</td>
## <td headers="stat_0_1" class="gt_row gt_center">13.5 (Â±1.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">13.6 (Â±1.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">13.5 (Â±1.0)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.011</td>
## <td headers="q.value_1" class="gt_row gt_center">0.7</td>
## <td headers="stat_0_2" class="gt_row gt_center">15.0 (Â±1.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">14.9 (Â±1.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">15.0 (Â±1.0)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.3</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Erythrocytes</td>
## <td headers="stat_0_1" class="gt_row gt_center">39.3 (Â±2.8)</td>
## <td headers="stat_1_1" class="gt_row gt_center">39.7 (Â±2.8)</td>
## <td headers="stat_2_1" class="gt_row gt_center">39.3 (Â±2.8)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value_1" class="gt_row gt_center">0.2</td>
## <td headers="stat_0_2" class="gt_row gt_center">43.2 (Â±3.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">43.1 (Â±3.2)</td>
## <td headers="stat_2_2" class="gt_row gt_center">43.2 (Â±3.0)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.4</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Gamma glutamyltransferase</td>
## <td headers="stat_0_1" class="gt_row gt_center">30.8 (Â±32.5)</td>
## <td headers="stat_1_1" class="gt_row gt_center">41.5 (Â±54.2)</td>
## <td headers="stat_2_1" class="gt_row gt_center">30.8 (Â±32.5)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.003</td>
## <td headers="stat_0_2" class="gt_row gt_center">45.2 (Â±47.3)</td>
## <td headers="stat_1_2" class="gt_row gt_center">77.0 (Â±125.0)</td>
## <td headers="stat_2_2" class="gt_row gt_center">45.2 (Â±47.0)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Glucose</td>
## <td headers="stat_0_1" class="gt_row gt_center">5.1 (Â±1.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">5.2 (Â±1.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">5.1 (Â±1.0)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.026</td>
## <td headers="stat_0_2" class="gt_row gt_center">5.2 (Â±1.3)</td>
## <td headers="stat_1_2" class="gt_row gt_center">5.5 (Â±1.9)</td>
## <td headers="stat_2_2" class="gt_row gt_center">5.2 (Â±1.3)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.003</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Haematocrit</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.020</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.2</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Haemoglobin</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.4 (Â±0.2)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.041</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.4 (Â±0.2)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.069</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">HbA1c</td>
## <td headers="stat_0_1" class="gt_row gt_center">35.8 (Â±5.8)</td>
## <td headers="stat_1_1" class="gt_row gt_center">37.4 (Â±6.3)</td>
## <td headers="stat_2_1" class="gt_row gt_center">35.8 (Â±5.7)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">36.5 (Â±7.3)</td>
## <td headers="stat_1_2" class="gt_row gt_center">38.5 (Â±9.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">36.5 (Â±7.3)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">HDL cholesterol</td>
## <td headers="stat_0_1" class="gt_row gt_center">1.6 (Â±0.4)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1.5 (Â±0.4)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1.6 (Â±0.4)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.2</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">1.3 (Â±0.3)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1.3 (Â±0.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1.3 (Â±0.3)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.11</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">High light scatter reticulocyte (%)</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.012</td>
## <td headers="q.value_1" class="gt_row gt_center">0.7</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.3 (Â±0.1)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value_2" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">High light scatter reticulocyte count</td>
## <td headers="stat_0_1" class="gt_row gt_center">2.0 (Â±1.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">2.1 (Â±0.6)</td>
## <td headers="stat_2_1" class="gt_row gt_center">2.0 (Â±1.0)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.10</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">1.9 (Â±1.3)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1.9 (Â±0.9)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1.9 (Â±1.3)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.4</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">IGF-1</td>
## <td headers="stat_0_1" class="gt_row gt_center">21.0 (Â±5.6)</td>
## <td headers="stat_1_1" class="gt_row gt_center">19.4 (Â±5.7)</td>
## <td headers="stat_2_1" class="gt_row gt_center">21.0 (Â±5.6)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">21.9 (Â±5.4)</td>
## <td headers="stat_1_2" class="gt_row gt_center">20.1 (Â±5.5)</td>
## <td headers="stat_2_2" class="gt_row gt_center">21.9 (Â±5.4)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Immature reticulocyte fraction</td>
## <td headers="stat_0_1" class="gt_row gt_center">29.7 (Â±7.2)</td>
## <td headers="stat_1_1" class="gt_row gt_center">29.4 (Â±6.7)</td>
## <td headers="stat_2_1" class="gt_row gt_center">29.7 (Â±7.2)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.3</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">27.9 (Â±7.3)</td>
## <td headers="stat_1_2" class="gt_row gt_center">27.1 (Â±7.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">27.9 (Â±7.3)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.023</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">LDL direct</td>
## <td headers="stat_0_1" class="gt_row gt_center">3.6 (Â±0.8)</td>
## <td headers="stat_1_1" class="gt_row gt_center">3.7 (Â±0.9)</td>
## <td headers="stat_2_1" class="gt_row gt_center">3.6 (Â±0.8)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.11</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">3.5 (Â±0.8)</td>
## <td headers="stat_1_2" class="gt_row gt_center">3.3 (Â±0.9)</td>
## <td headers="stat_2_2" class="gt_row gt_center">3.5 (Â±0.8)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.007</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Leukocyte count</td>
## <td headers="stat_0_1" class="gt_row gt_center">31.3 (Â±1.9)</td>
## <td headers="stat_1_1" class="gt_row gt_center">31.4 (Â±1.7)</td>
## <td headers="stat_2_1" class="gt_row gt_center">31.3 (Â±1.9)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.3</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">31.7 (Â±1.8)</td>
## <td headers="stat_1_2" class="gt_row gt_center">31.9 (Â±1.7)</td>
## <td headers="stat_2_2" class="gt_row gt_center">31.7 (Â±1.8)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value_2" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Lipoprotein A</td>
## <td headers="stat_0_1" class="gt_row gt_center">45.1 (Â±42.7)</td>
## <td headers="stat_1_1" class="gt_row gt_center">46.8 (Â±45.7)</td>
## <td headers="stat_2_1" class="gt_row gt_center">45.1 (Â±42.7)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.4</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">44.1 (Â±42.4)</td>
## <td headers="stat_1_2" class="gt_row gt_center">41.4 (Â±39.5)</td>
## <td headers="stat_2_2" class="gt_row gt_center">44.1 (Â±42.4)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.2</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Lymphocyte (%)</td>
## <td headers="stat_0_1" class="gt_row gt_center">34.4 (Â±1.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">34.3 (Â±0.9)</td>
## <td headers="stat_2_1" class="gt_row gt_center">34.4 (Â±1.1)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.11</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">34.6 (Â±1.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">34.6 (Â±0.9)</td>
## <td headers="stat_2_2" class="gt_row gt_center">34.6 (Â±1.0)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.6</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Lymphocyte count</td>
## <td headers="stat_0_1" class="gt_row gt_center">90.9 (Â±4.6)</td>
## <td headers="stat_1_1" class="gt_row gt_center">91.3 (Â±4.3)</td>
## <td headers="stat_2_1" class="gt_row gt_center">90.9 (Â±4.6)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.048</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">91.4 (Â±4.4)</td>
## <td headers="stat_1_2" class="gt_row gt_center">92.2 (Â±4.4)</td>
## <td headers="stat_2_2" class="gt_row gt_center">91.4 (Â±4.4)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.026</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MCH</td>
## <td headers="stat_0_1" class="gt_row gt_center">9.4 (Â±1.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">9.4 (Â±1.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">9.4 (Â±1.1)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.8</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">9.3 (Â±1.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">9.4 (Â±1.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">9.3 (Â±1.1)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.3</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MCHC</td>
## <td headers="stat_0_1" class="gt_row gt_center">105.6 (Â±7.6)</td>
## <td headers="stat_1_1" class="gt_row gt_center">105.8 (Â±8.3)</td>
## <td headers="stat_2_1" class="gt_row gt_center">105.6 (Â±7.6)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.5</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">106.3 (Â±7.6)</td>
## <td headers="stat_1_2" class="gt_row gt_center">107.5 (Â±7.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">106.3 (Â±7.6)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.046</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">MCV</td>
## <td headers="stat_0_1" class="gt_row gt_center">83.0 (Â±5.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">83.1 (Â±5.4)</td>
## <td headers="stat_2_1" class="gt_row gt_center">83.0 (Â±5.1)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.6</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">82.7 (Â±5.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">83.8 (Â±5.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">82.7 (Â±5.2)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.006</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Mean platelet volume</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.5 (Â±0.2)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.4 (Â±0.3)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.040</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.5 (Â±0.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.5 (Â±0.2)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.5 (Â±0.2)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.011</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Mean reticulocyte volume</td>
## <td headers="stat_0_1" class="gt_row gt_center">6.6 (Â±2.5)</td>
## <td headers="stat_1_1" class="gt_row gt_center">6.6 (Â±1.9)</td>
## <td headers="stat_2_1" class="gt_row gt_center">6.6 (Â±2.5)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.6</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">7.6 (Â±2.7)</td>
## <td headers="stat_1_2" class="gt_row gt_center">7.7 (Â±2.2)</td>
## <td headers="stat_2_2" class="gt_row gt_center">7.6 (Â±2.7)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.3</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Mean sphered cell volume</td>
## <td headers="stat_0_1" class="gt_row gt_center">4.2 (Â±1.4)</td>
## <td headers="stat_1_1" class="gt_row gt_center">4.4 (Â±1.5)</td>
## <td headers="stat_2_1" class="gt_row gt_center">4.2 (Â±1.4)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value_1" class="gt_row gt_center">0.3</td>
## <td headers="stat_0_2" class="gt_row gt_center">4.3 (Â±1.4)</td>
## <td headers="stat_1_2" class="gt_row gt_center">4.5 (Â±1.4)</td>
## <td headers="stat_2_2" class="gt_row gt_center">4.3 (Â±1.4)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.035</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Monocyte count</td>
## <td headers="stat_0_1" class="gt_row gt_center">60.6 (Â±8.2)</td>
## <td headers="stat_1_1" class="gt_row gt_center">61.1 (Â±7.5)</td>
## <td headers="stat_2_1" class="gt_row gt_center">60.6 (Â±8.2)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.2</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">61.2 (Â±8.5)</td>
## <td headers="stat_1_2" class="gt_row gt_center">61.8 (Â±8.5)</td>
## <td headers="stat_2_2" class="gt_row gt_center">61.2 (Â±8.5)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.14</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Monocyte percentage</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.3</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.0 (Â±0.0)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Neutrophill count</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.0 (Â±0.4)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.0 (Â±0.3)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.0 (Â±0.4)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.3</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.0 (Â±0.4)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.0 (Â±0.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.0 (Â±0.4)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Neutrophill percentage</td>
## <td headers="stat_0_1" class="gt_row gt_center">265.1 (Â±58.8)</td>
## <td headers="stat_1_1" class="gt_row gt_center">266.9 (Â±61.6)</td>
## <td headers="stat_2_1" class="gt_row gt_center">265.1 (Â±58.8)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.5</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">238.6 (Â±54.9)</td>
## <td headers="stat_1_2" class="gt_row gt_center">234.0 (Â±55.6)</td>
## <td headers="stat_2_2" class="gt_row gt_center">238.6 (Â±54.9)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.091</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Nucleated red blood cell (%)</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.6</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.2 (Â±0.0)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.14</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Nucleated red blood cell count</td>
## <td headers="stat_0_1" class="gt_row gt_center">16.4 (Â±0.5)</td>
## <td headers="stat_1_1" class="gt_row gt_center">16.5 (Â±0.5)</td>
## <td headers="stat_2_1" class="gt_row gt_center">16.4 (Â±0.5)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.058</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">16.6 (Â±0.5)</td>
## <td headers="stat_1_2" class="gt_row gt_center">16.6 (Â±0.5)</td>
## <td headers="stat_2_2" class="gt_row gt_center">16.6 (Â±0.5)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.2</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Phosphate</td>
## <td headers="stat_0_1" class="gt_row gt_center">1.2 (Â±0.1)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1.2 (Â±0.1)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1.2 (Â±0.1)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.2</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">1.1 (Â±0.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1.1 (Â±0.2)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1.1 (Â±0.2)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.070</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Platelet count</td>
## <td headers="stat_0_1" class="gt_row gt_center">4.3 (Â±0.3)</td>
## <td headers="stat_1_1" class="gt_row gt_center">4.4 (Â±0.3)</td>
## <td headers="stat_2_1" class="gt_row gt_center">4.3 (Â±0.3)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.12</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">4.7 (Â±0.4)</td>
## <td headers="stat_1_2" class="gt_row gt_center">4.7 (Â±0.4)</td>
## <td headers="stat_2_2" class="gt_row gt_center">4.7 (Â±0.4)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.006</td>
## <td headers="q.value_2" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Platelet crit</td>
## <td headers="stat_0_1" class="gt_row gt_center">13.5 (Â±1.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">13.6 (Â±1.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">13.5 (Â±1.0)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.14</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">13.5 (Â±0.9)</td>
## <td headers="stat_1_2" class="gt_row gt_center">13.6 (Â±1.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">13.5 (Â±0.9)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.062</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Platelet distribution width</td>
## <td headers="stat_0_1" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.2</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0.1 (Â±0.0)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.8</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Reticulocyte (%)</td>
## <td headers="stat_0_1" class="gt_row gt_center">1.3 (Â±0.9)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1.4 (Â±0.5)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1.3 (Â±0.9)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.3</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">1.4 (Â±0.9)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1.4 (Â±0.5)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1.4 (Â±0.9)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.6</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Reticulocyte count</td>
## <td headers="stat_0_1" class="gt_row gt_center">6.9 (Â±2.0)</td>
## <td headers="stat_1_1" class="gt_row gt_center">7.1 (Â±1.9)</td>
## <td headers="stat_2_1" class="gt_row gt_center">6.9 (Â±2.0)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.004</td>
## <td headers="q.value_1" class="gt_row gt_center">0.2</td>
## <td headers="stat_0_2" class="gt_row gt_center">6.9 (Â±2.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">7.2 (Â±1.9)</td>
## <td headers="stat_2_2" class="gt_row gt_center">6.9 (Â±2.1)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.038</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">SHBG</td>
## <td headers="stat_0_1" class="gt_row gt_center">60.3 (Â±28.6)</td>
## <td headers="stat_1_1" class="gt_row gt_center">58.3 (Â±27.0)</td>
## <td headers="stat_2_1" class="gt_row gt_center">60.3 (Â±28.6)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.13</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">41.3 (Â±16.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">43.6 (Â±16.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">41.3 (Â±16.1)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.004</td>
## <td headers="q.value_2" class="gt_row gt_center">0.2</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Testosterone</td>
## <td headers="stat_0_1" class="gt_row gt_center">2.3 (Â±2.3)</td>
## <td headers="stat_1_1" class="gt_row gt_center">2.6 (Â±2.5)</td>
## <td headers="stat_2_1" class="gt_row gt_center">2.3 (Â±2.3)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.017</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">11.6 (Â±3.8)</td>
## <td headers="stat_1_2" class="gt_row gt_center">11.4 (Â±4.1)</td>
## <td headers="stat_2_2" class="gt_row gt_center">11.6 (Â±3.8)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.3</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Total bilirubin</td>
## <td headers="stat_0_1" class="gt_row gt_center">8.2 (Â±3.6)</td>
## <td headers="stat_1_1" class="gt_row gt_center">8.1 (Â±3.7)</td>
## <td headers="stat_2_1" class="gt_row gt_center">8.2 (Â±3.6)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.5</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">10.2 (Â±4.8)</td>
## <td headers="stat_1_2" class="gt_row gt_center">10.0 (Â±4.6)</td>
## <td headers="stat_2_2" class="gt_row gt_center">10.2 (Â±4.8)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.4</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Total protein</td>
## <td headers="stat_0_1" class="gt_row gt_center">72.4 (Â±3.8)</td>
## <td headers="stat_1_1" class="gt_row gt_center">72.6 (Â±3.9)</td>
## <td headers="stat_2_1" class="gt_row gt_center">72.4 (Â±3.8)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.4</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">72.6 (Â±3.8)</td>
## <td headers="stat_1_2" class="gt_row gt_center">72.5 (Â±3.9)</td>
## <td headers="stat_2_2" class="gt_row gt_center">72.6 (Â±3.8)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.5</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Triglycerides</td>
## <td headers="stat_0_1" class="gt_row gt_center">1.6 (Â±0.8)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1.8 (Â±1.1)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1.6 (Â±0.8)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">2.0 (Â±1.1)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2.2 (Â±1.3)</td>
## <td headers="stat_2_2" class="gt_row gt_center">2.0 (Â±1.1)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.002</td>
## <td headers="q.value_2" class="gt_row gt_center">0.11</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Urate</td>
## <td headers="stat_0_1" class="gt_row gt_center">273.7 (Â±64.6)</td>
## <td headers="stat_1_1" class="gt_row gt_center">292.2 (Â±77.4)</td>
## <td headers="stat_2_1" class="gt_row gt_center">273.6 (Â±64.6)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">351.6 (Â±70.4)</td>
## <td headers="stat_1_2" class="gt_row gt_center">354.3 (Â±76.5)</td>
## <td headers="stat_2_2" class="gt_row gt_center">351.6 (Â±70.4)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.5</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Urea</td>
## <td headers="stat_0_1" class="gt_row gt_center">5.2 (Â±1.3)</td>
## <td headers="stat_1_1" class="gt_row gt_center">5.4 (Â±1.3)</td>
## <td headers="stat_2_1" class="gt_row gt_center">5.2 (Â±1.3)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.003</td>
## <td headers="q.value_1" class="gt_row gt_center">0.2</td>
## <td headers="stat_0_2" class="gt_row gt_center">5.6 (Â±1.4)</td>
## <td headers="stat_1_2" class="gt_row gt_center">5.6 (Â±1.4)</td>
## <td headers="stat_2_2" class="gt_row gt_center">5.6 (Â±1.4)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.6</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Vitamin D</td>
## <td headers="stat_0_1" class="gt_row gt_center">48.7 (Â±19.7)</td>
## <td headers="stat_1_1" class="gt_row gt_center">48.7 (Â±19.4)</td>
## <td headers="stat_2_1" class="gt_row gt_center">48.7 (Â±19.7)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">48.5 (Â±20.2)</td>
## <td headers="stat_1_2" class="gt_row gt_center">47.4 (Â±21.0)</td>
## <td headers="stat_2_2" class="gt_row gt_center">48.5 (Â±20.2)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.3</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="11"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Mean (Â±SD)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="11"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Welch Two Sample t-test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="11"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```r
# Stratified "ICD Table"
split_create_merge_tables(df_tbl_icd, table_name= "Table_ICD", feature="SEX", enforced_order=FALSE, remove_SEX=TRUE, export_RDS=TRUE, create_binary_table = TRUE)
```

```
## Warning for variable 'Acute renal failure':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic fatty liver':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic fibrosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic hepatic failure':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic hepatitis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic liver disease':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic liver disease, unspecified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Ascites':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Autoimmune hepatitis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Biliary cirrhosis, unspecified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Biliary Cyst':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Cholangitis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Chronic Hepatitis B':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Chronic Hepatitis C':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Chronic kidney disease':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Colorectal cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Complications of cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Disease of biliary tract, unspecified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Esophageal cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Fibrosis and cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Fistula of bile duct':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Gastric cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Granulomatous hepatitis, not elsewhere classified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatic fibrosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatic fibrosis and sclerosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatic sclerosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatomegaly with splenomegaly, not elsewhere classified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatomegaly, not elsewhere classified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatorenal Syndrome':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'HSM':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Inflammatory liver disease':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Inflammatory liver disease, unspecified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Jaundice':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'NAFLD':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'NASH':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Nonspecific reactive hepatitis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Obstruction of bile duct':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Oesophageal varices w bleeding':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Oesophageal varices w_o bleeding':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Other and unspecified cirrhosis of liver':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Other specified diseases of biliary tract':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Pancreatic cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Perforation of bile duct':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Primary biliary cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Secondary biliary cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Sepsis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Small intestine cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Spasm of sphincter of Oddi':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Splenomegaly, not elsewhere classified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="pmwtnqmgwu" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#pmwtnqmgwu table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #pmwtnqmgwu thead, #pmwtnqmgwu tbody, #pmwtnqmgwu tfoot, #pmwtnqmgwu tr, #pmwtnqmgwu td, #pmwtnqmgwu th {
##   border-style: none;
## }
## 
## #pmwtnqmgwu p {
##   margin: 0;
##   padding: 0;
## }
## 
## #pmwtnqmgwu .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #pmwtnqmgwu .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #pmwtnqmgwu .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #pmwtnqmgwu .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #pmwtnqmgwu .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #pmwtnqmgwu .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #pmwtnqmgwu .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #pmwtnqmgwu .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #pmwtnqmgwu .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #pmwtnqmgwu .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #pmwtnqmgwu .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #pmwtnqmgwu .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #pmwtnqmgwu .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #pmwtnqmgwu .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #pmwtnqmgwu .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #pmwtnqmgwu .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #pmwtnqmgwu .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #pmwtnqmgwu .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #pmwtnqmgwu .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #pmwtnqmgwu .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #pmwtnqmgwu .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #pmwtnqmgwu .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #pmwtnqmgwu .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #pmwtnqmgwu .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #pmwtnqmgwu .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #pmwtnqmgwu .gt_left {
##   text-align: left;
## }
## 
## #pmwtnqmgwu .gt_center {
##   text-align: center;
## }
## 
## #pmwtnqmgwu .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #pmwtnqmgwu .gt_font_normal {
##   font-weight: normal;
## }
## 
## #pmwtnqmgwu .gt_font_bold {
##   font-weight: bold;
## }
## 
## #pmwtnqmgwu .gt_font_italic {
##   font-style: italic;
## }
## 
## #pmwtnqmgwu .gt_super {
##   font-size: 65%;
## }
## 
## #pmwtnqmgwu .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #pmwtnqmgwu .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #pmwtnqmgwu .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #pmwtnqmgwu .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #pmwtnqmgwu .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #pmwtnqmgwu .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #pmwtnqmgwu .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 273225&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 273225<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 423&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 423<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 272802&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 272802<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Acute renal failure</td>
## <td headers="stat_0" class="gt_row gt_center">1,392 (0.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,386 (0.5%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.022</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">78 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">78 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic fatty liver</td>
## <td headers="stat_0" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic fibrosis</td>
## <td headers="stat_0" class="gt_row gt_center">5 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">5 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic hepatic failure</td>
## <td headers="stat_0" class="gt_row gt_center">20 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">20 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic hepatitis</td>
## <td headers="stat_0" class="gt_row gt_center">38 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">38 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic liver disease</td>
## <td headers="stat_0" class="gt_row gt_center">196 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">195 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.7</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic liver disease, unspecified</td>
## <td headers="stat_0" class="gt_row gt_center">137 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">136 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Arterial hypertension</td>
## <td headers="stat_0" class="gt_row gt_center">72,394 (26%)</td>
## <td headers="stat_1" class="gt_row gt_center">150 (35%)</td>
## <td headers="stat_2" class="gt_row gt_center">72,244 (26%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.002</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ascites</td>
## <td headers="stat_0" class="gt_row gt_center">816 (0.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">15 (3.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">801 (0.3%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Autoimmune hepatitis</td>
## <td headers="stat_0" class="gt_row gt_center">129 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">129 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Biliary cirrhosis, unspecified</td>
## <td headers="stat_0" class="gt_row gt_center">48 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">47 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.12</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Biliary Cyst</td>
## <td headers="stat_0" class="gt_row gt_center">8 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">8 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cholangitis</td>
## <td headers="stat_0" class="gt_row gt_center">187 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">29 (6.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">158 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cholelithiasis</td>
## <td headers="stat_0" class="gt_row gt_center">14,678 (5.4%)</td>
## <td headers="stat_1" class="gt_row gt_center">63 (15%)</td>
## <td headers="stat_2" class="gt_row gt_center">14,615 (5.4%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Chronic Hepatitis B</td>
## <td headers="stat_0" class="gt_row gt_center">85 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">85 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Chronic Hepatitis C</td>
## <td headers="stat_0" class="gt_row gt_center">153 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">153 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Chronic kidney disease</td>
## <td headers="stat_0" class="gt_row gt_center">2,137 (0.8%)</td>
## <td headers="stat_1" class="gt_row gt_center">10 (2.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,127 (0.8%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.041</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Colorectal cancer</td>
## <td headers="stat_0" class="gt_row gt_center">1,900 (0.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,894 (0.7%)</td>
## <td headers="p.value" class="gt_row gt_center">0.13</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Complications of cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">1,514 (0.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">38 (9.0%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,476 (0.5%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Disease of biliary tract, unspecified</td>
## <td headers="stat_0" class="gt_row gt_center">51 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">47 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">DM</td>
## <td headers="stat_0" class="gt_row gt_center">9,539 (3.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">30 (7.1%)</td>
## <td headers="stat_2" class="gt_row gt_center">9,509 (3.5%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.006</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_ALT.x</td>
## <td headers="stat_0" class="gt_row gt_center">17,692 (6.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">41 (9.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">17,651 (6.5%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.010</td>
## <td headers="q.value" class="gt_row gt_center">0.6</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_ALT.y</td>
## <td headers="stat_0" class="gt_row gt_center">17,692 (6.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">41 (9.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">17,651 (6.5%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.010</td>
## <td headers="q.value" class="gt_row gt_center">0.6</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AP.x</td>
## <td headers="stat_0" class="gt_row gt_center">45,310 (17%)</td>
## <td headers="stat_1" class="gt_row gt_center">115 (27%)</td>
## <td headers="stat_2" class="gt_row gt_center">45,195 (17%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AP.y</td>
## <td headers="stat_0" class="gt_row gt_center">45,310 (17%)</td>
## <td headers="stat_1" class="gt_row gt_center">115 (27%)</td>
## <td headers="stat_2" class="gt_row gt_center">45,195 (17%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AST.x</td>
## <td headers="stat_0" class="gt_row gt_center">15,361 (5.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">45 (11%)</td>
## <td headers="stat_2" class="gt_row gt_center">15,316 (5.6%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AST.y</td>
## <td headers="stat_0" class="gt_row gt_center">15,361 (5.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">45 (11%)</td>
## <td headers="stat_2" class="gt_row gt_center">15,316 (5.6%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_GGT.x</td>
## <td headers="stat_0" class="gt_row gt_center">41,886 (15%)</td>
## <td headers="stat_1" class="gt_row gt_center">105 (25%)</td>
## <td headers="stat_2" class="gt_row gt_center">41,781 (15%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_GGT.y</td>
## <td headers="stat_0" class="gt_row gt_center">41,886 (15%)</td>
## <td headers="stat_1" class="gt_row gt_center">105 (25%)</td>
## <td headers="stat_2" class="gt_row gt_center">41,781 (15%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_Liver_Enzymes.x</td>
## <td headers="stat_0" class="gt_row gt_center">51,905 (19%)</td>
## <td headers="stat_1" class="gt_row gt_center">121 (29%)</td>
## <td headers="stat_2" class="gt_row gt_center">51,784 (19%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_Liver_Enzymes.y</td>
## <td headers="stat_0" class="gt_row gt_center">51,905 (19%)</td>
## <td headers="stat_1" class="gt_row gt_center">121 (29%)</td>
## <td headers="stat_2" class="gt_row gt_center">51,784 (19%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Esophageal cancer</td>
## <td headers="stat_0" class="gt_row gt_center">168 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">166 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.015</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Fibrosis and cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">511 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">8 (1.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">503 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Fistula of bile duct</td>
## <td headers="stat_0" class="gt_row gt_center">10 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">10 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Gastric cancer</td>
## <td headers="stat_0" class="gt_row gt_center">154 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">152 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.010</td>
## <td headers="q.value" class="gt_row gt_center">0.6</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Granulomatous hepatitis, not elsewhere classified</td>
## <td headers="stat_0" class="gt_row gt_center">10 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">10 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatic fibrosis</td>
## <td headers="stat_0" class="gt_row gt_center">46 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">45 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.11</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatic fibrosis and sclerosis</td>
## <td headers="stat_0" class="gt_row gt_center">2 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">2 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatic sclerosis</td>
## <td headers="stat_0" class="gt_row gt_center">4 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">4 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatomegaly with splenomegaly, not elsewhere classified</td>
## <td headers="stat_0" class="gt_row gt_center">33 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">32 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.047</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatomegaly, not elsewhere classified</td>
## <td headers="stat_0" class="gt_row gt_center">97 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">95 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.032</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatorenal Syndrome</td>
## <td headers="stat_0" class="gt_row gt_center">12 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">12 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">HSM</td>
## <td headers="stat_0" class="gt_row gt_center">239 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">235 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">IBD</td>
## <td headers="stat_0" class="gt_row gt_center">3,347 (1.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">12 (2.8%)</td>
## <td headers="stat_2" class="gt_row gt_center">3,335 (1.2%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Inflammatory liver disease</td>
## <td headers="stat_0" class="gt_row gt_center">1,223 (0.4%)</td>
## <td headers="stat_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,217 (0.4%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.009</td>
## <td headers="q.value" class="gt_row gt_center">0.6</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Inflammatory liver disease, unspecified</td>
## <td headers="stat_0" class="gt_row gt_center">374 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">373 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Jaundice</td>
## <td headers="stat_0" class="gt_row gt_center">591 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">22 (5.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">569 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">NAFLD</td>
## <td headers="stat_0" class="gt_row gt_center">752 (0.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">748 (0.3%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.030</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">NASH</td>
## <td headers="stat_0" class="gt_row gt_center">67 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">66 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Nonspecific reactive hepatitis</td>
## <td headers="stat_0" class="gt_row gt_center">3 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">3 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Obstruction of bile duct</td>
## <td headers="stat_0" class="gt_row gt_center">435 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">39 (9.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">396 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Oesophageal varices w bleeding</td>
## <td headers="stat_0" class="gt_row gt_center">36 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">32 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Oesophageal varices w_o bleeding</td>
## <td headers="stat_0" class="gt_row gt_center">182 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">179 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.002</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Other and unspecified cirrhosis of liver</td>
## <td headers="stat_0" class="gt_row gt_center">327 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">321 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Other specified diseases of biliary tract</td>
## <td headers="stat_0" class="gt_row gt_center">623 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">15 (3.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">608 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Pancreatic cancer</td>
## <td headers="stat_0" class="gt_row gt_center">257 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">19 (4.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">238 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Perforation of bile duct</td>
## <td headers="stat_0" class="gt_row gt_center">8 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">8 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Primary biliary cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">219 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">216 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.013</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Secondary biliary cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">6 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">6 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Sepsis</td>
## <td headers="stat_0" class="gt_row gt_center">1,830 (0.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">29 (6.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,801 (0.7%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Small intestine cancer</td>
## <td headers="stat_0" class="gt_row gt_center">77 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">73 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Spasm of sphincter of Oddi</td>
## <td headers="stat_0" class="gt_row gt_center">15 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">15 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Splenomegaly, not elsewhere classified</td>
## <td headers="stat_0" class="gt_row gt_center">123 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">122 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.5</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Stomach disorders</td>
## <td headers="stat_0" class="gt_row gt_center">27,747 (10%)</td>
## <td headers="stat_1" class="gt_row gt_center">66 (16%)</td>
## <td headers="stat_2" class="gt_row gt_center">27,681 (10%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.018</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```
## Warning for variable 'Acute renal failure':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 2L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic fatty liver':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic fibrosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic hepatic failure':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic hepatitis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic liver disease':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Alcoholic liver disease, unspecified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Ascites':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Autoimmune hepatitis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Biliary cirrhosis, unspecified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Biliary Cyst':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Cholangitis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Chronic Hepatitis B':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Chronic Hepatitis C':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Colorectal cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Complications of cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Disease of biliary tract, unspecified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Esophageal cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Fibrosis and cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Fistula of bile duct':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Gastric cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Granulomatous hepatitis, not elsewhere classified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatic fibrosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## There was an error in 'add_p()/add_difference()' for variable 'Hepatic fibrosis and sclerosis', p-value omitted:
## Error in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : 'x' und 'y' müssen mindestens 2 Stufen haben
## Warning for variable 'Hepatic sclerosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatomegaly with splenomegaly, not elsewhere classified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatomegaly, not elsewhere classified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Hepatorenal Syndrome':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'HSM':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Inflammatory liver disease':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Inflammatory liver disease, unspecified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Jaundice':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'NAFLD':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'NASH':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Nonspecific reactive hepatitis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Obstruction of bile duct':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Oesophageal varices w bleeding':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Oesophageal varices w_o bleeding':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Other and unspecified cirrhosis of liver':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Other specified diseases of biliary tract':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Pancreatic cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Perforation of bile duct':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Primary biliary cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Secondary biliary cirrhosis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Sepsis':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## Warning for variable 'Small intestine cancer':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## There was an error in 'add_p()/add_difference()' for variable 'Spasm of sphincter of Oddi', p-value omitted:
## Error in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : 'x' und 'y' müssen mindestens 2 Stufen haben
## Warning for variable 'Splenomegaly, not elsewhere classified':
## simpleWarning in stats::chisq.test(x = structure(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, : Chi-Quadrat-Approximation kann inkorrekt sein
## add_q: Adjusting p-values with
## `stats::p.adjust(x$table_body$p.value, method = "bonferroni")`
```

```
## <div id="mpjuzxdzjh" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#mpjuzxdzjh table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #mpjuzxdzjh thead, #mpjuzxdzjh tbody, #mpjuzxdzjh tfoot, #mpjuzxdzjh tr, #mpjuzxdzjh td, #mpjuzxdzjh th {
##   border-style: none;
## }
## 
## #mpjuzxdzjh p {
##   margin: 0;
##   padding: 0;
## }
## 
## #mpjuzxdzjh .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #mpjuzxdzjh .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #mpjuzxdzjh .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #mpjuzxdzjh .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #mpjuzxdzjh .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #mpjuzxdzjh .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #mpjuzxdzjh .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #mpjuzxdzjh .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #mpjuzxdzjh .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #mpjuzxdzjh .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #mpjuzxdzjh .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #mpjuzxdzjh .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #mpjuzxdzjh .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #mpjuzxdzjh .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #mpjuzxdzjh .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #mpjuzxdzjh .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #mpjuzxdzjh .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #mpjuzxdzjh .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #mpjuzxdzjh .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #mpjuzxdzjh .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #mpjuzxdzjh .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #mpjuzxdzjh .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #mpjuzxdzjh .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #mpjuzxdzjh .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #mpjuzxdzjh .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #mpjuzxdzjh .gt_left {
##   text-align: left;
## }
## 
## #mpjuzxdzjh .gt_center {
##   text-align: center;
## }
## 
## #mpjuzxdzjh .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #mpjuzxdzjh .gt_font_normal {
##   font-weight: normal;
## }
## 
## #mpjuzxdzjh .gt_font_bold {
##   font-weight: bold;
## }
## 
## #mpjuzxdzjh .gt_font_italic {
##   font-style: italic;
## }
## 
## #mpjuzxdzjh .gt_super {
##   font-size: 65%;
## }
## 
## #mpjuzxdzjh .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #mpjuzxdzjh .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #mpjuzxdzjh .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #mpjuzxdzjh .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #mpjuzxdzjh .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #mpjuzxdzjh .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #mpjuzxdzjh .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 229001&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 229001<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 425&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 425<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 228576&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 228576<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Acute renal failure</td>
## <td headers="stat_0" class="gt_row gt_center">2,591 (1.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">15 (3.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,576 (1.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">286 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">283 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.007</td>
## <td headers="q.value" class="gt_row gt_center">0.4</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic fatty liver</td>
## <td headers="stat_0" class="gt_row gt_center">74 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">74 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic fibrosis</td>
## <td headers="stat_0" class="gt_row gt_center">18 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">18 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic hepatic failure</td>
## <td headers="stat_0" class="gt_row gt_center">84 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">83 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.4</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic hepatitis</td>
## <td headers="stat_0" class="gt_row gt_center">127 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">125 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.009</td>
## <td headers="q.value" class="gt_row gt_center">0.6</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic liver disease</td>
## <td headers="stat_0" class="gt_row gt_center">749 (0.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">742 (0.3%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic liver disease, unspecified</td>
## <td headers="stat_0" class="gt_row gt_center">509 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">506 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center">0.11</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Arterial hypertension</td>
## <td headers="stat_0" class="gt_row gt_center">80,216 (35%)</td>
## <td headers="stat_1" class="gt_row gt_center">214 (50%)</td>
## <td headers="stat_2" class="gt_row gt_center">80,002 (35%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ascites</td>
## <td headers="stat_0" class="gt_row gt_center">724 (0.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">11 (2.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">713 (0.3%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Autoimmune hepatitis</td>
## <td headers="stat_0" class="gt_row gt_center">28 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">28 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Biliary cirrhosis, unspecified</td>
## <td headers="stat_0" class="gt_row gt_center">15 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Biliary Cyst</td>
## <td headers="stat_0" class="gt_row gt_center">7 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">7 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cholangitis</td>
## <td headers="stat_0" class="gt_row gt_center">260 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">23 (5.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">237 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cholelithiasis</td>
## <td headers="stat_0" class="gt_row gt_center">5,276 (2.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">31 (7.3%)</td>
## <td headers="stat_2" class="gt_row gt_center">5,245 (2.3%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Chronic Hepatitis B</td>
## <td headers="stat_0" class="gt_row gt_center">161 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">160 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.7</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Chronic Hepatitis C</td>
## <td headers="stat_0" class="gt_row gt_center">297 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">296 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Chronic kidney disease</td>
## <td headers="stat_0" class="gt_row gt_center">3,007 (1.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">10 (2.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,997 (1.3%)</td>
## <td headers="p.value" class="gt_row gt_center">0.095</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Colorectal cancer</td>
## <td headers="stat_0" class="gt_row gt_center">2,588 (1.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">13 (3.1%)</td>
## <td headers="stat_2" class="gt_row gt_center">2,575 (1.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.026</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Complications of cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">1,628 (0.7%)</td>
## <td headers="stat_1" class="gt_row gt_center">37 (8.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,591 (0.7%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Disease of biliary tract, unspecified</td>
## <td headers="stat_0" class="gt_row gt_center">31 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">28 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">DM</td>
## <td headers="stat_0" class="gt_row gt_center">14,655 (6.4%)</td>
## <td headers="stat_1" class="gt_row gt_center">62 (15%)</td>
## <td headers="stat_2" class="gt_row gt_center">14,593 (6.4%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_ALT.x</td>
## <td headers="stat_0" class="gt_row gt_center">13,547 (5.9%)</td>
## <td headers="stat_1" class="gt_row gt_center">43 (10%)</td>
## <td headers="stat_2" class="gt_row gt_center">13,504 (5.9%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.022</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_ALT.y</td>
## <td headers="stat_0" class="gt_row gt_center">13,547 (5.9%)</td>
## <td headers="stat_1" class="gt_row gt_center">43 (10%)</td>
## <td headers="stat_2" class="gt_row gt_center">13,504 (5.9%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.022</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AP.x</td>
## <td headers="stat_0" class="gt_row gt_center">6,451 (2.8%)</td>
## <td headers="stat_1" class="gt_row gt_center">34 (8.0%)</td>
## <td headers="stat_2" class="gt_row gt_center">6,417 (2.8%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AP.y</td>
## <td headers="stat_0" class="gt_row gt_center">6,451 (2.8%)</td>
## <td headers="stat_1" class="gt_row gt_center">34 (8.0%)</td>
## <td headers="stat_2" class="gt_row gt_center">6,417 (2.8%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AST.x</td>
## <td headers="stat_0" class="gt_row gt_center">5,896 (2.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">29 (6.8%)</td>
## <td headers="stat_2" class="gt_row gt_center">5,867 (2.6%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AST.y</td>
## <td headers="stat_0" class="gt_row gt_center">5,896 (2.6%)</td>
## <td headers="stat_1" class="gt_row gt_center">29 (6.8%)</td>
## <td headers="stat_2" class="gt_row gt_center">5,867 (2.6%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_GGT.x</td>
## <td headers="stat_0" class="gt_row gt_center">38,334 (17%)</td>
## <td headers="stat_1" class="gt_row gt_center">122 (29%)</td>
## <td headers="stat_2" class="gt_row gt_center">38,212 (17%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_GGT.y</td>
## <td headers="stat_0" class="gt_row gt_center">38,334 (17%)</td>
## <td headers="stat_1" class="gt_row gt_center">122 (29%)</td>
## <td headers="stat_2" class="gt_row gt_center">38,212 (17%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_Liver_Enzymes.x</td>
## <td headers="stat_0" class="gt_row gt_center">44,946 (20%)</td>
## <td headers="stat_1" class="gt_row gt_center">129 (30%)</td>
## <td headers="stat_2" class="gt_row gt_center">44,817 (20%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_Liver_Enzymes.y</td>
## <td headers="stat_0" class="gt_row gt_center">44,946 (20%)</td>
## <td headers="stat_1" class="gt_row gt_center">129 (30%)</td>
## <td headers="stat_2" class="gt_row gt_center">44,817 (20%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Esophageal cancer</td>
## <td headers="stat_0" class="gt_row gt_center">510 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">510 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center">0.6</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Fibrosis and cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">547 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">9 (2.1%)</td>
## <td headers="stat_2" class="gt_row gt_center">538 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Fistula of bile duct</td>
## <td headers="stat_0" class="gt_row gt_center">7 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">6 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Gastric cancer</td>
## <td headers="stat_0" class="gt_row gt_center">373 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">370 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.030</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Granulomatous hepatitis, not elsewhere classified</td>
## <td headers="stat_0" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatic fibrosis</td>
## <td headers="stat_0" class="gt_row gt_center">73 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">73 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatic fibrosis and sclerosis</td>
## <td headers="stat_0" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatic sclerosis</td>
## <td headers="stat_0" class="gt_row gt_center">7 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">6 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatomegaly with splenomegaly, not elsewhere classified</td>
## <td headers="stat_0" class="gt_row gt_center">47 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">46 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.2</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatomegaly, not elsewhere classified</td>
## <td headers="stat_0" class="gt_row gt_center">155 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">152 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.002</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatorenal Syndrome</td>
## <td headers="stat_0" class="gt_row gt_center">37 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">37 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">HSM</td>
## <td headers="stat_0" class="gt_row gt_center">422 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">416 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">IBD</td>
## <td headers="stat_0" class="gt_row gt_center">3,041 (1.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">17 (4.0%)</td>
## <td headers="stat_2" class="gt_row gt_center">3,024 (1.3%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Inflammatory liver disease</td>
## <td headers="stat_0" class="gt_row gt_center">1,215 (0.5%)</td>
## <td headers="stat_1" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,208 (0.5%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Inflammatory liver disease, unspecified</td>
## <td headers="stat_0" class="gt_row gt_center">334 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">332 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Jaundice</td>
## <td headers="stat_0" class="gt_row gt_center">744 (0.3%)</td>
## <td headers="stat_1" class="gt_row gt_center">23 (5.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">721 (0.3%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">NAFLD</td>
## <td headers="stat_0" class="gt_row gt_center">845 (0.4%)</td>
## <td headers="stat_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2" class="gt_row gt_center">841 (0.4%)</td>
## <td headers="p.value" class="gt_row gt_center">0.12</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">NASH</td>
## <td headers="stat_0" class="gt_row gt_center">74 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">73 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.3</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Nonspecific reactive hepatitis</td>
## <td headers="stat_0" class="gt_row gt_center">1 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">1 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Obstruction of bile duct</td>
## <td headers="stat_0" class="gt_row gt_center">384 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">40 (9.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">344 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Oesophageal varices w bleeding</td>
## <td headers="stat_0" class="gt_row gt_center">66 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2" class="gt_row gt_center">63 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Oesophageal varices w_o bleeding</td>
## <td headers="stat_0" class="gt_row gt_center">334 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">327 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Other and unspecified cirrhosis of liver</td>
## <td headers="stat_0" class="gt_row gt_center">455 (0.2%)</td>
## <td headers="stat_1" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">448 (0.2%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Other specified diseases of biliary tract</td>
## <td headers="stat_0" class="gt_row gt_center">277 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">12 (2.8%)</td>
## <td headers="stat_2" class="gt_row gt_center">265 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Pancreatic cancer</td>
## <td headers="stat_0" class="gt_row gt_center">259 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">18 (4.2%)</td>
## <td headers="stat_2" class="gt_row gt_center">241 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Perforation of bile duct</td>
## <td headers="stat_0" class="gt_row gt_center">1 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">1 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Primary biliary cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">42 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">40 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Secondary biliary cirrhosis</td>
## <td headers="stat_0" class="gt_row gt_center">3 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">3 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">>0.9</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Sepsis</td>
## <td headers="stat_0" class="gt_row gt_center">2,022 (0.9%)</td>
## <td headers="stat_1" class="gt_row gt_center">24 (5.6%)</td>
## <td headers="stat_2" class="gt_row gt_center">1,998 (0.9%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Small intestine cancer</td>
## <td headers="stat_0" class="gt_row gt_center">98 (&lt;0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2" class="gt_row gt_center">92 (&lt;0.1%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Spasm of sphincter of Oddi</td>
## <td headers="stat_0" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="p.value" class="gt_row gt_center"><br /></td>
## <td headers="q.value" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Splenomegaly, not elsewhere classified</td>
## <td headers="stat_0" class="gt_row gt_center">235 (0.1%)</td>
## <td headers="stat_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2" class="gt_row gt_center">233 (0.1%)</td>
## <td headers="p.value" class="gt_row gt_center">0.11</td>
## <td headers="q.value" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Stomach disorders</td>
## <td headers="stat_0" class="gt_row gt_center">23,622 (10%)</td>
## <td headers="stat_1" class="gt_row gt_center">70 (16%)</td>
## <td headers="stat_2" class="gt_row gt_center">23,552 (10%)</td>
## <td headers="p.value" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value" class="gt_row gt_center">0.003</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="6"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```

```r
Table_ICD_stratified <- import_merge_tables(table_name= "Table_ICD", feature="SEX", levels = c("Female", "Male"))
```

```
## [1] "C:/Users/felix/OneDrive - Uniklinik RWTH Aachen/public/projects/cca/tables/Table_ICD_SEX_Female_ukb.RDS"
## [1] "C:/Users/felix/OneDrive - Uniklinik RWTH Aachen/public/projects/cca/tables/Table_ICD_SEX_Male_ukb.RDS"
## <div id="hbjmtpktoc" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
##   <style>#hbjmtpktoc table {
##   font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
##   -webkit-font-smoothing: antialiased;
##   -moz-osx-font-smoothing: grayscale;
## }
## 
## #hbjmtpktoc thead, #hbjmtpktoc tbody, #hbjmtpktoc tfoot, #hbjmtpktoc tr, #hbjmtpktoc td, #hbjmtpktoc th {
##   border-style: none;
## }
## 
## #hbjmtpktoc p {
##   margin: 0;
##   padding: 0;
## }
## 
## #hbjmtpktoc .gt_table {
##   display: table;
##   border-collapse: collapse;
##   line-height: normal;
##   margin-left: auto;
##   margin-right: auto;
##   color: #333333;
##   font-size: 16px;
##   font-weight: normal;
##   font-style: normal;
##   background-color: #FFFFFF;
##   width: auto;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #A8A8A8;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #A8A8A8;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_caption {
##   padding-top: 4px;
##   padding-bottom: 4px;
## }
## 
## #hbjmtpktoc .gt_title {
##   color: #333333;
##   font-size: 125%;
##   font-weight: initial;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-color: #FFFFFF;
##   border-bottom-width: 0;
## }
## 
## #hbjmtpktoc .gt_subtitle {
##   color: #333333;
##   font-size: 85%;
##   font-weight: initial;
##   padding-top: 3px;
##   padding-bottom: 5px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-color: #FFFFFF;
##   border-top-width: 0;
## }
## 
## #hbjmtpktoc .gt_heading {
##   background-color: #FFFFFF;
##   text-align: center;
##   border-bottom-color: #FFFFFF;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_bottom_border {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_col_headings {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_col_heading {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 6px;
##   padding-left: 5px;
##   padding-right: 5px;
##   overflow-x: hidden;
## }
## 
## #hbjmtpktoc .gt_column_spanner_outer {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: normal;
##   text-transform: inherit;
##   padding-top: 0;
##   padding-bottom: 0;
##   padding-left: 4px;
##   padding-right: 4px;
## }
## 
## #hbjmtpktoc .gt_column_spanner_outer:first-child {
##   padding-left: 0;
## }
## 
## #hbjmtpktoc .gt_column_spanner_outer:last-child {
##   padding-right: 0;
## }
## 
## #hbjmtpktoc .gt_column_spanner {
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: bottom;
##   padding-top: 5px;
##   padding-bottom: 5px;
##   overflow-x: hidden;
##   display: inline-block;
##   width: 100%;
## }
## 
## #hbjmtpktoc .gt_spanner_row {
##   border-bottom-style: hidden;
## }
## 
## #hbjmtpktoc .gt_group_heading {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   text-align: left;
## }
## 
## #hbjmtpktoc .gt_empty_group_heading {
##   padding: 0.5px;
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   vertical-align: middle;
## }
## 
## #hbjmtpktoc .gt_from_md > :first-child {
##   margin-top: 0;
## }
## 
## #hbjmtpktoc .gt_from_md > :last-child {
##   margin-bottom: 0;
## }
## 
## #hbjmtpktoc .gt_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   margin: 10px;
##   border-top-style: solid;
##   border-top-width: 1px;
##   border-top-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 1px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 1px;
##   border-right-color: #D3D3D3;
##   vertical-align: middle;
##   overflow-x: hidden;
## }
## 
## #hbjmtpktoc .gt_stub {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hbjmtpktoc .gt_stub_row_group {
##   color: #333333;
##   background-color: #FFFFFF;
##   font-size: 100%;
##   font-weight: initial;
##   text-transform: inherit;
##   border-right-style: solid;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
##   padding-left: 5px;
##   padding-right: 5px;
##   vertical-align: top;
## }
## 
## #hbjmtpktoc .gt_row_group_first td {
##   border-top-width: 2px;
## }
## 
## #hbjmtpktoc .gt_row_group_first th {
##   border-top-width: 2px;
## }
## 
## #hbjmtpktoc .gt_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hbjmtpktoc .gt_first_summary_row {
##   border-top-style: solid;
##   border-top-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_first_summary_row.thick {
##   border-top-width: 2px;
## }
## 
## #hbjmtpktoc .gt_last_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_grand_summary_row {
##   color: #333333;
##   background-color: #FFFFFF;
##   text-transform: inherit;
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hbjmtpktoc .gt_first_grand_summary_row {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-top-style: double;
##   border-top-width: 6px;
##   border-top-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_last_grand_summary_row_top {
##   padding-top: 8px;
##   padding-bottom: 8px;
##   padding-left: 5px;
##   padding-right: 5px;
##   border-bottom-style: double;
##   border-bottom-width: 6px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_striped {
##   background-color: rgba(128, 128, 128, 0.05);
## }
## 
## #hbjmtpktoc .gt_table_body {
##   border-top-style: solid;
##   border-top-width: 2px;
##   border-top-color: #D3D3D3;
##   border-bottom-style: solid;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_footnotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_footnote {
##   margin: 0px;
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hbjmtpktoc .gt_sourcenotes {
##   color: #333333;
##   background-color: #FFFFFF;
##   border-bottom-style: none;
##   border-bottom-width: 2px;
##   border-bottom-color: #D3D3D3;
##   border-left-style: none;
##   border-left-width: 2px;
##   border-left-color: #D3D3D3;
##   border-right-style: none;
##   border-right-width: 2px;
##   border-right-color: #D3D3D3;
## }
## 
## #hbjmtpktoc .gt_sourcenote {
##   font-size: 90%;
##   padding-top: 4px;
##   padding-bottom: 4px;
##   padding-left: 5px;
##   padding-right: 5px;
## }
## 
## #hbjmtpktoc .gt_left {
##   text-align: left;
## }
## 
## #hbjmtpktoc .gt_center {
##   text-align: center;
## }
## 
## #hbjmtpktoc .gt_right {
##   text-align: right;
##   font-variant-numeric: tabular-nums;
## }
## 
## #hbjmtpktoc .gt_font_normal {
##   font-weight: normal;
## }
## 
## #hbjmtpktoc .gt_font_bold {
##   font-weight: bold;
## }
## 
## #hbjmtpktoc .gt_font_italic {
##   font-style: italic;
## }
## 
## #hbjmtpktoc .gt_super {
##   font-size: 65%;
## }
## 
## #hbjmtpktoc .gt_footnote_marks {
##   font-size: 75%;
##   vertical-align: 0.4em;
##   position: initial;
## }
## 
## #hbjmtpktoc .gt_asterisk {
##   font-size: 100%;
##   vertical-align: 0;
## }
## 
## #hbjmtpktoc .gt_indent_1 {
##   text-indent: 5px;
## }
## 
## #hbjmtpktoc .gt_indent_2 {
##   text-indent: 10px;
## }
## 
## #hbjmtpktoc .gt_indent_3 {
##   text-indent: 15px;
## }
## 
## #hbjmtpktoc .gt_indent_4 {
##   text-indent: 20px;
## }
## 
## #hbjmtpktoc .gt_indent_5 {
##   text-indent: 25px;
## }
## </style>
##   <table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
##   <thead>
##     <tr class="gt_col_headings gt_spanner_row">
##       <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="2" colspan="1" scope="col" id="&lt;strong&gt;Characteristic&lt;/strong&gt;"><strong>Characteristic</strong></th>
##       <th class="gt_center gt_columns_top_border gt_column_spanner_outer" rowspan="1" colspan="5" scope="colgroup" id="Female">
##         <span class="gt_column_spanner">Female</span>
##       </th>
##       <th class="gt_center gt_columns_top_border gt_column_spanner_outer" rowspan="1" colspan="5" scope="colgroup" id="Male">
##         <span class="gt_column_spanner">Male</span>
##       </th>
##     </tr>
##     <tr class="gt_col_headings">
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 273225&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 273225<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 423&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 423<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 272802&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 272802<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;Overall&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;N = 229001&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>Overall</strong></p>
## <p>N = 229001<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 425&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>CCa</strong></p>
## <p>n = 425<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;No  CCa&lt;/strong&gt;&lt;/p&gt;&#10;&lt;p&gt;n = 228576&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;1&lt;/sup&gt;&lt;/span&gt;"><strong>No  CCa</strong></p>
## <p>n = 228576<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;p-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;2&lt;/sup&gt;&lt;/span&gt;"><strong>p-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></th>
##       <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="&lt;strong&gt;q-value&lt;/strong&gt;&lt;span class=&quot;gt_footnote_marks&quot; style=&quot;white-space:nowrap;font-style:italic;font-weight:normal;&quot;&gt;&lt;sup&gt;3&lt;/sup&gt;&lt;/span&gt;"><strong>q-value</strong><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span></th>
##     </tr>
##   </thead>
##   <tbody class="gt_table_body">
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Acute renal failure</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,392 (0.5%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,386 (0.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.022</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">2,591 (1.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">15 (3.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">2,576 (1.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic cirrhosis</td>
## <td headers="stat_0_1" class="gt_row gt_center">78 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">78 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">286 (0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">283 (0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.007</td>
## <td headers="q.value_2" class="gt_row gt_center">0.4</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic fatty liver</td>
## <td headers="stat_0_1" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">74 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">74 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic fibrosis</td>
## <td headers="stat_0_1" class="gt_row gt_center">5 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">5 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">18 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">18 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic hepatic failure</td>
## <td headers="stat_0_1" class="gt_row gt_center">20 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">20 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">84 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">83 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.4</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic hepatitis</td>
## <td headers="stat_0_1" class="gt_row gt_center">38 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">38 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">127 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">125 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.009</td>
## <td headers="q.value_2" class="gt_row gt_center">0.6</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic liver disease</td>
## <td headers="stat_0_1" class="gt_row gt_center">196 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">195 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.7</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">749 (0.3%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">742 (0.3%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Alcoholic liver disease, unspecified</td>
## <td headers="stat_0_1" class="gt_row gt_center">137 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">136 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.5</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">509 (0.2%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">506 (0.2%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.11</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Arterial hypertension</td>
## <td headers="stat_0_1" class="gt_row gt_center">72,394 (26%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">150 (35%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">72,244 (26%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.002</td>
## <td headers="stat_0_2" class="gt_row gt_center">80,216 (35%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">214 (50%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">80,002 (35%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Ascites</td>
## <td headers="stat_0_1" class="gt_row gt_center">816 (0.3%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">15 (3.5%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">801 (0.3%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">724 (0.3%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">11 (2.6%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">713 (0.3%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Autoimmune hepatitis</td>
## <td headers="stat_0_1" class="gt_row gt_center">129 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">129 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">28 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">28 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Biliary cirrhosis, unspecified</td>
## <td headers="stat_0_1" class="gt_row gt_center">48 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">47 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.12</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">15 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value_2" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Biliary Cyst</td>
## <td headers="stat_0_1" class="gt_row gt_center">8 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">8 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">7 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">7 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cholangitis</td>
## <td headers="stat_0_1" class="gt_row gt_center">187 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">29 (6.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">158 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">260 (0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">23 (5.4%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">237 (0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Cholelithiasis</td>
## <td headers="stat_0_1" class="gt_row gt_center">14,678 (5.4%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">63 (15%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">14,615 (5.4%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">5,276 (2.3%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">31 (7.3%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">5,245 (2.3%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Chronic Hepatitis B</td>
## <td headers="stat_0_1" class="gt_row gt_center">85 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">85 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">161 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">160 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.7</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Chronic Hepatitis C</td>
## <td headers="stat_0_1" class="gt_row gt_center">153 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">153 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">297 (0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">296 (0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Chronic kidney disease</td>
## <td headers="stat_0_1" class="gt_row gt_center">2,137 (0.8%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">10 (2.4%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">2,127 (0.8%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.041</td>
## <td headers="stat_0_2" class="gt_row gt_center">3,007 (1.3%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">10 (2.4%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">2,997 (1.3%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.095</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Colorectal cancer</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,900 (0.7%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,894 (0.7%)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.13</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">2,588 (1.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">13 (3.1%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">2,575 (1.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.026</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Complications of cirrhosis</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,514 (0.6%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">38 (9.0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,476 (0.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">1,628 (0.7%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">37 (8.7%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1,591 (0.7%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Disease of biliary tract, unspecified</td>
## <td headers="stat_0_1" class="gt_row gt_center">51 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">47 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">31 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">28 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">DM</td>
## <td headers="stat_0_1" class="gt_row gt_center">9,539 (3.5%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">30 (7.1%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">9,509 (3.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.006</td>
## <td headers="stat_0_2" class="gt_row gt_center">14,655 (6.4%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">62 (15%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">14,593 (6.4%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_ALT.x</td>
## <td headers="stat_0_1" class="gt_row gt_center">17,692 (6.5%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">41 (9.7%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">17,651 (6.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.010</td>
## <td headers="q.value_1" class="gt_row gt_center">0.6</td>
## <td headers="stat_0_2" class="gt_row gt_center">13,547 (5.9%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">43 (10%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">13,504 (5.9%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.022</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_ALT.y</td>
## <td headers="stat_0_1" class="gt_row gt_center">17,692 (6.5%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">41 (9.7%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">17,651 (6.5%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.010</td>
## <td headers="q.value_1" class="gt_row gt_center">0.6</td>
## <td headers="stat_0_2" class="gt_row gt_center">13,547 (5.9%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">43 (10%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">13,504 (5.9%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.022</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AP.x</td>
## <td headers="stat_0_1" class="gt_row gt_center">45,310 (17%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">115 (27%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">45,195 (17%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">6,451 (2.8%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">34 (8.0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">6,417 (2.8%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AP.y</td>
## <td headers="stat_0_1" class="gt_row gt_center">45,310 (17%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">115 (27%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">45,195 (17%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">6,451 (2.8%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">34 (8.0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">6,417 (2.8%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AST.x</td>
## <td headers="stat_0_1" class="gt_row gt_center">15,361 (5.6%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">45 (11%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">15,316 (5.6%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">5,896 (2.6%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">29 (6.8%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">5,867 (2.6%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_AST.y</td>
## <td headers="stat_0_1" class="gt_row gt_center">15,361 (5.6%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">45 (11%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">15,316 (5.6%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">5,896 (2.6%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">29 (6.8%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">5,867 (2.6%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_GGT.x</td>
## <td headers="stat_0_1" class="gt_row gt_center">41,886 (15%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">105 (25%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">41,781 (15%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">38,334 (17%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">122 (29%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">38,212 (17%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_GGT.y</td>
## <td headers="stat_0_1" class="gt_row gt_center">41,886 (15%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">105 (25%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">41,781 (15%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">38,334 (17%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">122 (29%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">38,212 (17%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_Liver_Enzymes.x</td>
## <td headers="stat_0_1" class="gt_row gt_center">51,905 (19%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">121 (29%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">51,784 (19%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">44,946 (20%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">129 (30%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">44,817 (20%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Elevated_Liver_Enzymes.y</td>
## <td headers="stat_0_1" class="gt_row gt_center">51,905 (19%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">121 (29%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">51,784 (19%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">44,946 (20%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">129 (30%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">44,817 (20%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Esophageal cancer</td>
## <td headers="stat_0_1" class="gt_row gt_center">168 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">166 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.015</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">510 (0.2%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">510 (0.2%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.6</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Fibrosis and cirrhosis</td>
## <td headers="stat_0_1" class="gt_row gt_center">511 (0.2%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">8 (1.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">503 (0.2%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">547 (0.2%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">9 (2.1%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">538 (0.2%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Fistula of bile duct</td>
## <td headers="stat_0_1" class="gt_row gt_center">10 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">10 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">7 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">6 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Gastric cancer</td>
## <td headers="stat_0_1" class="gt_row gt_center">154 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">152 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.010</td>
## <td headers="q.value_1" class="gt_row gt_center">0.6</td>
## <td headers="stat_0_2" class="gt_row gt_center">373 (0.2%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">370 (0.2%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.030</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Granulomatous hepatitis, not elsewhere classified</td>
## <td headers="stat_0_1" class="gt_row gt_center">10 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">10 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">14 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatic fibrosis</td>
## <td headers="stat_0_1" class="gt_row gt_center">46 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">45 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.11</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">73 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">73 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatic fibrosis and sclerosis</td>
## <td headers="stat_0_1" class="gt_row gt_center">2 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">2 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatic sclerosis</td>
## <td headers="stat_0_1" class="gt_row gt_center">4 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">4 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">7 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">6 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatomegaly with splenomegaly, not elsewhere classified</td>
## <td headers="stat_0_1" class="gt_row gt_center">33 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">32 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.047</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">47 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">46 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.2</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatomegaly, not elsewhere classified</td>
## <td headers="stat_0_1" class="gt_row gt_center">97 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">95 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.032</td>
## <td headers="stat_0_2" class="gt_row gt_center">155 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">152 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.002</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Hepatorenal Syndrome</td>
## <td headers="stat_0_1" class="gt_row gt_center">12 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">12 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">37 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">37 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">HSM</td>
## <td headers="stat_0_1" class="gt_row gt_center">239 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">235 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">422 (0.2%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">416 (0.2%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">IBD</td>
## <td headers="stat_0_1" class="gt_row gt_center">3,347 (1.2%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">12 (2.8%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">3,335 (1.2%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value_1" class="gt_row gt_center">0.3</td>
## <td headers="stat_0_2" class="gt_row gt_center">3,041 (1.3%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">17 (4.0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">3,024 (1.3%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Inflammatory liver disease</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,223 (0.4%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,217 (0.4%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.009</td>
## <td headers="q.value_1" class="gt_row gt_center">0.6</td>
## <td headers="stat_0_2" class="gt_row gt_center">1,215 (0.5%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1,208 (0.5%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;">0.005</td>
## <td headers="q.value_2" class="gt_row gt_center">0.3</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Inflammatory liver disease, unspecified</td>
## <td headers="stat_0_1" class="gt_row gt_center">374 (0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">373 (0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">334 (0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">332 (0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.3</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Jaundice</td>
## <td headers="stat_0_1" class="gt_row gt_center">591 (0.2%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">22 (5.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">569 (0.2%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">744 (0.3%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">23 (5.4%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">721 (0.3%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">NAFLD</td>
## <td headers="stat_0_1" class="gt_row gt_center">752 (0.3%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">748 (0.3%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;">0.030</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">845 (0.4%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">841 (0.4%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.12</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">NASH</td>
## <td headers="stat_0_1" class="gt_row gt_center">67 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">66 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.2</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">74 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">73 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.3</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Nonspecific reactive hepatitis</td>
## <td headers="stat_0_1" class="gt_row gt_center">3 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">3 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">1 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Obstruction of bile duct</td>
## <td headers="stat_0_1" class="gt_row gt_center">435 (0.2%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">39 (9.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">396 (0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">384 (0.2%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">40 (9.4%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">344 (0.2%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Oesophageal varices w bleeding</td>
## <td headers="stat_0_1" class="gt_row gt_center">36 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">32 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">66 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">63 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Oesophageal varices w_o bleeding</td>
## <td headers="stat_0_1" class="gt_row gt_center">182 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">179 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.002</td>
## <td headers="stat_0_2" class="gt_row gt_center">334 (0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">327 (0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Other and unspecified cirrhosis of liver</td>
## <td headers="stat_0_1" class="gt_row gt_center">327 (0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">321 (0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">455 (0.2%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">7 (1.6%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">448 (0.2%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Other specified diseases of biliary tract</td>
## <td headers="stat_0_1" class="gt_row gt_center">623 (0.2%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">15 (3.5%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">608 (0.2%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">277 (0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">12 (2.8%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">265 (0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Pancreatic cancer</td>
## <td headers="stat_0_1" class="gt_row gt_center">257 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">19 (4.5%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">238 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">259 (0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">18 (4.2%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">241 (0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Perforation of bile duct</td>
## <td headers="stat_0_1" class="gt_row gt_center">8 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">8 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">1 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Primary biliary cirrhosis</td>
## <td headers="stat_0_1" class="gt_row gt_center">219 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">3 (0.7%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">216 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.013</td>
## <td headers="stat_0_2" class="gt_row gt_center">42 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">40 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Secondary biliary cirrhosis</td>
## <td headers="stat_0_1" class="gt_row gt_center">6 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">6 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">3 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">3 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Sepsis</td>
## <td headers="stat_0_1" class="gt_row gt_center">1,830 (0.7%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">29 (6.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">1,801 (0.7%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">2,022 (0.9%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">24 (5.6%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">1,998 (0.9%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Small intestine cancer</td>
## <td headers="stat_0_1" class="gt_row gt_center">77 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">4 (0.9%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">73 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center"><0.001</td>
## <td headers="stat_0_2" class="gt_row gt_center">98 (&lt;0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">6 (1.4%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">92 (&lt;0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center"><0.001</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Spasm of sphincter of Oddi</td>
## <td headers="stat_0_1" class="gt_row gt_center">15 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">15 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">0 (0%)</td>
## <td headers="p.value_2" class="gt_row gt_center"><br /></td>
## <td headers="q.value_2" class="gt_row gt_center"><br /></td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Splenomegaly, not elsewhere classified</td>
## <td headers="stat_0_1" class="gt_row gt_center">123 (&lt;0.1%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">1 (0.2%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">122 (&lt;0.1%)</td>
## <td headers="p.value_1" class="gt_row gt_center">0.5</td>
## <td headers="q.value_1" class="gt_row gt_center">>0.9</td>
## <td headers="stat_0_2" class="gt_row gt_center">235 (0.1%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">2 (0.5%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">233 (0.1%)</td>
## <td headers="p.value_2" class="gt_row gt_center">0.11</td>
## <td headers="q.value_2" class="gt_row gt_center">>0.9</td></tr>
##     <tr><td headers="label" class="gt_row gt_left" style="font-weight: bold;">Stomach disorders</td>
## <td headers="stat_0_1" class="gt_row gt_center">27,747 (10%)</td>
## <td headers="stat_1_1" class="gt_row gt_center">66 (16%)</td>
## <td headers="stat_2_1" class="gt_row gt_center">27,681 (10%)</td>
## <td headers="p.value_1" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_1" class="gt_row gt_center">0.018</td>
## <td headers="stat_0_2" class="gt_row gt_center">23,622 (10%)</td>
## <td headers="stat_1_2" class="gt_row gt_center">70 (16%)</td>
## <td headers="stat_2_2" class="gt_row gt_center">23,552 (10%)</td>
## <td headers="p.value_2" class="gt_row gt_center" style="font-weight: bold;"><0.001</td>
## <td headers="q.value_2" class="gt_row gt_center">0.003</td></tr>
##   </tbody>
##   
##   <tfoot class="gt_footnotes">
##     <tr>
##       <td class="gt_footnote" colspan="11"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> n (%)</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="11"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Pearson’s Chi-squared test</td>
##     </tr>
##     <tr>
##       <td class="gt_footnote" colspan="11"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>3</sup></span> Bonferroni correction for multiple testing</td>
##     </tr>
##   </tfoot>
## </table>
## </div>
```



































#Check cause of missing data (merge df_y with every single df to check how many cases are lost)
Does not work yet

```r
setwd(project_path)
load("data/dataframes/df_y.RData")
inspect_na_contributions <- function(df_y, include_metabolomics = FALSE) {
  
  # Initial list of dataframes to merge
  dfs_to_merge <- list(df_covariates, df_diagnosis, df_blood, df_snp) # Removing df_eid for the analysis
  
  # Optionally add df_metabolomics
  if (include_metabolomics) {
    dfs_to_merge <- c(dfs_to_merge, list(df_metabolomics))
  }
  
  # Function to calculate number of rows dropped due to NA after merging with df_y
  dropped_rows_due_to_na <- function(df) {
    merged_df <- merge(df_y, df, by = "eid", all = TRUE)
    return(sum(!is.na(merged_df$status) & apply(merged_df[-1], 1, function(row) all(is.na(row)))))
  }
  
  # Apply function to each dataframe
  na_drop_counts <- sapply(dfs_to_merge, dropped_rows_due_to_na)
  
  # Assign names based on the condition of include_metabolomics
  if (include_metabolomics) {
    names(na_drop_counts) <- c("covariates", "icd", "blood", "snp", "metabolomics")
  } else {
    names(na_drop_counts) <- c("covariates", "icd", "blood", "snp")
  }
  
  return(na_drop_counts)
}

na_contributions <- inspect_na_contributions(df_y, include_metabolomics = TRUE)
```

```
## Warning in !is.na(merged_df$status) & apply(merged_df[-1], 1, function(row) all(is.na(row))): Länge des längeren Objektes
##  	 ist kein Vielfaches der Länge des kürzeren Objektes

## Warning in !is.na(merged_df$status) & apply(merged_df[-1], 1, function(row) all(is.na(row))): Länge des längeren Objektes
##  	 ist kein Vielfaches der Länge des kürzeren Objektes

## Warning in !is.na(merged_df$status) & apply(merged_df[-1], 1, function(row) all(is.na(row))): Länge des längeren Objektes
##  	 ist kein Vielfaches der Länge des kürzeren Objektes

## Warning in !is.na(merged_df$status) & apply(merged_df[-1], 1, function(row) all(is.na(row))): Länge des längeren Objektes
##  	 ist kein Vielfaches der Länge des kürzeren Objektes

## Warning in !is.na(merged_df$status) & apply(merged_df[-1], 1, function(row) all(is.na(row))): Länge des längeren Objektes
##  	 ist kein Vielfaches der Länge des kürzeren Objektes
```

```r
print(na_contributions)
```

```
##   covariates          icd        blood          snp metabolomics 
##            0            0            0            0            0
```






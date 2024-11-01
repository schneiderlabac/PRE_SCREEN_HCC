#########################################################################################################


################################## General functions ##############################################################
####### Removing patients that withdrew their approval
check_and_remove_withdrawals <- function(df, df_withdrawals) {
  withdrawal_eids <- df_withdrawals$eid # Extract eids from df_withdrawals
  before <- nrow(df) # Count the number of rows before removal
  if (any(df$eid %in% withdrawal_eids)) {
    df <- df[!df$eid %in% withdrawal_eids, ] # Remove rows with eids that are in df_withdrawals
    after <- nrow(df) # Count the number of rows after removal
    removed <- before - after # Calculate the number of rows removed
    warning_message <- sprintf("Patients with withdrawn consent have been removed from the dataframe. Rows before: %d, Rows after: %d, Rows removed: %d", before, after, removed)
    warning(warning_message)
  } else {
    cat("No patients with withdrawn consent in df. You may pass!\n")
  }
  return(df)
}

####### Sanity check
sanity <- function(df) {
  print("Number of NAs per column:")
  na_counts <- na_columnwise(df)                       # Use the new na_columnwise function
  print(summary(df))
  print(paste("Duplicated rows in df:", any(duplicated(df))))
  print(paste("Incomplete cases in df:", sum(!complete.cases(df))))
  
  print(paste("All values of df (except eid) are between 0 and 1:", sum(apply(df[, -1], 2, function(x) all(x >= 0 & x <= 1))) == (ncol(df) -1)))
  #print(paste("No negative eids in df:", apply(df[, 1], 2, function(x) all(x >= 0))))
  na_sums <- rowSums(is.na(df))
  hist(na_sums)
  print(paste("Variable type:", str(df)))
}


####### MinMax Normalization
minmax <- function(x, na.rm = TRUE) {
  return((x- min(x, na.rm = na.rm)) /(max(x, na.rm = na.rm)-min(x, na.rm = na.rm)))
}


######## Cap the max and min of a column that are outside of biological relevance
# e.g. a BMI of > 100 will be set to 100, followed by normalization

### Note: This function is not yet automatized onto the master_table!
cap_and_normalize <- function(df, df_cap) {
  for (i in 1:nrow(df_cap)) {
    col_name <- df_cap$column[i]
    if (col_name %in% names(df) && is.numeric(df[[col_name]])) {
      min_val <- df_cap$cap_min[i]
      max_val <- df_cap$cap_max[i]
      
      # Cap the values
      df[[col_name]] <- pmin(pmax(df[[col_name]], min_val), max_val)
      
      # Normalize to 0-1 scale
      df[[col_name]] <- (df[[col_name]] - min_val) / (max_val - min_val)
    }
  }
  return(df)
}


limit_df <- function(df, mapper = mapper) {
  report <- list()
  
  for (i in 1:nrow(mapper)) {
    col_name <- mapper$column[i]
    upper <- mapper$upper_limit[i]
    lower <- mapper$lower_limit[i]
    
    if (col_name %in% names(df) && (!is.na(upper) || !is.na(lower))) {
      original <- df[[col_name]]
      modified <- original  # Start with a copy of the original
      
      # Only apply limits to non-NA values
      non_na <- !is.na(original)
      
      if (!is.na(upper)) {
        modified[non_na] <- pmin(modified[non_na], upper, na.rm = TRUE)
      }
      if (!is.na(lower)) {
        modified[non_na] <- pmax(modified[non_na], lower, na.rm = TRUE)
      }
      
      # Count changes (excluding NA values)
      changes <- sum(original != modified & non_na, na.rm = TRUE)
      upper_changes <- sum(original > upper & non_na, na.rm = TRUE)
      lower_changes <- sum(original < lower & non_na, na.rm = TRUE)
      
      # Update the dataframe
      df[[col_name]] <- modified
      
      # Only add to report if changes were made
      if (changes > 0) {
        report[[col_name]] <- list(
          upper_limit = upper,
          lower_limit = lower,
          rows_exceeding_upper = upper_changes,
          rows_below_lower = lower_changes,
          total_modified_rows = changes,
          total_na_values = sum(is.na(original))
        )
      }
    }
  }
  
  # Print report only if there are entries
  if (length(report) > 0) {
    cat("Limit Application Report:\n")
    cat("-------------------------\n")
    for (col in names(report)) {
      cat(sprintf("Column: %s\n", col))
      cat(sprintf("  Upper limit: %s\n", report[[col]]$upper_limit))
      cat(sprintf("  Lower limit: %s\n", report[[col]]$lower_limit))
      cat(sprintf("  Rows exceeding upper limit: %d\n", report[[col]]$rows_exceeding_upper))
      cat(sprintf("  Rows below lower limit: %d\n", report[[col]]$rows_below_lower))
      cat(sprintf("  Total modified rows: %d\n", report[[col]]$total_modified_rows))
      cat(sprintf("  Total NA values (unmodified): %d\n", report[[col]]$total_na_values))
      cat("\n")
    }
  } else {
    cat("No limits were applied to any columns.\n")
  }
  
  return(df)
}


##### Check columns on amount of factor levels 
check_factor_levels <- function(df) {
  for (col_name in names(df)) {
    # Check if the column is a factor
    if (is.factor(df[[col_name]])) {
      # Check the number of levels
      if (nlevels(df[[col_name]]) > 2) {
        stop(sprintf("Error: The factor column '%s' has more than two levels.", col_name))
      }
    }
  }
  cat("All factor columns have two or fewer levels.\n")
}


######## Convert a dataframe to factors
convert_to_factor <- function(df) {
  df_converted <- df %>%
    mutate(across(-eid, ~factor(ifelse(as.character(.x) != "0", "1", "0"))))

  problematic_cols <- names(df_converted)[sapply(df_converted, function(col) {
  is.factor(col) && length(levels(col)) > 2
  })]
  # Warn for each problematic column
  if (length(problematic_cols) > 0) {
    for (col_name in problematic_cols) {
      warning(sprintf("Column '%s' has more than two levels after conversion.", col_name))
    }
  }
  print(str(df_converted))
  return(df_converted)
}



####### Count NAs per column
na_columnwise <- function(df) {
  na_counts <- colSums(is.na(df))
  for (col in names(na_counts)) {
    print(paste(col, ":", na_counts[col]))
  }
  return(na_counts)
}

#####Mediansplit
mediansplit <- function(df, col, na.rm=TRUE) {
  median <- median(col)
  y[col > median] <- 1
  y[col < median] <- 0
  return((y))
}

######## Count duplicated eids
count_eid_duplications <- function(df, df_name = "df") {
  duplicated_eids_count <- df %>%
    group_by(eid) %>%
    summarise(count = n(), .groups = 'drop') %>%
    filter(count > 1)
  
  # Print the message separately
  print(paste("Amount of duplicated eids in", df_name, ":", nrow(duplicated_eids_count)))
  
  return()
}







#########################################################################################################

########################### Subsetting control groups ##############################################################################





subset_patients_at_risk <- function(df, vec_risk_constellation, na.rm = TRUE) { #Function for subsetting
  df_patients_at_risk <- intersect(colnames(df), vec_risk_constellation) 
  
  return(df_patients_at_risk)
}


######### Import and merge df_y for brief orientation
innerjoin_df_y <- function(df) {
  df_name <- deparse(substitute(df)) #store the name of the df
  df_y_temp <- read.csv(paste0(project_path, "/data/dataframes/df_y.csv")) %>%
    dplyr::select(eid, status) %>%
    dplyr::filter(status == 1)
  df_joined <- inner_join(df, df_y_temp, by = "eid")
  positive_cases <- nrow(df_joined)
  message(df_name, " contains ", positive_cases, DOI, " cases.")
  
  return(df_joined)
}


check_current_par <- function() {
  # Load the current and predefined subset variables into the function (Group level)
  
  expected_subset <- project_configs[[project_key]]$par_subset 
  par_subset <- get("par_subset", envir = .GlobalEnv)
  if (!all(expected_subset %in% par_subset)) { 
    warning("Provided subset does not match the expected subset for the project key. Proceeding with provided subset.")
  }
    print("Current patients-at-risk-setting includes the following groups:")
    print(par_subset)
}



filter_rows_with_pos_entries <- function(df) {

  check_current_par()
  # Ensure relevant_columns are in the dataframe (Diagnosis level)
  vec_risk_constellation <- par_index$Diagnosis[par_index$Group %in% par_subset] #subset index for project-specific requirements
  existing_columns <- vec_risk_constellation[vec_risk_constellation %in% names(df)]
  df_name <- deparse(substitute(df)) #get name of df
  # Initialize a logical vector to flag rows with any positive entry in relevant columns
  positive_entry_rows <- rep(FALSE, nrow(df)) #empty vector
  
  for (col in existing_columns) {
    # Update the row flag if the entry is "1" for any of the relevant_columns
    positive_entry_rows <- positive_entry_rows | (df[[col]] == "1")
  }
  
  # Filter the dataframe to include only rows with at least one positive entry
  filtered_df <- df[positive_entry_rows, ]
  message(df_name, " contains ", nrow(filtered_df), " Patients at risk.")
  
  
  #innerjoin_df_y(df)
  
  return(filtered_df)
}


#########################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

######################### Imputation functions / Taking care of NA values ########################################################

#Function to Create df "df_cov_amount" that counts all present and missing values in df_covariates
summarize_na <- function(df_covariates) {
  # Initialize empty data frame
  df_cov_amount <- data.frame(
    covariate = character(ncol(df_covariates)),
    missing = numeric(ncol(df_covariates)),
    present = numeric(ncol(df_covariates)))
  
  colnames(df_cov_amount) <- c("covariate", "missing", "present") # Assign column names
  
  # Populate the data frame
  df_cov_amount$covariate <- colnames(df_covariates)
  df_cov_amount$missing <- colSums(is.na(df_covariates))
  df_cov_amount$present <- 502411 - df_cov_amount$missing
  print("Check df_cov_amount for updated number of NAs per column")
  return(df_cov_amount)
}

####### Function that will omit rows in a data frame "df" with more than a specified number of NA values:
omit.NA <- function(df, threshold) {
  # Find number of NA values per row
  na_counts <- rowSums(is.na(df))
  df <- df[na_counts <= threshold, ]
  return(df)
}

### Imputation functions

####### Mean Imputation
mean.impute <- function(x){
  x[is.na(x)] <- mean(x[!is.na(x)])  
  return(x)
}

####### Mode imputation (most common value) 
impute_mode <- function(data) {
  mode_cols <- names(data)[2:ncol(data)] # Identify columns to impute (all columns starting from the 2nd column)
  for(col in mode_cols) {
    mode_val <- as.numeric(names(which.max(table(data[[col]])))) # Find the mode (most frequent value) of the column
    data[[col]][is.na(data[[col]])] <- mode_val # Impute NAs with the mode value
  }
  return(data)
}

####### "Imputation" Adding new factor level for n/a SNPs
impute_snp <- function(df) {
  cols <- colnames(df)[2:ncol(df)]
  for(col in cols) {
    df[[col]] <- as.character(df[[col]])  # Convert to character
    df[[col]][is.na(df[[col]])] <- "unknown"
    df[[col]] <- as.factor(df[[col]])     # Convert back to factor
  }
  return(df)
}



# 1. Impute missing values in categorical columns with a constant value ("Unknown").
impute_categorical <- function(df_covariates, covariate_final, keep_untouched = c()) {
  # Fetch the categorical columns specified
  categorical_cols <- covariate_final$Assessment[covariate_final$Variable == "categorical"]
  
  # Exclude specified columns from the imputation process
  categorical_cols <- setdiff(categorical_cols, keep_untouched)
  
  for(col in categorical_cols) {
    if(col %in% colnames(df_covariates) & is.factor(df_covariates[[col]])) {
      # Proceed only if there are NA values in the column
      if(any(is.na(df_covariates[[col]]))) {
        # Add "Unknown" as a level to the factor only if it's not already a level
        if(!"Unknown" %in% levels(df_covariates[[col]])) {
          df_covariates[[col]] <- factor(df_covariates[[col]], levels = c(levels(df_covariates[[col]]), "Unknown"))
        }
        # Now replace NA values with "Unknown"
        df_covariates[[col]][is.na(df_covariates[[col]])] <- "Unknown"
      }
    }
  }
  return(df_covariates)
}

# 2. Impute missing values in continuous columns with the median.
impute_continuous <- function(data, covariate_final, keep_untouched = c()) {
  # Extract column names for continuous normalized and binned variables
  cont_normalized_cols <- covariate_final$Assessment[covariate_final$Variable == "cont_normalized"]
  cont_binned_cols <- covariate_final$Assessment[covariate_final$Variable == "cont_binned"]
  
  # Combine and exclude columns specified in keep_untouched
  all_cont_cols <- setdiff(c(cont_normalized_cols, cont_binned_cols), keep_untouched)
  
  for(col in all_cont_cols) {
    # Ensure the column exists in 'data' before attempting to impute
    if(col %in% names(data)) {
      # Calculate median only if there are NA values to replace
      if(any(is.na(data[[col]]))) {
        med_val <- median(data[[col]], na.rm = TRUE)
        data[[col]][is.na(data[[col]])] <- med_val
      }
    }
  }
  return(data)
}

# 3. Probabilistic imputation for ordinal columns.
impute_random <- function(data, covariate_final) {
  ordinal_cols <- covariate_final$Assessment[covariate_final$Variable == "ordinal"]
  
  for(col in ordinal_cols) {
    tab <- table(data[[col]], useNA = "no")  # Get the table of counts for each level
    probs <- tab / sum(tab)  # Calculate the probabilities of each level
    
    num_NA <- sum(is.na(data[[col]]))  # Number of NAs to replace
    imputed_values <- sample(names(probs), num_NA, replace = TRUE, prob = probs)  # Generate random values based on the probabilities
    
    data[[col]][is.na(data[[col]])] <- imputed_values  # Replace NAs with the imputed values
  }
  return(data)
}


adjust_outliers <- function(df, column_names) {
  # Ensure column_names is a character vector
  if (!is.character(column_names)) {
    stop("column_names must be a character vector")
  }
  
  # Ensure all specified columns exist in the dataframe
  missing_columns <- setdiff(column_names, colnames(df))
  if (length(missing_columns) > 0) {
    stop(paste("The following columns do not exist in the dataframe:", 
               paste(missing_columns, collapse = ", ")))
  }
  
  for (column_name in column_names) {
    # Calculate the 99.9th percentile
    quantile_999 <- quantile(df[[column_name]], 0.999, na.rm = TRUE)
    
    # Identify outliers
    outliers <- df[[column_name]] > quantile_999
    outliers_count <- sum(outliers, na.rm = TRUE)
    outliers_range <- range(df[[column_name]][outliers], na.rm = TRUE)
    
    # Replace outliers with the 99.9th percentile value
    df[[column_name]] <- ifelse(outliers, quantile_999, df[[column_name]])
    
    # Print the range of values that were cut
    cat("\nColumn:", column_name, "\n")
    cat("Outliers detected and adjusted to the 99.9th percentile limit:\n")
    cat("Number of outliers:", outliers_count, "\n")
    cat("Range of outliers:", paste(outliers_range, collapse = " to "), "\n")
    cat("99.9th percentile limit:", quantile_999, "\n")
  }
  
  return(df)
}






# Example usage
# Ensure that df_covariates exists and contains the Alk_g_d column
# if (exists("df_covariates")) {
#   df_covariates <- adjust_outliers(df_covariates, "Alk_g_d")
#   print("Outliers adjusted")
# } else {
#   print("The dataframe df_covariates does not exist.")
# }



#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
                              # Merge dataframes


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












#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

########################### Creating gtsummary tables ##############################################################################
# https://www.danieldsjoberg.com/gtsummary/articles/tbl_summary.html

create_table <- function(df_tbl, table_name, table_name_prefix="All", project_path=project_path, export_RDS=TRUE, head_only=FALSE, remove_SEX=TRUE, enforced_order=FALSE, biobank_key=biobank_key, create_binary_table=FALSE, adjust_p_values=TRUE, column_order = FALSE) {
  project_path <- get("project_path", envir = .GlobalEnv)
  biobank_key <- get("biobank_key", envir = .GlobalEnv)
  
  if (remove_SEX) { #optional: Remove the SEX column if no stratified analysis required
    df_tbl <- df_tbl %>% select(-SEX)
  }
  
  if (!isFALSE(enforced_order)) { #Cannot subset with = TRUE because will have vector passed when not false
    df_tbl <- df_tbl %>% select(status, all_of(enforced_order))
  }
  
  if (isFALSE(enforced_order)) { #Sort by alphabetical order of columnnames
    names <- sort(colnames(df_tbl))
    df_tbl <- df_tbl %>% select(all_of(names)) 
  }
  
  # Optional: Select the first 5 columns if head_only is TRUE
  if (head_only) {
    df_tbl <- df_tbl %>% select(1:5, status)
  }
  
  if (create_binary_table) { #label_list by default is a list of more accurate column descriptions. Instead, you can select boolean rows for shorter representation e.g. of ICD Codes
    names <- create_binary_list(df_tbl) #store names of columns with binary values
    
    value_list <- lapply(names, function(name) {               #create value and label in a specific format 
      formula <- as.formula(paste0("`", name, "` ~ '1'"))    # "AATD" ~ "1" per column defining the value to be plotted
      
      environment(formula) <- emptyenv()                    # AATD ~ "AATD" for label
      return(formula)
    })
    
    label_list <- lapply(names, function(name) {
      formula <- as.formula(paste0("`", name, "` ~ '", name, "'"))
      environment(formula) <- emptyenv()
      return(formula)
    })
  } else {
    # Default value and label logic if not creating binary table
    value_list <- NULL  # Assuming no specific value transformation needed
    label_list <- get("label_list", envir = .GlobalEnv)
  }
  
  if (!isFALSE(column_order)) {
    df_tbl$status <- factor(df_tbl$status, levels = column_order) #Orders levels of status according to desired order of columns in the table
  }
      # Get distinct values of the stratified column
  distinct_status_values <- df_tbl %>% distinct(status) %>% pull(status) %>% sort()
  
  # Some continuous columns with less variance will be plotted as categoricals, the following lines prevent that:
  continuous_columns <- c("Pack years", "Basophill (%)", "Eosinophill (%)", "Monocyte percentage", "Neutrophill count")
  
  # Filter the continuous_columns based on their presence in df_tbl
  continuous_columns_present <- continuous_columns[continuous_columns %in% colnames(df_tbl)]
  
  # Create the type_list based on the filtered columns
  type_list <- if (length(continuous_columns_present) > 0) {
    lapply(continuous_columns_present, function(col) {
      as.formula(paste0("`", col, "` ~ 'continuous'"))
    })
  } else {
    NULL
  }
  
  # Generate the summary table
  Table_gtsummary <- df_tbl %>%
    tbl_summary(
      by = status,
      type = type_list,
      value = value_list,
      label = label_list,
      statistic = list(
        all_continuous() ~ "{mean} (±{sd})",
        all_categorical() ~ "{n} ({p}%)"
      ),
      digits = all_continuous() ~ 1,
    ) %>%
    add_p(
      test = list(
        all_continuous() ~ "t.test",
        all_categorical() ~ "chisq.test"
      )
    ) %>%
    add_overall() %>%
    bold_labels() %>%
    italicize_levels() %>%
    bold_p() %>%
    modify_header(update = list(
      stat_0 ~ "**Overall** \n \n N = {N}",
      !!sym(paste0("stat_", 1)) ~ paste0("**", distinct_status_values[1], "** \n\n n = {n}"),
      !!sym(paste0("stat_", 2)) ~ paste0("**", distinct_status_values[2], "** \n\n n = {n}")
    ), text_interpret = "md")
  
  if (adjust_p_values) {
    Table_gtsummary <- Table_gtsummary %>%
      add_q(method = "bonferroni")
  }
  
  Table_gt <- as_gt(Table_gtsummary) %>%
    fmt_number(
      columns = c(-contains("p.value"), -contains("q.value")),  # Exclude p-value and q-value columns
      decimals = 2,
      locale = "en"  # Locale enforces '.' for decimals and ',' for thousands
    )
  # Print the table
  print(Table_gt)
  
  # Optional: Save the table as an RDS file
  if (export_RDS) {
    saveRDS(Table_gtsummary, file = paste0(project_path, "/tables/", table_name, "_", biobank_key, ".RDS"))
  }
  
  # Save HTML first, then use webshot for conversion to PNG or PDF
  
  html_path <- paste0(project_path, "/tables/", table_name, "_", biobank_key, ".html")
  gt::gtsave(Table_gt, filename = html_path)
  
  # webshot2::webshot(
  #   url = html_path,
  #   file = paste0(project_path, "/tables/", table_name, "_", biobank_key, ".png"),
  #   zoom = 2
  # )
  # 
  # # convert HTML to PDF
  # webshot2::webshot(
  #   url = html_path,
  #   file = paste0(project_path, "/tables/", table_name, "_", biobank_key, ".pdf")
  # )
  
  
  Table_gtsummary %>%
    as_flex_table() %>%
    save_as_docx(path = paste0(project_path, "/tables/", table_name, "_", biobank_key, ".docx"))
}



# Example usage
# create_table(df_tbl_1, "Table1", SEX = "Male", export_RDS = TRUE, head_only = TRUE)





split_create_merge_tables <- function(df, feature, table_name, project_path, enforced_order=FALSE, head_only=FALSE, remove_SEX=TRUE, export_RDS=TRUE, create_binary_table = FALSE, adjust_p_values=TRUE) {
  project_path <- get("project_path", envir = .GlobalEnv)
  biobank_key <- get("biobank_key", envir = .GlobalEnv)
  # Split the dataframe by the specified feature
  split_dfs <- split(df, df[[feature]])
  
  if (!isFALSE(enforced_order)) {
    enforced_order <- table1_order[table1_order != feature] #Remove feature to stratify from the order list
  }
  
  # Iterate over each split dataframe
  table_files <- c() # To store filenames of the saved tables for potential merging
  
  for (split_name in names(split_dfs)) {
    # Define table name based on prefix and split name
    merged_name <- paste(table_name, feature, split_name, sep="_")
    table_files <- c(table_files, paste0(table_name, "_", Sys.Date(), ".RDS"))
    
    # Use create_table to generate and save each table
    
    create_table(df_tbl = split_dfs[[split_name]], 
                 table_name = merged_name, 
                 export_RDS = export_RDS, 
                 head_only = head_only, 
                 remove_SEX = remove_SEX, 
                 enforced_order = enforced_order,
                 create_binary_table = create_binary_table,
                 adjust_p_values = adjust_p_values)
  }
  
  # # Optionally merge the tables if more than one split
  # if(export_RDS) {
  #   tab_spanner <- names(split_dfs)
  #   merge_saved_tables(table_files, project_path, tab_spanner)
  # }
  
  #gt::gtsave(Table_1_stratified, filename = paste0(project_path, "/tables/Table_stratified", table_name, "_", biobank_key, "_", Sys.Date(), ".html"))
}



import_merge_tables <- function(table_name, feature, levels,  tab_spanner) {

  project_path <- get("project_path", envir = .GlobalEnv)
  biobank_key <- get("biobank_key", envir = .GlobalEnv)
  
  # Initialize an empty list to store the imported tables
  imported_tables <- list()
  
  # Loop through each level and import the corresponding RDS file
  for (level in levels) {
    file_path <- paste0(project_path, "/tables/", table_name, "_", feature, "_", level, "_", biobank_key, ".RDS")
    print(file_path)
    
    imported_table <- readRDS(file_path)
    #imported_table <- imported_table %>% filter(Variable != feature)
    imported_tables[[level]] <- imported_table
  }

  # Merge the imported tables
  merged_table <- tbl_merge(
    tbls = imported_tables,
    tab_spanner = levels
  )
  print(merged_table)
  merged_table_gt <- as_gt(merged_table)
  
  
  # Save as HTML first, then transform HTML to png or pdf via webshot (otherwise via Google Chrome or manual conversion)
  html_path <- paste0(project_path, "/tables/", table_name, "_", feature, "_", biobank_key, ".html")
  gt::gtsave(merged_table_gt, filename = html_path)
  
  # Use webshot2 to convert HTML to PNG
  # webshot2::webshot(
  #   url = html_path,
  #   file = paste0(project_path, "/tables/", table_name, "_", feature, "_", biobank_key, ".png"),
  #   zoom = 2
  # )
  # 
  # # Use webshot2 to convert HTML to PDF
  # webshot2::webshot(
  #   url = html_path,
  #   file = paste0(project_path, "/tables/", table_name, "_", feature, "_", biobank_key, ".pdf")
  # )
  

  #Add as preferred
  #gt::gtsave(merged_table_gt, filename = paste0(project_path, "/tables/", table_name, "_", feature, "_", biobank_key, ".tex")) or .mw / .rtf
  merged_table %>%
    as_flex_table() %>%
    save_as_docx(path = paste0(project_path, "/tables/", table_name, "_", feature, "_", biobank_key, ".docx"))
  # Return the merged table
  return(merged_table_gt)
}



# This is a helper function that extracts all columns with binary input to get the lists in a way that for dichotomous (0,1) columns, only the row with TRUE will get displayed, reducing space in the table a lot!
create_binary_list <- function(df) {
  binary_cols <- sapply(df, function(x) all(x %in% c(0, 1)))
  names <- names(df)[binary_cols]
  
  return(names)
}


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################






###############################################################################################################################
#########################################ICD Related Functions/ Visuals###############################################################

#Summarize ICD codes
create_summary <- function(df) {
  total_rows <- nrow(df)
  summary_df <- df %>% 
    summarise(across(-eid, ~sum(. == "1", na.rm = TRUE))) %>% 
    pivot_longer(everything(), names_to = "Diagnosis", values_to = "Occurrence") %>% 
    mutate(Percentage = (Occurrence / total_rows) * 100) %>%
    arrange(desc(Occurrence))
  
  as.data.frame(summary_df)
}




barplot_diags <- function(df, diags_to_remove) {
  summary <- create_summary(df) %>%
    dplyr::filter(!Diagnosis %in% diags_to_remove)
  plot <- ggplot(summary, aes(x = reorder(Diagnosis, -Occurrence), y = Occurrence)) +
    geom_bar(stat = "identity", fill = "#69b3a2", width = 0.7) +  # Adjust the bar width
    geom_text(aes(label = Occurrence), position = position_nudge(x = 0, y = 100), hjust = 0, size = 3.5, family = "Arial") +  # Adjust label size and font
    scale_x_discrete(guide = guide_axis(n.dodge = 1)) +
    coord_flip() +
    theme_ipsum(base_size = 14) +  # Increase base_size for better readability
    theme(
      panel.grid.minor.y = element_blank(),
      panel.grid.major.y = element_line(color = "gray", linetype = "dotted"),  # Customize major grid lines
      legend.position = "none",
      axis.title = element_text(size = rel(1.2), family = "Arial"),  # Adjust axis title size
      plot.margin = unit(c(1, 1, 1, 1), "cm"),
      axis.title.y = element_text(size = rel(1.2))
    ) +
    labs(x = "", y = "Occurrence")  # Remove axis labels
  
  print(plot)
  
  
  # Save the plots
  ggsave(filename = file.path(project_path, paste0("supplement_visuals/ICD_Histogram_", Sys.Date(), ".png")), 
         plot = plot, width = 15, height = 9, bg= "white")
  ggsave(filename = file.path(project_path, paste0("supplement_visuals/ICD_Histogram_", Sys.Date(), ".svg")), 
         plot = plot, width = 15, height = 9, bg= "white")
  
}

##################################################################################################################################################################################


comparison_plot_diags <- function(data, base_size = 18) {
  # Reshape the data to long format
  data_long <- data %>%
    select(Diagnosis, Legend, starts_with("Percentage")) %>%
    pivot_longer(cols = starts_with("Percentage"), 
                 names_to = "Group", 
                 values_to = "Percentage") %>%
    mutate(Group = recode(Group, 
                          `Percentage.x` = "No Cancer", 
                          `Percentage.y` = "Cancer",
                          `Percentage` = "Patients at risk"),
           Group = factor(Group, levels = c("Cancer", "Patients at risk", "No Cancer")))
  
  # Determine top 15 diagnoses
  top_diagnoses <- data %>% 
    slice_head(n = 15) %>%
    pull(Diagnosis)
  
  data_long <- data_long %>%
    filter(Diagnosis %in% top_diagnoses)
  
  # Create the bar plot using the Legend column for the x-axis
  plot <- ggplot(data_long, aes(x = Legend, y = Percentage, fill = Group)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.75), width = 0.75) +
    scale_fill_manual(values = c("Cancer" = "#BE4F4F", "Patients at risk" = "#685A66", "No Cancer" = "#AFAFAF")) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(x = "Diagnosis", y = "Prevalence (%)") +
    theme_minimal(base_size = base_size, base_family = "Arial") +
    theme(
      legend.title = element_blank(),
      legend.position = c(0.3, 0.9),  # Top right corner
      legend.background = element_blank(),
      legend.text = element_text(size = base_size),
      legend.direction = "vertical",  # Ensure vertical alignment of legend texts
      axis.text = element_text(size = base_size, color="black"),
      axis.title.x = element_text(size = base_size + 2),
      axis.title.y = element_text(size = base_size + 2),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
      panel.border = element_rect(colour = "black", fill = NA, size = 1)
    )
  
  print(plot)
  
  # Save the plots
  ggsave(filename = file.path(project_path, paste0("supplement_visuals/ICD_Relations_", Sys.Date(), ".png")), 
         plot = plot, width = 10, height = 10, bg= "white", limitsize = FALSE)
  ggsave(filename = file.path(project_path, paste0("supplement_visuals/ICD_Relations_",  Sys.Date(), ".svg")), 
         plot = plot, width = 10, height = 10, bg= "transparent")
  
  # Text legend
  text_legend <- data_long %>% 
    distinct(Legend, Diagnosis) %>% 
    arrange(Legend) %>% 
    mutate(Legend_Diagnosis = paste(Legend, Diagnosis, sep = ": ")) %>%
    pull(Legend_Diagnosis) %>%
    paste(collapse = "\n")
  cat("Legend:\n", text_legend)
  
  writeLines(paste("Legend:\n", text_legend), file.path(project_path, "supplement_visuals", paste0("ICD_Relations_Legend_", Sys.Date(),  ".txt")))
  
  # Return the plot and the top diagnoses data
  list(plot = plot, top_diagnoses_data = data %>% filter(Diagnosis %in% top_diagnoses))
}


#################################################################################################################################################################################
                                # Cases of DOI Visuals


create_map_plot <- function(df_loc_counts, df_country, base_size = 12) {
  supplement_visuals_dir <- get("supplement_visuals_dir", envir = .GlobalEnv)
  
  # Create custom labels for the map
  custom_labels <- paste0(unique(df_country$location_country), " (n=", df_country$count, ")")
  
  # Make the map
  plot <- ggplot(df_loc_counts, aes(x = long, y = lat, size = (count*2), shape=in_validation)) +
    borders("world") +
    geom_point(aes(color = location_country),
    ) +
    scale_size(range = c(3, 8)) +
    scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17), labels = c("Training", "Validation")) + 
    scale_color_manual(values = c("England" = "grey", "Scotland" = "red", "Wales" = "blue"), labels = custom_labels) +
    geom_text_repel(aes(label = NA), nudge_x = 1.5, nudge_y = 1.5, box.padding = 0.5, segment.size = 0.3, size = 3) +
    coord_quickmap(xlim = c(-11, 2), ylim = c(49, 60)) +
    theme_minimal(base_size = base_size) +
    theme(
      legend.text = element_text(size = base_size),
      axis.text = element_blank(),
      axis.title = element_blank()
      
    ) +
    labs(title = paste("Distribution of", DOI, "Diagnosis in UK Biobank Centers"),
         x = "Longitude",
         y = "Latitude",
         size = "Number of Cases",
         color = "Country",
         shape= "Split") + 
    guides(
      color = guide_legend(override.aes = list(size = 6)),  
      shape = guide_legend(override.aes = list(size = 5))
    )
  
  # Print and save the plot
  print(plot)
  ggsave(filename = file.path(supplement_visuals_dir, paste0(DOI, "_occurrence_per_center.png")), plot = plot, width = 10, height = 8, bg= "white")
  ggsave(filename = file.path(supplement_visuals_dir, paste0(DOI, "_occurrence_per_center.pdf")), plot = plot, width = 10, height = 8, bg = "white")
}


# Example usage of create_map_plot function
# create_map_plot(df_loc_counts, "Some Disease", "path/to/visuals/dir")








#################################################################################################################################################################################



plot_included_discarded_cases <- function(df, base_size = 18) {
  supplement_visuals_dir <- get("supplement_visuals_dir", envir = .GlobalEnv)
  doi_included <- get("doi_included", envir = .GlobalEnv)
  doi_discarded <- get("doi_discarded", envir = .GlobalEnv)
  included_label <- paste("Included (n=", doi_included, ")", sep = "") # Custom labels for the legend
  discarded_label <- paste("Discarded (n=", doi_discarded, ")", sep = "")
  n_total <- nrow(df)

  plot <- ggplot(df, aes(x = year, fill = as.factor(discard))) +
    geom_histogram(binwidth = 1, color = "black", size=0.2, width= 0.5) +
    scale_fill_manual(values = c("grey", "#808080"), 
                      labels = c(included_label, discarded_label), 
                      name = "") + 
    
    xlab("Year") +
    ylab("Absolute number of cases") +
    ggtitle(paste("Year of", DOI, "Diagnosis")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = base_size), # Increase size to 150% 
          legend.position = c(0.3, 0.9), # upper left corner
          axis.title.x = element_text(size = base_size, vjust=-1), 
          axis.title.y = element_text(size = base_size), 
          legend.text = element_text(size = base_size), 
          axis.text.x = element_text(size = base_size, colour="black", vjust=0.1), 
          axis.text.y = element_text(size = base_size, colour="black"),
          axis.title.y.right = element_text(size = base_size, angle=90, vjust=-0.5),
          axis.text.y.right = element_text(size = base_size, colour = "black"),
          panel.grid.major = element_blank(), # Remove major grid lines
          panel.grid.minor = element_blank(), # Remove minor grid lines
          plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
          panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
          legend.spacing.y = unit(2, "cm"),
          legend.key = element_rect(colour = "white", fill = NA)) +
    scale_y_continuous(
      expand = c(0, 0), 
      limits = c(0, NA),
      sec.axis = sec_axis(
        ~. / n_total * 100000,
        name = "Incidence [n / 100.000]"
      )
    ) 
  guides(fill = guide_legend(override.aes = list(colour = "white")))
  
  print(plot)
  ggsave(filename = file.path(supplement_visuals_dir, paste0(DOI, "_yearly_cases.svg")), 
         plot = plot, width = 10, height = 10, bg = "transparent")
  
  
  }







#################################################################################################################################################################################
                  # Plot Stacked Bars

stacked_bars_time_comparison <- function(df, base_size=18, priority_order, color_map) {
  df <- df %>%
    mutate(Priority = match(Group, priority_order)) %>%
    arrange(Time, Priority) %>%
    group_by(Time) %>%
    mutate(LabelPos = cumsum(Count) - 0.5 * Count) %>%
    ungroup() %>%
    mutate(Group = factor(Group, levels = priority_order))
  
  max_x_value <- max(as.numeric(as.factor(df$Time))) + 1
  df$max_x <- max_x_value
  
  label_data <- df %>%
    filter(Order == 2) %>%
    distinct(Group, .keep_all = TRUE)
  
  plot <- ggplot(data = df, aes(x = Time, y = Count, fill = Group)) +
    geom_bar(stat = "identity", position = position_stack(vjust = 0.5, reverse = TRUE), width = 0.4) +  # Reduced width
    geom_text(aes(label = Count, y = LabelPos), size = base_size * 0.4, colour = "black", vjust = -0.3) +
    geom_text(aes(label = sprintf("%.0f%%", Percentage), y = LabelPos), size = base_size * 0.3, colour = "black", vjust = 1.2) +
    geom_text(data = distinct(label_data, Group, .keep_all = TRUE), 
              aes(x = max_x - 0.4, label = Group, y = LabelPos), hjust=0, size = base_size * 0.4, color = "black") +
    theme_minimal() +
    labs(title = "Etiology Over Time",
         y = "Count") +
    scale_fill_manual(values = color_map) +  # Use custom color map
    theme(plot.title = element_text(size=base_size, hjust = 0.5),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          axis.text.x = element_text(size = base_size, colour = "black"),
          axis.text.y = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.margin = margin(1, 300, 1, 10)) +
    coord_cartesian(clip = 'off')
  
  return(plot)
}






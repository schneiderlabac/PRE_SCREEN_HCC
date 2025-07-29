# -*- coding: utf-8 -*-
# =============================================================================
# PREPROCESSING UTILITIES FOR ALL OF US DATA
# =============================================================================
# 
# This file contains utility functions for preprocessing All of Us research
# data, including missing data analysis, imputation, normalization, and 
# data transformation functions.
#
# Author: Extracted from preprocess_23_04_25.ipynb
# Date: Created from notebook analysis
# =============================================================================



# =============================================================================
# ITEMS
# =============================================================================


# MISSING DATA ANALYSIS
# DATA IMPUTATION
# DATA NORMALIZATION
#  OUTLIER DETECTION AND ADJUSTMENT
# DATE AND TIME CONVERSION
# DATA TRANSFORMATION AND PROCESSING
# DATA VALIDATION AND QUALITY CONTROL
# REDUNDANCY ANALYSIS



# =============================================================================
# MISSING DATA ANALYSIS
# =============================================================================

#' Summarize Missing Data in Covariates
#' Analyzes missing data patterns in covariate datasets, with optional 
#' analysis of missing data in positive cases.


summarize_na <- function(df_covariates, df_y = NULL, rule = FALSE) {
#' @param df_covariates Data frame containing covariates with person identifiers
#' @param df_y Optional data frame containing outcome data with status column
#' @param rule Logical, if TRUE applies "<20" masking for small counts (default: FALSE)
#' @return Data frame with missing data summary by covariate
#' @examples
#' # Basic missing data summary
#' missing_summary <- summarize_na(my_covariates)
#'
#' # With outcome data and rule masking
#' missing_summary <- summarize_na(my_covariates, my_outcomes, rule = TRUE)


  # Get total number of rows
  total_rows <- nrow(df_covariates)
  
  # Initialize empty data frame
  df_cov_amount <- data.frame(
    covariate = character(ncol(df_covariates)),
    missing = numeric(ncol(df_covariates)),
    present = numeric(ncol(df_covariates)),
    missing_in_positive = numeric(ncol(df_covariates))
  )
  
  colnames(df_cov_amount) <- c("covariate", "missing", "present", "missing_in_positive")
  
  # Populate the data frame
  df_cov_amount$covariate <- colnames(df_covariates)
  df_cov_amount$missing <- colSums(is.na(df_covariates))
  df_cov_amount$present <- total_rows - df_cov_amount$missing
  
  # Calculate missing in positive class if df_y is provided
  if (!is.null(df_y)) {
    df_merged <- inner_join(df_covariates, df_y, by = eid) %>% filter(status == 1)
    df_cov_amount$missing_in_positive <- colSums(is.na(df_merged))
  } else {
    df_cov_amount$missing_in_positive <- NA
  }
  
  if (rule) {
    # Apply the "<20" rule
    apply_rule <- function(x) {
      ifelse(is.na(x), NA, ifelse(x < 20, "<20", as.character(x)))
    }
    
    df_cov_amount$missing <- apply_rule(df_cov_amount$missing)
    df_cov_amount$present <- apply_rule(df_cov_amount$present)
    df_cov_amount$missing_in_positive <- apply_rule(df_cov_amount$missing_in_positive)
    
    # Convert columns to character type
    df_cov_amount$missing <- as.character(df_cov_amount$missing)
    df_cov_amount$present <- as.character(df_cov_amount$present)
    df_cov_amount$missing_in_positive <- as.character(df_cov_amount$missing_in_positive)
  }
  
  print(paste("Total rows:", total_rows))
  if (!is.null(df_y)) {
    print(paste("Positive cases:", sum(df_y$status == 1)))
  }
  if (rule) {
    print("Note: Values below 20 are reported as '<20' to comply with All of Us regulations.")
  }
  
  return(df_cov_amount)
}

#' Check Missing Data in Positive Cases
#'
#' Convenience function to check missing data patterns specifically in positive cases.
#' Note: This function assumes df_y exists in the global environment.
#'
#' @param df Data frame to analyze
#' @param rule_status Logical, whether to apply the "<20" rule (default: TRUE)
#' @return Prints missing data summary for positive cases
check_missing_positives <- function(df, rule_status = TRUE) {
  df_merged <- inner_join(df, df_y, by = eid) %>% filter(status == 1)
  diag_na <- summarize_na(df_merged, rule = rule_status)
  print(diag_na)  
}

# =============================================================================
# DATA IMPUTATION
# =============================================================================

#' Impute Continuous Variables with Median
#'
#' Imputes missing values in continuous (numeric) variables using median imputation.
#' Skips the 'eid' column.
#'
#' @param data Data frame containing continuous variables
#' @return Data frame with imputed continuous variables
#' @examples
#' imputed_data <- impute_continuous(my_data)
impute_continuous <- function(data) {
  # Loop through all columns except 'eid'
  for(col in setdiff(names(data), "eid")) {
    # Check if the column is numeric (continuous)
    if(is.numeric(data[[col]])) {
      # Calculate median only if there are NA values to replace
      if(any(is.na(data[[col]]))) {
        med_val <- median(data[[col]], na.rm = TRUE)
        data[[col]][is.na(data[[col]])] <- med_val
      }
    }
  }
  return(data)
}

#' Impute Categorical Variables with Mode
#'
#' Imputes missing values in categorical (factor/character) variables using mode imputation.
#' Skips the 'eid' column.
#'
#' @param data Data frame containing categorical variables
#' @return Data frame with imputed categorical variables
#' @examples
#' imputed_data <- impute_categorical(my_data)
impute_categorical <- function(data) {
  # Function to calculate mode
  get_mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  
  # Loop through all columns except 'eid'
  for(col in setdiff(names(data), "eid")) {
    # Check if the column is a factor or character (categorical)
    if(is.factor(data[[col]]) || is.character(data[[col]])) {
      # Calculate mode only if there are NA values to replace
      if(any(is.na(data[[col]]))) {
        mode_val <- get_mode(data[[col]][!is.na(data[[col]])])
        data[[col]][is.na(data[[col]])] <- mode_val
      }
    }
  }
  return(data)
}

#' Impute All Variable Types
#'
#' Convenience function that applies both continuous and categorical imputation.
#'
#' @param data Data frame containing mixed variable types
#' @return Data frame with all variables imputed
#' @examples
#' fully_imputed_data <- impute_all(my_data)
impute_all <- function(data) {
  data <- impute_continuous(data)
  data <- impute_categorical(data)
  return(data)
}

# =============================================================================
# DATA NORMALIZATION
# =============================================================================

#' Min-Max Normalization
#'
#' Performs min-max normalization on a numeric vector, scaling values to [0,1].
#'
#' @param x Numeric vector to normalize
#' @param na.rm Logical, whether to remove NA values (default: TRUE)
#' @return Normalized numeric vector
#' @examples
#' normalized_values <- minmax(c(1, 5, 10, 15, 20))
minmax <- function(x, na.rm = TRUE) {
  return((x - min(x, na.rm = na.rm)) / (max(x, na.rm = na.rm) - min(x, na.rm = na.rm)))
}

# =============================================================================
# OUTLIER DETECTION AND ADJUSTMENT
# =============================================================================

#' Adjust Outliers to 99.9th Percentile
#'
#' Identifies and caps outliers at the 99.9th percentile for a specified column.
#' Provides detailed reporting of adjustments made.
#'
#' @param df Data frame containing the column to adjust
#' @param column_name Character string of column name to adjust
#' @return Data frame with outliers adjusted
#' @examples
#' adjusted_data <- adjust_outliers(my_data, "Pack years")
adjust_outliers <- function(df, column_name) {
  # Ensure the column exists in the dataframe
  if (!column_name %in% colnames(df)) {
    stop(paste("Column", column_name, "does not exist in the dataframe"))
  }

  # Calculate the 99.9th percentile
  quantile_999 <- quantile(df[[column_name]], 0.999, na.rm = TRUE)

  # Identify outliers
  outliers <- df[[column_name]] > quantile_999
  outliers_count <- sum(outliers, na.rm = TRUE)
  outliers_range <- range(df[[column_name]][outliers], na.rm = TRUE)

  # Replace outliers with the 99.9th percentile value
  df[[column_name]] <- ifelse(outliers, quantile_999, df[[column_name]])

  # Print the range of values that were cut
  cat("Outliers detected and adjusted to the 99.9th percentile limit:\n")
  cat("Number of outliers:", outliers_count, "\n")
  cat("Range of outliers:", outliers_range, "\n")
  cat("99.9th percentile limit:", quantile_999, "\n")

  return(df)
}

# =============================================================================
# DATE AND TIME UTILITIES
# =============================================================================

#' Extract Year from Datetime String
#'
#' Robustly extracts year from various datetime string formats.
#' Tries lubridate parsing first, then falls back to regex extraction.
#'
#' @param datetime_str Character string containing datetime information
#' @return Numeric year or NA if extraction fails
#' @examples
#' year_value <- extract_year("2023-05-15 14:30:00")
#' year_value <- extract_year("May 15, 2023")
extract_year <- function(datetime_str) {
  if (is.na(datetime_str)) {
    return(NA)
  }
  
  # Try parsing with lubridate
  parsed_date <- ymd_hms(datetime_str, quiet = TRUE)
  if (!is.na(parsed_date)) {
    return(year(parsed_date))
  }
  
  # If parsing fails, try extracting year with regex
  year_match <- str_extract(datetime_str, "\\d{4}")
  if (!is.na(year_match)) {
    return(as.numeric(year_match))
  }
  
  # If all else fails, return NA
  return(NA)
}

# =============================================================================
# DATA VALIDATION AND QUALITY CONTROL
# =============================================================================

#' Remove Rows with Excessive Missing Data
#'
#' Removes rows that exceed a specified threshold of missing values.
#' Provides detailed reporting of removal statistics.
#'
#' @param df Data frame to clean
#' @param threshold Integer, maximum number of NA values allowed per row
#' @return Data frame with rows containing excessive NAs removed
#' @examples
#' clean_data <- omit.NA(my_data, threshold = 25)
omit.NA <- function(df, threshold) {
  # Get initial number of rows
  initial_rows <- nrow(df)
  
  # Find number of NA values per row
  na_counts <- rowSums(is.na(df))
  df <- df[na_counts <= threshold, ]
  
  # Calculate number of rows removed
  rows_removed <- initial_rows - nrow(df)
  
  # Print information about removed rows
  cat(sprintf("Rows removed: %d (%.2f%%)\n", 
              rows_removed, 
              (rows_removed / initial_rows) * 100))
  cat(sprintf("Rows remaining: %d\n", nrow(df)))
  
  return(df)
}

#' Analyze Impact of Different NA Thresholds
#'
#' Analyzes how different NA thresholds affect dataset size and positive cases.
#' Useful for determining optimal data cleaning thresholds.
#'
#' @param df Data frame to analyze (must contain 'status' column)
#' @param max_threshold Maximum threshold to test (default: 30)
#' @param step Step size for threshold testing (default: 5)
#' @return Data frame with threshold analysis results
#' @examples
#' threshold_analysis <- analyze_na_thresholds(my_data, max_threshold = 40, step = 10)
analyze_na_thresholds <- function(df, max_threshold = 30, step = 5) {
  results <- data.frame(
    NA_Threshold = numeric(),
    Total_Rows = numeric(),
    Positive_Cases = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (threshold in seq(0, max_threshold, by = step)) {
    df_cleaned <- omit.NA(df, threshold)
    
    results <- rbind(results, data.frame(
      NA_Threshold = threshold,
      Total_Rows = nrow(df_cleaned),
      Positive_Cases = sum(df_cleaned$status, na.rm = TRUE)
    ))
  }
  
  return(results)
}

# =============================================================================
# DATA TRANSFORMATION AND PROCESSING
# =============================================================================

#' Apply Physiological Limits to Data
#'
#' Applies upper and lower limits to specified columns based on a mapping table.
#' Useful for ensuring biologically plausible values.
#'
#' @param df Data frame to limit
#' @param mapper Data frame with columns: column, upper_limit, lower_limit
#' @return List containing limited data frame and detailed report
#' @examples
#' # mapper <- data.frame(column = "BMI", upper_limit = 60, lower_limit = 10)
#' # result <- limit_df(my_data, mapper)
#' # limited_data <- result$df
limit_df <- function(df, mapper) {
  report <- list()
  
  for (i in 1:nrow(mapper)) {
    col_name <- mapper$column[i]
    upper <- mapper$upper_limit[i]
    lower <- mapper$lower_limit[i]
    
    if (col_name %in% names(df)) {
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
  
  # Print report
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
  
  return(list(df = df, report = report))
}

#' Create Summary Statistics for Continuous Variables
#'
#' Generates comprehensive summary statistics for numeric columns.
#'
#' @param df Data frame to summarize
#' @param columns_to_drop Character vector of columns to exclude
#' @param digits Number of decimal places for output (default: 4)
#' @return Data frame with summary statistics
#' @examples
#' summary_stats <- summarize_continuous_columns(my_data, c("eid", "status"))
summarize_continuous_columns <- function(df, columns_to_drop = c(), digits = 4) {
  # Remove specified columns
  df <- df %>% select(-any_of(columns_to_drop))
  
  # Select only numeric columns
  numeric_columns <- df %>% select_if(is.numeric) %>% names()
  
  # Calculate summary statistics for numeric columns
  summary_df <- df %>%
    select(all_of(numeric_columns)) %>%
    summarise(across(everything(), list(
      mean = ~mean(., na.rm = TRUE),
      median = ~median(., na.rm = TRUE),
      max = ~max(., na.rm = TRUE),
      min = ~min(., na.rm = TRUE)
    )))
  
  # Reshape the dataframe
  summary_df <- summary_df %>%
    pivot_longer(cols = everything(),
                 names_to = c("column", ".value"),
                 names_pattern = "(.*)_(mean|median|max|min)")
  
  # Reorder columns
  summary_df <- summary_df %>%
    select(column, mean, median, max, min)
  
  # Format numbers to avoid scientific notation
  summary_df <- summary_df %>%
    mutate(across(c(mean, median, max, min), 
                  ~format(round(., digits), nsmall = digits, scientific = FALSE)))
  
  return(summary_df)

}
# =============================================================================
merge_dataframes <- function(dfs_to_merge=NULL, include_metabolomics = FALSE, filter_par = FALSE, normalize = FALSE ) {
    
    convert_id_to_integer <- function(df) { # Function to ensure person_id is an integer
        df$eid <- as.integer(df$eid)
        return(df)
    }
    
    # Initial list of dataframes to merge
    if (is.null(dfs_to_merge)) {
        dfs_to_merge <- list(df_covariates, df_diagnosis, df_blood)
    }
    dfs_to_merge <- lapply(dfs_to_merge, convert_id_to_integer) # Convert 'eid' in each dataframe to integer

    if (include_metabolomics) { # Conditionally add df_metabolomics and/or else
        dfs_to_merge <- c(dfs_to_merge, list(df_metabolomics))
    }
    
    df_merged <- Reduce(function(x, y) merge(x, y, by = "eid", all = FALSE), dfs_to_merge)  # Merging the dataframes
    df_merged <- map_and_align(df_merged, paste0(data_path, "/columnmapper_aou.xlsx")    
    # Conditionally apply normalization function
    if (normalize==TRUE) {
        df_merged <- normalize_data(df_merged)
        if (!is.integer(df_merged$eid)) {
            stop("Error: 'eid' column is no longer an integer after normalization.")
        }
    }
    #df_merged <- df_merged %>% select(-all_of(diag_codes)) # Removing the column of interest
          
    # Filter the "population at risk (par) by a prespecified if required
    if (filter_par) {
        df_merged <- filter_rows_with_pos_entries(df_merged)
    }
    #df_merged <- df_merged %>% select(-all_of(vec_blood_risk))
        
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
        
    #if (normalize) {
    #    assign("raw", "", envir = .GlobalEnv)
    #} else {
    #    assign("raw", "_raw", envir = .GlobalEnv)
    #}

    # Remove NAs and return the dataframe
    #return(na.omit(df_merged))
    return(df_merged)
}
# =============================================================================

                   

# Convert units from AOU standard to UKB standard according to formula in conversion table                        
convert_units <- function(df, conversion_table) {
  # Create a copy of the dataframe to avoid modifying the original
  df_converted <- df

  # Iterate through each row in the conversion table
  for (i in 1:nrow(conversion_table)) {
    column <- conversion_table$column[i]
    formula <- conversion_table$adjust_unit[i]
    
    if (column %in% names(df_converted)) {
      if (formula != "x") {
        tryCatch({
          # Create a function from the formula string
          convert_func <- if (column == "HbA1c") {
            function(x) {
              # Set a lower bound for plausible HbA1c values (3%)
              x_cleaned <- pmax(x, 3)
              # Apply conversion
              (x_cleaned - 2.15) * 10.929
            }
          } else {
            function(x) eval(parse(text = formula))
          }
          
          # Store original values for reporting
          original_values <- df_converted[[column]]
          
          # Apply the conversion
          df_converted[[column]] <- sapply(df_converted[[column]], convert_func)
          
          # Print conversion information
          cat(sprintf("Column '%s' converted: %s\n", column, 
              if(column == "HbA1c") "(x - 2.15) * 10.929" else formula))
          cat(sprintf("  Original range: %f to %f\n", min(original_values, na.rm = TRUE), max(original_values, na.rm = TRUE)))
          cat(sprintf("  Converted range: %f to %f\n", min(df_converted[[column]], na.rm = TRUE), max(df_converted[[column]], na.rm = TRUE)))
          cat(sprintf("  Units changed from %s to %s\n", 
                      conversion_table$Unit_aou[i], 
                      conversion_table$Unit_ukb[i]))
        }, error = function(e) {
          cat(sprintf("Error converting column '%s': %s\n", column, e$message))
        })
      } else {
        cat(sprintf("Column '%s' not converted (no conversion needed)\n", column))
      }
    } else {
      cat(sprintf("Column '%s' not found in dataframe\n", column))
    }
  }
  
  return(df_converted)
}

# =============================================================================

limit_df <- function(df, mapper) {
  report <- list()
  
  for (i in 1:nrow(mapper)) {
    col_name <- mapper$column[i]
    upper <- mapper$upper_limit[i]
    lower <- mapper$lower_limit[i]
    
    if (col_name %in% names(df)) {
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
  
  # Print report
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
  
  return(list(df = df, report = report))
}

              

              

adjust_outliers <- function(df, column_names = NULL) {
  # Convert to data.table if it's not already
  if (!is.data.table(df)) {
    df <- as.data.table(df)
  }
  
  # If column_names is not provided, use all numeric columns except 'eid'
  if (is.null(column_names)) {
    column_names <- setdiff(names(df)[sapply(df, is.numeric)], "eid")
  } else {
    # Ensure column_names is a character vector
    column_names <- as.character(column_names)
    
    # Ensure all specified columns exist in the dataframe
    missing_columns <- setdiff(column_names, colnames(df))
    if (length(missing_columns) > 0) {
      stop(paste("The following columns do not exist in the dataframe:", 
                 paste(missing_columns, collapse = ", ")))
    }
    
    # Ensure all specified columns are numeric
    non_numeric_columns <- column_names[!sapply(df[, ..column_names], is.numeric)]
    if (length(non_numeric_columns) > 0) {
      stop(paste("The following columns are not numeric:", 
                 paste(non_numeric_columns, collapse = ", ")))
    }
  }
  
  # Sort column names alphabetically
  column_names <- sort(column_names)
  
  # Remove 'eid' from the sorted list if it's present
  column_names <- setdiff(column_names, "eid")
  
  cat("Columns will be processed in the following order:\n")
  cat(paste(column_names, collapse = ", "), "\n\n")
  
  for (column_name in column_names) {
    # Calculate the 99.9th percentile
    quantile_999 <- quantile(df[[column_name]], 0.999, na.rm = TRUE)
    
    # Identify outliers
    outliers <- df[[column_name]] > quantile_999
    outliers_count <- sum(outliers, na.rm = TRUE)
    outliers_range <- range(df[[column_name]][outliers], na.rm = TRUE)
    
    # Replace outliers with the 99.9th percentile value
    df[, (column_name) := fifelse(get(column_name) > quantile_999, quantile_999, get(column_name))]
    
    # Print the range of values that were cut
    cat("\nColumn:", column_name, "\n")
    cat("Outliers detected and adjusted to the 99.9th percentile limit:\n")
    cat("Number of outliers:", outliers_count, "\n")
    cat("Range of outliers:", paste(outliers_range, collapse = " to "), "\n")
    cat("99.9th percentile limit:", quantile_999, "\n")
  }
  
  return(df)
}




adjust_outlier_to_ukb <- function(df, conversion_table) {
  # Ensure the required columns exist in the conversion table
  required_cols <- c("column", "max_ukb")
  if (!all(required_cols %in% colnames(conversion_table))) {
    stop("Conversion table must contain 'column' and 'max_ukb' columns")
  }
  
  # Remove 'eid' from the conversion table if present
  conversion_table <- conversion_table[conversion_table$column != "eid", ]
  
  # Create a named vector of max_ukb values, converting to numeric
  max_ukb_values <- sapply(setNames(conversion_table$max_ukb, conversion_table$column), 
                           function(x) as.numeric(as.character(x)))
  
  # Iterate through each column in the conversion table
  for (col in conversion_table$column) {
    if (col %in% colnames(df) && col != "eid") {  # Explicit check to exclude 'eid'
      # Get the current max value
      current_max <- max(df[[col]], na.rm = TRUE)
      
      # Get the UKB max value
      ukb_max <- max_ukb_values[col]
      
      # Check if ukb_max is NA (conversion to numeric failed)
      if (is.na(ukb_max)) {
        warning(sprintf("Unable to convert max_ukb value for column '%s' to numeric. Skipping this column.", col))
        next
      }
      
      # Adjust values if current max is greater than UKB max
      if (current_max > ukb_max) {
        df[[col]] <- pmin(df[[col]], ukb_max)
        cat(sprintf("Column '%s' adjusted: max value changed from %s to %s\n", 
                    col, format(current_max, scientific = FALSE), format(ukb_max, scientific = FALSE)))
      } else {
        cat(sprintf("Column '%s' not adjusted: current max (%s) is not greater than UKB max (%s)\n", 
                    col, format(current_max, scientific = FALSE), format(ukb_max, scientific = FALSE)))
      }
    } else if (col == "eid") {
      cat("Column 'eid' skipped as per requirement.\n")
    } else {
      warning(sprintf("Column '%s' not found in the dataframe", col))
    }
  }
  
  return(df)
}


# =============================================================================

map_and_align <- function(df, mapper_path) {
  # Read the mapper Excel file
  mapper_df <- read_excel(mapper_path)
  
  # Create a named vector for renaming, excluding NA mappings
  rename_vector <- setNames(mapper_df$column_ukb, mapper_df$column_aou)
  rename_vector <- rename_vector[!is.na(rename_vector) & !is.na(names(rename_vector))]
    
    # Keep only mappings for columns actually in df
  rename_vector <- rename_vector[names(rename_vector) %in% names(df)]
  
  # Identify columns to remove (those with NA in column_ukb or column_aou)
  columns_to_remove <- unique(c(
    mapper_df$column_aou[is.na(mapper_df$column_ukb)],
    mapper_df$column_aou[is.na(mapper_df$column_aou)]
  ))
  columns_to_remove <- columns_to_remove[!is.na(columns_to_remove)]  # Remove NA values
  
  # Rename columns
  df_renamed <- df %>%
    rename_with(~ ifelse(.x %in% names(rename_vector), rename_vector[.x], .x), everything())
  
  # Remove columns with NA mappings
  df_final <- df_renamed %>%
    select(-any_of(intersect(columns_to_remove, names(df_renamed))))
  
  # Identify columns that weren't renamed (no mapping found)
  unmapped_cols <- setdiff(names(df_final), mapper_df$column_ukb[!is.na(mapper_df$column_ukb)])
  
  # Print summary
  cat("Summary of map_and_align:\n")
  cat(sprintf("- %d columns renamed\n", sum(names(df) %in% names(rename_vector))))
  cat(sprintf("- %d columns identified for removal due to NA mapping\n", length(columns_to_remove)))
  cat(sprintf("- %d columns actually removed\n", ncol(df_renamed) - ncol(df_final)))
  cat(sprintf("- %d columns left unmapped\n", length(unmapped_cols)))
  
  # Print removed columns
  if (length(columns_to_remove) > 0) {
    cat("\nColumns identified for removal due to NA mapping:\n")
    print(columns_to_remove)
  }
  
  # Print unmapped columns
  if (length(unmapped_cols) > 0) {
    cat("\nColumns not renamed (no mapping found):\n")
    print(unmapped_cols)
  }
  
  return(df_final)
}



# DATA VALIDATION AND QUALITY CONTROL
# =============================================================================
#' Assign status labels to a longitudinal blood dataset with a diagnosis horizon
#'
#' @param df_x Dataframe containing person_id, year, and features
#' @param df_y Dataframe containing diagnoses (must include person_id, year_of_diag, status = 1)
#' @param horizon Integer, number of years after blood measurement to consider for DOI (default = 5)
#'
#' @return A labeled dataframe with label column: 1 = case, 0 = control, 2 = omit


assign_labels_with_horizon <- function(df_x, df_y, horizon = 5, join_column = "eid") {

  df_all <- df_x %>%
    left_join(df_y, by = join_column) %>%
    mutate(
    year_of_diag = as.numeric(year_of_diag),
      year = as.numeric(year),
      year_diff = year_of_diag - year,
      status = case_when(
        is.na(status) ~ 0,
        year_diff < 0 ~ 2,
        year_diff <= horizon ~ 1,
        year_diff > horizon ~ 0
      )
    )
    print(head(df_all))

  # Print diagnostics
  cat("📊 Status distribution:\n")
  print(table(df_all$status, useNA = "ifany"))

  cat(sprintf(
    "\n✅ Labeling complete with %d rows:\n",
    nrow(df_all)
  ))
  cat(sprintf("- %d positives (status = 1)\n", sum(df_all$status == 1, na.rm = TRUE)))
  cat(sprintf("- %d controls (status = 0)\n", sum(df_all$status == 0, na.rm = TRUE)))
  cat(sprintf("- %d omitted due to DOI before visit (status = 2)\n", sum(df_all$status == 2, na.rm = TRUE)))

  return(df_all)
}


# =============================================================================



#Visualization function to plot included and discarded cases by year
plot_included_discarded_cases <- function(df,
                                          year_col = "year_of_diag",
                                          target_col = "status",
                                          include_value = 1,
                                          discard_value = 2,
                                          base_size = 18,
                                          output_path = "visuals",
                                         filename = "discarded_cases.svg") {

  # Font setup
  if (systemfonts::system_fonts() %>% filter(family == "Arial") %>% nrow() > 0) {
    font_family <- "Arial"
  } else {
    font_family <- "Open Sans"
    font_add_google("Open Sans", "Open Sans")
  }

  showtext_auto()

  # Generate summary counts
  n_included <- sum(df[[target_col]] == include_value, na.rm = TRUE)
  n_discarded <- sum(df[[target_col]] == discard_value, na.rm = TRUE)
    n_total <- length(df)
    get("DOI", envir = .GlobalEnv)
    

  # Labels
  included_label <- paste0("Included (n=", n_included, ")")
  discarded_label <- paste0("Discarded (n=", n_discarded, ")")

  # Create plot
  plot <- ggplot(df %>% filter(.data[[target_col]] %in% c(include_value, discard_value)),
                 aes(x = .data[[year_col]], fill = as.factor(.data[[target_col]]))) +
    geom_histogram(binwidth = 1, color = "black", size = 0.2, width = 0.5) +
    scale_fill_manual(values = c("grey", "#808080"),
                      labels = c(included_label, discarded_label),
                      name = "") +
    xlab("Year") +
    ylab("Absolute number of cases") +
    ggtitle(paste("Year of", DOI, "Diagnosis")) +
    theme_minimal(base_family = font_family) +
    theme(text = element_text(family = font_family),
          plot.title = element_text(hjust = 0.5, size = base_size),
          legend.position = c(0.3, 0.9),
          axis.title.x = element_text(size = base_size),
          axis.title.y = element_text(size = base_size),
          axis.title.y.right = element_text(size = base_size, angle = 90, vjust = -0.5),
          axis.text.y.right = element_text(size = base_size, colour = "black"),
          legend.text = element_text(size = base_size),
          axis.text.x = element_text(size = base_size, colour = "black", vjust = 0.1),
          axis.text.y = element_text(size = base_size, colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
          panel.border = element_rect(colour = "black", fill = NA, linewidth = 1.5),
          legend.spacing.y = unit(2, "cm"),
          legend.key = element_rect(colour = "white", fill = NA)) +
    scale_y_continuous(
      expand = c(0, 0),
      limits = c(0, NA),
      sec.axis = sec_axis(
        ~ . / ifelse(is.null(n_total), 1, n_total) * 100000,
        name = "Incidence [n / 100.000]"
      )
    ) +
    guides(fill = guide_legend(override.aes = list(colour = "white")))

  print(plot)
    filename <- paste0(output_path, "/", DOI, "_", filename)
  ggsave(filename = filename, plot = plot, width = 10, height = 10, bg = "transparent")
}



# =============================================================================
# Normalization function to scale data
# =============================================================================
normalize_data_ukb <- function(df, conversion_table, vec_all) {
  # Ensure the required columns exist in the conversion table
  required_cols <- c("column", "max_ukb")
  if (!all(required_cols %in% colnames(conversion_table))) {
    stop("Conversion table must contain 'column' and 'max_ukb' columns")
  }
  
  # Create a named vector of max_ukb values, converting to numeric
  max_ukb_values <- sapply(setNames(conversion_table$max_ukb, conversion_table$column), 
                           function(x) as.numeric(as.character(x)))
  
  # Function to perform min-max normalization using UKB max
  minmax_ukb <- function(x, max_ukb) {
    if (is.na(max_ukb) || max_ukb == 0) {
      warning(sprintf("Invalid max_ukb value for column. Skipping normalization."))
      return(x)
    }
    return(x / max_ukb)
  }
  
  # Iterate through each column in vec_all
  for (col in vec_all) {
    if (col %in% names(df) && col != "eid" && is.numeric(df[[col]])) {
      if (col %in% names(max_ukb_values)) {
        ukb_max <- max_ukb_values[col]
        df[[col]] <- minmax_ukb(df[[col]], ukb_max)
        cat(sprintf("Column '%s' normalized using UKB max: %s\n", 
                    col, format(ukb_max, scientific = FALSE)))
      } else {
        warning(sprintf("UKB max value not found for column '%s'. Skipping normalization.", col))
      }
    } else if (col == "eid") {
      cat("Column 'eid' skipped for normalization.\n")
    } else if (!col %in% names(df)) {
      warning(sprintf("Column '%s' not found in the dataframe", col))
    } else if (!is.numeric(df[[col]])) {
      warning(sprintf("Column '%s' is not numeric. Skipping normalization.", col))
    }
  }
  
  return(df)
}

normalize_data_aou <- function(df, vec_covariates) {
  # Function to perform standard min-max normalization
  minmax <- function(x) {
    if (length(unique(x)) == 1) return(x)  # Return as is if all values are the same
    return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
  }
  
  # Iterate through each column in vec_covariates
  for (col in vec_covariates) {
    if (col %in% names(df) && col != "eid" && is.numeric(df[[col]])) {
      df[[col]] <- minmax(df[[col]])
      cat(sprintf("Column '%s' normalized using standard min-max.\n", col))
    } else if (col == "eid") {
      cat("Column 'eid' skipped for normalization.\n")
    } else if (!col %in% names(df)) {
      warning(sprintf("Column '%s' not found in the dataframe", col))
    } else if (!is.numeric(df[[col]])) {
      warning(sprintf("Column '%s' is not numeric. Skipping normalization.", col))
    }
  }
  
  return(df)
}

                           



# =============================================================================
#Patients at risk filtering
#' Filter Rows with Positive Entries in Specific Columns

filter_rows_with_pos_entries <- function(df) {
  
  df_name <- deparse(substitute(df)) #get name of df
      # Dimension and positive status BEFORE
  cat("\n", df_name, " - Dimensions before filtering:", dim(df), "\n")
  if ("status" %in% names(df)) {
    cat("Positive status (status == 1) before filtering:", sum(df$status == 1, na.rm = TRUE), "\n")
  } else {
    cat("Warning: No 'status' column found in the dataframe!\n")
  }
    
    
  # Ensure relevant_columns are in the dataframe (Diagnosis level)
  vec_risk_constellation <- par_index$Diagnosis[par_index$Group %in% par_subset] #subset index for project-specific requirements
  existing_columns <- vec_risk_constellation[vec_risk_constellation %in% names(df)]
    
  
  # Initialize a logical vector to flag rows with any positive entry in relevant columns
  positive_entry_rows <- rep(FALSE, nrow(df)) #empty vector
  
  for (col in existing_columns) {
    # Update the row flag if the entry is "1" for any of the relevant_columns
    positive_entry_rows <- positive_entry_rows | (df[[col]] == "1")
  }
  
  # Filter the dataframe to include only rows with at least one positive entry
  filtered_df <- df[positive_entry_rows, ]
    
      # Dimension and positive status AFTER
  cat("\n", df_name, " - Dimensions after filtering:", dim(filtered_df), "\n")
  if ("status" %in% names(filtered_df)) {
    cat("Positive status (status == 1) after filtering:", sum(filtered_df$status == 1, na.rm = TRUE), "\n")
  }
    
  message(df_name, " contains ", nrow(filtered_df), " Patients at risk.")
  
  
  #innerjoin_df_y(df)
  
  return(filtered_df)
}


















# =============================================================================
# END OF PREPROCESSING UTILITIES
# =============================================================================



print("All functions from preprocessing_functions.r successfully loaded.)")

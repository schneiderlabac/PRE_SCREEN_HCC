#########################################################################################


library(tidyverse)
library(data.table)
library(ggthemes)
library(extrafont)
options(java.parameters = "-Xmx8000m")
library(hrbrthemes)


#########################################################################################
#################################ICD Related Functions/ Visuals########################################################

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

comparison_plot_diags <- function(data, control, comparison) {
  # Reshape the data to long format
  data_long <- data %>%
    select(Diagnosis, Legend, Percentage.x, Percentage.y) %>%
    pivot_longer(cols = starts_with("Percentage"), 
                 names_to = "Group", 
                 values_to = "Percentage") %>%
    mutate(Group = recode(Group, `Percentage.x` = control, `Percentage.y` = comparison))
  
  # Determine top 15 diagnoses
  top_diagnoses <- data %>% 
    slice_head(n = 15) %>%
    pull(Diagnosis)
  
  data_long <- data_long %>%
    filter(Diagnosis %in% top_diagnoses)
  
  # Create the bar plot using the Legend column for the x-axis
  plot <- ggplot(data_long, aes(x = Legend, y = Percentage, fill = Group)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.8) +
    scale_fill_manual(values = c("Cancer" = "#BE4F4F", "No Cancer" = "#AFAFAF", "Patients at risk" = "#685A66")) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(x = "Diagnosis", y = "Prevalence (%)") +
    theme_minimal(base_size = 18, base_family = "Arial") +
    theme(
      legend.title = element_blank(),
      legend.position = c(0.8, 0.9), 
      legend.background = element_blank(),
      legend.text = element_text(size = 32, family = "Arial", color = "black"),
      axis.text = element_text(size = 28, family = "Arial", color = "black"),
      axis.title.x = element_text(size = 32, family = "Arial", color = "black", vjust = -1),
      axis.title.y = element_text(size = 32, family = "Arial", color = "black", vjust = 3),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(0.5,0.5,0.5,0.5, "cm"),
      panel.border = element_rect(colour = "black", fill = NA, size = 1)
    ) +
    guides(fill = guide_legend(reverse = TRUE, override.aes = list(size = 6)))
  
  print(plot)
  # Save the plots
  ggsave(filename = file.path(project_path, paste0("supplement_visuals/ICD_Relations_", control, "_", comparison, Sys.Date(), ".png")), 
         plot = plot, width = 10, height = 10, bg= "white", limitsize = FALSE)
  ggsave(filename = file.path(project_path, paste0("supplement_visuals/ICD_Relations_", control, "_", comparison,  Sys.Date(), ".svg")), 
         plot = plot, width = 10, height = 10, bg= "white")
  
  # Text legend
  text_legend <- top_diagnoses_data %>% 
    distinct(Legend, Diagnosis) %>% 
    arrange(Legend) %>% 
    mutate(Legend_Diagnosis = paste(Legend, Diagnosis, sep = ": ")) %>%
    pull(Legend_Diagnosis) %>%
    paste(collapse = "\n")
  cat("Legend:\n", text_legend)
  
  writeLines(paste("Legend:\n", text_legend), file.path(project_path, "supplement_visuals", paste0("ICD_Relations_Legend_", Sys.Date(), control, "_", comparison, ".txt")))
  
  # Return the plot and the top diagnoses data
  list(plot = plot, top_diagnoses_data = data %>% filter(Diagnosis %in% top_diagnoses))
}
  





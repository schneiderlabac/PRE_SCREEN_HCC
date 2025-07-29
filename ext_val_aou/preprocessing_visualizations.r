# #######################################################################
# ##########       Visuals for extraction of raw data ###################
# #######################################################################



plot_included_discarded_cases <- function(df, base_size = 18, n_total) {
    if (systemfonts::system_fonts() %>% filter(family == "Arial") %>% nrow() > 0) {
    font_family <- "Arial"
  } else {
    font_family <- "Open Sans"
    font_add_google("Open Sans", "Open Sans")
  }
  
  showtext_auto()
  doi_included <- get("doi_included", envir = .GlobalEnv)
  doi_discarded <- get("doi_discarded", envir = .GlobalEnv)
  included_label <- paste("Included (n=", doi_included, ")", sep = "") # Custom labels for the legend
  discarded_label <- paste("Discarded (n=", doi_discarded, ")", sep = "")

  plot <- ggplot(df, aes(x = year, fill = as.factor(discard))) +
    geom_histogram(binwidth = 1, color = "black", size=0.2, width= 0.5) +
    scale_fill_manual(values = c("grey", "#808080"), 
                      labels = c(included_label, discarded_label), 
                      name = "") + 
    
    xlab("Year") +
    ylab("Absolute number of cases") +
    ggtitle(paste("Year of", DOI, "Diagnosis")) +
    theme_minimal(base_family = font_family) +
    theme(text = element_text(family = font_family),
          plot.title = element_text(hjust = 0.5, size = base_size), # Increase size to 150% 
          legend.position = c(0.3, 0.9), # upper left corner
          axis.title.x = element_text(size = base_size), 
          axis.title.y = element_text(size = base_size),
          axis.title.y.right = element_text(size = base_size, angle=90, vjust=-0.5),
          axis.text.y.right = element_text(size = base_size, colour = "black"),
          legend.text = element_text(size = base_size), 
          axis.text.x = element_text(size = base_size, colour="black", vjust=0.1), 
          axis.text.y = element_text(size = base_size, colour="black"),
          panel.grid.major = element_blank(), # Remove major grid lines
          panel.grid.minor = element_blank(), # Remove minor grid lines
          plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
          panel.border = element_rect(colour = "black", fill = NA, linewidth = 1.5),
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
  ggsave(filename = paste0("data/", DOI, "_yearly_cases.svg"), plot = plot, width = 10, height = 10, bg = "transparent")
  
  }






stacked_bars_time_comparison <- function(df, base_size=18) {
  priority_order <- c("Cirrhosis", "Viral Hepatitis", "CLD", "No Liver disease")
  
  df <- df %>%
    mutate(Priority = priority(Group)) %>%
    arrange(Time, Priority) %>%
    group_by(Time) %>%
    mutate(LabelPos = cumsum(Count) - 0.5 * Count) %>% # Calculate label positions on cumulative sum column for label positioning
    ungroup() %>%
    mutate(Group = factor(Group, levels = priority_order))
  print(head(df))
  
  max_x_value <- max(as.numeric(as.factor(df$Time))) + 1
  df$max_x <- max_x_value  # add max_x to the dataframe
  
  label_data <- df %>%
    filter(Order == 2) %>%
    distinct(Group, .keep_all = TRUE)
  
  plot <- ggplot(data = df, aes(x = Time, y = Count, fill = Group)) +
    geom_bar(stat = "identity", position = position_stack(vjust = 0.5, reverse = TRUE), width=0.55 ) +
    geom_text(aes(label = Count, y = LabelPos), size = base_size * 0.4, colour = "black", vjust = -0.3) +
    geom_text(aes(label = sprintf("%.0f%%", Percentage), y = LabelPos), size = base_size * 0.3, colour = "black", vjust = 1.2) +
    geom_text(data = distinct(label_data, Group, .keep_all = TRUE), 
              aes(x = max_x - 0.4, label = Group, y = LabelPos), hjust=0, size = base_size * 0.4, color = "black") +
    theme_minimal() +
    labs(title = "Etiology Over Time",
         y = "Count") +
    scale_fill_brewer(palette = "Set3") +
    theme(plot.title = element_text(size=base_size, hjust = 0.5),
          axis.title.x =element_blank(),
          axis.title.y = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          axis.text.x = element_text(size = base_size, colour = "black"),
          axis.text.y = element_blank(),
          panel.grid.major = element_blank(), # Remove major grid lines
          panel.grid.minor = element_blank(), # Remove minor grid lines
          plot.margin = margin(1, 300, 1, 10)) +
    coord_cartesian(clip = 'off')

  print(plot)
  #print(df)
  ggsave(filename = paste0(DOI, "/", DOI, "_etiology_over_time.svg"), 
         plot = plot, width = 8, height = 10, bg = "transparent")
}




print("All functions from preprocessing_visualizations.r successfully loaded.)")

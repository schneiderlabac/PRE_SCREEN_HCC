library(ggplot2)

custom_theme <- theme(
  legend.title = element_blank(),
  legend.position = c(0.8, 0.9),
  legend.background = element_blank(),
  legend.text = element_text(size = 28, family = "Arial", color = "black"),
  axis.text = element_text(size = 28, family = "Arial", color = "black"),
  axis.title.x = element_text(size = 32, family = "Arial", color = "black", vjust = -1),
  axis.title.y = element_text(size = 32, family = "Arial", color = "black", vjust = 3),
  panel.grid.major.x = element_blank(),
  panel.grid.major.y = element_blank(),
  panel.grid.minor = element_blank(),
  plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
  panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)
)

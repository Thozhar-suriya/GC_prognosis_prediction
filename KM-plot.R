library("ipred")
library("survival")
library("survivalROC")
library("glmnet")
library('kernlab')
library('caret')
library(survminer)
library(survival)
library(dplyr)
library(tidyverse)
library(ggsurvplot)

##data Preparation
meta <- read.csv("Survival_data_1.csv", header = T, row.names=1)
data <- read.csv("Clust_lab.csv", header = T, row.names=1)
data1 <- merge(data, meta, by = "row.names")
dim(data1)
data1 <- data1 %>% drop_na()
write.csv(data1, "data - Copy.csv")


## KM plot original

data <- read.csv("data - Copy.csv", header = T, row.names = 1)
View(data)
kmg <- survfit(Surv(time/364, status) ~ Cluster_Label, data=data)

data$Cluster_Label <- factor(data$Cluster_Label, 
                           levels = c("G1", "G2"))
                           
p <- ggsurvplot(kmg, 
                data = data, 
                conf.int = FALSE, 
                pval = TRUE, 
                pval.method = TRUE,
                pval.size = 4,                   # Size of p-value text
                pval.method.size = 4,            # Size of p-value method text
                pval.fontface = "bold",          # Bold p-value
                pval.method.fontface = "bold",   # Bold p-value method
                risk.table = FALSE, 
                xlab = "Year", 
                ylab = "Survival probability",
                legend.labs = c("G1", "G2"), 
                legend.title = "Group",
                palette = c("dodgerblue2", "orchid2"),  
                ggtheme = theme_classic2(base_size = 12, base_family = "Arial"),
                font.family = "Arial")

# Further customize fonts using ggpar
ggpar(p, 
      font.main = c(11, "bold"),      # Title font
      font.x = c(11, "bold"),         # X-axis font
      font.y = c(11, "bold"),     
      font.caption = c(9, "bold"),    # Caption font
      font.legend = c(12, "bold"),    # Legend font
      font.tickslab = c(10, "bold"))  # Axis tick labels

ggsave(filename = "survival_plot1.pdf", 
       plot = p$plot, 
       width = 7.5, 
       height = 4.5, 
       units = "in", 
       dpi = 300, 
       device = cairo_pdf)
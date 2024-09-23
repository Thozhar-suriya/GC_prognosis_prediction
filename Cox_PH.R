setwd('D:')
library('plyr')
library("ipred")
library("survival")
library("survivalROC")
library("glmnet")
library("dplyr")
library("tidyr")
library(survminer)

data=read.csv("Encoded.csv",row.names= 1)
dim(data)
#View(data)
mysurv=Surv(data$time,data$status)
mysurv

Unicox <- function(x){
  fml <- as.formula(paste0('mysurv~', x))
  gcox <- coxph(fml, data)
  cox_sum <- summary(gcox)
  HR <- round(cox_sum$coefficients[,2],2)
  PValue <- round(cox_sum$coefficients[,5],4)
  CI <- paste0(round(cox_sum$conf.int[,3:4],2),collapse='-')
  Uni_cox <- data.frame('Characteristics' = x,
                        'Hazard Ratio' = HR,
                        'CI95' = CI,
                        'P value' = PValue)
  return(Uni_cox)
}

VarNames <-colnames(data)[3:ncol(data)]
Univar <- lapply(VarNames, Unicox)
Univar <- ldply(Univar, data.frame)
Univar
write.csv(Univar, "Univariate.csv")


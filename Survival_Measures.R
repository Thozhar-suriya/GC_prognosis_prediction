library(survival)
library(survminer)
library(survcomp)
library(scoring)
library(pec)
library(survivalROC)
library(rms)
library("ipred")
library('kernlab')
library('caret')


##C-index for predicted
Sur2 <- Surv(data$time, data$status)
fit <- coxph(Sur2 ~ risk, data = data)
cindex <- survcomp::concordance.index(x=predict(fit),
                                      surv.time = data$time,
                                      surv.event= data$status,
                                      method = "noether" )
print(cindex$c.index)


#New Brier score
library(rms)

cox_model <- coxph(Surv(time, status) ~ risk, data=data)
brier_score <- brier(cox_model)
print(brier_score)
mean_brier_score <- mean(brier_score$brier)
cat("brier_score:", mean_brier_score, "\n")


##P-Value
kmg <- surv_fit(Surv(time/365, status) ~ risk, data=data)
surv_pvalue(kmg)

high_samples <- sum(data$risk > 0.728)
low_samples <- sum(data$risk < 0.728)

p <- ggsurvplot(kmg, data=data, conf.int=FALSE, pval=TRUE, pval.method=TRUE, risk.table=FALSE, xlab = "Year", ylab = "Survival probability",
                legend.labs=c("High", "Low"), legend.title="Risk", palette=c("dodgerblue2", "orchid2"),  ggtheme = theme_classic2(base_size=10, base_family = "Arial"),
                font.family = "Arial")
ggpar(p, 
      font.main = c(10, "bold"),
      font.x = c(10, "bold"),
      font.y = c(10, "bold"),
      font.caption = c(10, "bold"), 
      font.legend = c(10, "bold"), 
      font.tickslab = c(10, "bold"))


cat("C-index:", ci$concordance, "\n")
cat("brier_score:", mean_brier_score, "\n")
cat("p_value:", surv_pvalue(kmg)$pval.txt, "\n")





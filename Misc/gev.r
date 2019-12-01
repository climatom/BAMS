library(ismev)
library(extRemes)
library(ncdf4)
library(testit)
library(MASS)

# Associate filname for ann-max TW and read in
fin <- "/home/lunet/gytm3/Everest2019/Research/BAMS/extreme_series.csv"
data <- read.csv(fin,sep=",")
# Reverese series for temp 
data["y_sum"] <- data["t_sum"]*-1
data["y_col"] <- data["t_col"]*-1
# Compute max annual series
vs=c("y_sum","y_col","ws_sum","ws_col")
ann <- as.data.frame(matrix(ncol=4, nrow=40))
names(ann) <- vs
ann["y_sum"]<-aggregate(y_sum~year,data,max)["y_sum"]
ann["y_col"]<-aggregate(y_col~year,data,max)["y_col"]
ann["ws_sum"]<-aggregate(ws_sum~year,data,max)["ws_sum"]
ann["ws_col"]<-aggregate(ws_col~year,data,max)["ws_col"]
write.csv(ann,"/home/lunet/gytm3/Everest2019/Research/BAMS/ann_max.csv")

# Fit GEV models 
fit_temp_summit<-gev.fit(ann$y_sum)
fit_temp_col<-gev.fit(ann$y_col)
fit_ws_summit<-gev.fit(ann$ws_sum)
fit_ws_col<-gev.fit(ann$ws_col)

# Write out the coefficients (temp summit, temp col, ws summit, ws col)
coefs <- matrix(ncol=3, nrow=4)
coefs[1,] <- c(fit_temp_summit$mle)
coefs[2,] <- c(fit_temp_col$mle)
coefs[3,] <- c(fit_ws_summit$mle)
coefs[4,] <- c(fit_ws_col$mle)
write.matrix(coefs,"/home/lunet/gytm3/Everest2019/Research/BAMS/GEV_coefs.csv",sep=",")





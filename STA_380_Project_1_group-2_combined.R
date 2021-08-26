library(randomForest)
library(dplyr)
library(Metrics)
library(caret)
library(corrplot)
library(caret)
library(e1071)

library(reshape2)
library(ggplot2)
library(class) ## a library with lots of classification tools
library(kknn) ## knn library


#################KNN########################

jfk_to <- read.csv("M1_final.csv")

n=dim(jfk_to)[1]
set.seed(1)

#If run on complete data set, code takes nearly 4 hours to run.
#Comment this line to not reduce the dataset. 
ind = sample(1:n,0.03*n) #Reduce to 3% of Original Data for faster run! Comment to remove this
jfk_to <- jfk_to[ind,]  #Reduce to 3% of Original Data for faster run! Comment to remove this
n=dim(jfk_to)[1]  #Reduce to 3% of Original Data for faster run! Comment to remove this

#Label Encoding
jfk_to$OP_UNIQUE_CARRIER= as.integer(as.factor(jfk_to$OP_UNIQUE_CARRIER))
jfk_to$DEST = as.integer(as.factor(jfk_to$DEST         ))
jfk_to$Wind = as.integer(as.factor(jfk_to$Wind        ))
jfk_to$Condition = as.integer(as.factor(jfk_to$Condition ))


#Data Cleanup
jfk_to$Dew.Point <- sub("[^0-9]", "", jfk_to$Dew.Point)
jfk_to$Dew.Point <- as.numeric(jfk_to$Dew.Point)

#Split Dataset in 75:25 Ratio for Test and Train Data.
ind = sample(1:n,0.75*n) #Reduce to 10% of Original Data
jfk_train <- jfk_to[ind,]
jfk_test <- jfk_to[-ind,]



#Scaling Standard Normalization/ Z factor
jfk_standardized <- as.data.frame(scale(jfk_train[-c(5)]))




#Draw Corelation Between Variables
cormat <- round(cor(jfk_standardized),2)
melted_cormat <- melt(cormat)
head(melted_cormat)

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

upper_tri <- get_upper_tri(cormat)

melted_cormat <- melt(upper_tri, na.rm = TRUE)
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 12, hjust = 1))+
  coord_fixed()

ggheatmap +
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 1.5) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))


#Corelation for TAXI_OUT VARIABLE ONLY
TAXI_OUT_CORR=melted_cormat[melted_cormat$Var2 == "TAXI_OUT" & melted_cormat$Var1 != "TAXI_OUT", ]
TAXI_OUT_CORR$absvalue=abs(as.numeric(TAXI_OUT_CORR$value))
TAXI_OUT_CORR <- TAXI_OUT_CORR[order(-TAXI_OUT_CORR$absvalue),]

p<-ggplot(data=TAXI_OUT_CORR, aes(x = reorder(Var1, -value^2), y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=value), vjust=-0.3, size=2.5)+
  theme_minimal()


p + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


#KNN Algorithm and KFold Verification
#Prepare Train and Test Data sets

n=dim(jfk_standardized)[1]

Y = jfk_standardized$TAXI_OUT
jfk_standardized=jfk_standardized[ , !(names(jfk_standardized) %in% c("TAXI_OUT"))]
JFKData = jfk_standardized

#Train and Test Variables for the purpose of KNN
train = data.frame(Y,JFKData)
test = data.frame(Y,JFKData)


#Run for In order of Correlation coefficient
# 1 Variable - Top Corelation coeffecient Features
# 2 Variable - Top 2 Correlation coeefiicient Features considered and so on
# 3 Variable
# .....
# All Variables


colsUsed <-c()
bestMSE <- c()
cols <- c()
bestMSEkv<-c()
#For each Feature Combination
for(var in TAXI_OUT_CORR$Var1){
  JFKData = jfk_standardized
  cols <- c(cols,var)
  print(cols)
  print('---------------')
  train = data.frame(Y,JFKData[,cols])
  test = data.frame(Y,JFKData[,cols])
  
  
  
  kcv = 10
  n0 = round(n/kcv,0)
  out_MSE = matrix(0,kcv,350)
  
  used = NULL
  set = 1:n
  
  j=1
  for(j in 1:kcv){
    
    if(n0<length(set)){val = sample(set,n0)}
    if(n0>=length(set)){val=set}
    
    train_i = train[-val,]
    test_i = test[val,]
    
    for(i in 1:350){
      
      near = kknn(Y~.,train_i,test_i,k=i,kernel = "rectangular")
      aux = mean((test_i[,1]-near$fitted)^2)
      
      out_MSE[j,i] = aux
    }
    
    used = union(used,val)
    set = (1:n)[-used]
    
    cat(j,'\n')
    
  }
  
  
  
  mMSE = apply(out_MSE,2,mean)
  
  plot(log(1/(1:350)),sqrt(mMSE),xlab="Complexity (log(1/k))",ylab="out-of-sample RMSE",col=4,lwd=2,type="l",cex.lab=1.2,main=paste("kfold(",kcv,")"))
  best = which.min(mMSE)
  text(log(1/best),sqrt(mMSE[best])+0.1,paste("k=",best),col=2,cex=1.2)
  text(log(1/2),sqrt(mMSE[2])+0.3,paste("k=",2),col=2,cex=1.2)
  text(log(1/100)+0.4,sqrt(mMSE[100]),paste("k=",100),col=2,cex=1.2)
  
  
  bestMSE <- c(bestMSE, min(mMSE))
  bestMSEkv<-c(bestMSEkv, best)
}

checkIndexMinimumMSE=which.min(bestMSE)
i=1
finalFeatureCombination <- c()
for(var in TAXI_OUT_CORR$Var1){
  JFKData = jfk_standardized
  finalFeatureCombination <- c(finalFeatureCombination,var)
  
  if(i==checkIndexMinimumMSE){
    break;
  }
  i=i+1
}

finalkValue=bestMSEkv[checkIndexMinimumMSE]

#TEST AGAINST OUT OF SAME 25% DATA
Ytrain = jfk_train$TAXI_OUT
Ytest = jfk_test$TAXI_OUT


jfk_train_standardized <- as.data.frame(scale(jfk_train[-c(23,5)]))
jfk_test_standardized <- as.data.frame(scale(jfk_test[-c(23,5)]))

train= data.frame(Ytrain,jfk_train_standardized)
test= data.frame(Ytest,jfk_test_standardized)

near = kknn(Ytrain~.,train,test,k=20,kernel = "rectangular")
aux = mean((test[,1]-near$fitted)^2)

rmse_final=sqrt(aux)
print(rmse_final)



#################Boosting########################

#importing data, converting, and dropping tail numbers ############################

flight_data=read.csv("M1_final.csv")
flight_data_df=data.frame(flight_data)
flight_data_df$Dew.Point=as.integer(flight_data_df$Dew.Point)
str(flight_data_df)
flight_data_df$OP_UNIQUE_CARRIER=as.factor(flight_data_df$OP_UNIQUE_CARRIER)
flight_data_df$TAIL_NUM=as.factor(flight_data_df$TAIL_NUM)
flight_data_df$DEST=as.factor(flight_data_df$DEST)
flight_data_df$Wind=as.factor(flight_data_df$Wind)
flight_data_df$Condition=as.factor(flight_data_df$Condition)
str(flight_data_df)
flight_data_df=flight_data_df[,-5]


#splitting of data ##################################################################
set.seed(1234)
n=nrow(flight_data_df)
n1=floor(n/2)
n2=floor(n/4)
n3=n-n1-n2
ii = sample(1:n,n)
fdtrain=flight_data_df[ii[1:n1],]
fdval = flight_data_df[ii[n1+1:n2],]
fdtest = flight_data_df[ii[n1+n2+1:n3],]


# boosting model with model comparison table #######################################

library(gbm)

set.seed(1234)
idv = c(4,6,10)
ntv = c(500,1000,5000)
lamv=c(0.01,.1,.2)
parmb = expand.grid(idv,ntv,lamv)
colnames(parmb) = c('tdepth','ntree','lam')
print(parmb)
nset = nrow(parmb)
olb = rep(0,nset)
ilb = rep(0,nset)
bfitv = vector('list',nset)
#this loop takes at least 10 minutes, but it works
for(i in 1:nset) {
  cat('doing boost ',i,' out of ',nset,'\n')
  tempboost = gbm(TAXI_OUT~.,data=fdtrain ,distribution='gaussian',
                  interaction.depth=parmb[i,1],n.trees=parmb[i,2],shrinkage=parmb[i,3])
  ifit = predict(tempboost,n.trees=parmb[i,2])
  ofit=predict(tempboost,newdata=fdval,n.trees=parmb[i,2])
  olb[i] = sum((fdval$TAXI_OUT-ofit)^2)
  ilb[i] = sum((fdtrain$TAXI_OUT-ifit)^2)
  bfitv[[i]]=tempboost
}
ilb = round(sqrt(ilb/nrow(fdtrain)),3); olb = round(sqrt(olb/nrow(fdval)),3)

results_table=cbind(parmb,olb,ilb)
results_table



# running best params with test set to get OOB RMSE, importance graph, ######################
fdtrainval = rbind(fdtrain,fdval)
#this single boost model is about 5 minutes long
finb = gbm(TAXI_OUT~.,data=fdtrainval ,distribution='gaussian', interaction.depth=10,n.trees=5000,shrinkage=0.01)
finbpred=predict(finb,newdata=fdtest,n.trees=5000)
finbrmse = sqrt(sum((fdtest$TAXI_OUT-finbpred)^2)/nrow(fdtest))
cat('finbrmse: ',finbrmse,'\n')
plot(fdtest$TAXI_OUT,finbpred,xlab='test TAXI_OUT',ylab='boost pred'); abline(0,1,col='red',lwd=2)


p=ncol(fdtrain)-1 #want number of variables for later
vsum=summary(finb) #this will have the variable importance info
row.names(vsum)=NULL #drop varable names from rows.

print(vsum)

plot(vsum$rel.inf,axes=F,pch=16,col='red'); axis(1,labels=vsum$var,at=1:p, las=2); axis(2); for(i in 1:p) lines(c(i,i),c(0,vsum$rel.inf[i]),lwd=4,col='blue')


#################################################################
### Bagging
#################################################################
# Bagging over the JFK Airport Data
# Bring in data as csv and set seed
M1_final = read.csv('M1_final.csv', stringsAsFactors = TRUE)
library(randomForest)
set.seed(1)

# Modify data to streamline calculations
airdata = subset(M1_final, select = -c(5,6))
airdata$Dew.Point = as.numeric(airdata$Dew.Point)
airdata$Wind.Gust = as.numeric(airdata$Wind.Gust)
airdata$Wind.Speed = as.numeric(airdata$Wind.Speed)
str(airdata)
attach(airdata)

# set up for loop for determining RSME with different parameters
ntreev = c(100,500,1000)
train = sample(1:nrow(airdata), nrow(airdata)*.75, replace = TRUE)
a.train = airdata[train,]
test = airdata[-train,]
n = nrow(test)
ntreev = c(100, 500, 1000)
nset = length(ntreev)
fmat = matrix(0,n,nset)

# loop over parameters with bagging
for (i in 1:nset) {
  cat('Taxi Out bagging:',i,'\n')
  bag.final = randomForest(TAXI_OUT~., data = a.train,mtry = 20, ntree = ntreev[i],maxnodes= 75,importance = TRUE)
  fmat[,i] = predict(bag.final, newdata = test)
}
# RSME calculation fmat[,i], where i = 1,2,3 and 1 = 100, 2 = 500, 3 = 1000
sqrt(mean((fmat[,3]-test$TAXI_OUT)^2))

# getting variable importance for optimal model
bag.final = randomForest(TAXI_OUT~., data = train,mtry = 20, ntree = 100,maxnodes= 50,importance = TRUE)
varImpPlot(bag.final, main = 'Variable Importance \n for Optimal Model')

#plotting error vs. ntrees
plot(bag.final)

# plot airline carriers vs. taxi_out time
plot(M1_final$OP_UNIQUE_CARRIER, TAXI_OUT)

#############################################################

#############################################################
# Random Forest Regressor
# author : Vivek Mehendiratta
#############################################################

set.seed(1)

# Creating a function which imports Taxi data and wrangles it
getTaxiData = function() {
  taxi = read.csv("M1_final.csv", stringsAsFactors = TRUE)
  
  # Cleaning Dew.Point variable
  a = c(
    -1,-2,-3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 67
  )
  
  taxi = taxi[taxi$Dew.Point %in% as.character(a), ]
  taxi$Dew.Point = as.integer(taxi$Dew.Point)
  
  return(taxi)
  
}

taxi = getTaxiData()

str(taxi)

# Running Random Forest on raw data without wrangling
rf.taxi = randomForest(
  formula = TAXI_OUT ~ .,
  data = taxi,
  mtry = 10,
  importance = TRUE,
  maxnodes = 15,
  ntree = 100,
  type = "regression"
)

# Error occured : Can not handle categorical predictors 
# with more than 53 categories.

# Cleaning Dew.Point variable

taxi = getTaxiData()

# To handle this error, changing the factors where levels are more than 53 
# to label encoding
# Label Encoding
taxi$TAIL_NUM = as.integer(taxi$TAIL_NUM)
taxi$DEST = as.integer(taxi$DEST)

# Random Forest Modeling
# to make this work, using maxnodes paramter in random forest
rf.taxi = randomForest(
  formula = TAXI_OUT ~ .,
  data = taxi,
  mtry = 10,
  maxnodes = 15,
  importance = TRUE,
  type = "regression"
)

which.min(rf.taxi$mse)

plot(rf.taxi)
rf.taxi

importance (rf.taxi)
varImpPlot (rf.taxi)

# TAIL_NUM appears to be very week indicator, may be it is because of Label Encoding
# One-hot encoding

taxi = getTaxiData()

dummy = dummyVars(" ~ .", data = taxi)
taxi_onehot = data.frame(predict(dummy, newdata = taxi))
print(dim(taxi_onehot))

# Running Random forest on one-hot encoded variable
rf.taxi.onehot = randomForest(
  formula = TAXI_OUT ~ .,
  data = taxi_onehot,
  mtry = 48,
  maxnodes = 15,
  importance = TRUE,
  type = "regression"
)

which.min(rf.taxi.onehot$mse)

plot(rf.taxi.onehot)
rf.taxi.onehot

importance(rf.taxi.onehot)
varImpPlot(rf.taxi.onehot)

# TAIL_NUM, DEST does not have much affect, removing them
# Removing "DAY_OF_MONTH", "DAY_OF_WEEK", "MONTH" as it won't help me predict for the current time

taxi = getTaxiData()

drop_columns = c()
drop_columns = append(drop_columns,
                      c("TAIL_NUM", "DEST", "DAY_OF_MONTH", "DAY_OF_WEEK", "MONTH"))

taxi = taxi %>% select(-one_of(drop_columns))

str(taxi)

rf.taxi = randomForest(
  formula = TAXI_OUT ~ .,
  data = taxi,
  mtry = 8,
  maxnodes = 15,
  importance = TRUE,
  type = "regression"
)

which.min(rf.taxi$mse)

plot(rf.taxi)
rf.taxi

importance (rf.taxi)
varImpPlot (rf.taxi)

# Train Test split
# 75/25 split

taxi = getTaxiData()

drop_columns = c()
drop_columns = append(drop_columns,
                      c("TAIL_NUM", "DEST", "DAY_OF_MONTH", "DAY_OF_WEEK", "MONTH"))

taxi = taxi %>% select(-one_of(drop_columns))

train_ind = sample(1:nrow(taxi), 21000)

rf_test = taxi[-train_ind, ]


# Cross Validation
# 5-fold CV
training_control = trainControl(method = "cv",
                                number = 5,
                                search = "random")
set.seed(1)

# Run the model
rf_cv = train(
  TAXI_OUT ~ .,
  data = taxi,
  subset = train_ind,
  method = "rf",
  metric = "RMSE",
  maxnodes = 15,
  trControl = training_control
)

# Print the results
print(rf_cv)

# Parameter tuning : maxnodes, ntrees
cat("Parameter search by trying multiple parameters of max nodes and ntree")
tuneGrid = expand.grid(.mtry = 5)
for (maxnode_ in c(15, 20, 25, 50, 75)) {
  for (ntree_ in c(100, 500, 1000)) {
    cat("maxnode : ", maxnode_, '\t ntree : ', ntree_, '\n')
    rf_best_param = train(
      TAXI_OUT ~ .,
      data = taxi,
      subset = train_ind,
      method = "rf",
      metric = "RMSE",
      type = "Regression",
      tuneGrid = tuneGrid,
      trControl = training_control,
      ntree = ntree_,
      maxnodes = maxnode_,
      importance = TRUE
    )
    print(rf_best_param)
    
    pred = predict(rf_best_param, newdata = rf_test)
    predicted_rmse = sqrt(sum((rf_test$TAXI_OUT - pred)^2)/nrow(rf_test))
    cat('\nOOB RMSE : ', predicted_rmse, '\n\n')
  }
}

best_ntree = 1000
best_maxnode = 75

rf_best = randomForest(
  TAXI_OUT ~ .,
  data = taxi,
  subset = train_ind,
  type = "regression",
  mtry =5,
  ntree = best_ntree,
  maxnodes = best_maxnode,
  importance = TRUE
)

pred = predict(rf_best, newdata = rf_test)
predicted_rmse = sqrt(sum((rf_test$TAXI_OUT - pred)^2)/nrow(rf_test))
cat('\nOOB RMSE : ', predicted_rmse, '\n\n')

optimum_ntree = 100
optimum_maxnode = 50

rf_optimum = randomForest(
  TAXI_OUT ~ .,
  data = taxi,
  subset = train_ind,
  type = "regression",
  mtry =5,
  ntree = optimum_ntree,
  maxnodes = optimum_ntree,
  importance = TRUE
)

pred = predict(rf_optimum, newdata = rf_test)
predicted_rmse = sqrt(sum((rf_test$TAXI_OUT - pred)^2)/nrow(rf_test))
cat('\nOOB RMSE : ', predicted_rmse, '\n\n')

plot(rf_best)
plot(rf_optimum)

varImpPlot(rf_best)
varImpPlot(rf_optimum)

plot(taxi$TAXI_OUT~taxi$OP_UNIQUE_CARRIER, las = 3)
plot(taxi$TAXI_OUT~taxi$Condition, las = 3)


# Keeping top 4 variables and TAXI_OUT variable
keep_columns = c()
keep_columns = append(keep_columns,
                      c("Wind", "OP_UNIQUE_CARRIER", "sch_dep", "Condition", "TAXI_OUT"))

taxi = taxi %>% select(one_of(keep_columns))

rf_test_top4 = taxi[-train_ind, ]

rf_top4 = randomForest(
  TAXI_OUT ~ .,
  data = taxi,
  subset = train_ind,
  type = "regression",
  mtry = 2,
  ntree = optimum_ntree,
  maxnodes = optimum_ntree,
  importance = TRUE
)

pred = predict(rf_top4, newdata = rf_test_top4)
predicted_rmse = sqrt(sum((rf_test_top4$TAXI_OUT - pred)^2)/nrow(rf_test_top4))
cat('\nOOB RMSE : ', predicted_rmse, '\n\n')

plot(rf_top4)

######################################################
############ Multi Linear Regression #################
######################################################
##### Get access to tidyverse for graphical purposes
library(tidyverse)

##### Read in the csv file that was downloaded from Kaggle
mydata = read.csv('C:\\Users\\smkal\\Desktop\\Project\\JFK.csv', header = TRUE)

##### Converting Dew.Point column to int but still having issues with it
##### when trying to run regression  
mydata$Dew.Point = as.integer(mydata$Dew.Point)

##### Attach the file to have easy access
attach(mydata)
View(mydata)

##### Look at some of the data
head(mydata)
table(mydata$TAXI_OUT)
str(mydata)

##### Begin fitting the linear regression models using the lm() method
##### Looking for meaningful coefficients using a single predictor variable
model_single1 = lm(TAXI_OUT ~ MONTH)
summary(model_single1)
### Coefficient is 0.0254

model_single2 = lm(TAXI_OUT ~ DAY_OF_MONTH)
summary(model_single2)
### Coefficient is -0.0188

model_single3 = lm(TAXI_OUT ~ DAY_OF_WEEK)
summary(model_single3)
### Coefficient is 0.098 

model_single4 = lm(TAXI_OUT ~ DEP_DELAY)
summary(model_single4)
### Coefficient is 0.006

model_single5 = lm(TAXI_OUT ~ CRS_ELAPSED_TIME)
summary(model_single5)
### Coefficient is 0.004

model_single6 = lm(TAXI_OUT ~ DISTANCE)
summary(model_single6)
### Coefficient is 4.612e-04

model_single7 = lm(TAXI_OUT ~ CRS_DEP_M)
summary(model_single7)
### Coefficient is 1.027e-03

model_single8 = lm(TAXI_OUT ~ DEP_TIME_M)
summary(model_single8)
### Coefficient is 1.027e-03

model_single9 = lm(TAXI_OUT ~ CRS_ARR_M)
summary(model_single9)
### Coefficient is 1.416e-03

model_single10 = lm(TAXI_OUT ~ Temperature)
summary(model_single10)
### Coefficient is -0.057563

##### The summary here prints out a lot of values? Will leave this out
##### during multiple regression
model_single11 = lm(TAXI_OUT ~ Dew.Point)
summary(model_single11)
### Coefficient is ?

model_single12 = lm(TAXI_OUT ~ Humidity)
summary(model_single12)
### Coefficient is -0.008

model_single13 = lm(TAXI_OUT ~ Wind.Speed)
summary(model_single13)
### Coefficient is 0.068

model_single14 = lm(TAXI_OUT ~ Wind.Gust)
summary(model_single14)
### Coefficient is 0.055

### One of the best predictor variables
model_single15 = lm(TAXI_OUT ~ Pressure)
summary(model_single15)
plot(mydata$Pressure, mydata$TAXI_OUT)
abline(model_single15, col = "red", lwd = 2)
### Coefficient is -1.372

### One of the best predictor variables
model_single16 = lm(TAXI_OUT ~ sch_dep)
summary(model_single16)
plot(mydata$sch_dep, mydata$TAXI_OUT)
abline(model_single16, col = "red", lwd = 2)
### Coefficient is 0.137

model_single17 = lm(TAXI_OUT ~ sch_arr)
summary(model_single17)
plot(mydata$sch_arr, mydata$TAXI_OUT)
abline(model_single17, col = "red", lwd = 2)
### Coefficient is 0.053

##### Fitting models with multiple predictor variables
### X variables are month, day of month, day of week
model_multi_1 = lm(TAXI_OUT ~ MONTH + DAY_OF_MONTH + DAY_OF_WEEK)
summary(model_multi_1)

### X variables are scheduled departure, scheduled arrival
model_multi_2 = lm(TAXI_OUT ~ sch_dep + sch_arr)
summary(model_multi_2)

### X variables are scheduled departure, scheduled arrival, pressure
model_multi_3 = lm(TAXI_OUT ~ sch_dep + sch_arr + Pressure)
summary(model_multi_3)

### X variables are pressure, wind speed, wind gust
model_multi_4 = lm(TAXI_OUT ~ Pressure + Wind.Speed + Wind.Gust)
summary(model_multi_4)

### X variables are all quantitative columns
model_all = lm(TAXI_OUT ~ MONTH + DAY_OF_MONTH + DAY_OF_WEEK + DEP_DELAY + CRS_ELAPSED_TIME
               + DISTANCE + CRS_DEP_M + DEP_TIME_M + CRS_ARR_M + Temperature + Humidity
               + Wind.Speed + Wind.Gust + Pressure + sch_dep + sch_arr)
coef(model_all)
summary(model_all)

##### Visualize each model using plot()
plot(model_multi_1)
plot(model_multi_2)
plot(model_multi_3)
plot(model_multi_4)
plot(model_all)

##### Making predictions from actual values from first row where TAXI_OUT = 14
predict(model_multi_1, data.frame(MONTH = 11, DAY_OF_MONTH = 1, DAY_OF_WEEK = 5), interval = "confidence")

### Best results from multiple regression occur here with model_multi_2, though still not very accurate
predict(model_multi_2, data.frame(sch_dep = 9, sch_arr = 17), interval = "confidence")

predict(model_multi_3, data.frame(sch_dep = 9, sch_arr = 17, Pressure = 29.86), interval = "confidence")

predict(model_multi_4, data.frame(Pressure = 29.86, Wind.Speed = 25, Wind.Gust = 38), interval = "confidence")

predict(model_all, data.frame(MONTH = 11, DAY_OF_MONTH = 1, DAY_OF_WEEK = 5, DEP_DELAY = -1, 
                              CRS_ELAPSED_TIME = 124, DISTANCE = 636, CRS_DEP_M = 324, DEP_TIME_M = 323,
                              CRS_ARR_M = 448, Temperature = 48, Humidity = 58, Wind.Speed = 25,
                              Wind.Gust = 38, Pressure = 29.86, sch_dep = 9, 
                              sch_arr = 17), interval = "confidence")

##### Predicting with one of the simple linear models
predict(model_single16, data.frame(sch_dep = 9), interval = "confidence")
#############################################################
#############################################################
# Random Forest Regressor
# author : Vivek Mehendiratta
#############################################################


library(randomForest)
library(dplyr)
library(Metrics)
library(caret)
library(corrplot)
library(caret)
library(e1071)

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
tuneGrid = expand.grid(.mtry = 50)
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
  mtry =10,
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
  mtry =10,
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

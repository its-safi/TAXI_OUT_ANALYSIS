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

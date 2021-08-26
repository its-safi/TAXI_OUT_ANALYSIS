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



library(dplyr)
library(reshape2)
library(ggplot2)
library(class) ## a library with lots of classification tools
library(kknn) ## knn library

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

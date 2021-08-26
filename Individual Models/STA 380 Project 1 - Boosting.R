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
rm(list=ls())
setwd("~/club mahindra")
library(data.table)
library(stringi)
library(caret)
library(catboost)
traindata<-traindata[,c(-1:-4)]
traindata<-traindata[,c(-16)]
traindata$channel_code<-as.factor(traindata$channel_code)
traindata$main_product_code<-as.factor(traindata$main_product_code)
traindata$persontravellingid<-as.factor(traindata$persontravellingid)
traindata$resort_region_code<-as.factor(traindata$resort_region_code)
traindata$resort_type_code<-as.factor(traindata$resort_type_code)
traindata$room_type_booked_code<-as.factor(traindata$room_type_booked_code)
traindata$season_holidayed_code<-as.factor(traindata$season_holidayed_code)
traindata$state_code_residence<-as.factor(traindata$state_code_residence)
traindata$state_code_resort<-as.factor(traindata$state_code_resort)
traindata$member_age_buckets<-as.factor(traindata$member_age_buckets)
traindata$booking_type_code<-as.factor(traindata$booking_type_code)
traindata$cluster_code<-as.factor(traindata$cluster_code)
traindata$reservationstatusid_code<-as.factor(traindata$reservationstatusid_code)
traindata$resort_id<-as.factor(traindata$resort_id)




train_control <- trainControl(method="repeatedcv", number=5,repeats=5,verboseIter = TRUE)
set.seed(25)
p<-createDataPartition(y = traindata$amount_spent_per_room_night_scaled,times = 2,p = 0.8)
trainsample<-traindata[p[[1]],]
val<-data.frame(sapply(X = traindata,FUN = function(x) table(is.na(x)),simplify = TRUE))

memory.limit(100000)
subs<-trainsample[complete.cases(trainsample),]

modelcaretcat2<-train(y = trainsample[,19],x = trainsample[,-19],trControl=train_control, method=catboost.caret )


#rfmod <- train(y = subs[,19],x = subs[,-19],method="rf",ntree=4)






testdata<-read.csv("test.csv")
testdata$booking_date<-as.Date(x = testdata$booking_date,"%d/%m/%y")
testdata$checkin_date<-as.Date(x = testdata$checkin_date,"%d/%m/%y")
testdata$checkout_date<-as.Date(x = testdata$checkout_date,"%d/%m/%y")
testdata$daysinadvance<-as.numeric(testdata$checkin_date-testdata$booking_date)
testdata$duration<-as.numeric(testdata$checkout_date-testdata$checkin_date)
testdata<-testdata[,c(-1:-4)]
testdata<-testdata[,c(-16)]
testdata$channel_code<-as.factor(testdata$channel_code)
testdata$main_product_code<-as.factor(testdata$main_product_code)
testdata$persontravellingid<-as.factor(testdata$persontravellingid)
testdata$resort_region_code<-as.factor(testdata$resort_region_code)
testdata$resort_type_code<-as.factor(testdata$resort_type_code)
testdata$room_type_booked_code<-as.factor(testdata$room_type_booked_code)
testdata$season_holidayed_code<-as.factor(testdata$season_holidayed_code)
testdata$state_code_residence<-as.factor(testdata$state_code_residence)
testdata$state_code_resort<-as.factor(testdata$state_code_resort)
testdata$member_age_buckets<-as.factor(testdata$member_age_buckets)
testdata$booking_type_code<-as.factor(testdata$booking_type_code)
testdata$cluster_code<-as.factor(testdata$cluster_code)
testdata$reservationstatusid_code<-as.factor(testdata$reservationstatusid_code)
testdata$resort_id<-as.factor(testdata$resort_id)
testdata2<-read.csv("test.csv")

tab<-predict(modelcaretcat2,newdata = testdata,type = "raw")

submission<-cbind(reservation_id=as.character(testdata2[,1]),amount_spent_per_room_night_scaled=tab)
write.csv(x = submission,file = "submission2.csv",row.names = FALSE)




###############New feature predicting cost per day

traindata<-traindata[,-21]
modelcaretcat3<-train(y = traindata[,19],x = traindata[,-19],trControl=train_control, method=catboost.caret )


modelcaretcat3$finalModel

modelcaretcat2$finalModel



tab<-predict(modelcaretcat3,newdata = testdata,type = "raw")

submission<-cbind(reservation_id=as.character(testdata2[,1]),amount_spent_per_room_night_scaled=tab)
write.csv(x = submission,file = "submission3.csv",row.names = FALSE)

########################working with smaller samples
train_control <- trainControl(method="cv", number=5,verboseIter = TRUE)
p2<-createDataPartition(y = traindata$amount_spent_per_room_night_scaled,times = 4,p = 0.2)
samp1<-traindata[p2[[1]],]
modelcaretcat4<-train(y = samp1[,19],x = samp1[,-19],trControl=train_control, method=catboost.caret )

tab<-predict(modelcaretcat4,newdata = testdata,type = "raw")

submission<-cbind(reservation_id=as.character(testdata2[,1]),amount_spent_per_room_night_scaled=tab)
write.csv(x = submission,file = "submission4.csv",row.names = FALSE)


p3<-createDataPartition(y = traindata$amount_spent_per_room_night_scaled,times = 4,p = 0.4)
samp1<-traindata[p3[[1]],]
modelcaretcat5<-train(y = samp1[,19],x = samp1[,-19],trControl=train_control, method=catboost.caret )

tab<-predict(modelcaretcat5,newdata = testdata,type = "raw")

submission<-cbind(reservation_id=as.character(testdata2[,1]),amount_spent_per_room_night_scaled=tab)
write.csv(x = submission,file = "submission5.csv",row.names = FALSE)


p3<-createDataPartition(y = traindata$amount_spent_per_room_night_scaled,times = 1,p = 0.75)
samp1<-traindata[p3[[1]],]
modelcaretcat6<-train(y = samp1[,19],x = samp1[,-19],trControl=train_control, method=catboost.caret )

tab<-predict(modelcaretcat6,newdata = testdata,type = "raw")

submission<-cbind(reservation_id=as.character(testdata2[,1]),amount_spent_per_room_night_scaled=tab)
write.csv(x = submission,file = "submission6.csv",row.names = FALSE)


####
p3<-createDataPartition(y = traindata$amount_spent_per_room_night_scaled,times = 1,p = 0.85)
samp1<-traindata[p3[[1]],]
samp2<
modelcaretcat7<-train(y = samp1[,19],x = samp1[,-19],trControl=train_control, method=catboost.caret )

tab<-predict(modelcaretcat7,newdata = testdata,type = "raw")

submission<-cbind(reservation_id=as.character(testdata2[,1]),amount_spent_per_room_night_scaled=tab)
write.csv(x = submission,file = "submission7.csv",row.names = FALSE)

####
p3<-createDataPartition(y = traindata$amount_spent_per_room_night_scaled,times = 1,p = 0.85)
samp1<-traindata[p3[[1]],]
samp2<-traindata[-p3[[1]],]

trainpool<-catboost.load_pool(data = samp1,label = samp1$amount_spent_per_room_night_scaled,cat_features = c(1,2,5,6,7,8,10,11,12,14,15,16,17,18))
testpool<-catboost.load_pool(data = samp2,label = samp2$amount_spent_per_room_night_scaled,cat_features = c(1,2,5,6,7,8,10,11,12,14,15,16,17,18))

test2<-cbind(testdata[,1:18],amount_spent_per_room_night_scaled=1,testdata[,19:20])
valpool<-catboost.load_pool(data = test2,cat_features = c(1,2,5,6,7,8,10,11,12,14,15,16,17,18))

fit_params<-list(use_best_model=TRUE)
modelcaretcat8<-catboost.train(learn_pool = trainpool,test_pool = testpool,params = fit_params)
predcatbase<-catboost.predict(model = modelcaretcat8,pool =catboost.load_pool(samp2))

submission<-data.frame(reservation_id=as.character(testdata2[,1]),amount_spent_per_room_night_scaled=tab)
write.csv(x = submission,file = "submission8.csv",row.names = FALSE)



testpool<-catboost.load_pool(data = testdata,label = samp2$amount_spent_per_room_night_scaled,cat_features = c(1,2,5,6,7,8,10,11,12,14,15,16,17,18))



samp1<-samp1[,-19]
samp1$amount_spent_per_room_night_scaled<-traindata[p3[[1]],19]
trainpool<-catboost.load_pool(data = samp1,label = samp1$amount_spent_per_room_night_scaled,cat_features = c(1,2,5,6,7,8,10,11,12,14,15,16,17,18))
testpool<-catboost.load_pool(data = testdata,label = samp1$amount_spent_per_room_night_scaled,cat_features = c(1,2,5,6,7,8,10,11,12,14,15,16,17,18))

modelcaretcat9<-catboost.train(learn_pool = trainpool,params = fit_params)

predictcat<-catboost.predict(model = modelcaretcat9,pool = catboost.load_pool(testdata,label =NULL, cat_features = c(1,2,5,6,7,8,10,11,12,14,15,16,17,18)))






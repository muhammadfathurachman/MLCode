# Ru

library(lattice)
library(DAAG)
library(rpart.plot)
library(rpart)
library(mlbench)
library(rattle)
library(e1071)
library(RColorBrewer)
setwd("G:/COMPUTER SCIENCE LECTURE/MACHINE LEARNING UI/TUGAS 1")
car <- read.csv("car_e.csv",sep = ";", header = TRUE)
iris
#Create K-Folds Validation.
folds <- cut(seq(1,nrow(car)),breaks=10,labels=FALSE)

accTree <- c()
accBayes <- c()
accBN<-c()

tableTree <-c()
tableBayes <-c()
tableBN <-c()

tableError <- matrix(data = NA, nrow = 10, ncol = 6)
BayesNet <- make_Weka_classifier("weka/classifiers/bayes/BayesNet")

for(i in 1:10) 
{
  #PartitionData using K-Folds Validation
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- car[testIndex,]
  trainData <- car[-testIndex,]
  
  #Predition
  fit <- rpart(class~., data = trainData, method = "class",control=rpart.control(minsplit=4, minbucket=1, cp=0.001))
  predictionTree <- predict(fit, testData[,1:6], type = "class")
  tableTree<- table(predictionTree, testData$class)
  accTree[i] <- (tableTree[1,1]+tableTree[2,2]+tableTree[3,3]+tableTree[4,4])/nrow(testData)
  tableError[i,1] <- accTree[i]
  tableError[i,2] <- 1-accTree[i]
  
  #NaiveBayes
  modelBayes<- naiveBayes(class~., data = trainData)
  predictionBayes <- predict(modelBayes, testData[,1:6], type="class")
  tableBayes<- table(predictionBayes, testData$class)
  accBayes[i] <- (tableBayes[1,1]+tableBayes[2,2]+tableBayes[3,3]+tableBayes[4,4])/nrow(testData)
  tableError[i,3] <- accBayes[i]
  tableError[i,4] <- 1-accBayes[i]
  
  #Bayesian Network
  carBN <- BayesNet(class~., data = trainData)
  predictionBN <- predict(carBN, testData[,1:6])
  tableBN <- table(predictionBN, testData$class)
  accBN[i] <- (tableBN[1,1]+tableBN[2,2]+tableBN[3,3]+tableBN[4,4])/nrow(testData)
  tableError[i,5] <- accBN[i]
  tableError[i,6] <- 1-accBN[i]
}
colMeans(tableError)


### Induction Tree 

weather <- read.csv("weather.csv", header = TRUE, sep = ";")

p <- rpart(PlayTennis~., data = weather,method = "class", control=rpart.control(minsplit=2, minbucket=1, cp=0.001))

#Naive Bayes
netTab <- c()
for(i in 1:10) 
{
  #PartitionData using K-Folds Validation
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- car[testIndex,]
  trainData <- car[-testIndex,]
  model <- naiveBayes(class~., data = trainData)
  prediction <- predict(model, testData[,1:6], type = "class")
}

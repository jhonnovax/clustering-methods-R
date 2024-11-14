#------------------------
# Load dataset
#------------------------
library(caTools)
data <- read.csv("Special Topics/Individual Project/employee-attrition.csv")
head(data)

#------------------------
# Imbalance checking
#------------------------
library(ROSE)
table(data$Attrition)
data <- ovun.sample(Attrition~., data, p=0.5, seed=1, method="under")$data
table(data$Attrition)


#------------------------
# Desicion Tree
#------------------------
library(caTools)
library(caret)
library(rpart)
library(rpart.plot)
library(dplyr)

# Getting accuracy of the dataset
set.seed(42)
sample_split <- sample.split(Y = data$Attrition, SplitRatio = 0.8)
train_set <- subset(x = data, sample_split == TRUE)
test_set <- subset(x = data, sample_split == FALSE)
nrow(train_set)
nrow(test_set)
model <- rpart(Attrition ~ . , data = train_set, method = "class")
rpart.plot(model)
importances <- varImp(model)
importances %>%
  arrange(desc(Overall))
preds <- predict(model, test_set, type = "class")
confusionMatrix(factor(test_set$Attrition), factor(preds));


#------------------------
# PCA
#------------------------
library(psych)

numfunc = function (data) {
  data1=na.omit (data)
  m1=dim(data1)
  for(i in 1:m1[2]) {
    if(class(data1[, i]) == "numeric"| class(data1[, i]) == "integer") {data1[, i] =data1[, i]}
    else{data1[, i]=rep(NA, m1[1]) }}
  data2=na.omit(t(data1))
  data1=t(data2)
  list("data1"=data1)
}
myData3 <- numfunc(data)$data1

fa.parallel(myData3, fa="pc", n.iter=100, show.legend = TRUE)
abline(h=1,col=3)
pc31=principal (myData3, nfactor=2)
pc31
Q31=round(pc31$weights,2)
Q31
pc32=principal (myData3, nfactor=3)
pc32
Q32=round(pc32$weights,2)
Q32

head(myData3)
C1 <- 0.03*myData3[, 1]+0.42*myData3[, 2]+0.48*myData3[, 3]+0.01*myData3[,4]+0.02*myData3[,5]-0.01*myData3[,6]-0.01*myData3[,7]+0.39*myData3[,8]
C2 <- -0.08*myData3[, 1]-0.06*myData3[, 2]-0.01*myData3[, 3]+0.33*myData3[,4]-0.03*myData3[,5]-0.66*myData3[,6]+0.61*myData3[,7]+0.08*myData3[,8]
C3 <- 0.70*myData3[, 1]+0.02*myData3[, 2]-0.01*myData3[, 3]-0.25*myData3[,4]-0.54*myData3[,5]+0.11*myData3[,6]+0.32*myData3[,7]-0.01*myData3[,8]
C4 <- data$Attrition

myData4 <- data.frame(C1, C2, C3, C4)
head (myData4)

sample_split <- sample.split(Y = myData4$C4, SplitRatio = 0.8)
train_set1 <- subset(x = myData4, sample_split == TRUE)
test_set1 <- subset(x = myData4, sample_split == FALSE)
model1 <- rpart (C4 ~ . , data = train_set1, method = "class")
model1
rpart.plot(model1)
importances <- varImp(model1)
importances %>%
  arrange(desc(Overall))
preds1 <- predict(model1, newdata = test_set1, type = "class")
confusionMatrix(factor(test_set1$C4), preds1);

#------------------------
# Enconding categorical values
#------------------------
test_set$Gender <- as.numeric(as.factor(test_set$Gender))
test_set$Job.Role <- as.numeric(as.factor(test_set$Job.Role))
test_set$Work.Life.Balance <- as.numeric(as.factor(test_set$Work.Life.Balance))
test_set$Job.Satisfaction <- as.numeric(as.factor(test_set$Job.Satisfaction))
test_set$Performance.Rating <- as.numeric(as.factor(test_set$Performance.Rating))
test_set$Overtime <- as.numeric(as.factor(test_set$Overtime))
test_set$Education.Level <- as.numeric(as.factor(test_set$Education.Level))
test_set$Marital.Status <- as.numeric(as.factor(test_set$Marital.Status))
test_set$Job.Level <- as.numeric(as.factor(test_set$Job.Level))
test_set$Company.Size <- as.numeric(as.factor(test_set$Company.Size))
test_set$Remote.Work <- as.numeric(as.factor(test_set$Remote.Work))
test_set$Leadership.Opportunities <- as.numeric(as.factor(test_set$Leadership.Opportunities))
test_set$Innovation.Opportunities <- as.numeric(as.factor(test_set$Innovation.Opportunities))
test_set$Company.Reputation <- as.numeric(as.factor(test_set$Company.Reputation))
test_set$Employee.Recognition <- as.numeric(as.factor(test_set$Employee.Recognition))
head(test_set)


#------------------------
# Scaling/Normalizing
#------------------------
test_set[, -24] <- scale(test_set[, -24])
head(test_set)


#------------------------
# Correlations
#------------------------
library(corrplot)
cor_matrix <- cor(test_set[, -24])
corrplot(cor_matrix, method = "color")


#------------------------
# Removing target
#------------------------
target_values <- test_set$Attrition
test_set$Attrition <- NULL
head(test_set)

#------------------------
# KMeans
#------------------------
kmeans1 <- kmeans(test_set, 3)
kmeans1

# Calculate the best number of clusters testing in a range of (2:15)
wss <- (nrow(test_set)-1) * sum(apply (test_set, 2, var))
for (i in 2:15) wss [i] <- sum(kmeans (test_set, centers=i) $withiness)
plot(1:15, wss, type="b", col=10)

# Plot Clusters and Identify Centroids
plot(test_set[c("Monthly.Income", "Distance.from.Home")], col=kmeans1$cluster)
points(kmeans1$centers[, c("Monthly.Income", "Distance.from.Home") ], col=1:3, pch=8, cex=2)


#------------------------
# NBClust
#------------------------
library(NbClust)
nc = NbClust(test_set, min.nc=2, max.nc=10, method="kmeans")
Bar <- table(nc$Best.n[1,])
Bar


#------------------------
# PAMK
#------------------------
library(fpc)
pamk1 <- pamk(test_set, 2:10)
pamk1

layout(matrix(c(1, 2),1, 2))
plot(pamk1$pamobject)
layout(matrix(1))


#------------------------
# DBSCAN
#------------------------
# Function to calculate eps against any data set
eps <- function(mydata) {
  dist <- dbscan::kNNdist(mydata, ncol(mydata)+1)
  dist <- dist [order (dist) ]
  dist <- dist/ max(dist)
  ddist <- diff(dist)/(1/length(dist))
  EPS <- dist[length(ddist)-length(ddist [ddist>1])]
  print (EPS)
}
eps_value <- eps(test_set) 
eps_value
eps_value <- 4.3

# To see the best value for eps
library(dbscan)
kNNdistplot(test_set, k=4+1)
abline(h=4.5, col='green')
abline(h=eps_value, col=2, lty=5)

# Chart of clusters
db1 <- dbscan(test_set, eps=eps_value, minPts=5)
db1
hullplot(test_set, db1)
plot(test_set, col=db1$cluster)

# Chart of clusters with no noise/outliers
noise=which(db1$cluster == 0)
noise
data2 <- test_set[-noise,]
eps_noise <- eps(data2)
eps_noise
eps_noise <- 4.5
db2 <- dbscan(data2, eps=eps_noise, minPts=ncol(data2)+1)
db2
hullplot(data2, db2)
plot(data2, col=db1$cluster)


#------------------------
# Hierarchical
#------------------------
library(tidyverse)
library(factoextra)

d <- dist(test_set, method="euclidean")
hc1 <- hclust(d, method="complete")
hc1
plot(hc1, cex=0.6)
rect.hclust (hc1, k=4, border=1:4)
fviz_cluster(list(data=test_set, cluster=cutree(hc1, k=4)), labelsize = 7)

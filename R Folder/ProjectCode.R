rm(list=ls())
library(caret)
library(MASS)
library(e1071)
library(class)
library(pROC)
library(tree)


data = read.csv("adult_U.csv", header = TRUE)
dim(data)
head(data)

is.na(data)
colSums(is.na(data)) #About 6465 NA values produced from this function
rowSums(is.na(data))
data1 = na.omit(data)

data2 = data1
data2["educational.num"]=NULL
data2["fnlwgt"]=NULL
is.na(data2) = data2=='?'
is.na(data2) = data2==' ?'
data2 = na.omit(data2)


summary(data1)

#Histograms of Quantitative variables
hist(data1$age)
hist(data1$fnlwgt)
hist(data1$capital.gain)
hist(data1$capital.loss)
hist(data1$hours.per.week)
hist(data1$educational.num)
hist(data1$income)

#Frequencies of Qualititative variables
table(data1$workclass)
table(data1$native.country)
table(data1$education)
table(data1$marital.status)
table(data1$occupation)
table(data1$relationship)
table(data1$race)
table(data1$gender)
table(data1$income)


#Make variables factors
str(data1)
workclass=as.factor(data1$workclass)
education=as.factor(data1$education)
marital.status=as.factor(data1$marital.status)
occupation=as.factor(data1$occupation)
relationship=as.factor(data1$relationship)
race=as.factor(data1$race)
gender=as.factor(data1$gender)
native.country=as.factor(data1$native.country)
income=as.factor(data1$income)
##################################################
###Restructuring the dataset
data2$native.country[data1$native.country=="Cambodia"] = "Asia"
data2$native.country[data1$native.country=="Canada"] = "North-America"    
data2$native.country[data1$native.country=="China"] = "Asia"       
data2$native.country[data1$native.country=="Columbia"] = "South-America"    
data2$native.country[data1$native.country=="Cuba"] = "South-America"        
data2$native.country[data1$native.country=="Dominican-Republic"] = "North-America"
data2$native.country[data1$native.country=="Ecuador"] = "South-America"     
data2$native.country[data1$native.country=="El-Salvador"] = "South-America" 
data2$native.country[data1$native.country=="England"] = "Europe"
data2$native.country[data1$native.country=="France"] = "Europe"
data2$native.country[data1$native.country=="Germany"] = "Europe"
data2$native.country[data1$native.country=="Greece"] = "Europe"
data2$native.country[data1$native.country=="Guatemala"] = "North-America"
data2$native.country[data1$native.country=="Haiti"] = "North-America"
data2$native.country[data1$native.country=="Holand-Netherlands"] = "Europe"
data2$native.country[data1$native.country=="Honduras"] = "North-America"
data2$native.country[data1$native.country=="Hong"] = "Asia"
data2$native.country[data1$native.country=="Hungary"] = "Europe"
data2$native.country[data1$native.country=="India"] = "Asia"
data2$native.country[data1$native.country=="Iran"] = "Asia"
data2$native.country[data1$native.country=="Ireland"] = "Europe"
data2$native.country[data1$native.country=="Italy"] = "Europe"
data2$native.country[data1$native.country=="Jamaica"] = "North-America"
data2$native.country[data1$native.country=="Japan"] = "Asia"
data2$native.country[data1$native.country=="Laos"] = "Asia"
data2$native.country[data1$native.country=="Mexico"] = "North-America"
data2$native.country[data1$native.country=="Nicaragua"] = "North-America"
data2$native.country[data1$native.country=="Outlying-US(Guam-USVI-etc)"] = "North-America"
data2$native.country[data1$native.country=="Peru"] = "South-America"
data2$native.country[data1$native.country=="Philippines"] = "Asia"
data2$native.country[data1$native.country=="Poland"] = "Europe"
data2$native.country[data1$native.country=="Portugal"] = "Europe"
data2$native.country[data1$native.country=="Puerto-Rico"] = "North-America"
data2$native.country[data1$native.country=="Scotland"] = "Europe"
data2$native.country[data1$native.country=="South"] = "Europe"
data2$native.country[data1$native.country=="Taiwan"] = "Asia"
data2$native.country[data1$native.country=="Thailand"] = "Asia"
data2$native.country[data1$native.country=="Trinadad&Tobago"] = "North-America"
data2$native.country[data1$native.country=="United-States"] = "United-States"
data2$native.country[data1$native.country=="Vietnam"] = "Asia"
data2$native.country[data1$native.country=="Yugoslavia"] = "Europe"
##################################################
#set.seed(1210)
#n= dim(data1)[1]
#train.index = sample(n, 0.7*n)
#train = data1[train.index, ]
#test = data1[-train.index, ]
#test.y = test$income

#n= dim(data2)[1]
#train.index = sample(n, 0.7*n)
#train = data2[train.index, ]
#test = data2[-train.index, ]
#test.y = test$income

#str(data1)
#sapply(data1,class)
#glm= glm(income~., data = train, family = binomial)

#str(data2)
#sapply(data2,class)
#glm2= glm(income~. -relationship, data = train, family = binomial)
####################################################

set.seed(1210)
dim(data1)
sample = sample(1:45222, replace = FALSE)
train.index = sample[1:(0.7*45222)]
test.index = sample[(0.7*45222+1):45222]
train = data[train.index, ]
test= data[test.index, ]
glm.fits = glm(income ~ age + occupation + education + hours.per.week + gender, data = train, family = binomial)
summary(glm.fits)
glm.fits2 = glm(income ~ age + occupation + education + hours.per.week + gender, data = test, family = binomial)
summary(glm.fits2)
##########################################################

test.y = test$income
glm.class = rep("0", 0.3*n)
glm.probs = predict(glm.fits, test, type='response')
glm.class[glm.probs > .5] = "1"
glm.sum=confusionMatrix(data=as.factor(glm.class), reference= as.factor(test.y), positive="1")
glm.sum

#######################################
#LDA

lda.fit = lda(income ~ age + occupation + education + hours.per.week + gender, data =train)
lda.pred = predict(lda.fit, test)
lda.class = lda.pred$class
lda.sum=confusionMatrix(data=as.factor(lda.class), reference= as.factor(test.y), positive="1")
lda.sum

#######################################
#QDA

qda.fit = qda(income ~ age + occupation + education + hours.per.week + gender, data = train)
qda.class = predict(qda.fit, test)$class
qda.sum=confusionMatrix(data=as.factor(qda.class), reference= as.factor(test.y), positive="1")
qda.sum

#######################################
#Naive Bayes

nb.fit = naiveBayes(income ~ age + occupation + education + hours.per.week + gender, data = train)
nb.class = predict(nb.fit, test)
nb.sum=confusionMatrix(data=as.factor(nb.class), reference= as.factor(test.y), positive="1")
nb.sum

#######################################

glm.roc=roc(response= test.y, predictor= glm.probs, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(glm.roc)

lda.roc = roc(response= test.y, predictor=lda.pred$posterior[,1], plot = TRUE, print.auc = TRUE)  #ROC curve
auc(lda.roc)

qda.pred = predict(qda.fit, test)
qda.roc = roc(response= test.y, predictor=qda.pred$posterior[,1], plot = TRUE, print.auc = TRUE)
auc(qda.roc)



ggroc(list(glm=glm.roc, lda=lda.roc, qda=qda.roc))

#######################################################
###Classification Tree
High = factor(ifelse(data1$income<1, "No", "Yes"))
Treedata = data.frame(data1, data1$income)
Treedata = data.frame(data1, High)


n = nrow(Treedata)
set.seed(1210)
train.index= sample(1:n, 0.7*n)
train = Treedata[train.index,]
test = Treedata[-train.index,]
y.test = data1$income[-train.index]
y.test = High[-train.index]

tree= tree(income~ age + occupation + education + hours.per.week + gender, data = train)
plot(tree)
text(tree, pretty = 0)
summary(tree)

data1$income = as.character(data1$income)
pred = predict(tree, newdata = test)
pred
confusionMatrix(data=pred, reference = y.test)


income = as.character(income)
y.test = test$income
y.test= as.factor(y.test)


tree= tree(as.factor(income)~., train)
summary(tree)
plot(tree)
text(tree, pretty = 0)

yhat.tree = predict(tree, test, type='class')
confusionMatrix(yhat.tree, y.test)
yprob.tree = predict(tree, test)

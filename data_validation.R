library(xgboost)
library(Matrix)
setwd("C:/Users/Xiaoyu Sun/Desktop/kaggle")
# ---------------------------------------------------
# Load
orig.train <- read.csv("dtrain.csv", stringsAsFactors = F)
orig.test <- read.csv("dvalid.csv", stringsAsFactors = F)


# ---------------------------------------------------
# Features
feature.names <- names(orig.train)
#feature.names <- feature.names[-grep('^ID$', feature.names)]
feature.names <- feature.names[-grep('^TARGET$', feature.names)]
feature.formula <- formula(paste('TARGET ~ ', paste(feature.names, collapse = ' + '), sep = ''))

# ---------------------------------------------------
#Build a xgb.DMatrix object

data <- sparse.model.matrix(feature.formula, data = orig.train)
sparseMatrixColNamesTrain <- colnames(data)
DMMatrixTrain <- xgb.DMatrix(data, label = orig.train[, 'TARGET'])
rm(data)
dvalid <- xgb.DMatrix(sparse.model.matrix(feature.formula, data = orig.test),
                      label = orig.test[, 'TARGET'])



# ---------------------------------------------------
watchlist <- list(valid = dvalid, train = DMMatrixTrain)
bst.cv = xgb.cv(params=params, data=DMMatrixTrain,  nfold = 5,metrics="auc", nrounds=200)

# ---------------------------------------------------
auc<-rep(0,20)
for(i in 1:20){
params <- list(booster = "gbtree", objective = "binary:logistic",
               max_depth = 8, eta = 0.05,
               colsample_bytree = 0.65, subsample = 0.95)
model <- xgb.train(params = params, data = DMMatrixTrain,
                   nrounds = 300, early.stop.round = 50,
                   eval_metric = 'auc', maximize = T,
                   watchlist = watchlist, print.every.n = 100)

auc[i]<- model$bestScore
}
sum(auc)/20

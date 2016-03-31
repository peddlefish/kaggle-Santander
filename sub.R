library(xgboost)
library(Matrix)
setwd("C:/Users/Xiaoyu Sun/Desktop/kaggle")
# ---------------------------------------------------
# Load
orig.train <- read.csv("train_filtered.csv", stringsAsFactors = F)
orig.test <- read.csv("test_filtered.csv", stringsAsFactors = F)
#sample.submission <- read.csv("../input/sample_submission.csv", stringsAsFactors = F)
test  <- read.csv("test.csv")
test.id <- test$ID
orig.test$TARGET<--1

orig.train.row <- nrow(orig.train)
merged <- rbind(orig.train, orig.test)
# ---------------------------------------------------
merged <- read.csv("merged.csv", stringsAsFactors = F)
# Split
merged$id<-NULL
train <- merged[merged$TARGET != -1, ]
test <- merged[merged$TARGET == -1, ]

# ---------------------------------------------------
# Features
feature.names <- names(train)
#feature.names <- feature.names[-grep('^ID$', feature.names)]
feature.names <- feature.names[-grep('^TARGET$', feature.names)]
feature.formula <- formula(paste('TARGET ~ ', paste(feature.names, collapse = ' + '), sep = ''))



# ---------------------------------------------------
# Matrix


data <- sparse.model.matrix(feature.formula, data = train)
sparseMatrixColNamesTrain <- colnames(data)
dtrain <- xgb.DMatrix(data, label = orig.train[, 'TARGET'])
rm(data)

dtest <- sparse.model.matrix(feature.formula, data = test)

watchlist <- list( train = dtrain)

# ---------------------------------------------------
# XGBOOST
params <- list(booster = "gbtree", objective = "binary:logistic",
               max_depth = 8, eta = 2/500,
               colsample_bytree = 0.85, subsample = 0.95)
model <- xgb.train(params = params, data = dtrain,
                   nrounds = 500, early.stop.round = 50,
                   eval_metric = 'auc', maximize = T,
                   watchlist = watchlist, print.every.n = 10)

pred <- predict(model, dtest)
# ---------------------------------------------------
# XGBOOST
params <- list(booster = "gbtree", objective = "binary:logistic",
               max_depth = 8, eta = 0.05,
               colsample_bytree = 0.65, subsample = 0.95)
model <- xgb.train(params = params, data = dtrain,
                   nrounds = 500, early.stop.round = 50,
                   eval_metric = 'auc', maximize = T,
                   watchlist = watchlist, print.every.n = 10)

pred <- predict(model, dtest)

# ---------------------------------------------------
# SAVE
submission <- data.frame(ID = test.id, TARGET = pred)
write.csv(submission, 'xgboost_avg.csv', row.names=FALSE, quote = FALSE)

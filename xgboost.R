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


# ---------------------------------------------------
# Convert
feature.train.names <- names(orig.train)[-1]



merged$flag<-NULL
# ---------------------------------------------------
# Features
feature.names <- names(orig.train)
#feature.names <- feature.names[-grep('^ID$', feature.names)]
feature.names <- feature.names[-grep('^TARGET$', feature.names)]
feature.formula <- formula(paste('TARGET ~ ', paste(feature.names, collapse = ' + '), sep = ''))

# ---------------------------------------------------
# Hyperparameter Tuning

#Build a xgb.DMatrix object
# ---------------------------------------------------
# Matrix
library(xgboost)
library(Matrix)
options(scipen=999)


# ---------------------------------------------------
# Load
orig.train <- read.csv("../input/train.csv", stringsAsFactors = F)
orig.test <- read.csv("../input/test.csv", stringsAsFactors = F)
sample.submission <- read.csv("../input/sample_submission.csv", stringsAsFactors = F)

# ---------------------------------------------------
# Merge
orig.test$TARGET <- -1
merged <- rbind(orig.train, orig.test)

# ---------------------------------------------------
# Convert
feature.train.names <- names(orig.train)[-1]
for (f in feature.train.names) {
  if (class(merged[[f]]) == "numeric") {
    merged[[f]] <- merged[[f]] / max(merged[[f]])
  } else if (class(merged[[f]]) == "integer") {
    u <- unique(merged[[f]])
    if (length(u) == 1) {
      merged[[f]] <- NULL
    } else if (length(u) < 200) {
      merged[[f]] <- factor(merged[[f]])
    }
  }
}

# ---------------------------------------------------
# Split
train <- merged[merged$TARGET != -1, ]
test <- merged[merged$TARGET == -1, ]


# ---------------------------------------------------
# Features
feature.names <- names(merged)
feature.names <- feature.names[-grep('^TARGET$', feature.names)]
feature.formula <- formula(paste('TARGET ~', paste(feature.names, collapse = ' + '), sep = ''))
# ---------------------------------------------------
# Matrix
indexes <- sample(seq_len(nrow(merged)), floor(nrow(merged)*0.85))

data <- sparse.model.matrix(feature.formula, data = merged[indexes, ])
sparseMatrixColNamesTrain <- colnames(data)
dtrain <- xgb.DMatrix(data, label = merged[indexes, 'TARGET'])
rm(data)
dvalid <- xgb.DMatrix(sparse.model.matrix(feature.formula, data = merged[-indexes, ]),
                      label = merged[-indexes, 'TARGET'])


watchlist <- list(valid = dvalid, train = dtrain)
  # ---------------------------------------------------
  # XGBOOST
 # params <- list(booster = "gbtree", objective = "binary:logistic",
   #              max_depth = 8, eta = 0.02,
  #               colsample_bytree = 0.65, subsample = 1.0)
  model <- xgb.train(params = params, data = dtrain,verbose = 0,
                    nrounds = 500, early.stop.round = 50,
                    eval_metric = 'auc', maximize = T,
                    watchlist = watchlist, print.every.n = 100)
  
 # pred <- predict(model, dtest)
# Parameters Range Setting
  searchGridSubCol <- expand.grid(subsample = c(0.5, 0.75,1.0), 
                                  colsample_bytree = c(0.6, 0.8,1.0),
                                  MaxDepth = c(6,7,8,9,10))
  ntrees <- 500

aucErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentMaxDepth <- parameterList[["MaxDepth"]]
  print(currentMaxDepth)
  xgboostModelCV <- xgb.cv(data =  DMMatrixTrain, nrounds = ntrees, nfold = 5, print.every.n = 200, 
                          verbose = TRUE, "eval_metric" = "auc",
                          #"objective" = 'multi:softmax',
                           "max.depth" = currentMaxDepth, "eta" = 2/ntrees,                               
                           "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate)
  
  xvalidationScores <- as.data.frame(xgboostModelCV)
  #Save rmse of the last iteration
  auc <- tail(xvalidationScores$test.auc.mean, 1)
  
  return(c(auc, currentSubsampleRate, currentColsampleRate,currentMaxDepth))
  
})
write.csv(aucErrorsHyperparameters, 'aucErrorsHyperparameters.csv', row.names=FALSE, quote = FALSE)
# ---------------------------------------------------

# SAVE
submission <- data.frame(ID = test.id, TARGET = pred)
write.csv(submission, 'xgboost_1_simple.csv', row.names=FALSE, quote = FALSE)

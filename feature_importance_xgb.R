library(Ckmeans.1d.dp)
setwd("C:/Users/Xiaoyu Sun/Desktop/kaggle")
# ---------------------------------------------------
# Load
orig.train <- read.csv("train_filtered.csv", stringsAsFactors = F)
orig.test <- read.csv("test_filtered.csv", stringsAsFactors = F)

#train.tar <- orig.train$TARGET
#orig.train$TARGET <- NULL 

# ---------------------------------------------------
# Features
feature.names <- names(orig.train)
feature.names <- feature.names[-grep('^TARGET$', feature.names)]
feature.formula <- formula(paste('TARGET ~', paste(feature.names, collapse = ' + '), sep = ''))
# ---------------------------------------------------
# Matrix
indexes <- sample(seq_len(nrow(orig.train)), floor(nrow(orig.train)*0.85))

data <- sparse.model.matrix(feature.formula, data = orig.train[indexes, ])
sparseMatrixColNamesTrain <- colnames(data)
dtrain <- xgb.DMatrix(data, label = orig.train[indexes, 'TARGET'])
rm(data)
dvalid <- xgb.DMatrix(sparse.model.matrix(feature.formula, data = orig.train[-indexes, ]),
                      label = orig.train[-indexes, 'TARGET'])


watchlist <- list(valid = dvalid, train = dtrain)
# ---------------------------------------------------
# XGBOOST
params <- list(booster = "gbtree", objective = "binary:logistic",
            max_depth = 8, eta = 0.02,
              colsample_bytree = 0.65, subsample = 1.0)

bst <- xgb.train(params = params, data = dtrain,verbose = 0,
                   nrounds = 500, early.stop.round = 50,
                   eval_metric = 'auc', maximize = T,
                   watchlist = watchlist, print.every.n = 100)

#train$data@Dimnames[[2]] represents the column names of the sparse matrix.
bst = xgboost(param=param, data = dtrain, label = y, nrounds=nround)
# Compute feature importance matrix
importance_matrix <- xgb.importance(feature.names, model = bst)
importance_matrix_names <- data.frame(importance_matrix[1:50,])$Feature
# Nice graph
xgb.plot.importance(importance_matrix[1:10,])
# ---------------------------------------------------
# ---------------------------------------------------
# write top 50 features to csv

# Features
feature.names <- importance_matrix_names
write.csv(orig.train[c("TARGET",feature.names)],"train_filtered_2.csv",row.names = F)
#feature.names <- feature.names[-grep('^TARGET$', feature.names)]












feature.formula <- formula(paste('TARGET ~', paste(feature.names, collapse = ' + '), sep = ''))
# ---------------------------------------------------
# Matrix
indexes <- sample(seq_len(nrow(orig.train)), floor(nrow(orig.train)*0.85))

data <- sparse.model.matrix(feature.formula, data = orig.train[indexes, ])
sparseMatrixColNamesTrain <- colnames(data)
dtrain <- xgb.DMatrix(data, label = orig.train[indexes, 'TARGET'])
rm(data)
dvalid <- xgb.DMatrix(sparse.model.matrix(feature.formula, data = orig.train[-indexes, ]),
                      label = orig.train[-indexes, 'TARGET'])


watchlist <- list(valid = dvalid, train = dtrain)

# XGBOOST
params <- list(booster = "gbtree", objective = "binary:logistic",
               max_depth = 8, eta = 0.02,
               colsample_bytree = 0.65, subsample = 1.0)

bst <- xgb.train(params = params, data = dtrain,verbose = 0,
                 nrounds = 500, early.stop.round = 50,
                 eval_metric = 'auc', maximize = T,
                 watchlist = watchlist, print.every.n = 100)

library(xgboost)
library(Matrix)
library(foreach)
library(doParallel)
setwd("C:/Users/Xiaoyu Sun/Desktop/kaggle")
orig.train <- read.csv("train_filtered_1.csv", stringsAsFactors = F)

# ----------------------------------------------------------------------
#################################################################
######## 1. Function Define #####################################
#################################################################

#1.1 AUC function
my_auc <- function(true_Y, probs) {
  N <- length(true_Y)
  if (length(probs) != N)
    return (NULL) # error
  if (is.factor(true_Y))
    true_Y <- as.numeric(as.character(true_Y))
  roc_y <- true_Y[order(probs, decreasing = FALSE)]
  stack_x = cumsum(roc_y == 1) / sum(roc_y == 1)
  stack_y = cumsum(roc_y == 0) / sum(roc_y == 0)
  auc = sum((stack_x[2:N] - stack_x[1:(N - 1)]) * stack_y[2:N])
  return(auc)
}

#1.2 Stratified Sampling
strsamp <- function(train_1, train_0){
  indexes_1<- sample(seq_len(nrow(train_1)), floor(nrow(train_1)*0.8))
  dtrain_1 <-train_1[indexes_1, ]
  dvalid_1 <-train_1[-indexes_1, ]
  
  indexes_0<- sample(seq_len(nrow(train_0)), floor(nrow(train_0)*0.8))
  dtrain_0 <-train_0[indexes_0, ]
  dvalid_0 <-train_0[-indexes_0, ]
  
  dtrain<-rbind(dtrain_1,dtrain_0)
  dvalid<-rbind(dvalid_1,dvalid_0)
  return(list(dtrain=dtrain, dvalid=dvalid))
}

#1.3 Shuffle train and valid data set
shuffle <- function(dtrain, dvalid){
  set.seed(2*3*5*7*11*13*17)
  shuffle <- sample(nrow(dtrain), nrow(dtrain))
  dtrain <- dtrain[shuffle,] 
  shuffle <- sample(nrow(dvalid), nrow(dvalid))
  dvalid <- dvalid[shuffle,] 
  return(list(dtrain=dtrain,dvalid=dvalid))
}

#1.4 Generate train and valid matrix for xgboost training
gen_matrix <- function(shuffle_result,feature.formula){
  data <- sparse.model.matrix(feature.formula, data =  shuffle_result$dtrain)
  sparseMatrixColNamesTrain <- colnames(data)
  DMMatrixTrain <- xgb.DMatrix(data, label =  shuffle_result$dtrain[, 'TARGET'])
  rm(data)
  dvalid <- xgb.DMatrix(sparse.model.matrix(feature.formula, data =  shuffle_result$dvalid),
                        label =  shuffle_result$dvalid[, 'TARGET'])
  return(list(dtrain=DMMatrixTrain,dvalid=dvalid))
}

#1.5 straitified cross validation
str_cv <- function(train_1,train_0, nrounds = 500, max.depth,subsample, colsample_bytree){
  auc<-rep(0,5)
  for(i in 1:5){
    # Matrix
    
    strsamp_result <- strsamp(train_1,train_0)
    shuffle_result<-shuffle(strsamp_result$dtrain,strsamp_result$dvalid)
    dvalid.target <-  shuffle_result$dvalid$TARGET
    xgboost_result <- gen_matrix(shuffle_result,feature.formula)
    
    #xgboost
    watchlist <- list(train = xgboost_result$dtrain,valid = xgboost_result$dvalid)
    
    params <- list(booster = "gbtree", "max.depth" = max.depth,  "eta" = 2/nrounds,                               
                   "subsample" = subsample, "colsample_bytree" = colsample_bytree)
    set.seed(i*17+142)
    model <- xgb.train(params = params, data = xgboost_result$dtrain,
                       early.stop.round = 50,
                       nrounds = nrounds,
                       objective = "binary:logistic",
                       verbose = TRUE,
                       eval_metric = 'auc', maximize = T,
                       watchlist = watchlist, print.every.n = 200)
    bst <- model$bestInd
    pred_val <- predict(model, xgboost_result$dvalid, ntreelimit = bst)
    auc[i] <- my_auc(dvalid.target, pred_val)
    cat("\n*** AUC ~ ", auc[i],"\n")
  }
  return(sum(auc)/5)
}
# ----------------------------------------------------------------------
####################################################################
###########  2. Data Engineering  ##################################
####################################################################

#2.1 If No Engineering
train<-orig.train

#2.2 If # Categorical feature != 0 is 1, delete it.
cat("If # Categorical feature != 0 is 1, delete it.\n")
column_index<-NULL
a<- ncol(orig.train)-1
for(i in 1:a){
  if(length(which(as.vector(orig.train[,i])!=0))<2){
    column_index[length(column_index)+1]<- i
    print(i)
  }
}
train<-orig.train[-column_index ]

#2.3 Removing highly correlated features
cat("removing highly correlated features\n")
train<-orig.train
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if ((cor(train[[f1]] , train[[f2]])) > 0.99) {
      cat(f1, "and", f2, "are highly correlated \n")
      toRemove <- c(toRemove, f2)
    }
  }
}
train.names  <- setdiff(names(train), toRemove)
train        <- train[,train.names]
toRemove
cat("-------------------------\n")
removed <- ncol0-ncol(train)
cat("\n ",removed," features have been removed\n")



# ----------------------------------------------------------------------
#################################################################
######## 3. Xgboost Preset ######################################
#################################################################

#split train by target
train_1 <- train[train$TARGET==1,]
train_0 <- train[train$TARGET==0,]

# Parameter Preset
feature.names <- names(train)
feature.names <- feature.names[-grep('^TARGET$', feature.names)]
feature.formula <- formula(paste('TARGET ~ ', paste(feature.names, collapse = ' + '), sep = ''))

#setup parallel backend
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl, cores = detectCores() - 1)

ls<-foreach(i = 7:9 , .packages = c("xgboost","Matrix")) %dopar% {

# Parameters Range Setting
searchGridSubCol <- expand.grid(subsample = 0.5,#seq(0.5, 1, 0.05), 
                                colsample_bytree = 0.5,#seq(0.5, 1,0.05),
                                MaxDepth = c(i))
ntrees <- 500
# ----------------------------------------------------------------------
#################################################################
######## 4. Parameter Tunning ###################################
#################################################################

aucErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentMaxDepth <- parameterList[["MaxDepth"]]
  print(currentMaxDepth)
  auc_cv<-str_cv(train_1,train_0, nrounds = ntrees, max.depth=currentMaxDepth,subsample=currentSubsampleRate, colsample_bytree=currentColsampleRate)
  return(c(auc_cv, currentSubsampleRate, currentColsampleRate,currentMaxDepth))
})
write.csv(aucErrorsHyperparameters, paste0('auc_Hyperparameters_',i,'.csv'), row.names=FALSE, quote = FALSE)
}
stopCluster(cl)



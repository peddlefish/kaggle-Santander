library(xgboost)
library(Matrix)
setwd("C:/Users/Xiaoyu Sun/Desktop/kaggle")
train <- read.csv("train.csv")
test  <- read.csv("test.csv")

##### Removing IDs
train$ID <- NULL
test.id <- test$ID
test$ID <- NULL

##### Extracting TARGET
train.y <- train$TARGET
train$TARGET <- NULL

##### 0 count per line
#count0 <- function(x) {
#    return( sum(x == 0) )
#}
#train$n0 <- apply(train, 1, FUN=count0)
#test$n0 <- apply(test, 1, FUN=count0)

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}


##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(train), toRemove)

train <- train[, feature.names]
test <- test[, feature.names]


train$TARGET <- train.y


#train <- sparse.model.matrix(TARGET ~ ., data = train)

write.csv(train, "train_filtered.csv", row.names = F)
write.csv(test, "test_filtered.csv", row.names = F)

# Matrix
#train<-orig.train
#test<-orig.test
# Features
train_1 <- train[train$TARGET==1,]
indexes_1<- sample(seq_len(nrow(train_1)), floor(nrow(train_1)*0.85))
dtrain_1 <-train_1[indexes_1, ]
dvalid_1 <-train_1[-indexes_1, ]

train_0 <- train[train$TARGET==0,]
indexes_0<- sample(seq_len(nrow(train_0)), floor(nrow(train_0)*0.85))
dtrain_0 <-train_0[indexes_0, ]
dvalid_0 <-train_0[-indexes_0, ]

dtrain<-rbind(dtrain_1,dtrain_0)
dvalid<-rbind(dvalid_1,dvalid_0)

write.csv(dtrain, "dtrain.csv", row.names = F)
write.csv(dvalid, "dvalid.csv", row.names = F)

# stratified cross validation

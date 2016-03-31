library(xgboost)
library(Matrix)
setwd("C:/Users/Xiaoyu Sun/Desktop/kaggle")
# ---------------------------------------------------
# Load
train <- read.csv("train_filtered.csv", stringsAsFactors = F)
ncol0<-ncol(train)
## 3 #################################################
### Removing highly correlated features
cat("removing highly correlated features\n")
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if ((cor(train[[f1]] , train[[f2]])) > 0.999) {
      cat(f1, "and", f2, "are highly correlated \n")
      toRemove <- c(toRemove, f2)
    }
  }
}
train.names  <- setdiff(names(train), toRemove)
train        <- train[,train.names]
test.names   <- setdiff(names(test), toRemove)
test         <- test[,test.names]
toRemove
cat("-------------------------\n")
####################################################
removed <- ncol0-ncol(train)
cat("\n ",removed," features have been removed\n")
write.csv(train, "train_filtered_1.csv", row.names = F)
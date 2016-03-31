library(xgboost)
library(Matrix)
library(plyr)

setwd("C:/Users/Xiaoyu Sun/Desktop/kaggle")
# ---------------------------------------------------
# Load
train <- read.csv("dtrain.csv", stringsAsFactors = F)
valid <- read.csv("dvalid.csv", stringsAsFactors = F)
#sample.submission <- read.csv("../input/sample_submission.csv", stringsAsFactors = F)
#test  <- read.csv("test.csv")
#test.id <- test$ID
train$flag<-1
valid$flag<--1
merged <- rbind(train, valid)

# Split
merged$id<-NULL
orig.train <- merged[merged$flag != -1, ]
orig.train$flag <- NULL 
orig.test <- merged[merged$flag == -1, ]
orig.test$flag <- NULL

# ---------------------------------------------------
#2. numeric to level. Label out and Clean categorical features;
# delete "num_var" variables:
merged<-merged[ -c(68:127) ]



# one-hot-encoding categorical features
#num_feats = c('gender', 'education', 'employer')
#dummies <- dummyVars(~ gender +  education + employer, data = df_all)
#df_all_ohe <- as.data.frame(predict(dummies, newdata = df_all))
#df_all_combined <- cbind(df_all[,-c(which(colnames(df_all) %in% ohe_feats))],df_all_ohe)df_all_combined$agena <- as.factor(ifelse(df_all_combined$age < 0,1,0))

# ---------------------------------------------------
#3. If # Categorical feature != 0 is 1, delete it.
column_index<-NULL
a<- ncol(merged)-1
for(i in 1:a){
  if(length(which(as.vector(merged[,i])!=0))<2){
    column_index[length(column_index)+1]<- i
    print(i)
  }
}
merged<-merged[-column_index ]

# ---------------------------------------------------
#4. Add frequecy counts for each variable
a<- ncol(merged)-1
b<- nrow(merged)
df <- data.frame(matrix(0,ncol = a, nrow = b))
colnames(df) <- paste0(colnames(merged)[1:a], "_freq")
merged$id  <- 1:nrow(merged)
for(i in 1:237){
y<-count(merged, colnames(merged)[i])
out <- merge(x = merged, y = y, all.x = TRUE)
df[,i]<-out[order(out$id), ]$freq
}
merged1<-cbind(merged, df)
merged <- merged1

# ---------------------------------------------------
#5. Add average target for each variable
a<- ncol(merged)-1
b<- nrow(merged)
df1 <- data.frame(matrix(0,ncol = a, nrow = b))
colnames(df1) <- paste0(colnames(merged)[1:a], "_avg.tar")
merged$id  <- 1:nrow(merged)
merged_train <- merged[1:orig.train.row,]
for(i in 1:a){
col_formula <- as.formula(sprintf("TARGET ~ %s", colnames(merged)[i]))  

dff<-aggregate(col_formula, data=merged_train, FUN=mean)
colnames(dff)[2] <- c(paste0(colnames(merged)[i], "_avg.tar"))
out <- merge(x = merged, y = dff, all.x = TRUE)
df1[,i]<-out[order(out$id), ][,ncol(out)]
}
merged1<-cbind(merged, df1)
merged <- merged1
#6. Oversampling

#7. Cost Matrix
write.csv(merged, 'merged.csv', row.names=FALSE, quote = FALSE)

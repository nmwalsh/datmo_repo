# XGBoost Model with K-fold CV
# Nicholas Walsh

library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

set.seed(100)

#Assign train and test data
df_train <- read.csv("Desktop/numerai_datasets/numerai_training_data.csv", head=T)
df_test <- read.csv("Desktop/numerai_datasets/numerai_tournament_data.csv", head=T)

#labels <- colnames(df_train)
df_train <- train[,grep("feature|target",names(train))]
#train <- train[,grep("feature",names(train))]
df_test  <- test[,grep("id|feature",names(test))]
#label <- train[,grep("id",names(train))]
#df_test2  <- test[,grep("feature",names(test))]
#df_all <- rbind(df_train,df_test)
#labels <- colnames(df_train)

X <- df_train
y <- df_train$target
X_test <- df_test




best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:100) {
  param <- list(objective = "binary:logistic",
                eval_metric = "logloss",
                #num_class = 12,
                label = y,
                max_depth = sample(3:7, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 1000
  cv.nfold = 5
  seed.number = 1
  set.seed(seed.number)
  mdcv <- xgb.cv(data=data.matrix(X[]), label = y, params = param, nthread=3, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stopping_rounds=8, maximize=FALSE)
  
  min_logloss = min(mdcv$evaluation_log[, ..test.logloss.mean])
  min_logloss_index = which.min(mdcv$evaluation_log[, ..test.logloss.mean])
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}

nround = best_logloss_index
set.seed(best_seednumber)
md <- xgb.train(data=data.matrix(X[]), params=best_param, nrounds=nround, nthread=3)
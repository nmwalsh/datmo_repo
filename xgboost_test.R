# XGBoost Model
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


df_train <- train[,grep("feature|target",names(train))]
df_test  <- test[,grep("id|feature",names(test))]


X <- df_train
y <- df_train$target
X_test <- df_test


xgb <- xgboost(data = data.matrix(X[]),
               label = y,
               eta = 0.1,
               max_depth = 3,
               nround = 500,
               subsample = 0.5,
               colsample_bytree = .9,
               seed = 1,
               eval_metric = "logloss",
               #objective = "multi:softprob",
               objective = "binary:logistic",
               #num_class = 1,
               nthread = 3,
               early_stopping_rounds = 20
)

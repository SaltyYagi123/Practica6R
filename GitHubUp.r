#Practica 6 Decision Trees Yago Tobio Souto 
#El objetivo de la practica es realizar una extensi�n del analisis realizado 
#en la pr�ctica 5 de KNN y K-Means mediante t�cnicas de arboles de decisi�n y ensembles. 

rm(list = ls())

#Incorporamos la libreria en la cual obtendremos el pack de datos Credit 
library(ISLR)
library(ggplot2)
library(tidyverse)      #manipulacion de datos y visualizaciones
library("car")  # Para VIF
library("psych")  # Para multi.hist
library("corrplot") #Para corrplot
library("plot3D")
library("modelr")  # Para add_predictions
library(ISLR)
library(lattice)#Histogramas
library(dummies)
library(glmnet)
library(tidyverse)      #manipulacion de datos y visualizaciones
library(ROCR)
library(caret)
library(lift) #for lift curve
library(dplyr)
library(class)    
library(cluster) # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(dummies) #for creration of dummy variables
library(caret) #for confusion matrix
library(RSNNS) #for normalization
library(data.tree)

data <- read.csv("letter-recognition.data", header = FALSE, sep = ",", na.strings = "NA")
#Tenemos que sacar las propiedades del archivo de datos 
summary(data)
str(data)

#Se nos ha encargado hacer el estudio para los casos de las letras X, Y y Z. 
data <- na.omit(data)

#Por lo tanto, debemos de quitar todos los datos que no sean X, Y o Z 
#Paso 1.- Obtener el indice para los valores de X, Y o Z 
index_x <- grep("X",data$V1)
index_y <- grep("Y", data$V1)
index_z <- grep("Z", data$V1)

index_filter <- c(index_x, index_y, index_z)

#Ahora ya tenemos todos los datos filtrados tal que T solo valdr� X,Y o Z 
data_filtered <- data[index_filter,]


#Al solo tener 3 niveles en T, vamos a convertirlo a una variable factor 
data_filtered$V1 <- as.factor(data_filtered$V1)
str(data_filtered)

names(data_filtered) <- c('lettr','x-box','y-box','width','high','onpix','x-bar','y-bar'
                       ,'x2bar','y2bar','xybar', 'x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx')

data_filtered <- data_filtered %>% relocate(lettr, .after = yegvx)

set.seed(123)

#Creation of training and test datasets 
index <- sample(nrow(data_filtered), round(0.8*nrow(data_filtered)))
train <- data_filtered[index, ]
test <- data_filtered[-index, ]
train_label <- data_filtered[index, 17]
test_label <- data_filtered[-index, 17]



table(train$lettr == "X")/nrow(train)
table(test$lettr == "X")/nrow(test)
table(train$lettr == "Y")/nrow(train)
table(test$lettr == "Y")/nrow(test)
table(train$lettr == "Z")/nrow(train)
table(test$lettr == "Z")/nrow(test)



################## Pasamos ID3 ##################################
library(data.tree)

#Definimos puridad 
IsPure <- function(data){
  length(unique(data[, ncol(data)])) == 1
}

#Definition of entropy
Entropy <- function( vls ) {
  res <- vls/sum(vls) * log2(vls/sum(vls))
  res[vls == 0] <- 0
  -sum(res)
}

Entropy(c(10, 0))
Entropy(c(0, 7))
Entropy(c(3, 7))


#Information Gain
InformationGain <- function( tble ) {
  tble <- as.data.frame.matrix(tble)
  entropyBefore <- Entropy(colSums(tble))
  s <- rowSums(tble)
  entropyAfter <- sum (s / sum(s) * apply(tble, 
                                          MARGIN = 1, FUN = Entropy ))
  informationGain <- entropyBefore - entropyAfter
  return (informationGain)
}

#Testeamos Information Gain 
InformationGain(table(data_filtered[,c('lettr', 'x-bar')]))
InformationGain(table(data_filtered[, c('lettr', 'y-bar')]))

#Aqui podemos observar el cambio de entrop�a para estos dos separadores, los cuales 
#tendriamos que hacerlos a mano como alternativas.

#ID3 Code 
TrainID3 <- function(node, data) {
  
  node$obsCount <- nrow(data)
  
  #if the data-set is pure (e.g. all toxic), then
  if (IsPure(data)) {
    #construct a leaf having the name of the pure feature (e.g. 'toxic')
    child <- node$AddChild(unique(data[,ncol(data)]))
    node$feature <- tail(names(data), 1)
    child$obsCount <- nrow(data)
    child$feature <- ''
  } else {
    #chose the feature with the highest information gain (e.g. 'color')
    ig <- sapply(colnames(data)[-ncol(data)], 
                 function(x) InformationGain(
                   table(data[,x], data[,ncol(data)])
                 )
    )
    feature <- names(ig)[ig == max(ig)][1]
    
    node$feature <- feature
    
    #take the subset of the data-set having that feature value
    childObs <- split(data[,!(names(data) %in% feature)], data[,feature], drop = TRUE)
    
    for(i in 1:length(childObs)) {
      #construct a child having the name of that feature value (e.g. 'red')
      child <- node$AddChild(names(childObs)[i])
      
      #call the algorithm recursively on the child and the subset      
      TrainID3(child, childObs[[i]])
    }
  }
}

#.Debemos de tener en cuenta que dicha variable que deseamos clasificar 
#Debe de ser la ultima en nuestro dataset 
#Tras esto, pasamos a construir el arbol ID3 para predecir la variable lettr 
treeID3 <- Node$new("lettr")
TrainID3(treeID3, train)
print(treeID3, "feature", "obsCount")



# Prediction function
Predict <- function(tree, features) {
  if (tree$children[[1]]$isLeaf) return (tree$children[[1]]$name)
  child <- tree$children[[features[[tree$feature]]]]
  return ( Predict(child, features))
}

########## Procedemos CON C50 ####################################2
library(C50)

tree_C50 <- C5.0(lettr ~ ., data = train, 
                 control = C5.0Control(noGlobalPruning = FALSE, CF = 0.25))
# Observamos que si que deseamos podar, y nuestro factor es de 0.25
# Cuanto mas alto CF, menos podaremos

summary(tree_C50)


plot(tree_C50, trial = 0, subtree = NULL)

predictions <- predict(tree_C50, newdata = test, type = "class")
table(prediction = predictions, real = test$lettr)
error_class <- mean(predictions != test$lettr)
paste("Classification error is: ", 100*error_class, "%")


ruleModel <- C5.0(lettr ~., data = train, rules = TRUE)

summary(ruleModel)


###################### Procedemos con CART ##################################
library(rpart)
library(rpart.plot)
library(caret)

tree_CART <- rpart(formula = lettr ~ ., data = train, method = 'class')
print(tree_CART)

summary(tree_CART)


rpart.plot(tree_CART, type = 1, branch = 0, tweak = 1.55, fallen.leaves = TRUE, 
           varlen = 0, faclen = 0)

prp(tree_CART, faclen = 3, clip.facs = TRUE, tweak = 1.2, extra = 101)

rpart.rules(tree_CART, style = "tall", cover = TRUE, nn = TRUE, clip.facs = TRUE)

#Prediction of the training cases from its respective dataset 
pred_train <- predict(tree_CART, newdata = train, type = "class")
caret::confusionMatrix(pred_train, train$lettr)

pred_test <- predict(tree_CART, newdata = test, type = "class")
caret::confusionMatrix(pred_test, test$lettr)

printcp(tree_CART, digits = 4)
plotcp(tree_CART, lty = 2, col = "red", upper = "size")
plotcp(tree_CART, lty = 2, col = "red", upper = "splits")

#Pruning analysis con 0.02224
tree_pruned <- prune(tree_CART, cp = 0.02224)
rpart.plot(tree_pruned, type = 1, branch = 0, tweak = 1, 
           fallen.leaves = TRUE, 
           varlen = 0, faclen = 0)

##### Predicciones de CART con dataset de entrenamiento y test ###########

pred_train <- predict(tree_pruned, newdata =  train, type = "class")
caret::confusionMatrix(pred_train, train$lettr)

error_class <- mean(pred_train != train$lettr)
error_class

predictions <- predict(tree_pruned, newdata = test, type = "class")
caret::confusionMatrix(predictions, test$lettr)



##### PROCEDEMOS CON ENSEMBLES ###########
library(MASS)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(caret)
library(rsample)
library(randomForest)
library(ggpubr)

index <- initial_split(data_filtered, prop=0.8)
train_bag<- training(index)
test_bag<- testing(index)



##### 1.- Bagging #######
#Create the bagging model 
bagging_model <- randomForest(formula = lettr ~., data = train_bag, mtry = 16)

rm(list = ls())
library(dplyr)
library(tidyverse)      #data manipulation and visualization
library(class)          # to call class package for kNN
library(caret)          # for building the model

set.seed(123)

letterData <- read.csv(file = "letter-recognition.data",     #Name of text file.
                         sep = ",",                #Separation character.
                         header = FALSE,            #If column names are in the first row.
                         na.strings = "NA",        #Character to be marked as missing value.
                         stringsAsFactors = FALSE) #¿convert string to factors?

summary(letterData)

letterData<-filter(letterData, V1=="X"|V1=="Y"|V1=="Z")
letterData$V1=as.factor(letterData$V1)

names(letterData) <- c('lettr','x.box','y.box','width','high','onpix','x.bar','y.bar','x2bar','y2bar','xybar',
                       'x2ybr','xy2br','x.ege','xegvy','y.ege','yegvx')

letterData <- letterData %>% relocate(lettr, .after = yegvx)

library(MASS)       # for obtaining data
library(tidyverse)  # for data processing
library(rpart)      # for CART decision tree
library(rpart.plot) # for plotting CART
library(caret)      # for confusion matrix and more
library(rsample)    # for data splitting
library(randomForest)  # For bagging and randomforest
library(ggpubr)

#Creating a training and test datasets
set.seed(123)
letterData_split<- initial_split(letterData, prop=0.8)
letterData_train<- training(letterData_split)
letterData_test<- testing(letterData_split)

################# Bagging
# Creating the bagging model
bagging_model<- randomForest(formula=lettr~ ., data=letterData_train, 
                             mtry=16)  #16 are all the predictors
# ntree= xx assign the number of tress, by default 500
#Resul of bagging model
print(bagging_model)


################# #Prediction ############

#Prediction of the training cases from the train dataset
# Confusion Matrix of the real and estimated values
pred_train <- predict(bagging_model, newdata = letterData_train)

confusion_matrix<-table(letterData_train$lettr,pred_train,
                        dnn=c("observations", "predictions"))
confusion_matrix


#Prediction of the test cases from the test dataset
pred_test <- predict(bagging_model, newdata = letterData_test)

confusion_matrix<-table(letterData_test$lettr,pred_test,
                        dnn=c("observations", "predictions"))
confusion_matrix


# Creating the bagging model for the whole data set
bagging_model_fulldataset<- randomForest(formula=lettr  ~ ., data=letterData, 
                                         mtry=16, importance=TRUE)  #16 are all the predictors
# ntree= xx assign the number of tress, by default 500


#Result of bagging model
print(bagging_model_fulldataset)


### Indicators
# Importance indexes
importance_pred <- as.data.frame(importance(bagging_model_fulldataset, 
                                            scale = TRUE))
importance_pred <- rownames_to_column(importance_pred, 
                                      var = "variable")

p1 <- ggplot(data = importance_pred, 
             aes(x = reorder(variable, MeanDecreaseAccuracy), 
                 y = MeanDecreaseAccuracy, fill = MeanDecreaseAccuracy)) +
  labs(x = "variable", title = "Accuracy Reduction") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "bottom")

p2 <- ggplot(data = importance_pred, 
             aes(x = reorder(variable, MeanDecreaseGini), 
                 y = MeanDecreaseGini, fill = MeanDecreaseGini)) +
  labs(x = "variable", title = "Gini Reduction") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "bottom")

ggarrange(p1,p2)

importance_pred

#Selecting the optimun number of trees
oob_mse<-data.frame(oob_mse=bagging_model_fulldataset$err.rate,
                    trees=seq_along(bagging_model_fulldataset$err.rate))
ggplot(data=oob_mse[1:500,], aes(x=trees[1:500], y=oob_mse.OOB[1:500]))+
  geom_line()+
  labs(title = "Evolution of OOB vs number of trees", x="n. trees")+
  theme_bw()


# Testing results with 75 trees
bagging_model_75<- randomForest(formula=lettr  ~ ., data=letterData, 
                                 mtry=16, ntree=75,
                                 importance=TRUE) 
#Result of bagging model
print(bagging_model_75)

#Prediction of the training cases from the train dataset PRUNING
pred_train <- predict(bagging_model_75, newdata = letterData_train, type ="class")
confusionMatrix(pred_train, letterData_train$lettr)

table(prediction=pred_train, real= letterData_train$lettr)

error_classification <- mean(pred_train != letterData_train$lettr)

paste("The classification error is:", 100*error_classification, "%",
      sum(pred_train==letterData_train$lettr),
      "correct classified cases from", length(pred_train))

#Prediction of new cases from the test dataset
predictions <- predict(bagging_model_75, newdata = letterData_test, type ="class")
confusionMatrix(predictions, letterData_test$lettr)

table(prediction=predictions, real= letterData_test$lettr)

error_classification <- mean(predictions != letterData_test$lettr)

paste("The classification error is:", 100*error_classification, "%",
      sum(predictions==letterData_test$lettr),
      "correct classified cases from", length(predictions))


# Creating the random forest model
# m=root(16)
bagging_model<- randomForest(formula=lettr  ~ ., data=letterData_train, 
                             mtry=4)  #4 from 16 predictors will be selected
print(bagging_model)


#Tunning the m number of predictors
tuning_rf_mtry <- function(df, y, ntree = 500){
  # This function returns the out-of-bag-MSE of a RandomForest model
  # in function of the number of predictors evaluated
  
  
  # Arguments:
  #   df = data frame with predictors and variable to predict
  #   y  = name of the variable to predict
  #   ntree = number of trees created by the randomForest algorithm
  
  require(dplyr)
  max_predictors <- ncol(df) - 1
  n_predictors   <- rep(NA, max_predictors)
  oob_err_rate   <- rep(NA, max_predictors)
  for (i in 1:max_predictors) {
    set.seed(123)
    f <- formula(paste(y,"~ ."))
    model_rf <- randomForest(formula = f, data = df, mtry = i, ntree = ntree)
    n_predictors[i] <- i
    oob_err_rate[i] <- tail(model_rf$err.rate[,1], n = 1)
  }
  results <- data_frame(n_predictors, oob_err_rate)
  return(results)
}

hiperparameter_mtry <-  tuning_rf_mtry(df = letterData, y = "lettr")
hiperparameter_mtry %>% arrange(oob_err_rate)

ggplot(data = hiperparameter_mtry, aes(x = n_predictors, y = oob_err_rate)) +
  scale_x_continuous(breaks = hiperparameter_mtry$n_predictors) +
  geom_line() +
  geom_point() +
  geom_point(data = hiperparameter_mtry %>% arrange(oob_err_rate) %>% head(1),
             color = "red") +
  labs(title = "Evolution of the out-of-bag-error vs m",
       x = "number of predictors used") +
  theme_bw()


# Creating the random forest model
bagging_model<- randomForest(formula=lettr  ~ ., data=letterData_train, 
                             mtry=1)  #1 from 16 predictors will be selected


#Result of random forest model
print(bagging_model)



tuning_rf_nodesize <- function(df, y, size = NULL, ntree = 500){
  # This funstion returns the out-of-bag-MSE of a random forestmodel
  # in function of the minimum size of the terminal nodes (nodesize).
  
  
  # Arguments:
  #   df = data frame with predictors and variable to predict
  #   y  = name of the variable to predict
  #   size= evaluated sizes
  #   ntree = number of trees created by the randomForest algorithm
  
  
  require(dplyr)
  if (is.null(size)){
    size <- seq(from = 1, to = nrow(df), by = 5)
  }
  
  oob_err_rate <- rep(NA, length(size))
  for (i in seq_along(size)) {
    set.seed(123)
    f <- formula(paste(y,"~ ."))
    model_rf <- randomForest(formula = f, data = df, mtry = 5, ntree = ntree,
                             nodesize = i)
    oob_err_rate[i] <- tail(model_rf$err.rate[, 1],n = 1)
  }
  results <- data_frame(size, oob_err_rate)
  return(results)
}

hiperparameter_nodesize <-  tuning_rf_nodesize(df = letterData, y = "lettr",
                                               size = c(1:20))
hiperparameter_nodesize %>% arrange(oob_err_rate)


ggplot(data = hiperparameter_nodesize, aes(x = size, y = oob_err_rate)) +
  scale_x_continuous(breaks = hiperparameter_nodesize$size) +
  geom_line() +
  geom_point() +
  geom_point(data = hiperparameter_nodesize %>% arrange(oob_err_rate) %>% head(1),
             color = "red") +
  labs(title = "Evoluci�n del out-of-bag-error vs nodesize",
       x = "number of observationes in terminal nodes") +
  theme_bw()


model_randomforest <- randomForest(lettr ~ ., data = letterData_train,
                                   mtry = 1 , ntree = 500, nodesize = 1,
                                   importance = TRUE)
oob_error_rate <- data.frame(oob_error_rate = model_randomforest$err.rate[,1],
                             trees = seq_along(model_randomforest$err.rate[,1]))


ggplot(data = oob_error_rate, aes(x = trees, y = oob_error_rate )) +
  geom_line() +
  labs(title = "Evolutionof the out-of-bag-error vs trees number",
       x = "number of trees") +
  theme_bw()


# Creating the random forest model
rf_model<- randomForest(formula=lettr  ~ ., data=letterData_train, 
                        mtry=3, ntree=425, nodesize=1,
                        importance=TRUE, norm.votes=TRUE)



#Result of random forest model
print(rf_model)

### Indicators
# Importance indexes
importance_pred <- as.data.frame(importance(rf_model, scale = TRUE))
importance_pred <- rownames_to_column(importance_pred, 
                                      var = "variable")

print(importance_pred)


p1 <- ggplot(data = importance_pred, 
             aes(x = reorder(variable, MeanDecreaseAccuracy), 
                 y = MeanDecreaseAccuracy, fill = MeanDecreaseAccuracy)) +
  labs(x = "variable", title = "Accuracy Reduction") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "bottom")

p2 <- ggplot(data = importance_pred, 
             aes(x = reorder(variable, MeanDecreaseGini), 
                 y = MeanDecreaseGini, fill = MeanDecreaseGini)) +
  labs(x = "variable", title = "Purity Reduction (Gini)") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "bottom")

ggarrange(p1,p2)

#Prediction of the training cases from the train dataset PRUNING
pred_train <- predict(rf_model, newdata = letterData_train, type ="class")
caret::confusionMatrix(pred_train, letterData_train$lettr)

table(prediction=pred_train, real= letterData_train$lettr)

error_classification <- mean(pred_train != letterData_train$lettr)

paste("The classification error is:", 100*error_classification, "%",
      sum(pred_train==letterData_train$lettr),
      "correct classified cases from", length(pred_train))

#Prediction of new cases from the test dataset
predictions <- predict(rf_model, newdata = letterData_test, type ="class")
caret::confusionMatrix(predictions, letterData_test$lettr)

table(prediction=predictions, real= letterData_test$lettr)

error_classification <- mean(predictions != letterData_test$lettr)

paste("The classification error is:", 100*error_classification, "%",
      sum(predictions==letterData_test$lettr),
      "correct classified cases from", length(predictions))

########### Boosting ##################

library(gbm)        # for Boosting algorithms

train <- sample(1:nrow(letterData), size = nrow(letterData)*0.8)
cv_error  <- vector("numeric")
n_trees <- vector("numeric")
shrinkage <- vector("numeric")

# Shrinkage optimization
for (i in c(0.001, 0.01, 0.1)) {
  set.seed(123)
  tree_boosting <- gbm(lettr ~ ., data = letterData[train, ],
                       distribution = "multinomial",
                       n.trees = 20000,
                       interaction.depth = 1,
                       shrinkage = i,
                       n.minobsinnode = 10,
                       bag.fraction = 0.5,
                       cv.folds = 5)
  cv_error  <- c(cv_error, tree_boosting$cv.error)
  n_trees <- c(n_trees, seq_along(tree_boosting$cv.error))
  shrinkage <- c(shrinkage, rep(i, length(tree_boosting$cv.error)))
}
error <- data.frame(cv_error, n_trees, shrinkage)

ggplot(data = error, aes(x = n_trees, y = cv_error,
                         color = as.factor(shrinkage))) +
  geom_smooth() +
  labs(title = "Evolution of the cv-error", color = "shrinkage") + 
  theme_bw() +
  theme(legend.position = "bottom")
#escogemos 0.01

# Selecting tree complexity
cv_error  <- vector("numeric")
n_trees <- vector("numeric")
interaction.depth <- vector("numeric")
for (i in c(3, 4, 5, 8)) {
  set.seed(123)
  tree_boosting <- gbm(lettr ~ ., data = letterData[train, ],
                       distribution = "multinomial",
                       n.trees = 5000,
                       interaction.depth = i,
                       shrinkage = 0.01,
                       n.minobsinnode = 10,
                       bag.fraction = 0.5,
                       cv.folds = 5)
  
  cv_error  <- c(cv_error, tree_boosting$cv.error)
  n_trees <- c(n_trees, seq_along(tree_boosting$cv.error))
  interaction.depth <- c(interaction.depth,
                         rep(i, length(tree_boosting$cv.error)))
}
error <- data.frame(cv_error, n_trees, interaction.depth)

ggplot(data = error, aes(x = n_trees, y = cv_error,
                         color = as.factor(interaction.depth))) +
  geom_smooth() +
  labs(title = "Evolution of the cv-error", color = "interaction.depth") + 
  theme_bw() +
  theme(legend.position = "bottom")
#escogemos 8

#Selecting the minimum number of observations
cv_error  <- vector("numeric")
n_trees <- vector("numeric")
n.minobsinnode <- vector("numeric")
for (i in c(2, 3, 5, 10)) {
  tree_boosting <- gbm(lettr ~ ., data = letterData[train, ],
                       distribution = "multinomial",
                       n.trees = 5000,
                       interaction.depth = 8,
                       shrinkage = 0.01,
                       n.minobsinnode = i,
                       bag.fraction = 0.5,
                       cv.folds = 5)
  cv_error  <- c(cv_error, tree_boosting$cv.error)
  n_trees <- c(n_trees, seq_along(tree_boosting$cv.error))
  n.minobsinnode <- c(n.minobsinnode,
                      rep(i, length(tree_boosting$cv.error)))
}
error <- data.frame(cv_error, n_trees, n.minobsinnode)

ggplot(data = error, aes(x = n_trees, y = cv_error,
                         color = as.factor(n.minobsinnode))) +
  geom_smooth() +
  labs(title = "Evolution of the cv-error", color = "n.minobsinnode") + 
  theme_bw() +
  theme(legend.position = "bottom")
#escogemos 10

# Determination of the number of trees to use
set.seed(123)
tree_boosting <- gbm(lettr ~ ., data = letterData[train, ],
                     distribution = "multinomial",
                     n.trees = 10000,
                     interaction.depth = 8,
                     shrinkage = 0.01,
                     n.minobsinnode = 10,
                     bag.fraction = 0.5,
                     cv.folds = 5)
error <- data.frame(cv_error = tree_boosting$cv.error,
                    n_trees = seq_along(tree_boosting$cv.error))
ggplot(data = error, aes(x = n_trees, y = cv_error)) +
  geom_line(color = "blue") +
  geom_point(data = error[which.min(error$cv_error),], color = "red") +
  labs(title = "Evolution of the cv-error") + 
  theme_bw() 

error[which.min(error$cv_error),]
#1000 trees

#Using caret for finding all the parameters
library(caret)

set.seed(123)
validation <- trainControl(## 10-fold CV
  method = "cv",
  number = 10)

tuning_grid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                            n.trees = c(100, 1000, 2000, 3000), 
                            shrinkage = c(0.1, 0.01, 0.001),
                            n.minobsinnode = c(1, 10, 20))

set.seed(123)
best_model <- train(lettr ~ ., data = letterData[train, ], 
                    method = "gbm", 
                    trControl = validation, 
                    verbose = FALSE, 
                    tuneGrid = tuning_grid)

# Showing the best valures for the hyperparameters 
best_model$bestTune

#best model fitting
set.seed(123)
tree_boosting <- gbm(lettr ~ ., data = letterData[train, ],
                     distribution = "multinomial",
                     n.trees = 1000,
                     interaction.depth = 8,
                     shrinkage = 0.01,
                     n.minobsinnode = 10,
                     bag.fraction = 0.5)

summary(tree_boosting)

# Plotting the influence of each predictor
importance_pred <- summary(tree_boosting, plotit = FALSE)
ggplot(data = importance_pred, aes(x = reorder(var, rel.inf), 
                                   y = rel.inf,
                                   fill = rel.inf)) +
  labs(x = "variable", title = "MSE reduction") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "bottom")

#Prediction of the test dataset
predictions<- predict(object = tree_boosting, newdata = letterData[-train,], class="response")
predictions <- apply(predictions, 1, which.max)
predictions<-replace(predictions, predictions==1, "A")
predictions<-replace(predictions, predictions==2, "B")
predictions<-replace(predictions, predictions==3, "C")

table(prediction=predictions, real= letterData[-train, "lettr"])

error_classification <- mean(predictions != letterData[-train, "lettr"])

paste("The classification error is:", 100*error_classification, "%",
      sum(predictions==letterData[-train, "lettr"]),
      "correct classified cases from", length(predictions))








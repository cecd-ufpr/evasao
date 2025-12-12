library(caret)
library(caretEnsemble)
library(tidyverse)
library(kernelshap)
library(shapviz)

ev <- read.csv2("evasao.csv")
ev$evasao <- as.factor(ev$evasao) #dropout variable
ev$Area_the <- as.factor(ev$Area_the)
ev$grau <- as.factor(ev$grau)
#-------------------------------------------------------------------------------
# Balancing data
set.seed(123)
evadidos <- subset(ev, evasao == 1) 
n_evadidos <- subset(ev, evasao == 0)
sorteio <- sample(1:nrow(n_evadidos), sum(ev$evasao==1))
n_evadidos_sorteados <- n_evadidos[sorteio,]
dados_balanceados <- rbind(evadidos, n_evadidos_sorteados)
#-------------------------------------------------------------------------------
# Comparative
dataset<- db[, c("evasao", "Area_the","ira","proprep","grau")]
x <- dataset %>%
  mutate(Area_the = recode(Area_the, 
                           "CL" = 1, "ET" = 2, "LS" = 3, "PH" = 4, "BE" = 5, 
                           "AH" = 6, "SS" = 7, "PS" = 8, "CS" = 9, "ED" = 10, "LW" = 11))
y <- dataset %>%
  mutate(grau = recode(grau, 
                       "Bacharelado" = 1, "ABI" = 2, "Tecnol\xf3gico" = 3, "Licenciatura" = 4 ))
dataset$the <- x$Area_the
dataset$the <- as.factor(dataset$the)
dataset$grau <- y$grau
dataset$grau <- as.factor(dataset$grau)
dataset$class<- make.names(dataset$evasao)
treino = dataset[indice_treino, ]
teste = dataset[-indice_treino, ]
levels(treino$evasao) <- c("Nao", "Sim")
levels(teste$evasao) <- c("Nao", "Sim")
ctrl <- trainControl(
  method = "cv",      # k-fold
  number = 10,        # number of folds
  savePredictions = "final",
  classProbs = TRUE
)
# Methods
model_list <- list(
  # Logistic Regression
  glm = caretModelSpec(
    method = "glm",
    family = "binomial",
    preProcess = c("center", "scale")
  ),
  # SVM Radial
  svmRadial = caretModelSpec(
    method = "svmRadial",
    tuneGrid = data.frame(C = 1, sigma = 0.1),  
    preProcess = c("center", "scale")
  ),
  # Decision Tree
  rpart = caretModelSpec(
    method = "rpart",
    tuneGrid = data.frame(cp = 0.01)  
  ),
  # Random Forest
  rf = caretModelSpec(
    method = "rf",
    tuneGrid = data.frame(mtry = sqrt(ncol(treino) - 1)),  
    ntree = 100  
  ),
  #Stochastic Gradient Boosting - SGB
  gbm = caretModelSpec(
    method = "gbm",
    tuneGrid = expand.grid(
      n.trees = 100,       # número de árvores
      interaction.depth = 3,  
      shrinkage = 0.1,     
      n.minobsinnode = 10  
    ),
    preProcess = c("center", "scale"),
    verbose = FALSE       
  ),
  # Neural Net
  nnet = caretModelSpec(
    method = "nnet",
    tuneGrid = expand.grid(
      size = 5,        
      decay = 0.1    
    ),
    preProcess = c("center", "scale"),
    trace = FALSE     
  )
)

# Trainning
ensemble_models <- caretList(
  evasao ~ ira + Area_the + proprep + grau,  # Fórmula do modelo
  data = treino,
  trControl = ctrl,
  metric = "ROC",  # Otimizar pela AUC (Area Under the Curve)
  tuneList = model_list
)

# Results
results <- resamples(ensemble_models)
summary(results)  
dotplot(results, metric = "Accuracy")  

# Confusion Matrix
for (model_name in names(ensemble_models)) {
  cat("\n--- Modelo:", model_name, "---\n")
  pred <- predict(ensemble_models[[model_name]], newdata = teste)
  print(confusionMatrix(pred, teste$evasao))  
}

#-------------------------------------------------------------------------------
# SHAP Values
# Pred variables
X <- treino[, c("ira", "Area_the", "proprep", "grau")]
# Pred function
pred_fun <- function(model, newdata) {
  predict(model, newdata, type = "prob")[, "Sim"]
}
# Extract SHAP Values
shap_rl <- kernelshap(ensemble_models$glm, X = X, bg_X = X[1:100, ], pred_fun = pred_fun)
shap_rpart <- kernelshap(ensemble_models$rpart, X = X, bg_X = X[1:100, ], pred_fun = pred_fun)
shap_rf <- kernelshap(ensemble_models$rf, X = X, bg_X = X[1:100, ], pred_fun = pred_fun)
shap_gbm <- kernelshap(ensemble_models$gbm, X = X, bg_X = X[1:100, ], pred_fun = pred_fun)
shap_svm <- kernelshap(ensemble_models$svm, X = X, bg_X = X[1:100, ], pred_fun = pred_fun)
shap_rn <- kernelshap(ensemble_models$nnet, X = X, bg_X = X[1:100, ], pred_fun = pred_fun)

# Beeswarm plot
sv_rl <- shapviz(shap_rl)
sv_importance(sv_rl, kind = "bee")

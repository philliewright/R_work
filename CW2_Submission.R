# BE AWARE THIS CODE IS USING 10% SUBSET OF THE DATA
# FOR MARKING PURPOSES.
# THEREFORE THE RESULTS MAY DIFFER FROM THE REPORT!

# Time to Run using Subset: 12mins 45seconds on average 

# Contents
# ----------------
# 1. Preliminary work
#    1.1 Importing Libraries and Data
# 
# 2. Data Preparation and Exploration
#    2.1 Splitting Data
#    2.2 Principal Component Analysis (PCA)
#    2.3 Plotting PCA
# 
# 3. Decision Tree Model
#    3.1 Building the Decision Tree Model
#    3.2 Hyperparameter Tuning
# 
# 4. Random Forest Model
#    4.1 Classification
#    4.2 Hyperparameter Tuning
# 
# 5. XGBoost Model
#    5.1 Classification
#    5.2 Hyperparameter Tuning
# 
# 6. Model Comparison
#    6.1 Comparing Decision Tree, Random Forest, and XGBoost Models
# 
# 7. Regression Analysis Setup
#    7.1 Importing Data
#    7.2 Data Splitting and PCA for Regression
# 
# 8. Regression Analysis
#    8.1 Multiple Linear Regression
#    8.2 Polynomial Regression Analysis
#    8.3 XGBoosted Regression
# 
# 9. Alternative Approach with Non-preprocessed Data
#    9.1 Overview
# 
# 10. Data Preprocessing for New Approach
#     10.1 Min-Max Scaling
#     10.2 Logarithmic Transformation of Shares
# 
# 11. Principal Component Analysis for New Data
#     11.1 Data Splitting and PCA
# 
# 12. Regression with New Data
#     12.1 Boosted Regression Analysis



rm(list=ls())

#----------------------------------
# 1. Preliminary Work
# 1.1 Importing Libraries and Data
#----------------------------------
#install.packages("dplyr")
#install.packages("xgboost")
#install.packages("ggplot2")
#install.packages("tidyverse")
#install.packages("randomForest")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("caret")
#install.packages("patchwork")

# Import libraries
library(dplyr)
library(xgboost)
library(ggplot2)
library(tidyverse)
library(randomForest)
library(rpart)
library(rpart.plot)
library(caret)  # for conf matrix
library(patchwork)#

set.seed(123) # for the fractioning of the data
# Load data
data_folder <- "./"

# Set the working directory to the parent directory of the script
current_directory <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(current_directory)

# Check current wd and list the files
print(getwd())
list.files()

# Read the csv file with the Natural language processing results for the news data
file_path <- file.path(data_folder, "scaled_outlier_removed.csv")
if (file.exists(file_path)) {
  scaled_data <- read.csv(file_path, header = TRUE)
} else {
  stop("Error: scaled_outlier_removed.csv file does not exist in the specified directory.")
}

# Add thresholds and remove 'shares' column for classification
thresholds <- quantile(scaled_data$shares, probs = c(0, 0.30,0.65, 1), na.rm = TRUE)
scaled_data$popularity_thresholds <- cut(scaled_data$shares, breaks = thresholds, labels = c("Poor",  "Average", "Good"), include.lowest = TRUE)
summary(scaled_data)
scaled_data <- scaled_data %>% select(-shares) 

scaled_data <- scaled_data %>% sample_frac(.1)  # Use only 10% of the data for marking

# -----------------------------------
# 2. Data Preparation and Exploration
# -----------------------------------
# -----------------------------------
# 2.1 Splitting Data
# -----------------------------------
# Split data into training and testing sets before PCA
set.seed(123)
sample_idx <- sample(c(TRUE, FALSE), nrow(scaled_data), replace = TRUE, prob = c(0.8, 0.2))
train_data <- scaled_data[sample_idx, ]
test_data <- scaled_data[!sample_idx, ]

train_labels <- train_data$popularity_thresholds
test_labels <- test_data$popularity_thresholds

# ----------------------------------------
# 2.2 Principal Component Analysis (PCA)
# ----------------------------------------
# Performing PCA on training data
train_data_no_label <- train_data %>% select(-popularity_thresholds)
pca_result_train <- prcomp(train_data_no_label, scale. = TRUE)
cumulative_variance_train <- cumsum(pca_result_train$sdev^2) / sum(pca_result_train$sdev^2)
n_components_train <- which(cumulative_variance_train >= 0.95)[1]

# Transforming the train and test data using PCA above
test_data_no_label <- test_data %>% select(-popularity_thresholds)
test_pca <- predict(pca_result_train, newdata = test_data_no_label)
train_pca <- predict(pca_result_train, newdata = train_data_no_label) 

train_pca_combined <- data.frame(train_pca, popularity_thresholds = train_labels)
test_pca_combined <- data.frame(test_pca, popularity_thresholds = test_labels)

# ----------------------------------------
# 2.3 Plotting PCA
# ----------------------------------------

# Visualising feature importance, shows how much varience is explained per PC
sd <- pca_result_train$sdev
vr <- sd^2
pr_vr <- vr/sum(vr)
plot(cumsum(pr_vr), xlab="Component", ylab="Proportion of variance", main="Variance Explained by Principal Components")

loadings <- pca_result_train$rotation
feature_importance <- rowSums(abs(loadings))
sorted_features <- names(feature_importance[order(-feature_importance)])

# Displays the sorted featurea
sorted_features

# Checks the distribution over both training and test sets
# Need them to be fairly equal so the training isnt unbalenced
barplot(table(train_labels), main = "Distribution of Popularity Thresholds in Training Data", col = "green")
barplot(table(test_labels), main = "Distribution of Popularity Thresholds in Test Data", col = "darkgreen")

#---------------------------------------------------
# 3.1 Building the Decision Tree Model
#---------------------------------------------------

# Training the decision tree model
train_rpart_model <- function(data, cp_value) {
  model.rpart <- rpart(popularity_thresholds ~ ., method = "class", data = data, cp = cp_value)
  return(model.rpart)
}


# Ploting the trained model
plot_rpart_model <- function(model) {
  plot(model)
  text(model)
  prp(model)
}

# Make predictions with the trained model
predict_rpart_model <- function(model, test_data) {
  pred_tree <- predict(model, test_data, type = "class")
  return(pred_tree)
}

# Evaluate the model's performance
evaluate_model <- function(predictions, true_values) {
  conf_matrix <- table(predictions, true_values)
  accuracy_value <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision_value <- diag(conf_matrix) / rowSums(conf_matrix)
  recall_value <- diag(conf_matrix) / colSums(conf_matrix)
  f1_score <- 2 * (precision_value * recall_value) / (precision_value + recall_value)
  return(list(accuracy = accuracy_value, 
              precision = mean(precision_value, na.rm = TRUE), recall = mean(recall_value, na.rm = TRUE),
              f1_score = mean(f1_score, na.rm = TRUE)))
}

# Train and evaluate the decision tree model
cp_value <- 0.0008
model <- train_rpart_model(train_pca_combined, cp_value)
plot_rpart_model(model)
predictions <- predict_rpart_model(model, test_pca_combined)
evaluate_model(predictions, test_pca_combined$popularity_thresholds)

#---------------------------------------------
# 3.2 Hyperparameter tuning for decision tree
#----------------------------------------------

ctrl <- trainControl(method = "cv", number = 10)
cp_grid <- 10^seq(-6, -2, by = 0.1)
model_cv <- train(popularity_thresholds ~ ., method = "rpart", data = train_data, 
                  trControl = ctrl, tuneGrid = data.frame(cp = cp_grid))
plot(model_cv)

# Print the best tuning parameters
print(model_cv$bestTune)


# Runs the decision tree model using the best cp value
best_cp <- 0.0003981072
final_decision_tree_model <- rpart(popularity_thresholds ~ ., data = train_data, method = "class", cp = best_cp)

# Predictions on the test set
test_predictions <- predict(final_decision_tree_model, newdata = test_data, type = "class")

# Evaluate the model

confusion_matrix <- confusionMatrix(test_predictions, test_data$popularity_thresholds)
print(confusion_matrix)

#------------------------------------------
# 4.1 Random Forest Classification
#------------------------------------------


# Function to train the initial Random Forest model
train_initial_rf <- function(data, ntree = 500, mtry = NULL, importance = TRUE) {
  if (is.null(mtry)) {
    mtry <- floor(sqrt(ncol(data) - 1))
  }
  
  set.seed(123)
  rf <- randomForest(popularity_thresholds ~ ., data = data, ntree = ntree, mtry = mtry, importance = importance)
  print(rf)
  return(rf)
}

# Function to tune the Random Forest hyperparameters
tune_rf <- function(data, control, grid, ntree = 500, nodesize = 10) {
  rf_model_tuning <- train(popularity_thresholds ~ ., data = data, method = "rf", trControl = control, tuneGrid = grid, ntree = ntree, nodesize = nodesize)
  
  # Plotting
  rf_results <- rf_model_tuning$results
  ggplot(rf_results, aes(x = mtry, y = Accuracy)) +
    geom_line() +
    geom_point() +
    labs(title = "Random Forest Accuracy for Different mtry values",
         x = "mtry",
         y = "Accuracy") +
    theme_minimal()
  
  return(rf_model_tuning)
}

# Function to evaluate the Random Forest model
evaluate_rf <- function(model, test_data) {
  predictions <- predict(model, newdata = test_data)
  confusion_mat_rf <- confusionMatrix(predictions, test_data$popularity_thresholds)
  
  # Get the metrics
  # You need to define the 'evaluate_model' function before using it here
  tuned_rf_metrics <- evaluate_model(predictions, test_data$popularity_thresholds)
  print(tuned_rf_metrics)
  
  return(list(confusion_mat = confusion_mat_rf, metrics = tuned_rf_metrics))
}

#------------------------------------------
# 4.2 RF Classification Tuning
#------------------------------------------
# Grid for hyperparameter tuning
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)
rf_grid <- expand.grid(mtry = c(34,36,38,40)) 


# Tune the hyperparameters
rf_model_tuning <- tune_rf(train_pca_combined, ctrl, rf_grid)

# Get the best mtry
best_mtry <- rf_model_tuning$bestTune$mtry
best_mtry

# Train the final model using the best mtry
final_rf <- train_initial_rf(train_pca_combined, mtry = best_mtry)

# Evaluate the final model
rf_final_eval_results <- evaluate_rf(final_rf, test_pca_combined)

rf_results <- rf_model_tuning$results
ggplot(rf_results, aes(x = mtry, y = Accuracy)) +
  geom_line() +
  geom_point() +
  labs(title = "Random Forest Accuracy for Different mtry values",
       x = "mtry",
       y = "Accuracy") +
  theme_minimal()



#---------------------------------
# 5.1 XGBOOST Classification
#----------------------------------
levels(train_labels) <- 0:(length(unique(train_labels))-1)
train_labels_num <- as.numeric(train_labels) - 1

levels(test_labels) <- 0:(length(unique(test_labels))-1)
test_labels_num <- as.numeric(test_labels) - 1

# Create the xgb.DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(train_pca_combined %>% select(-popularity_thresholds)), label = train_labels_num)
dtest <- xgb.DMatrix(data = as.matrix(test_pca_combined %>% select(-popularity_thresholds)), label = test_labels_num)

# Classification with xgboost
params_cls <- list(
  booster = "gbtree",
  eta = 0.1,
  max_depth = 6,
  objective = "multi:softprob",
  num_class = length(unique(train_labels)),
  eval_metric = "mlogloss"
)

xgb_model_cls <- xgb.train(params = params_cls, data = dtrain, nrounds = 100, watchlist = list(train=dtrain, test=dtest), early_stopping_rounds = 10, verbose = 1)

# Predict on the test set
preds_cls <- predict(xgb_model_cls, dtest)
preds_cls_matrix <- matrix(preds_cls, ncol = length(unique(train_labels)))
predicted_class <- max.col(preds_cls_matrix) - 1
predicted_class


# confusion matrix, accuracy, precision, recall etc

cm_stats <- confusionMatrix(as.factor(predicted_class), as.factor(test_labels_num), mode = "everything")
print(cm_stats)

#-----------------------------------
# 5.2 XGBOOST Classification Tuning
#-----------------------------------

# Hyperparameter tuning, setting the Grid
# Using less than in reportfor marking demonstration putposes
etas <- c(0.1, 0.5)
max_depths <- 4:6
results_list <- list()

for(eta in etas) {
  for(depth in max_depths) {
    params <- list(
      booster = "gbtree",
      eta = eta,
      max_depth = depth,
      objective = "multi:softprob",
      num_class = length(unique(train_labels)),
      eval_metric = "mlogloss"
    )
    model <- xgb.train(params = params, 
                       data = dtrain, 
                       nrounds = 100, 
                       watchlist = list(train=dtrain, test=dtest), 
                       early_stopping_rounds = 10, 
                       verbose = 1)
    evals_result <- model$evaluation_log
    min_mlogloss <- tail(evals_result$test_mlogloss, n=1)
    results_list[[paste("eta", eta, "depth", depth, sep = "_")]] <- list(
      model = model,
      mlogloss = min_mlogloss
    )
  }
}

mlogloss_values <- sapply(results_list, function(x) x$mlogloss)
param_combinations <- names(mlogloss_values)
etas_extracted <- sapply(param_combinations, function(x) as.numeric(strsplit(x, "_")[[1]][2]))
depths_extracted <- sapply(param_combinations, function(x) as.numeric(strsplit(x, "_")[[1]][4]))
plotting_df <- data.frame(ETA = etas_extracted, MaxDepth = depths_extracted, MLogLoss = mlogloss_values)

# PLotting the different hyperparameters so we can visualise the improvements
ggplot(plotting_df, aes(x = MaxDepth, y = MLogLoss, color = factor(ETA))) +
  geom_line() +
  geom_point() +
  labs(title = "MLogLoss for Different Hyperparameters",
       x = "Max Depth",
       y = "MLogLoss",
       color = "ETA") +
  theme_minimal()

# extracting the details for best model
best_params_name <- names(which.min(mlogloss_values))
best_model <- results_list[[best_params_name]]$model

best_preds <- predict(best_model, dtest)
best_preds_matrix <- matrix(best_preds, ncol = length(unique(train_labels)))
best_predicted_class <- max.col(best_preds_matrix) - 1
best_predicted_labels <- factor(best_predicted_class, levels = 0:(length(unique(train_labels))-1), labels = unique(train_labels))

# final metrics for the best XGBoost classifier

xgb_best_metrics <- evaluate_model(best_predicted_labels, test_labels)
xgb_best_metrics




#----------------------------------------------
# 6.1 Comparinging all 3 classification models
#----------------------------------------------

# Create an empty data frame to store the metrics
model_metrics <- data.frame(
  Model = character(),
  accuracy = numeric(),
  precision = numeric(),
  recall = numeric(),
  f1_score = numeric(),
  stringsAsFactors = FALSE
)

# Add these metrics to your model_metrics data frame
model_metrics <- data.frame(Model=character(), Accuracy=numeric(),
                            Precision=numeric(), Recall=numeric(),
                            F1_Score=numeric(), stringsAsFactors=FALSE)

tree_metrics <- evaluate_model(test_predictions, test_data$popularity_thresholds)
new_row <- data.frame(Model="BestDecisionTree", Accuracy=tree_metrics$accuracy, Precision=mean(tree_metrics$precision, na.rm=TRUE), Recall=mean(tree_metrics$recall, na.rm=TRUE), F1_Score=mean(tree_metrics$f1_score, na.rm=TRUE))
model_metrics <- rbind(model_metrics, new_row)

# Add metrics for the final random forest model
rf_metrics <- evaluate_rf(final_rf, test_pca_combined)$metrics
new_row <- data.frame(Model = "BestRandomForest", 
                      Accuracy = rf_metrics$accuracy, 
                      Precision = mean(rf_metrics$precision, na.rm = TRUE), 
                      Recall = mean(rf_metrics$recall, na.rm = TRUE), 
                      F1_Score = mean(rf_metrics$f1_score, na.rm = TRUE))
model_metrics <- rbind(model_metrics, new_row)

# Add metrics for the best XGBoost model
xgb_metrics <- evaluate_model(best_predicted_labels, test_labels)
new_row <- data.frame(Model = "BestXGBoost", 
                      Accuracy = xgb_metrics$accuracy, 
                      Precision = mean(xgb_metrics$precision, na.rm = TRUE), 
                      Recall = mean(xgb_metrics$recall, na.rm = TRUE), 
                      F1_Score = mean(xgb_metrics$f1_score, na.rm = TRUE))
model_metrics <- rbind(model_metrics, new_row)

print(model_metrics)


# Rename the columns to match the metrics
colnames(model_metrics) <- c("Model", "Accuracy", "Precision", "Recall", "F1_Score")
model_metrics
# Convert the data to a longer format
model_metrics_long <- model_metrics %>%
  pivot_longer(cols = -Model, names_to = "metric", values_to = "value")

# Create the bar plot
ggplot(model_metrics_long, aes(x = Model, y = as.numeric(value), fill = metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Comparison", x = "Model", y = "Value") +
  theme_minimal() +
  scale_fill_manual(values = c("Accuracy" = "darkred", "Precision" = "lightblue", "Recall" = "darkgreen", "F1_Score" = "darkorange"),
                    labels = c("Accuracy", "Precision", "Recall", "F1 Score")) +
  guides(fill = guide_legend(title = "Metrics"))


#--------------------------------------
# 7 Setting up for Regression Analysis
# ------------------------------------
# clean the environment
rm(list=ls())

# ----------------------------------
# 7.1 Importing data again
# ----------------------------------
# Set the working directory to the parent directory of the script
current_directory <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(current_directory)

data_folder <- "./"

# Check current wd and list the files
print(getwd())
list.files()

# Re-read the csv file with the Natural language processing results for the news data
file_path <- file.path(data_folder, "scaled_outlier_removed.csv")
if (file.exists(file_path)) {
  scaled_data <- read.csv(file_path, header = TRUE)
} else {
  stop("Error: scaled_outlier_removed.csv file does not exist in the specified directory.")
}


scaled_data

scaled_data <- scaled_data %>% sample_frac(.1)  # Use only 10% of the data for marking

# ---------------------------------------------
# 7.2 Split data and Perform PCA for regression
# ----------------------------------------------

# Split data into training and testing sets before PCA
set.seed(123)
sample_idx <- sample(c(TRUE, FALSE), nrow(scaled_data), replace = TRUE, prob = c(0.8, 0.2))
train_data <- scaled_data[sample_idx, ]
test_data <- scaled_data[!sample_idx, ]

# Perform PCA on training data
train_data_no_shares <- train_data %>% select(-shares)
pca_result_train <- prcomp(train_data_no_shares, scale. = TRUE)
cumulative_variance_train <- cumsum(pca_result_train$sdev^2) / sum(pca_result_train$sdev^2)
n_components_train <- which(cumulative_variance_train >= 0.95)[1]

# Transform train and test data using PCA components
test_data_no_shares <- test_data %>% select(-shares)
test_pca <- predict(pca_result_train, newdata = test_data_no_shares)
train_pca <- predict(pca_result_train, newdata = train_data_no_shares) 

# Add back 'shares' to PCA datasets
train_pca_combined <- data.frame(train_pca, shares = train_data$shares)
test_pca_combined <- data.frame(test_pca, shares = test_data$shares)

# ----------------------------------
# 8. Regression Analysis
# ----------------------------------

# Simple Linear Regression using the first principal component
lm_model <- lm(shares ~ PC1, data=train_pca_combined)
summary(lm_model)

#The simple linear regression model using only the first principal component
#(PC1) as a predictor isn't a good fit for predicting shares. The model 
#accounts for a negligible amount of variance in shares and the predictor
#isn't statistically significant. Therefore we thought it was beneficial to consider more
#predictors and use Multiple Regression Analysis

# ----------------------------------
# 8.1 Multiple Linear Regression
# ----------------------------------
evaluate_regression <- function(model, test_data, actual_values) {
  predictions <- predict(model, newdata = test_data)
  
  residuals <- actual_values - predictions
  
  rmse <- sqrt(mean(residuals^2))
  mae <- mean(abs(residuals))
  r2 <- summary(model)$r.squared
  adj_r2 <- summary(model)$adj.r.squared
  
  return(list(RMSE = rmse, MAE = mae, R2 = r2, AdjustedR2 = adj_r2))
}

# Empty data frame to store results
results_reg_eval <- data.frame(Num_PCs = integer(), RMSE = numeric(),
                               MAE = numeric(), R2 = numeric(), 
                               AdjustedR2 = numeric())

for(i in 1:ncol(train_pca_combined) - 1) {  # -1 to exclude the 'shares' column
  formula_str <- paste("shares ~", paste(names(train_pca_combined)[1:i],
                                         collapse = " + "))
  
  model <- lm(as.formula(formula_str), 
              data = train_pca_combined)
  
  metrics <- evaluate_regression(model, test_pca_combined,
                                 test_pca_combined$shares)
  results_reg_eval <- rbind(results_reg_eval, c(i, metrics$RMSE, 
                                       metrics$MAE, metrics$R2, 
                                       metrics$AdjustedR2))
}

results_reg_eval

# Plotting the results
# Renaming the columns of the results dataframe
colnames(results_reg_eval) <- c("Num_PCs", "RMSE", "MAE", "R2", "AdjustedR2")


# Creates a plot for RMSE and MAE
plot_rmse_mae <- ggplot(results_reg_eval, aes(x = Num_PCs)) +
  geom_line(aes(y = RMSE, color = "RMSE")) +
  geom_line(aes(y = MAE, color = "MAE")) +
  labs(title = "RMSE and MAE vs. Number of PCs",
       y = "Metric Value") +
  theme_minimal() +
  scale_color_manual(values = c("RMSE" = "red", "MAE" = "blue"))

# Creates a plot for R-squared and Adjusted R-squared
plot_r_squared <- ggplot(results_reg_eval, aes(x = Num_PCs)) +
  geom_line(aes(y = R2, color = "R2")) +
  geom_line(aes(y = AdjustedR2, color = "AdjustedR2")) +
  labs(title = "R-squared and Adjusted R-squared vs. Number of PCs",
       y = "Metric Value") +
  theme_minimal() +
  scale_color_manual(values = c("R2" = "green", "AdjustedR2" = "purple"))

# Combine the two plots using patchwork
final_plot <- plot_rmse_mae + plot_r_squared
final_plot


# best number of PCs for multiple linear regression
best_num_pcs <- results_reg_eval[which.min(results_reg_eval$RMSE), "Num_PCs"]
best_num_pcs

# fit best reg model
formula_str_best <- paste("shares ~", paste(names(train_pca_combined)[1:best_num_pcs], collapse = " + "))
best_model_mult_reg <- lm(as.formula(formula_str_best), data = train_pca_combined)

# Metrics for best Polynomial Regression Model
best_mult_reg_metrics <- evaluate_regression(best_model_mult_reg, test_pca_combined, test_pca_combined$shares)
print(best_mult_reg_metrics)

# ---------------------------------------------------
# 8.2 Polynomial Regression Analysis
# ---------------------------------------------------
evaluate_poly_reg <- function(num_pcs, degree, train_data, test_data) {
  formula_str <- paste("shares ~", paste(paste("poly(", names(train_data)[1:num_pcs], ", degree = ", degree, ")", collapse = " + "), sep = ""))
  
  model <- lm(as.formula(formula_str), data = train_data)
  
  predictions <- predict(model, newdata = test_data)
  
  residuals <- test_data$shares - predictions
  
  rmse <- sqrt(mean(residuals^2))
  mae <- mean(abs(residuals))
  r2 <- summary(model)$r.squared
  adj_r2 <- summary(model)$adj.r.squared
  
  return(c(num_pcs, degree, rmse, mae, r2, adj_r2))
}

num_pcs_range <- 35:40  # 1 to 40 PCs
degree_range <- 5:7  # evaluate polynomial degrees 1 to 10

combinations_list <- expand.grid(Num_PCs = num_pcs_range, Degree = degree_range)

results_poly_reg_eval_list <- lapply(1:nrow(combinations_list), function(i) {
  evaluate_poly_reg(as.numeric(combinations_list[i, "Num_PCs"]), 
                      as.numeric(combinations_list[i, "Degree"]), 
                      train_pca_combined, 
                      test_pca_combined)
})

results_poly_reg_eval <- do.call(rbind, results_poly_reg_eval_list)
colnames(results_poly_reg_eval) <- c("Num_PCs", "Degree", "RMSE", "MAE", "R2", "AdjustedR2")

print(results_poly_reg_eval)

# Identify Optimal Model
best_model <- results_poly_reg_eval[which.min(results_poly_reg_eval[,3]),]
print(best_model)

# Fit the best model to the entire training dataset directly
best_formula <- as.formula(paste(
  "shares ~", 
  paste(
    paste("poly(", names(train_pca_combined)[1:as.numeric(best_model[1])], ", degree = ", best_model[2], ")", collapse = " + "),
    sep = ""
  )
))


best_model <- lm(best_formula, data = train_pca_combined)

# Make predictions on the test dataset
test_predictions <- predict(best_model, newdata = test_pca_combined)

# Calculate evaluation metrics on the test data
test_residuals <- test_pca_combined$shares - test_predictions
test_rmse <- sqrt(mean(test_residuals^2))
test_mae <- mean(abs(test_residuals))
test_r2 <- summary(best_model)$r.squared
test_adj_r2 <- summary(best_model)$adj.r.squared

# Print the evaluation metrics on the test data
cat("Test RMSE:", test_rmse, "\n")
cat("Test MAE:", test_mae, "\n")
cat("Test R-squared:", test_r2, "\n")
cat("Test Adjusted R-squared:", test_adj_r2, "\n")


# Visualisation to
# plot RMSE across different polynomial degrees for a fixed number of PCs:
results_poly_reg_eval <- as.data.frame(results_poly_reg_eval)

ggplot(results_poly_reg_eval[results_poly_reg_eval$Num_PCs == 40, ], aes(x = Degree, y = RMSE)) +
  geom_point() +
  geom_line() +
  labs(title = "RMSE for Different Polynomial Degrees with 40 PCs",
       x = "Polynomial Degree",
       y = "RMSE") +
  theme_minimal()


# trying with 7 PCs
ggplot(results_poly_reg_eval[results_poly_reg_eval$Num_PCs == 7, ], aes(x = Degree, y = RMSE)) +
  geom_point() +
  geom_line() +
  labs(title = "RMSE for Different Polynomial Degrees with 10? PCs",
       x = "Polynomial Degree",
       y = "RMSE") +
  theme_minimal()


# Identify Optimal Model
best_model <- results_poly_reg_eval[which.min(results_poly_reg_eval$RMSE),]
print(best_model)



# Heatmap of RMSE
ggplot(results_poly_reg_eval, aes(x = Num_PCs, y = Degree, fill = RMSE)) +
  geom_tile() +
  labs(title = "Heatmap of RMSE",
       x = "Number of Principal Components",
       y = "Polynomial Degree") +
  theme_minimal()

# Heatmap of MAE
ggplot(results_poly_reg_eval, aes(x = Num_PCs, y = Degree, fill = MAE)) +
  geom_tile() +
  labs(title = "Heatmap of MAE",
       x = "Number of Principal Components",
       y = "Polynomial Degree") +
  theme_minimal()

# Heatmap of R2
ggplot(results_poly_reg_eval, aes(x = Num_PCs, y = Degree, fill = R2)) +
  geom_tile() +
  labs(title = "Heatmap of R2",
       x = "Number of Principal Components",
       y = "Polynomial Degree") +
  theme_minimal()

# Heatmap of Adjusted R2
ggplot(results_poly_reg_eval, aes(x = Num_PCs, y = Degree, fill = AdjustedR2)) +
  geom_tile() +
  labs(title = "Heatmap of Adjusted R2",
       x = "Number of Principal Components",
       y = "Polynomial Degree") +
  theme_minimal()



# ------------------------------------
# 8.3 XGBoosted Regression
# ------------------------------------

# Convert data to xgb.DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(train_pca_combined[, -ncol(train_pca_combined)]), label = train_pca_combined$shares)
dtest <- xgb.DMatrix(data = as.matrix(test_pca_combined[, -ncol(test_pca_combined)]), label = test_pca_combined$shares)

# Define a function to evaluate the regression model
evaluate_xgboost_reg <- function(model, dtest, test_data) {
  preds <- predict(model, dtest)
  residuals <- test_data$shares - preds
  rmse <- sqrt(mean(residuals^2))
  mae <- mean(abs(residuals))
  
  r2 <- 1 - (sum(residuals^2) / sum((test_data$shares - mean(test_data$shares))^2))
  adj_r2 <- 1 - ((1 - r2) * ((nrow(test_data) - 1) / (nrow(test_data) - ncol(test_data) + 1)))
  
  return(list(RMSE = rmse, MAE = mae, R2 = r2, AdjustedR2 = adj_r2))
}

# Initial Regression with xgboost
params <- list(
  booster = "gbtree",
  eta = 0.1,
  max_depth = 6,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

xgb_model_reg <- xgb.train(params = params, 
                           data = dtrain, 
                           nrounds = 100, 
                           watchlist = list(train=dtrain, test=dtest), 
                           early_stopping_rounds = 10, 
                           verbose = 1)

# Get all evaluation metrics for the first run
first_run_metrics <- evaluate_xgboost_reg(xgb_model_reg, dtest, test_pca_combined)
print(first_run_metrics)

# Hyperparameter tuning through grid search
# Define hyperparameter grid for random search
# Again grid search is limited for demonstration purposes for marking
param_grid <- list(
  eta = c(0.3, 0.5),
  max_depth = c( 6, 7),
  min_child_weight = c( 5, 7),
  subsample = c( 0.9, 1),
  colsample_bytree = c(0.7, 1)
)

# Random search function
random_search <- function(param_grid, n_iter = 50) {
  results <- list()
  for(i in 1:n_iter) {
    params <- list(
      booster = "gbtree",
      eta = sample(param_grid$eta, 1),
      max_depth = sample(param_grid$max_depth, 1),
      min_child_weight = sample(param_grid$min_child_weight, 1),
      subsample = sample(param_grid$subsample, 1),
      colsample_bytree = sample(param_grid$colsample_bytree, 1),
      objective = "reg:squarederror",
      eval_metric = "rmse"
    )
    res <- tryCatch({
      model <- xgb.train(params = params, 
                         data = dtrain, 
                         nrounds = 100, 
                         watchlist = list(train=dtrain, test=dtest), 
                         early_stopping_rounds = 10, 
                         verbose = 0)
      list(success = TRUE, params = params, best_iteration = model$best_ntreelimit, best_score = model$best_score)
    }, error = function(e) {
      list(success = FALSE, message = e$message)
    })
    results[[i]] <- res
  }
  return(results)
}


# Perform random search
set.seed(123)
search_results <- random_search(param_grid, n_iter = 50)



# Extract best model parameters
best_model_info <- search_results[[which.min(sapply(search_results, function(x) x$best_score))]]
best_params <- best_model_info$params
best_iteration <- best_model_info$best_iteration

# Train best model on full data
best_model <- xgb.train(params = best_params, 
                        data = dtrain, 
                        nrounds = best_iteration, 
                        watchlist = list(train=dtrain, test=dtest))

# Evaluate best model

best_model_metrics <- evaluate_xgboost_reg(best_model, dtest, test_pca_combined)
print(best_model_metrics)



# Extract RMSE values from the evaluation log of the best model
train_rmse_best <- best_model$evaluation_log$train_rmse
test_rmse_best <- best_model$evaluation_log$test_rmse

#Data frame with the extracted RMSEs
rmse_data_best <- data.frame(
  Iteration = 1:length(train_rmse_best),
  Train_RMSE = train_rmse_best,
  Test_RMSE = test_rmse_best
)

# Plot RMSE using ggplot2
ggplot(rmse_data_best, aes(x = Iteration)) +
  geom_line(aes(y = Train_RMSE, color = "Train RMSE")) +
  geom_line(aes(y = Test_RMSE, color = "Test RMSE")) +
  labs(title = "RMSE over Iterations for Best XGBoost Model",
       y = "RMSE",
       x = "Iteration",
       color = "Dataset") +
  theme_minimal()

# residual analysis

# Compute the residuals
preds_best_model <- predict(best_model, dtest)

residuals <- test_pca_combined$shares - preds_best_model

# Plot residuals vs predicted values
plot(preds_best_model, residuals, main="Residuals vs Fitted", xlab="Fitted values", ylab="Residuals", pch=20)
abline(h = 0, col = "red") # Adds a horizontal line at y=0 for reference

# Scatter plot of predicted vs actual values
plot(test_pca_combined$shares, preds_best_model, main="Actual vs Predicted", xlab="Actual values", ylab="Predicted values", pch=20)
abline(a=0, b=1, col="red") # Adds a y=x reference line

# Histogram of residuals
hist(residuals, main="Histogram of Residuals", xlab="Residuals", breaks=30)

# QQ-plot of residuals
qqnorm(residuals, main="QQ-plot of Residuals")
qqline(residuals)

range(train_pca_combined$shares)


#----------------------------------------------------------------
# 9.1 New attempt using non preprocessed Data
#---------------------------------------------------------------

# Load data
data_folder <- "./"

# Set the working directory to the parent directory of the script
current_directory <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(current_directory)

# Check current wd and list the files
print(getwd())
list.files()

# Read the csv file for new data
file_path <- file.path(data_folder, "new_data_attempt.csv")
if (file.exists(file_path)) {
  new_data <- read.csv(file_path, header = TRUE)
} else {
  stop("Error: new_data_attempt.csv file does not exist in the specified directory.")
}

summary(new_data)

new_data <- new_data %>% sample_frac(.1)  # Use only 10% of the data

#---------------------------------------------------
# 10.1 Scaling the data - Min-Max scaling
# -------------------------------------------------

# Create a function to perform Min-Max scaling
min_max_scaling <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

# Apply Min-Max scaling to the selected columns
scaled_data <- new_data

# Define a threshold for Z-scores (e.g., 3 standard deviations)
z_score_threshold <- 3

# Calculate Z-scores for numeric columns (excluding shares and categorical variables)
numeric_columns <- sapply(new_data, is.numeric)
columns_to_scale <- setdiff(names(new_data)[numeric_columns], c("shares", "data_channel_is_lifestyle", 
                                                                "data_channel_is_entertainment", 
                                                                "data_channel_is_bus", "data_channel_is_socmed",
                                                                "data_channel_is_tech", "data_channel_is_world",
                                                                "weekday_is_monday","weekday_is_tuesday",
                                                                "weekday_is_wednesday","weekday_is_thursday",
                                                                "weekday_is_friday","weekday_is_saturday","weekday_is_sunday"))

# Apply Min-Max scaling to the selected columns
for (col_name in columns_to_scale) {
  scaled_data[[col_name]] <- min_max_scaling(new_data[[col_name]])
}

# Calculates the Z-scores for the scaled columns
scaled_z_scores <- scale(scaled_data[, columns_to_scale])

# Identifies outliers based on Z-scores
scaled_outliers <- abs(scaled_z_scores) > z_score_threshold

# Remove outliers entirely from the scaled data
scaled_data <- scaled_data[!apply(scaled_outliers, 1, any), ]

# Summary of the data after handling outliers and scaling
summary(scaled_data)


#---------------------------------------------------------------------------------------
# 10.2 logarithmically transform shares as 75th percentile value much less than highest value
#--------------------------------------------------------------------------------------

scaled_data$shares <- log(scaled_data$shares)

summary(scaled_data)


regression_data <- scaled_data
summary(regression_data)

# ----------------------------------
# 11.1 Split data and Perform PCA
# ----------------------------------

# Split data into training and testing sets before PCA
set.seed(123)
sample_idx <- sample(c(TRUE, FALSE), nrow(regression_data), replace = TRUE, prob = c(0.8, 0.2))
train_data <- regression_data[sample_idx, ]
test_data <- regression_data[!sample_idx, ]

# Perform PCA on training data
train_data_no_shares <- train_data %>% select(-shares)
pca_result_train <- prcomp(train_data_no_shares, scale. = TRUE)
cumulative_variance_train <- cumsum(pca_result_train$sdev^2) / sum(pca_result_train$sdev^2)
n_components_train <- which(cumulative_variance_train >= 0.95)[1]

# Transform train and test data using PCA components
test_data_no_shares <- test_data %>% select(-shares)
test_pca <- predict(pca_result_train, newdata = test_data_no_shares)
train_pca <- predict(pca_result_train, newdata = train_data_no_shares) 

# Add back 'shares' to PCA datasets
train_pca_combined <- data.frame(train_pca, shares = train_data$shares)
test_pca_combined <- data.frame(test_pca, shares = test_data$shares)
# ------------------------------------
# 12. New Boosted Regression with new data
# ------------------------------------

# Convert data to xgb.DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(train_pca_combined[, -ncol(train_pca_combined)]), label = train_pca_combined$shares)
dtest <- xgb.DMatrix(data = as.matrix(test_pca_combined[, -ncol(test_pca_combined)]), label = test_pca_combined$shares)

# Define a function to evaluate the regression model
evaluate_xgboost_reg <- function(model, dtest, test_data) {
  preds <- predict(model, dtest)
  residuals <- test_data$shares - preds
  rmse <- sqrt(mean(residuals^2))
  mae <- mean(abs(residuals))
  
  r2 <- 1 - (sum(residuals^2) / sum((test_data$shares - mean(test_data$shares))^2))
  adj_r2 <- 1 - ((1 - r2) * ((nrow(test_data) - 1) / (nrow(test_data) - ncol(test_data) + 1)))
  
  return(list(RMSE = rmse, MAE = mae, R2 = r2, AdjustedR2 = adj_r2))
}

# Initial Regression with xgboost
params <- list(
  booster = "gbtree",
  eta = 0.1,
  max_depth = 6,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

xgb_model_reg <- xgb.train(params = params, 
                           data = dtrain, 
                           nrounds = 100, 
                           watchlist = list(train=dtrain, test=dtest), 
                           early_stopping_rounds = 10, 
                           verbose = 1)

# Get all evaluation metrics for the first run
first_run_metrics <- evaluate_xgboost_reg(xgb_model_reg, dtest, test_pca_combined)
print(first_run_metrics)

# Hyperparameter tuning through grid search


# Define hyperparameter grid for random search
#
# Again, less grid used for demonstration purposes

param_grid <- list(
  eta = c(0.3, 0.5),
  max_depth = c( 7, 8),
  min_child_weight = c(5, 7),
  subsample = c(0.9, 1),
  colsample_bytree = c(0.9, 1)
)

# Random search function
random_search <- function(param_grid, n_iter = 50) {
  results <- list()
  for(i in 1:n_iter) {
    params <- list(
      booster = "gbtree",
      eta = sample(param_grid$eta, 1),
      max_depth = sample(param_grid$max_depth, 1),
      min_child_weight = sample(param_grid$min_child_weight, 1),
      subsample = sample(param_grid$subsample, 1),
      colsample_bytree = sample(param_grid$colsample_bytree, 1),
      objective = "reg:squarederror",
      eval_metric = "rmse"
    )
    res <- tryCatch({
      model <- xgb.train(params = params, 
                         data = dtrain, 
                         nrounds = 100, 
                         watchlist = list(train=dtrain, test=dtest), 
                         early_stopping_rounds = 10, 
                         verbose = 0)
      list(success = TRUE, params = params, best_iteration = model$best_ntreelimit, best_score = model$best_score)
    }, error = function(e) {
      list(success = FALSE, message = e$message)
    })
    results[[i]] <- res
  }
  return(results)
}


# Perform random search
set.seed(123)
search_results <- random_search(param_grid, n_iter = 50)



# Extract best model parameters
best_model_info <- search_results[[which.min(sapply(search_results, function(x) x$best_score))]]
best_params <- best_model_info$params
best_iteration <- best_model_info$best_iteration

# Train best model on full data
best_model <- xgb.train(params = best_params, 
                        data = dtrain, 
                        nrounds = best_iteration, 
                        watchlist = list(train=dtrain, test=dtest))

# Evaluate best model

best_model_metrics <- evaluate_xgboost_reg(best_model, dtest, test_pca_combined)
print(best_model_metrics)



# Extract RMSE values from the evaluation log of the best model
train_rmse_best <- best_model$evaluation_log$train_rmse
test_rmse_best <- best_model$evaluation_log$test_rmse

# Create a data frame with the extracted RMSEs
rmse_data_best <- data.frame(
  Iteration = 1:length(train_rmse_best),
  Train_RMSE = train_rmse_best,
  Test_RMSE = test_rmse_best
)

# Plot RMSE using ggplot2
ggplot(rmse_data_best, aes(x = Iteration)) +
  geom_line(aes(y = Train_RMSE, color = "Train RMSE")) +
  geom_line(aes(y = Test_RMSE, color = "Test RMSE")) +
  labs(title = "RMSE over Iterations for Best XGBoost Model",
       y = "RMSE",
       x = "Iteration",
       color = "Dataset") +
  theme_minimal()

# residual analysis

# Compute the residuals
preds_best_model <- predict(best_model, dtest)

residuals <- test_pca_combined$shares - preds_best_model

diagnostic_data <- data.frame(
  Residuals = residuals,
  Predicted = preds_best_model,
  Actual = test_pca_combined$shares
)

# Write the data frame to a CSV file
write.csv(diagnostic_data, "log_diagnostic_data.csv", row.names = FALSE)
# Plot residuals vs predicted values
plot(preds_best_model, residuals, main="Residuals vs Fitted", xlab="Fitted values", ylab="Residuals", pch=20)
abline(h = 0, col = "red") # Adds a horizontal line at y=0 for reference

# Scatter plot of predicted vs actual values
plot(test_pca_combined$shares, preds_best_model, main="Actual vs Predicted", xlab="Actual values", ylab="Predicted values", pch=20)
abline(a=0, b=1, col="red") # Adds a y=x reference line

# Histogram of residuals
hist(residuals, main="Histogram of Residuals", xlab="Residuals", breaks=30)

# QQ-plot of residuals
qqnorm(residuals, main="QQ-plot of Residuals")
qqline(residuals)

range(train_pca_combined$shares)

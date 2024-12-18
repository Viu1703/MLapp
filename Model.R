# Libraries ----
library(caret)
library(randomForest)

# Load and preprocess the dataset ----
# Update the file path as needed for your local directory
ds <- read.csv("Alzheimer's disease processed dataset.csv")  
ds$PatientID <- NULL
ds$DoctorInCharge <- NULL

# Convert Diagnosis to factor
ds$Diagnosis <- as.factor(ds$Diagnosis)

# Define the selected features based on your requirement ----
selected_features <- c("MemoryComplaints", "Forgetfulness", "Disorientation", 
                       "Confusion", "Depression", "FamilyHistoryAlzheimers", 
                       "Age", "BehavioralProblems", 
                       "CardiovascularDisease", "PhysicalActivity", "Diagnosis")  # Include Diagnosis for training

# Subset dataset with selected features
ds <- ds[, selected_features]

# Split the dataset ----
set.seed(123)
trainIndex <- createDataPartition(ds$Diagnosis, p = 0.7, list = FALSE)
train_data <- ds[trainIndex, ]
test_data <- ds[-trainIndex, ]

# Train the initial Random Forest model ----
rf_model <- randomForest(Diagnosis ~ ., data = train_data, ntree = 100, mtry = 3)

# Retraining with cross-validation and hyperparameter tuning ----
set.seed(123)
train_control_rf <- trainControl(method = "cv", number = 10)  # Cross-validation setup

# Tuning grid for mtry parameter
tune_grid_rf <- expand.grid(mtry = seq(1, length(selected_features) - 1, by = 1))

# Train a tuned Random Forest model using caret
tuned_rf_model <- train(Diagnosis ~ ., data = train_data, method = "rf", 
                        trControl = train_control_rf, tuneGrid = tune_grid_rf)

# Display best hyperparameter value for RF
print(tuned_rf_model$bestTune)

# Predicting on test data with tuned RF model
rf_predictions_tuned <- predict(tuned_rf_model, test_data)
conf_matrix_rf_tuned <- confusionMatrix(rf_predictions_tuned, test_data$Diagnosis)

cat("\nConfusion Matrix for Tuned RF:\n")
print(conf_matrix_rf_tuned)

# Save the tuned model as an .rds file in the output directory ----
# Adjust the path as needed to save in your desired directory
saveRDS(tuned_rf_model, "rf_model_tuned4.rds")  

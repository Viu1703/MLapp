import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

# Load and preprocess the dataset
# Update the file path as needed for your local directory
ds = pd.read_csv("alz9dataset.csv")

# Print out the columns to check their names
print(ds.columns)

# Drop unnecessary columns if they exist
columns_to_drop = ['PatientID', 'DoctorInCharge']
ds.drop(columns=[col for col in columns_to_drop if col in ds.columns], inplace=True)

# Convert Diagnosis to category
ds['Diagnosis'] = ds['Diagnosis'].astype('category')

# Define selected features based on your requirement
selected_features = [
    "MemoryComplaints", "Forgetfulness", "Disorientation", 
    "Confusion", "Depression", "FamilyHistoryAlzheimers", 
    "Age", "BehavioralProblems", 
    "CardiovascularDisease", "PhysicalActivity", "Diagnosis"
]

# Subset dataset with selected features
ds = ds[selected_features]

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(ds, test_size=0.3, random_state=123, stratify=ds['Diagnosis'])

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_features=3, random_state=123)
rf_model.fit(train_data.drop(columns='Diagnosis'), train_data['Diagnosis'])

# Predicting on test data with tuned RF model
rf_predictions_tuned = rf_model.predict(test_data.drop(columns='Diagnosis'))
conf_matrix_rf_tuned = confusion_matrix(test_data['Diagnosis'], rf_predictions_tuned)

print("\nConfusion Matrix for Tuned RF:\n")
print(conf_matrix_rf_tuned)

# Save the tuned model as a .pkl file in the output directory
joblib.dump(rf_model, "rf_model_tuned5.pkl")

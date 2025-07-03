# ==============================================================================
# SCRIPT: train_models.py
# PURPOSE: To train, evaluate, and save three different regression models
#          for predicting 'Total_Time_to_Compliance'.
#
# MODELS TRAINED:
#   1. RandomForestRegressor
#   2. GradientBoostingRegressor
#   3. XGBRegressor (XGBoost)
#
# GENERATED FILES:
#   1. random_forest_model.joblib: The saved RandomForest model.
#   2. gradient_boosting_model.joblib: The saved GradientBoosting model.
#   3. xgboost_model.joblib: The saved XGBoost model.
# ==============================================================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. CONFIGURATION ---
DATASET_PATH = 'data/ml_dataset_final.csv'
TARGET_COLUMN = 'Total_Time_to_Compliance'

# Columns to drop before training.
# 'Company_ID' is just an identifier, and 'Optimal_Path' is a string we can't use as a feature.
COLUMNS_TO_DROP = ['Company_ID', 'Optimal_Path', TARGET_COLUMN]

# --- 2. LOAD AND PREPARE DATA ---
print("--- Phase 1: Loading and Preparing Data ---")
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: The dataset file '{DATASET_PATH}' was not found.")
    print("Please make sure you have run 'build_final_dataset.py' first.")
    exit()

# Separate features (X) from the target (y)
X = df.drop(columns=COLUMNS_TO_DROP)
y = df[TARGET_COLUMN]

# Split data into training and testing sets
# We use 80% of the data for training and 20% for testing.
# `random_state` ensures we get the same split every time we run the script, for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data loaded successfully. Training with {len(X_train)} samples, testing with {len(X_test)} samples.\n")


# --- 3. DEFINE AND TRAIN MODELS ---
print("--- Phase 2: Defining and Training Models ---")

# We will define our three models here.
# The parameters (n_estimators, max_depth, etc.) are common starting points.
# For a real-world project, these would be fine-tuned using techniques like GridSearchCV.
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# This dictionary will store the performance metrics for each model
results = {}

for name, model in models.items():
    print(f"Training {name} model...")
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the unseen test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store the results
    results[name] = {'MAE': mae, 'R2 Score': r2}
    
    print(f"{name} model trained.")
    print(f"  - Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"  - R2 Score: {r2:.2f}\n")
    
    # Save the trained model to its own file
    output_filename = f"{name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, output_filename)
    print(f"  - Model saved to '{output_filename}'\n")

# --- 4. DISPLAY FINAL RESULTS ---
print("--- Phase 3: Final Results Summary ---")
results_df = pd.DataFrame(results).T # Transpose to have models as rows
print(results_df.sort_values(by='MAE')) # Sort by MAE to see the best performing model at the top
print("\n--- Model Training Complete ---")
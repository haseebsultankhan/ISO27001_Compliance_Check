# ==============================================================================
# SCRIPT: train_models.py (v2 - Expanded Model Suite)
# PURPOSE: To train, evaluate, and save a wide variety of regression models
#          to find the best performer for our task.
# ==============================================================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Import a wider range of models ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# --- Import tools for preprocessing ---
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- 1. CONFIGURATION ---
DATASET_PATH = 'data/ml_dataset_final.csv'
TARGET_COLUMN = 'Total_Time_to_Compliance'
COLUMNS_TO_DROP = ['Company_ID', 'Optimal_Path', TARGET_COLUMN]

# --- 2. LOAD AND PREPARE DATA ---
print("--- Phase 1: Loading and Preparing the NEW Data ---")
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: The dataset file '{DATASET_PATH}' was not found.")
    print("Please make sure you have run 'build_final_dataset.py' first.")
    exit()

X = df.drop(columns=COLUMNS_TO_DROP)
y = df[TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data loaded successfully. Training with {len(X_train)} samples, testing with {len(X_test)} samples.\n")


# --- 3. DEFINE AND TRAIN EXPANDED SUITE OF MODELS ---
print("--- Phase 2: Defining and Training Expanded Suite of Models ---")

# For models like SVR that are sensitive to feature scaling, we create a pipeline.
# This pipeline first scales the data and then applies the model.
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()), # Step 1: Scale the features
    ('svr', SVR())                # Step 2: Apply Support Vector Regressor
])

# Define our expanded list of models
models = {
    # Classic Linear Models
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42),
    
    # Tree-based Models
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    
    # Support Vector Machine Model
    "SVR": svr_pipeline
}

results = {}

for name, model in models.items():
    print(f"--- Training {name} model ---")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'R2 Score': r2}
    
    print(f"{name} model trained.")
    print(f"  - Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"  - R2 Score: {r2:.2f}")
    
    output_filename = f"{name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, output_filename)
    print(f"  - Model saved to '{output_filename}'\n")

# --- 4. DISPLAY FINAL RESULTS ---
print("--- Phase 3: Final Results Summary ---")
results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='MAE'))
print("\n--- All Model Training Complete ---")

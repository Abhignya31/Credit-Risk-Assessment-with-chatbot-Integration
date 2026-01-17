# train_model.py

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

# --- Configuration ---
DATASET_PATH = 'dataset/cs-training.csv' 
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Models will be saved to: {MODELS_DIR}")

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATASET_PATH, index_col=0) 
except FileNotFoundError:
    print(f"ERROR: Training data not found at {DATASET_PATH}. Please ensure the file is present.")
    exit()

# --- 2. Preprocessing & Feature Engineering ---
# Handle missing values 
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df.fillna(0, inplace=True)

# Define the 12 features that the model will be trained on
TRAINING_FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents'
]

# Add two engineered features
df['debt_to_income'] = df['MonthlyIncome'] / (df['DebtRatio'] + 1e-6)
df['age_to_credit'] = df['age'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1e-6)
TRAINING_FEATURES.extend(['debt_to_income', 'age_to_credit']) # 12 features total

TARGET = 'SeriousDlqin2yrs'
X = df[TRAINING_FEATURES]
y = df[TARGET]

# --- 3. Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=TRAINING_FEATURES) 

# --- 4. Model Training (Advanced ML) ---
# Calculate the imbalance ratio (to fix "wrong prediction" / extreme scores)
ratio = y.value_counts()[0] / y.value_counts()[1] 

model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42,
    n_estimators=150, 
    max_depth=5,
    scale_pos_weight=ratio # Crucial for handling class imbalance
)
model.fit(X_scaled_df, y)

print("--- Model Training Complete ---")

# --- 5. Create ROBUST SHAP Explainer ---
# Use shap.Explainer for maximum compatibility when saving/loading.
explainer = shap.Explainer(model, X_scaled_df) 
print("SHAP Explainer created using the unified syntax.")

# --- 6. Save Artifacts (Pickle) ---
MODELS_DIR_PATH = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR_PATH, exist_ok=True)
model_path = os.path.join(MODELS_DIR_PATH, 'xgb_model.pkl')
scaler_path = os.path.join(MODELS_DIR_PATH, 'scaler.pkl')
explainer_path = os.path.join(MODELS_DIR_PATH, 'shap_explainer.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_path}")

with open(explainer_path, 'wb') as f:
    pickle.dump(explainer, f)
print(f"SHAP Explainer saved to {explainer_path}")

print("\nSUCCESS: All required files have been generated.")
# CreditRisk_WebApp_Pro
Advanced Credit Risk Assessment Web App with Virtual Loan Assistant, document upload, insurance options, and SHAP explainability.

## How to run
1. Create and activate a Python venv (Python 3.8+ recommended):
   ```bash
   python -m venv credit-risk-env
   credit-risk-env\Scripts\activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your trained model and scaler in `models/` as `xgb_model.pkl` and `scaler.pkl`. If not present, the app will use a safe heuristic fallback.
4. Put Kaggle dataset `cs-training.csv` in `dataset/` if you want to (re)train a model.
5. Run the app:
   ```bash
   streamlit run app/main.py
   ```

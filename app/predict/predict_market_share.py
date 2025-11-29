# app/api/predict.py
import numpy as np
import joblib

# โหลดโมเดล XGBoost
xgb_model = joblib.load("models/xgb_model.pkl")

def predict_market_share(price: float, marketing_budget: float, competitors_share: float) -> float:
    X = np.array([[price, marketing_budget, competitors_share]])
    pred = xgb_model.predict(X)[0]
    return float(pred)

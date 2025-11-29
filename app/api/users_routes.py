import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

router = APIRouter()

# -------------------------
# โหลดโมเดล XGBoost ด้วย joblib
# -------------------------
xgb_model = joblib.load("models/xgb_model.pkl")

# -------------------------
# Market Share Prediction API
# -------------------------
class MarketShareInput(BaseModel):
    price: float
    marketing_budget: float
    competitors_share: float

class MarketShareResult(BaseModel):
    predicted_share: float

@router.post("/market-share/predict", response_model=MarketShareResult)
def predict_market_share(input_data: MarketShareInput):
    X = np.array([[input_data.price, input_data.marketing_budget, input_data.competitors_share]])
    pred = xgb_model.predict(X)[0]
    return {"predicted_share": float(pred)}

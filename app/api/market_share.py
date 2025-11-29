# app/api/market_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from predict.predict_market_share import predict_market_share

router = APIRouter()

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
def predict_endpoint(input_data: MarketShareInput):
    try:
        pred = predict_market_share(
            price=input_data.price,
            marketing_budget=input_data.marketing_budget,
            competitors_share=input_data.competitors_share
        )
        return {"predicted_share": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

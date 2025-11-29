from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from predict.predict_ev import predict_ev

router = APIRouter()

# -------------------------
# EV Prediction API
# -------------------------
class EVInput(BaseModel):
    city: str
    fixed_price: float
    charging_amount: float

class PredictionResult(BaseModel):
    prediction: float

@router.post("/predict", response_model=PredictionResult)
def predict_endpoint(ev_data: EVInput):
    try:
        pred = predict_ev(ev_data.city, ev_data.fixed_price, ev_data.charging_amount)
        return {"prediction": pred}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

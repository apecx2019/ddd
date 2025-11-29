import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
# ---------------------------
# โหลดโมเดล
# ---------------------------
lin_model = joblib.load("linear_regression_model.pkl")

# ---------------------------
# Dictionary เมือง
# ---------------------------
city_dict = {
    "Bangkok": "city_Bangkok",
    "Chiang Mai": "city_Chiang Mai",
    "Hat Yai": "city_Hat Yai",
    "Khon Kaen": "city_Khon Kaen",
    "Nakhon Ratchasima": "city_Nakhon Ratchasima",
    "Pattaya": "city_Pattaya",
    "Phuket": "city_Phuket",
    "Surat Thani": "city_Surat Thani",
    "Ubon Ratchathani": "city_Ubon Ratchathani",
    "Udon Thani": "city_Udon Thani"
}

# ---------------------------
# Dictionary seasons
# ---------------------------
season_dict = {
    1: "season_Q1", 2: "season_Q1", 3: "season_Q1",
    4: "season_Q2", 5: "season_Q2", 6: "season_Q2",
    7: "season_Q3", 8: "season_Q3", 9: "season_Q3",
    10: "season_Q4", 11: "season_Q4", 12: "season_Q4"
}

# ---------------------------
# ค่าเฉลี่ย feature
# ---------------------------
avg_values = {
    "ev_market_share": 0.362475,
    "avg_ev_price": 48498.552858,
    "avg_ice_price": 25150.850950,
    "gasoline_price": 4.601550,
    "electricity_price": 0.168725,
    "public_charging_points": 1052.120833,
    "fast_charging_points": 140.235000,
    "ev_subsidy_amount": 2152.947833,
    "purchase_subsidy_active": 0.733333,
    "gdp_per_capita": 24057.330208,
    "unemployment_rate": 4.010900,
    "avg_ev_range_km": 299.414167,
    "ev_sales_last_month": 7431.429167,
    "ev_sales_last_year_same_month": 6749.874167
}

# ---------------------------
# Columns ของโมเดล
# ---------------------------
feature_cols = [
    'ev_market_share', 'avg_ev_price', 'avg_ice_price', 'gasoline_price',
    'electricity_price', 'public_charging_points', 'fast_charging_points',
    'ev_subsidy_amount', 'purchase_subsidy_active', 'gdp_per_capita',
    'unemployment_rate', 'avg_ev_range_km', 'ev_sales_last_month',
    'ev_sales_last_year_same_month', 'city_Bangkok', 'city_Chiang Mai',
    'city_Hat Yai', 'city_Khon Kaen', 'city_Nakhon Ratchasima',
    'city_Pattaya', 'city_Phuket', 'city_Surat Thani',
    'city_Ubon Ratchathani', 'city_Udon Thani', 'season_Q1', 'season_Q2',
    'season_Q3', 'season_Q4'
]

# ---------------------------
# เตรียม feature
# ---------------------------
def prepare_features(city: str, month: int) -> pd.DataFrame:
    input_df = pd.DataFrame([avg_values])

    # one-hot city
    for city_name, col_name in city_dict.items():
        input_df[col_name] = 1 if city_name.lower() == city.lower() else 0

    # one-hot season
    season_col = season_dict.get(month, "season_Q1")
    for q in ["season_Q1", "season_Q2", "season_Q3", "season_Q4"]:
        input_df[q] = 1 if q == season_col else 0

    # fill missing columns
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    final_df = input_df[feature_cols]
    return final_df

# ---------------------------
# ฟังก์ชันทำนาย
# ---------------------------
def predict_ev(city: str, month: int) -> float:
    X = prepare_features(city, month)
    prediction = lin_model.predict(X)
    return float(prediction[0])

# ---------------------------
# Request Model (รับเดือนจากผู้ใช้)
# ---------------------------
class PredictRequest(BaseModel):
    city: str
    month: int   # <<<<<< user ส่งเดือนมาเอง

# ---------------------------
# API Endpoint
# ---------------------------
@router.post("/predict")
async def predict(request: PredictRequest):

    city = request.city
    month = request.month

    # validate city
    if city not in city_dict:
        raise HTTPException(status_code=400, detail=f"City '{city}' is not supported.")

    # validate month
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")

    prediction = predict_ev(city, month)

    return {
        "city": city,
        "month": month,
        "season": season_dict[month],
        "predicted_ev_sales": prediction
    }

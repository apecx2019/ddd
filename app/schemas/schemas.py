from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    city: str
    year: int
    months: List[int] | None = None
    avg_ev_price: float | None = None
    gasoline_price: float | None = None
    public_charging_points: int | None = None
    ev_market_share: float | None = None

class SalesResponse(BaseModel):
    months: List[int]
    predicted_sales: List[float]

class MarketShareResponse(BaseModel):
    months: List[int]
    predicted_market_share: List[float]

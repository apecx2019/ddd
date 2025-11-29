from fastapi import FastAPI
from api.ev_routes import router as ev_router
from api.market_share import router as market_share

app = FastAPI(title="My FastAPI Project")

# Register router
app.include_router(market_share, prefix="/api/market_share", tags=["market_share"])
app.include_router(ev_router, prefix="/api/ev_router", tags=["EV Charging"])

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI!"}

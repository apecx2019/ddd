from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.ev_routes import router as ev_router
from api.market_share import router as market_share

app = FastAPI(title="My FastAPI Project")

# ---------------------------
# CORS configuration
# ---------------------------
origins = [
    "http://localhost:8080",  # frontend ที่รันบน port 8080
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # อนุญาต origin
    allow_credentials=True,      # อนุญาต cookies/auth headers
    allow_methods=["*"],         # อนุญาตทุก HTTP method (GET, POST, etc.)
    allow_headers=["*"],         # อนุญาตทุก header
)

# ---------------------------
# Register routers
# ---------------------------
app.include_router(market_share, prefix="/api/market_share", tags=["market_share"])
app.include_router(ev_router, prefix="/api/ev_router", tags=["EV Charging"])

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI!"}

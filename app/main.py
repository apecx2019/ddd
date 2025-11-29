from fastapi import FastAPI
from api.ev_routes import router as products_router

app = FastAPI(title="My FastAPI Project")

app.include_router(products_router, prefix="/api/products", tags=["Products"])

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI!"}

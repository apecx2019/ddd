# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# import numpy as np
# import os
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(BASE_DIR, "data", "th_ev_sales_smart_city_cleaned.csv")
# SALES_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_sales_log.pkl")
# RF_SALES_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_sales_log.pkl")
# BEST_MODEL_INFO_PATH = os.path.join(BASE_DIR, "models", "best_sales_model_info.pkl")
# MKT_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_mktshare.pkl")

# os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# app = FastAPI(title="EV Forecast API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8080", "http://localhost:5173", "*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------
# # Data preprocessing
# # -----------------------
# def preprocess(df: pd.DataFrame):
#     df = df.copy()
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df = df.dropna(subset=['date'])
#     df['year'] = df['date'].dt.year
#     df['month'] = df['date'].dt.month
#     if 'season' in df.columns:
#         season_map = {'Q1':1,'Q2':2,'Q3':3,'Q4':4}
#         df['season'] = df['season'].map(season_map).fillna(((df['month']-1)//3)+1)
#     else:
#         df['season'] = ((df['month']-1)//3)+1
#     for col in ['ev_sales','ev_market_share','avg_ev_price','gasoline_price','public_charging_points']:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
#     return df

# # Load CSV
# if os.path.exists(DATA_PATH):
#     raw_df = pd.read_csv(DATA_PATH)
#     df = preprocess(raw_df)
# else:
#     print(f"Data file not found at {DATA_PATH}. Endpoints will error until file is provided.")
#     df = pd.DataFrame()

# # City one-hot columns
# CITY_COLS = []
# if not df.empty:
#     cities = sorted(df['city'].dropna().unique().tolist())
#     CITY_COLS = [f"city_{c}" for c in cities]

# FEATURE_COLS = ['year','month','season','ev_market_share','avg_ev_price','gasoline_price','public_charging_points'] + CITY_COLS

# def build_row(city, year, month, ev_market_share, avg_ev_price, gasoline_price, public_charging_points):
#     row = {
#         'year': int(year),
#         'month': int(month),
#         'season': int(((int(month)-1)//3)+1),
#         'ev_market_share': float(ev_market_share),
#         'avg_ev_price': float(avg_ev_price),
#         'gasoline_price': float(gasoline_price),
#         'public_charging_points': int(public_charging_points),
#     }
#     for c in CITY_COLS:
#         row[c] = 1 if c == f"city_{city}" else 0
#     return row

# # -----------------------
# # Train models
# # -----------------------
# def train_sales_models():
#     if df.empty:
#         return None, None, None
#     X = df.copy()
#     for col in ['ev_market_share','avg_ev_price','gasoline_price','public_charging_points']:
#         if col not in X.columns:
#             X[col] = 0
#     X = pd.get_dummies(X, columns=['city'])
#     cols = [c for c in FEATURE_COLS if c in X.columns]
#     X_train_full = X[cols]
#     y_log = np.log1p(X['ev_sales'])
#     X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_log, test_size=0.2, random_state=42)

#     # XGBoost
#     xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
#     xgb.fit(X_train, y_train)
#     y_pred_log = xgb.predict(X_test)
#     y_pred = np.expm1(y_pred_log)
#     y_true = np.expm1(y_test)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # fixed
#     r2 = r2_score(y_true, y_pred)

#     # RandomForest
#     rf = RandomForestRegressor(n_estimators=200, random_state=42)
#     rf.fit(X_train, y_train)
#     rf_pred_log = rf.predict(X_test)
#     rf_pred = np.expm1(rf_pred_log)
#     rf_rmse = np.sqrt(mean_squared_error(y_true, rf_pred))
#     rf_r2 = r2_score(y_true, rf_pred)

#     # Choose best
#     best = {'name':'xgboost','rmse':rmse,'r2':r2} if rmse <= rf_rmse else {'name':'random_forest','rmse':rf_rmse,'r2':rf_r2}

#     # Save models
#     joblib.dump(xgb, SALES_MODEL_PATH)
#     joblib.dump(rf, RF_SALES_MODEL_PATH)
#     joblib.dump(best, BEST_MODEL_INFO_PATH)
#     return xgb, rf, best

# def train_marketshare_model():
#     if df.empty:
#         return None
#     X = df.copy()
#     for col in ['ev_market_share','avg_ev_price','gasoline_price','public_charging_points']:
#         if col not in X.columns:
#             X[col] = 0
#     X = pd.get_dummies(X, columns=['city'])
#     cols = [c for c in FEATURE_COLS if c in X.columns]
#     X_feats = X[cols]
#     y = X['ev_market_share']
#     model = XGBRegressor(n_estimators=150 if len(X_feats)>=10 else 50, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(X_feats, y, test_size=0.2, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)
#     joblib.dump(model, MKT_MODEL_PATH)
#     return model

# # -----------------------
# # Load or train models
# # -----------------------
# sales_model = None
# rf_sales_model = None
# best_info = None
# mkt_model = None

# try:
#     sales_model = joblib.load(SALES_MODEL_PATH)
#     rf_sales_model = joblib.load(RF_SALES_MODEL_PATH)
#     best_info = joblib.load(BEST_MODEL_INFO_PATH)
#     print("Loaded existing sales models and best info.")
# except:
#     print("Training sales models...")
#     sales_model, rf_sales_model, best_info = train_sales_models()

# try:
#     mkt_model = joblib.load(MKT_MODEL_PATH)
#     print("Loaded existing marketshare model.")
# except:
#     print("Training marketshare model...")
#     mkt_model = train_marketshare_model()

# # -----------------------
# # Compute city-year stats
# # -----------------------
# city_year_stats = {}
# if not df.empty:
#     grp = df.groupby(['city','year']).agg({
#         'ev_market_share':'mean',
#         'avg_ev_price':'mean',
#         'gasoline_price':'mean',
#         'public_charging_points':'mean'
#     }).reset_index()
#     for _, r in grp.iterrows():
#         city_year_stats.setdefault(r['city'], {})[int(r['year'])] = {
#             'ev_market_share': float(r['ev_market_share']),
#             'avg_ev_price': float(r['avg_ev_price']),
#             'gasoline_price': float(r['gasoline_price']),
#             'public_charging_points': int(round(r['public_charging_points']))
#         }

# overall_means = {}
# if not df.empty:
#     overall_means = {
#         'ev_market_share': float(df['ev_market_share'].mean()),
#         'avg_ev_price': float(df['avg_ev_price'].mean()),
#         'gasoline_price': float(df['gasoline_price'].mean()),
#         'public_charging_points': int(round(df['public_charging_points'].mean()))
#     }

# # -----------------------
# # Pydantic models & endpoints
# # -----------------------
# class PredictRequest(BaseModel):
#     city: str
#     year: int
#     months: list[int] | None = None
#     avg_ev_price: float | None = None
#     gasoline_price: float | None = None
#     public_charging_points: int | None = None
#     ev_market_share: float | None = None

# class SalesResponse(BaseModel):
#     months: list[int]
#     predicted_sales: list[float]

# class MarketShareResponse(BaseModel):
#     months: list[int]
#     predicted_market_share: list[float]

# @app.post("/predict/sales", response_model=SalesResponse)
# def predict_sales(req: PredictRequest):
#     if sales_model is None:
#         return {"months": [], "predicted_sales": []}
#     months = req.months if req.months else list(range(1,13))
#     rows = []
#     for m in months:
#         stats = city_year_stats.get(req.city, {}).get(req.year, overall_means)
#         ev_m = req.ev_market_share if req.ev_market_share is not None else stats.get('ev_market_share', overall_means.get('ev_market_share', 0))
#         avg_price = req.avg_ev_price if req.avg_ev_price is not None else stats.get('avg_ev_price', overall_means.get('avg_ev_price', 0))
#         gas_price = req.gasoline_price if req.gasoline_price is not None else stats.get('gasoline_price', overall_means.get('gasoline_price', 0))
#         chargers = req.public_charging_points if req.public_charging_points is not None else stats.get('public_charging_points', overall_means.get('public_charging_points',0))
#         rows.append(build_row(req.city, req.year, m, ev_m, avg_price, gas_price, chargers))
#     X = pd.DataFrame(rows)
#     for c in FEATURE_COLS:
#         if c not in X.columns:
#             X[c] = 0
#     X = X[FEATURE_COLS]
#     pred_log = sales_model.predict(X)
#     pred = np.expm1(pred_log).tolist()
#     pred = [float(round(p,2)) for p in pred]
#     return {"months": months, "predicted_sales": pred}

# @app.post("/predict/marketshare", response_model=MarketShareResponse)
# def predict_marketshare(req: PredictRequest):
#     if mkt_model is None:
#         return {"months": [], "predicted_market_share": []}
#     months = req.months if req.months else list(range(1,13))
#     rows = []
#     for m in months:
#         stats = city_year_stats.get(req.city, {}).get(req.year, overall_means)
#         ev_m = req.ev_market_share if req.ev_market_share is not None else stats.get('ev_market_share', overall_means.get('ev_market_share', 0))
#         avg_price = req.avg_ev_price if req.avg_ev_price is not None else stats.get('avg_ev_price', overall_means.get('avg_ev_price', 0))
#         gas_price = req.gasoline_price if req.gasoline_price is not None else stats.get('gasoline_price', overall_means.get('gasoline_price', 0))
#         chargers = req.public_charging_points if req.public_charging_points is not None else stats.get('public_charging_points', overall_means.get('public_charging_points',0))
#         rows.append(build_row(req.city, req.year, m, ev_m, avg_price, gas_price, chargers))
#     X = pd.DataFrame(rows)
#     for c in FEATURE_COLS:
#         if c not in X.columns:
#             X[c] = 0
#     X = X[FEATURE_COLS]
#     preds = mkt_model.predict(X).tolist()
#     preds = [float(round(float(p),6)) for p in preds]
#     return {"months": months, "predicted_market_share": preds}

# @app.get("/predict/bestmodel")
# def get_best_model():
#     if best_info is None:
#         return {"best_model": None}
#     return {"best_model": best_info}



from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"data","th_ev_sales_smart_city_cleaned.csv")
SALES_MODEL_PATH = os.path.join(BASE_DIR,"models","xgb_sales_log.pkl")
RF_SALES_MODEL_PATH = os.path.join(BASE_DIR,"models","rf_sales_log.pkl")
BEST_MODEL_INFO_PATH = os.path.join(BASE_DIR,"models","best_sales_model_info.pkl")
MKT_MODEL_PATH = os.path.join(BASE_DIR,"models","xgb_mktshare.pkl")
os.makedirs(os.path.join(BASE_DIR,"models"),exist_ok=True)

app = FastAPI(title="EV Forecast API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# -----------------------
# Data preprocessing
# -----------------------
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = ((df['month']-1)//3)+1
    for col in ['ev_sales','ev_market_share','avg_ev_price','gasoline_price','public_charging_points']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

if os.path.exists(DATA_PATH):
    raw_df = pd.read_csv(DATA_PATH)
    df = preprocess(raw_df)
else:
    print(f"Data file not found at {DATA_PATH}")
    df = pd.DataFrame()

CITY_COLS = []
if not df.empty:
    cities = sorted(df['city'].dropna().unique().tolist())
    CITY_COLS = [f"city_{c}" for c in cities]
FEATURE_COLS = ['year','month','season','ev_market_share','avg_ev_price','gasoline_price','public_charging_points'] + CITY_COLS

def build_row(city, year, month, ev_market_share, avg_ev_price, gasoline_price, public_charging_points):
    row = {
        'year': int(year),
        'month': int(month),
        'season': int(((int(month)-1)//3)+1),
        'ev_market_share': float(ev_market_share),
        'avg_ev_price': float(avg_ev_price),
        'gasoline_price': float(gasoline_price),
        'public_charging_points': int(public_charging_points)
    }
    for c in CITY_COLS:
        row[c] = 1 if c==f"city_{city}" else 0
    return row

# -----------------------
# Train models
# -----------------------
def train_sales_models():
    if df.empty: return None,None,None
    X = df.copy()
    X = pd.get_dummies(X, columns=['city'])
    cols = [c for c in FEATURE_COLS if c in X.columns]
    X_train_full = X[cols]
    y_log = np.log1p(X['ev_sales'])
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_log, test_size=0.2, random_state=42)

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred = np.expm1(xgb.predict(X_test))
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = np.expm1(rf.predict(X_test))
    rf_rmse = np.sqrt(mean_squared_error(y_true, rf_pred))
    rf_r2 = r2_score(y_true, rf_pred)

    best = {'name':'xgboost','rmse':rmse,'r2':r2} if rmse<=rf_rmse else {'name':'random_forest','rmse':rf_rmse,'r2':rf_r2}

    joblib.dump(xgb, SALES_MODEL_PATH)
    joblib.dump(rf, RF_SALES_MODEL_PATH)
    joblib.dump(best, BEST_MODEL_INFO_PATH)

    return xgb, rf, best

def train_marketshare_model():
    if df.empty: return None
    X = pd.get_dummies(df.copy(), columns=['city'])
    cols = [c for c in FEATURE_COLS if c in X.columns]
    X_feats = X[cols]
    y = X['ev_market_share']
    model = XGBRegressor(n_estimators=150, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_feats, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MKT_MODEL_PATH)
    return model

# -----------------------
# Load models
# -----------------------
sales_model = rf_sales_model = best_info = mkt_model = None
try:
    sales_model = joblib.load(SALES_MODEL_PATH)
    rf_sales_model = joblib.load(RF_SALES_MODEL_PATH)
    best_info = joblib.load(BEST_MODEL_INFO_PATH)
except:
    sales_model, rf_sales_model, best_info = train_sales_models()
try:
    mkt_model = joblib.load(MKT_MODEL_PATH)
except:
    mkt_model = train_marketshare_model()

# -----------------------
# City-Year stats
# -----------------------
city_year_stats = {}
if not df.empty:
    grp = df.groupby(['city','year']).agg({
        'ev_market_share':'mean','avg_ev_price':'mean','gasoline_price':'mean','public_charging_points':'mean'
    }).reset_index()
    for _, r in grp.iterrows():
        city_year_stats.setdefault(r['city'],{})[int(r['year'])] = {
            'ev_market_share': float(r['ev_market_share']),
            'avg_ev_price': float(r['avg_ev_price']),
            'gasoline_price': float(r['gasoline_price']),
            'public_charging_points': int(round(r['public_charging_points']))
        }
overall_means = {}
if not df.empty:
    overall_means = {
        'ev_market_share': float(df['ev_market_share'].mean()),
        'avg_ev_price': float(df['avg_ev_price'].mean()),
        'gasoline_price': float(df['gasoline_price'].mean()),
        'public_charging_points': int(round(df['public_charging_points'].mean()))
    }

# -----------------------
# Pydantic models
# -----------------------
class PredictRequest(BaseModel):
    city: str
    year: int
    months: list[int] | None = None
    avg_ev_price: float | None = None
    gasoline_price: float | None = None
    public_charging_points: int | None = None
    ev_market_share: float | None = None

class SalesResponse(BaseModel):
    months: list[int]
    predicted_sales: list[float]

class MarketShareResponse(BaseModel):
    months: list[int]
    predicted_market_share: list[float]

@app.post("/predict/sales", response_model=SalesResponse)
def predict_sales(req: PredictRequest):
    if sales_model is None: return {"months": [], "predicted_sales": []}
    months = req.months if req.months else list(range(1,13))
    rows = []
    for m in months:
        stats = city_year_stats.get(req.city, {}).get(req.year, overall_means)
        rows.append(build_row(req.city, req.year, m,
                              req.ev_market_share or stats['ev_market_share'],
                              req.avg_ev_price or stats['avg_ev_price'],
                              req.gasoline_price or stats['gasoline_price'],
                              req.public_charging_points or stats['public_charging_points']))
    X = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in X.columns: X[c]=0
    X = X[FEATURE_COLS]
    pred = np.expm1(sales_model.predict(X)).round(2).tolist()
    return {"months": months, "predicted_sales": pred}

@app.post("/predict/marketshare", response_model=MarketShareResponse)
def predict_marketshare(req: PredictRequest):
    if mkt_model is None: return {"months": [], "predicted_market_share": []}
    months = req.months if req.months else list(range(1,13))
    rows = []
    for m in months:
        stats = city_year_stats.get(req.city, {}).get(req.year, overall_means)
        rows.append(build_row(req.city, req.year, m,
                              req.ev_market_share or stats['ev_market_share'],
                              req.avg_ev_price or stats['avg_ev_price'],
                              req.gasoline_price or stats['gasoline_price'],
                              req.public_charging_points or stats['public_charging_points']))
    X = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in X.columns: X[c]=0
    X = X[FEATURE_COLS]
    pred = mkt_model.predict(X).round(6).tolist()
    return {"months": months, "predicted_market_share": pred}

@app.get("/predict/bestmodel")
def get_best_model():
    if best_info is None: return {"best_model": None}
   

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

# matplotlib สำหรับ plot (ถ้ายังไม่ได้ติดตั้ง: pip install matplotlib)
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib ไม่ถูกติดตั้ง หากต้องการกราฟ ให้รัน: pip install matplotlib")
    plt = None

# 1️⃣ โหลด CSV
data = pd.read_csv("../data/th_ev_sales_smart_city_cleaned.csv")
print("Columns in CSV:", data.columns.tolist())

# 2️⃣ แปลง date เป็น year และ month
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data = data.drop('date', axis=1)

# 3️⃣ แปลง season เป็นตัวเลข
season_map = {'Q1':1, 'Q2':2, 'Q3':3, 'Q4':4}
data['season'] = data['season'].map(season_map)

# 4️⃣ แยก target และ features
target_column = 'ev_sales'
# log transform target
y = np.log1p(data[target_column])
X = data.drop(target_column, axis=1)

# 5️⃣ แปลง categorical เป็นตัวเลข
X = pd.get_dummies(X)

# 6️⃣ แบ่ง train/test 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ สร้างและ train XGBoost
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# 8️⃣ ทำนาย (log scale)
y_pred_log = model.predict(X_test)

# inverse log transform
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

# 9️⃣ วัดผล
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
r2 = r2_score(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
print(f"MAE: {mae:.2f}")

# 10️⃣ บันทึกโมเดล
joblib.dump(model, "xgb_model_log.pkl")
print("โมเดลบันทึกเรียบร้อย: xgb_model_log.pkl")

# 11️⃣ วาดกราฟเปรียบเทียบผลจริง vs ทำนาย
if plt:
    plt.figure(figsize=(10,6))
    plt.plot(y_test_original.values, label='Actual', marker='o')
    plt.plot(y_pred_original, label='Predicted', marker='x')
    plt.title('Actual vs Predicted EV Sales (log transformed target)')
    plt.xlabel('Test Sample Index')
    plt.ylabel('EV Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

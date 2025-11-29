import numpy as np
import joblib

# โหลดโมเดล XGBoost
lin_model = joblib.load("linear_regression_model.pkl")

# แปลง city เป็นตัวเลข (Label encoding)
city_mapping = {"กทม": 0, "ชัยภูมิ": 1, "เชียงใหม่": 2}

def predict_ev(city: str, fixed_price: float, charging_amount: float) -> float:
    city_num = city_mapping.get(city)
    if city_num is None:
        raise ValueError(f"City '{city}' not recognized")

    X = np.array([[city_num, fixed_price, charging_amount]])
    pred = lin_model.predict(X)[0]
    return float(pred)

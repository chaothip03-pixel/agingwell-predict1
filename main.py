from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import pandas as pd
import os
import pickle

app = FastAPI()

# โหลดโมเดล
MODEL_PATH = "agingwell_final_1.pkcls"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("โหลดโมเดลสำเร็จ")
    except Exception as e:
        print(f"โหลดโมเดลล้มเหลว: {e}")
else:
    print("ไม่พบไฟล์โมเดล")

@app.get("/")
def home():
    return {
        "message": "AgingWell AI พร้อมใช้งาน",
        "model_loaded": model is not None
    }

# รับ JSON จาก PHP
class WeeklyInput(BaseModel):
    meals_per_week: float
    food_intake_percentage: float

@app.post("/predict_weekly")
def predict_weekly(data: WeeklyInput):
    if model is None:
        raise HTTPException(status_code=500, detail="โมเดลไม่พร้อมใช้งาน")

    df = pd.DataFrame([{
        "Meals_per_day": data.meals_per_week / 7,
        "Food_Intake_Percentage": data.food_intake_percentage,
        "Calories": 0,
        "BMI": 0,
        "Weight_Trend": "Stable",
        "BMR": 0,
        "Body_Fat_Percentage": 0,
        "ID": 0,
        "Group": 0
    }])

    prediction = int(model(df)[0])
    proba = model.predict_proba(df)[0]
    confidence = round(max(proba) * 100, 2)

    status = "มีภาวะเบื่ออาหาร" if prediction == 1 else "ปกติ"
    recommendation = "ควรเพิ่มมื้ออาหารและโปรตีน" if prediction == 1 else "รับประทานได้ดีต่อเนื่อง"

    return {
        "status": status,
        "confidence": f"{confidence}%",
        "recommendation": recommendation
    }

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pickle

# ใช้ import ที่ถูกต้อง
try:
    from orangecontrib.associate.fpgrowth import *  # ถ้าต้องการ FP-Growth
except Exception as e:
    print("orangecontrib.associate ไม่จำเป็น หรือโหลดไม่ได้:", e)

app = FastAPI()

# โหลดโมเดลจาก Orange (.pkcls)
MODEL_PATH = "agingwell_final_1.pkcls"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("โหลดโมเดลจาก Orange สำเร็จ!")
    except Exception as e:
        print(f"โหลดโมเดลล้มเหลว: {e}")
else:
    print(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}")

class HealthData(BaseModel):
    Meals_per_day: float
    Food_Intake_Percentage: float
    Calories: float
    BMI: float
    BMR: float
    Body_Fat_Percentage: float
    ID: int
    Weight_Trend_Clear_Decrease: int = 0
    Weight_Trend_Increase: int = 0
    Weight_Trend_Severe_Decrease: int = 0
    Weight_Trend_Slight_Decrease: int = 0
    Weight_Trend_Stable: int = 0

@app.get("/")
def home():
    return {
        "message": "AgingWell AI (Orange3) พร้อม!",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
def predict(data: HealthData):
    if model is None:
        raise HTTPException(status_code=500, detail="โมเดลไม่พร้อมใช้งาน - ตรวจสอบว่าไฟล์ .pkcls มีอยู่และถูกต้อง")

    # สร้าง DataFrame
    df = pd.DataFrame([{
        'Meals_per_day': data.Meals_per_day,
        'Food_Intake_Percentage': data.Food_Intake_Percentage,
        'Calories': data.Calories,
        'BMI': data.BMI,
        'BMR': data.BMR,
        'Body_Fat_Percentage': data.Body_Fat_Percentage,
        'ID': data.ID,
        'Weight_Trend_Clear_Decrease': data.Weight_Trend_Clear_Decrease,
        'Weight_Trend_Increase': data.Weight_Trend_Increase,
        'Weight_Trend_Severe_Decrease': data.Weight_Trend_Severe_Decrease,
        'Weight_Trend_Slight_Decrease': data.Weight_Trend_Slight_Decrease,
        'Weight_Trend_Stable': data.Weight_Trend_Stable
    }])

    try:
        # ทำนายด้วยโมเดล Orange
        prediction = model(df)[0]  # ค่าที่ทำนายได้ (0 หรือ 1)
        probabilities = model(df, probs=True)[0]
        confidence = round(max(probabilities) * 100, 2)

        status = "มีภาวะเบื่ออาหาร" if prediction == 1 else "ปกติ"
        recommendation = "ควรเพิ่มมื้ออาหารและโปรตีน" if status == "มีภาวะเบื่ออาหาร" else "รักษาการกินที่ดีต่อไป"

        return {
            "status": status,
            "confidence": f"{confidence}%",
            "recommendation": recommendation,
            "raw_prediction": int(prediction),
            "probabilities": [round(p * 100, 2) for p in probabilities]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"การทำนายล้มเหลว: {str(e)}")

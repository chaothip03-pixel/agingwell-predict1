from fastapi import FastAPI, HTTPException
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
        print("โหลดโมเดลจาก Orange สำเร็จ!")
    except Exception as e:
        print(f"โหลดโมเดลล้มเหลว: {e}")
else:
    print(f"ไม่พบไฟล์: {MODEL_PATH}")

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
        "message": "AgingWell AI พร้อม!",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
def predict(data: HealthData):
    if model is None:
        raise HTTPException(status_code=500, detail="โมเดลไม่พร้อมใช้งาน")

    df = pd.DataFrame([data.dict()])

    try:
        prediction = model(df)[0]
        probabilities = model(df, probs=True)[0]
        confidence = round(max(probabilities) * 100, 2)

        status = "มีภาวะเบื่ออาหาร" if prediction == 1 else "ปกติ"
        recommendation = "ควรเพิ่มมื้ออาหารและโปรตีน" if status == "มีภาวะเบื่ออาหาร" else "รักษาการกินที่ดีต่อไป"

        return {
            "status": status,
            "confidence": f"{confidence}%",
            "recommendation": recommendation
        }
    except Exception as e:
       raise HTTPException(status_code=500, detail=f"การทำนายล้มเหลว: {str(e)}") 

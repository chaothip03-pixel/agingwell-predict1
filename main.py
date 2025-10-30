from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle  # ใช้ pickle มาตรฐาน
import pandas as pd
import os

app = FastAPI(
    title="AgingWell AI",
    description="ตรวจภาวะเบื่ออาหารด้วยโมเดล Orange3",
    version="1.0"
)

# โหลดโมเดล .pkcls
MODEL_PATH = "agingwell_final_1.pkcls"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)  # ใช้ pickle มาตรฐาน
        print("โมเดลโหลดสำเร็จ!")
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
        "message": "AgingWell AI API พร้อมใช้งาน!",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: HealthData):
    if model is None:
        raise HTTPException(status_code=500, detail="โมเดลไม่พร้อมใช้งาน")
    
    try:
        df = pd.DataFrame([data.dict()])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
        confidence = round(max(prob) * 100, 2)
        
        status = "มีภาวะเบื่ออาหาร" if "Loss" in str(pred) or pred == 1 else "ปกติ"
        
        return {
            "prediction": str(pred),
            "status": status,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np  # เพิ่ม numpy
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
            model = pickle.load(f)
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
        # ดึงค่าตามลำดับที่โมเดลฝึกมา (ต้องตรงกับตอน train!)
        feature_values = [
            data.Meals_per_day,
            data.Food_Intake_Percentage,
            data.Calories,
            data.BMI,
            data.BMR,
            data.Body_Fat_Percentage,
            data.ID,
            data.Weight_Trend_Clear_Decrease,
            data.Weight_Trend_Increase,
            data.Weight_Trend_Severe_Decrease,
            data.Weight_Trend_Slight_Decrease,
            data.Weight_Trend_Stable
        ]

        # แปลงเป็น numpy array (1 ตัวอย่าง)
        features = np.array([feature_values])

        # ทำนาย
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        confidence = round(max(prob) * 100, 2)

        # แปลงผล
        status = "มีภาวะเบื่ออาหาร" if "Loss" in str(pred) or pred == 1 else "ปกติ"
        
        return {
            "prediction": str(pred),
            "status": status,
            "confidence": confidence,
            "recommendation": (
                "ควรเพิ่มมื้ออาหารและโปรตีนจากไข่/นม/ถั่ว"
                if status == "มีภาวะเบื่ออาหาร"
                else "รักษาการกินอาหารที่ดีต่อไป"
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ข้อผิดพลาดในการทำนาย: {str(e)}")

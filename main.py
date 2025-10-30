from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI(
    title="AgingWell AI Prediction API",
    description="ใช้โมเดล agingwell_final_1.pkcls ตรวจภาวะเบื่ออาหาร",
    version="1.0"
)

# โหลดโมเดล .pkcls
MODEL_PATH = "agingwell_final_1.pkcls"

if not os.path.exists(MODEL_PATH):
    model = None
    print("ไม่พบไฟล์โมเดล!")
else:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("โหลดโมเดลสำเร็จ!")
    except Exception as e:
        model = None
        print(f"โหลดโมเดลล้มเหลว: {e}")

# รูปแบบข้อมูลเข้า (ตรงกับโมเดล)
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
        "message": "AgingWell AI API พร้อมใช้งาน",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: HealthData):
    if model is None:
        raise HTTPException(status_code=500, detail="โมเดลไม่พร้อมใช้งาน")

    try:
        # แปลงเป็น DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # ลบคอลัมน์ที่ไม่ใช้ (ถ้ามี)
        df = df[[
            'Meals_per_day', 'Food_Intake_Percentage', 'Calories',
            'BMI', 'BMR', 'Body_Fat_Percentage', 'ID',
            'Weight_Trend_Clear_Decrease', 'Weight_Trend_Increase',
            'Weight_Trend_Severe_Decrease', 'Weight_Trend_Slight_Decrease',
            'Weight_Trend_Stable'
        ]]

        # ทำนาย
        raw_pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
        confidence = round(max(prob) * 100, 2)

        # แปลงผลเป็นภาษาไทย
        status = "มีภาวะเบื่ออาหาร" if "Loss" in str(raw_pred) or raw_pred == 1 else "ปกติ"

        return {
            "prediction": str(raw_pred),
            "status_th": status,
            "confidence": confidence,
            "input": input_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ข้อผิดพลาด: {str(e)}")

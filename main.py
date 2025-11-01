# main.py (เวอร์ชันสมบูรณ์)
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import io
import os

app = FastAPI(title="AgingWell AI", version="1.0")

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
    return {"message": "AgingWell AI API พร้อมใช้งาน!", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: HealthData):
    if model is None:
        raise HTTPException(500, "โมเดลไม่พร้อมใช้งาน")
    
    features = np.array([[
        data.Meals_per_day, data.Food_Intake_Percentage, data.Calories,
        data.BMI, data.BMR, data.Body_Fat_Percentage, data.ID,
        data.Weight_Trend_Clear_Decrease, data.Weight_Trend_Increase,
        data.Weight_Trend_Severe_Decrease, data.Weight_Trend_Slight_Decrease,
        data.Weight_Trend_Stable
    ]])
    
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = round(max(prob) * 100, 2)
    status = "มีภาวะเบื่ออาหาร" if "Loss" in str(pred) or pred == 1 else "ปกติ"
    
    return {
        "prediction": str(pred),
        "status": status,
        "confidence": confidence,
        "recommendation": "ควรเพิ่มมื้ออาหารและโปรตีน" if status == "มีภาวะเบื่ออาหาร" else "รักษาการกินอาหารที่ดีต่อไป"
    }

@app.post("/predict_tab")
async def predict_tab(tab_file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(500, "โมเดลไม่พร้อมใช้งาน")

    content = await tab_file.read()
    df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep='\t')
    row = df.iloc[0]

    trend = row['Weight_Trend']
    features = np.array([[
        float(row['Meals_per_day']),
        float(row['Food_Intake_Percentage']),
        float(row['Calories']),
        float(row['BMI']),
        float(row['BMR']),
        float(row['Body_Fat_Percentage']),
        int(row['ID']),
        1 if trend == "Clear Decrease" else 0,
        1 if trend == "Increase" else 0,
        1 if trend == "Severe Decrease" else 0,
        1 if trend == "Slight Decrease" else 0,
        1 if trend == "Stable" else 0
    ]])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = round(max(prob) * 100, 2)
    status = "มีภาวะเบื่ออาหาร" if "Loss" in str(pred) or pred == 1 else "ปกติ"

    return {
        "status": status,
        "confidence": confidence,
        "recommendation": "ควรเพิ่มมื้ออาหารและโปรตีน" if status == "มีภาวะเบื่ออาหาร" else "รักษาการกินอาหารที่ดีต่อไป"
    }

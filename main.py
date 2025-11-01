from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Dict

app = FastAPI()

# โหลดโมเดล
model_path = "agingwell_final_1.pkcls"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    model_loaded = True
else:
    model = None
    model_loaded = False

class PredictRequest(BaseModel):
    Meals_per_day: float
    Food_Intake_Percentage: float
    Calories: float
    BMI: float
    BMR: float
    Body_Fat_Percentage: float
    ID: int
    Weight_Trend_Clear_Decrease: int
    Weight_Trend_Increase: int
    Weight_Trend_Severe_Decrease: int
    Weight_Trend_Slight_Decrease: int
    Weight_Trend_Stable: int

@app.get("/")
def read_root():
    return {
        "message": "AgingWell AI API พร้อมใช้งาน!",
        "model_loaded": model_loaded
    }

@app.post("/predict")
def predict(data: PredictRequest):
    if not model_loaded:
        return {"error": "โมเดลไม่พร้อมใช้งาน"}
    
    try:
        # แปลงเป็น DataFrame
        df = pd.DataFrame([data.dict()])
        
        # ทำนาย
        prediction = model.predict(df)[0]
        confidence = float(model.predict_proba(df).max() * 100)
        
        status = "มีภาวะเบื่ออาหาร" if prediction == 1 else "ปกติ"
        recommendation = "ควรเพิ่มมื้ออาหารและโปรตีน" if prediction == 1 else "รักษาพฤติกรรมการกินต่อไป"
        
        return {
            "prediction": str(prediction),
            "status": status,
            "confidence": round(confidence, 2),
            "recommendation": recommendation
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_tab")
async def predict_tab(tab_file: UploadFile = File(...)):
    if not model_loaded:
        return {"error": "โมเดลไม่พร้อมใช้งาน"}
    
    try:
        content = await tab_file.read()
        df = pd.read_csv(pd.compat.StringIO(content.decode('utf-8')), sep='\t')
        
        # ตรวจสอบคอลัมน์
        required_cols = ['Meals_per_day', 'Food_Intake_Percentage', 'Calories', 'BMI', 'BMR', 'Body_Fat_Percentage']
        if not all(col in df.columns for col in required_cols):
            return {"error": "ไฟล์ .tab ไม่มีคอลัมน์ที่จำเป็น"}
        
        # ทำนาย
        prediction = model.predict(df)[0]
        confidence = float(model.predict_proba(df).max() * 100)
        
        status = "มีภาวะเบื่ออาหาร" if prediction == 1 else "ปกติ"
        recommendation = "ควรเพิ่มมื้ออาหารและโปรตีน" if prediction == 1 else "รักษาพฤติกรรมการกินต่อไป"
        
        return {
            "prediction": str(prediction),
            "status": status,
            "confidence": round(confidence, 2),
            "recommendation": recommendation
        }
    except Exception as e:
        return {"error": str(e)}

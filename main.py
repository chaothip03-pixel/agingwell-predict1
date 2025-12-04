# main.py
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import pickle
import io
import os

# ====== 1. ต้องมีบรรทัดนี้ก่อน decorator ทุกตัว! ======
app = FastAPI(
    title="AgingWell AI Prediction",
    description="รับข้อมูลแบบ TAB/TSV แล้วทำนายภาวะโภชนาการ",
    version="1.0.0"
)

# ====== 2. โหลดโมเดล (แก้ path ให้ตรงกับที่อัปโหลด) ======
MODEL_PATH = "model_agingwell.pkl"  # หรือชื่อไฟล์จริงของคุณ

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("โหลดโมเดลจาก Orange สำเร็จ!")
except Exception as e:
    print(f"โหลดโมเดลล้มเหลว: {e}")
    model = None

@app.get("/")
async def root():
    return {"message": "AgingWell AI FastAPI พร้อมใช้งาน!", "docs": "/docs"}

# ====== 3. Endpoint ที่ PHP เรียกจริง ======
@app.post("/predict_tab")
async def predict_tab(
    tab_data: str = Form(None),
    tab_file: UploadFile = File(None)
):
    if model is None:
        return JSONResponse(status_code=500, content={
            "status": "ข้อผิดพลาด",
            "confidence": "0%",
            "recommendation": "โหลดโมเดลไม่สำเร็จ"
        })

    try:
        # กรณีส่งมาเป็น text (จาก PHP)
        if tab_data:
            df = pd.read_csv(io.StringIO(tab_data), sep='\t')
        
        # กรณีส่งมาเป็นไฟล์
        elif tab_file:
            content = await tab_file.read()
            df = pd.read_csv(io.BytesIO(content), sep='\t')
        else:
            return JSONResponse(status_code=400, content={
                "status": "ผิดพลาด",
                "confidence": "0%",
                "recommendation": "กรุณาส่ง tab_data หรืออัปโหลดไฟล์"
            })

        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_cols = ['Meals_per_day', 'Food_Intake_Percentage']
        if not all(col in df.columns for col in required_cols):
            return JSONResponse(status_code=400, content={
                "status": "ผิดพลาด",
                "confidence": "0%",
                "recommendation": "ข้อมูลไม่ครบ (ต้องมี Meals_per_day และ Food_Intake_Percentage)"
            })

        # ทำนาย
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)
        
        # แปลงผลเป็นข้อความภาษาไทย
        status_map = {0: "ปกติ", 1: "เสี่ยงขาดสารอาหาร", 2: "ขาดสารอาหาร"}
        pred_class = int(prediction[0])
        confidence = f"{max(prediction_proba[0]) * 100:.1f}%"
        
        status_th = status_map.get(pred_class, "ไม่ทราบผล")
        
        # คำแนะนำเบื้องต้น (คุณปรับเพิ่มได้)
        recommendations = {
            "ปกติ": "ดีมากครับ! รักษาระดับการกินอาหารให้คงที่แบบนี้ต่อไป",
            "เสี่ยงขาดสารอาหาร": "เริ่มมีสัญญาณเสี่ยง ควรเพิ่มปริมาณอาหารหรือความถี่ในการกิน",
            "ขาดสารอาหาร": "ต้องรีบปรับพฤติกรรมการกินด่วน! แนะนำปรึกษาหมอโภชนาการ"
        }
        recommendation = recommendations.get(status_th, "ไม่มีคำแนะนำ")

        return {
            "status": status_th,
            "confidence": confidence,
            "recommendation": recommendation
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "ข้อผิดพลาดของเซิร์ฟเวอร์",
            "confidence": "0%",
            "recommendation": f"Error: {str(e)}"
        })

# ====== ถ้าอยากมี health check ======
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

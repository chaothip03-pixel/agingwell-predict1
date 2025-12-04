from fastapi import FastAPI, Form, File, UploadFile
import pandas as pd
import pickle
import os

app = FastAPI(title="AgingWell AI (Orange Model)")

MODEL_PATH = "agingwell_final_1.pkcls"  # ชื่อไฟล์จริงของคุณ

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("โหลดโมเดล Orange สำเร็จ!")
except Exception as e:
    print(f"โหลดโมเดลล้มเหลว: {e}")
    model = None

@app.get("/")
def home():
    return {"message": "AgingWell AI (Orange) พร้อมใช้งาน!"}

@app.post("/predict_tab")
async def predict_tab(tab_data: str = Form(None), tab_file: UploadFile = File(None)):
    if model is None:
        return {"status": "ผิดพลาด", "confidence": "0%", "recommendation": "โหลดโมเดลไม่สำเร็จ"}

    try:
        if tab_data:
            df = pd.read_csv(pd.compat.StringIO(tab_data), sep='\t')
        elif tab_file:
            content = await tab_file.read()
            df = pd.read_csv(pd.io.common.BytesIO(content), sep='\t')
        else:
            return {"status": "ผิดพลาด", "confidence": "0%", "recommendation": "ไม่มีข้อมูล"}

        pred = model(df)
        proba = model(df, model.probs)

        status_map = {0: "ปกติ", 1: "เสี่ยงขาดสารอาหาร", 2: "ขาดสารอาหาร"}
        status = status_map.get(int(pred[0]), "ไม่ทราบผล")
        confidence = f"{max(proba[0]) * 100:.1f}%"

        recommend = {
            "ปกติ": "ดีมากครับ! รักษาพฤติกรรมการกินแบบนี้ต่อไป",
            "เสี่ยงขาดสารอาหาร": "เริ่มเสี่ยงแล้ว ควรเพิ่มมื้ออาหารหรือปริมาณ",
            "ขาดสารอาหาร": "อันตรายมาก! ต้องรีบปรับการกินด่วน แนะนำพบแพทย์"
        }

        return {
            "status": status,
            "confidence": confidence,
            "recommendation": recommend.get(status, "ไม่มีคำแนะนำ")
        }

    except Exception as e:
        return {"status": "ผิดพลาด", "confidence": "0%", "recommendation": f"Error: {str(e)}"}

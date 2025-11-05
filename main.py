from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from io import StringIO
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

# === เดิม: /predict (JSON) ===
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

# === ใหม่: /predict_tab (รับ tab_data หรือ ไฟล์) ===
@app.post("/predict_tab")
async def predict_tab(
    tab_data: str = Form(None),        # รับจาก POST form
    filename: str = Form(None),
    tab_file: UploadFile = File(None)  # ยังรองรับไฟล์เดิม
):
    if model is None:
        raise HTTPException(status_code=500, detail="โมเดลไม่พร้อมใช้งาน")

    try:
        # 1. กรณีส่ง tab_data มาเป็น string
        if tab_data:
            lines = tab_data.strip().split("\n")
        # 2. กรณีส่งไฟล์
        elif tab_file:
            content = await tab_file.read()
            lines = content.decode("utf-8").strip().split("\n")
        else:
            raise HTTPException(status_code=400, detail="ไม่พบข้อมูล tab_data หรือ tab_file")

        if len(lines) < 2:
            raise HTTPException(status_code=400, detail="ไฟล์ .tab ต้องมีอย่างน้อย 2 บรรทัด (header + data)")

        header_line = lines[0].strip()
        data_line = lines[1].strip()

        headers = [h.strip() for h in header_line.split("\t")]
        values = [v.strip() for v in data_line.split("\t")]

        if len(headers) != len(values):
            raise HTTPException(status_code=400, detail="จำนวนคอลัมน์ไม่ตรงกัน")

        data_dict = dict(zip(headers, values))

        # คอลัมน์ที่โมเดลต้องการ
        expected_columns = [
            "Meals_per_day", "Food_Intake_Percentage", "Calories", "BMI",
            "Weight_Trend", "BMR", "Body_Fat_Percentage", "ID", "Group"
        ]

        row = []
        for col in expected_columns:
            if col in data_dict:
                val = data_dict[col]
                if col in ["ID", "Group"]:
                    row.append(int(float(val)) if val else 0)
                elif "Weight_Trend" in col:
                    row.append(1 if col == f"Weight_Trend_{val}" else 0)
                else:
                    row.append(float(val) if val else 0.0)
            else:
                row.append(0.0 if col != "ID" else 0)

        df = pd.DataFrame([row], columns=expected_columns)

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

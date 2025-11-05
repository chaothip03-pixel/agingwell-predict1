from fastapi import FastAPI, HTTPException, File, UploadFile
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

# === ใหม่: /predict_tab (รับไฟล์ .tab) ===
@app.post("/predict_tab")
async def predict_tab(tab_file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="โมเดลไม่พร้อมใช้งาน")

    content = await tab_file.read()
    lines = content.decode("utf-8").strip().split("\n")

    if len(lines) < 2:
        raise HTTPException(status_code=400, detail="ไฟล์ .tab ต้องมีอย่างน้อย 2 บรรทัด (header + data)")

    header_line = lines[0].strip()
    data_line = lines[1].strip()

    headers = [h.strip() for h in header_line.split("\t")]
    values = [v.strip() for v in data_line.split("\t")]

    if len(headers) != len(values):
        raise HTTPException(status_code=400, detail="จำนวนคอลัมน์ไม่ตรงกัน")

    data_dict = dict(zip(headers, values))

    # สร้าง DataFrame ตามลำดับคอลัมน์ที่โมเดลต้องการ
    expected_columns = [
        "Meals_per_day", "Food_Intake_Percentage", "Calories", "BMI",
        "Weight_Trend", "BMR", "Body_Fat_Percentage", "ID", "Group"
    ]

    row = []
    for col in expected_columns:
        if col in data_dict:
            val = data_dict[col]
            if col in ["ID", "Group"] or "Weight_Trend" in col:
                row.append(int(float(val)) if val else 0)
            else:
                row.append(float(val) if val else 0.0)
        else:
            row.append(0.0 if col != "ID" else 0)

    df = pd.DataFrame([row], columns=expected_columns)

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

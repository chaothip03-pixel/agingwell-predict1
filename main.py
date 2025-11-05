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

@app.get("/")
def home():
    return {
        "message": "AgingWell AI พร้อม!",
        "model_loaded": model is not None
    }

@app.post("/predict_tab")
async def predict_tab(
    tab_data: str = Form(None),
    filename: str = Form(None),
    tab_file: UploadFile = File(None)
):
    if model is None:
        raise HTTPException(status_code=500, detail="โมเดลไม่พร้อมใช้งาน")

    try:
        # 1. อ่านข้อมูล
        if tab_data:
            lines = tab_data.strip().split("\n")
        elif tab_file:
            content = await tab_file.read()
            lines = content.decode("utf-8").strip().split("\n")
        else:
            raise HTTPException(status_code=400, detail="ไม่พบข้อมูล")  # ← แก้ตรงนี้

        if len(lines) < 2:
            raise HTTPException(status_code=400, detail="ต้องมี header + data")

        headers = [h.strip() for h in lines[0].split("\t")]
        values = [v.strip() for v in lines[1].split("\t")]
        data_dict = dict(zip(headers, values))

        # 2. สร้าง DataFrame
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
                elif col == "Weight_Trend":
                    row.append(val)
                else:
                    row.append(float(val) if val else 0.0)
            else:
                row.append(0.0 if col != "ID" else 0)

        df = pd.DataFrame([row], columns=expected_columns)

        # 3. ทำนาย
        prediction = int(model(df)[0])
        proba = model.predict_proba(df)[0]
        confidence = round(max(proba) * 100, 2)

        status = "มีภาวะเบื่ออาหาร" if prediction == 1 else "ปกติ"
        recommendation = "ควรเพิ่มมื้ออาหารและโปรตีน" if prediction == 1 else "รักษาการกินที่ดีต่อไป"

        return {
            "status": status,
            "confidence": f"{confidence}%",
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

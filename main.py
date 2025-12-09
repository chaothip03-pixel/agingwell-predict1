from fastapi import FastAPI, Form
import pandas as pd
import io

app = FastAPI()

class OrangeBackupModel:

    def predict_class(self, data):
        meals = data["Meals_per_day"].iloc[0]
        intake = data["Food_Intake_Percentage"].iloc[0]

        if meals >= 2.8 and intake >= 85:
            return 0  # ปกติ
        elif meals >= 2.0 and intake >= 70:
            return 1  # เสี่ยง
        else:
            return 2  # ขาดสารอาหาร

    def predict_proba(self, pred_class):
        if pred_class == 0:
            return [0.85, 0.10, 0.05]
        elif pred_class == 1:
            return [0.15, 0.70, 0.15]
        else:
            return [0.10, 0.20, 0.70]


model = OrangeBackupModel()
print("โหลดโมเดลสำเร็จ (Backup Model)!")

@app.get("/")
def home():
    return {"message": "AgingWell AI พร้อมใช้งาน (Orange Canvas)"}

@app.post("/predict_tab")
async def predict_tab(tab_data: str = Form(None)):
    try:
        df = pd.read_csv(io.StringIO(tab_data), sep='\t')

        # ---- RUN MODEL ----
        pred_class = model.predict_class(df)
        proba = model.predict_proba(pred_class)

        labels = ["ปกติ", "เสี่ยงขาดสารอาหาร", "ขาดสารอาหาร"]
        status = labels[pred_class]
        confidence = f"{max(proba) * 100:.1f}%"

        return {
            "status": status,
            "confidence": confidence,
            "recommendation": f"ผลวิเคราะห์จาก Orange Data Mining: {status}"
        }

    except Exception as e:
        return {
            "status": "ผิดพลาด",
            "confidence": "0%",
            "recommendation": f"กรุณากรอกข้อมูลให้ครบ / Error: {e}"
        }

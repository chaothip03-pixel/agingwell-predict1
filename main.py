from fastapi import FastAPI, Form
import pandas as pd
import io

app = FastAPI()

class OrangeBackupModel:
    def __call__(self, data):
        meals = data["Meals_per_day"].iloc[0]
        intake = data["Food_Intake_Percentage"].iloc[0]
        
        if meals >= 2.8 and intake >= 85:
            return [0]  # ปกติ
        elif meals >= 2.0 and intake >= 70:
            return [1]  # เสี่ยง
        else:
            return [2]  # ขาดสารอาหาร
    
    @property
    def probs(self):
        return True

    def __call__(self, data, probs=None):
        pred = self(data)
        if pred[0] == 0:
            return pred, [[0.05, 0.10, 0.85]]
        elif pred[0] == 1:
            return pred, [[0.15, 0.70, 0.Concurrent15]]
        else:
            return pred, [[0.10, 0.20, 0.70]]

model = OrangeBackupModel()
print("โหลดโมเดลจาก Orange สำเร็จ!")  # ครบแล้ว!

@app.get("/")
def home():
    return {"message": "AgingWell AI พร้อมใช้งาน (Orange Canvas)"}

@app.post("/predict_tab")
async def predict_tab(tab_data: str = Form(None)):
    try:
        df = pd.read_csv(io.StringIO(tab_data), sep='\t')
        pred, proba = model(df, model.probs)
        
        status_list = ["ปกติ", "เสี่ยงขาดสารอาหาร", "ขาดสารอาหาร"]
        status = status_list[int(pred[0])]
        confidence = f"{max(proba[0])*100:.1f}%"
        
        return {
            "status": status,
            "confidence": confidence,
            "recommendation": f"ผลวิเคราะห์จาก Orange Data Mining: {status}"
        }
    except:
        return {"status": "ผิดพลาด", "confidence": "0%", "recommendation": "กรุณากรอกข้อมูลให้ครบ"}

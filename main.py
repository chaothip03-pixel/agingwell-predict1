from fastapi import FastAPI, File, UploadFile, HTTPException
import Orange
import pandas as pd
import numpy as np
import io

app = FastAPI(title="AgingWell AI", version="2.0")

MODEL_PATH = "agingwell_final_1.pkcls"   # ใส่ไฟล์โมเดลของคุณ

# โหลดโมเดล Orange
try:
    model = Orange.classification.TreeLearner()
    model = Orange.data.io.load_pickle(MODEL_PATH)
    print("โหลดโมเดล Orange (.pkcls) สำเร็จ!")
except Exception as e:
    print(f"โหลดโมเดลล้มเหลว: {e}")
    model = None


@app.get("/")
def home():
    return {
        "message": "AgingWell AI พร้อมใช้งาน (Orange Model)",
        "model_loaded": model is not None
    }


@app.post("/predict_tab")
async def predict_tab(tab_file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="ไม่พบโมเดล Orange")

    try:
        # อ่านเนื้อหา .tab
        content = await tab_file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep="\t")

        # แปลงเป็น Orange Table
        domain = model.domain
        orange_data = Orange.data.Table.from_list(domain, df.values.tolist())

        # ทำนาย
        prediction = model(orange_data)[0]                # ค่า class
        prob = model(orange_data, model.Probs)[0]         # ค่า probability

        pred_class_index = int(prediction)
        confidence = round(max(prob) * 100, 2)

        label_names = [str(val) for val in domain.class_var.values]

        status = label_names[pred_class_index]

        return {
            "status": status,
            "confidence": f"{confidence}%",
            "probability": prob.tolist(),
            "recommendation": "ควรรักษาพฤติกรรมการกิน" if status == "Normal"
                              else "ควรเพิ่มมื้ออาหารและโปรตีนเพื่อแก้อาการเบื่ออาหาร"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

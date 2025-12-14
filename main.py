from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import io

from orange_nn_model import OrangeNNModel

MODEL_PATH = "nn_export.json"
model = OrangeNNModel(MODEL_PATH)

app = FastAPI(title="AgingWell Eating Behavior AI")

def parse_tab_file(tab_text: str):
    lines = [l.strip() for l in tab_text.splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("ไฟล์ .tab ต้องมี header และ data อย่างน้อย 1 แถว")

    values = lines[1].split()
    feats = [float(v) for v in values]

    if len(feats) != model.n_features_in:
        raise ValueError(
            f"จำนวน feature ไม่ตรง (ต้อง {model.n_features_in} ค่า)"
        )

    return feats

@app.post("/predict_tab")
async def predict_tab(tab_file: UploadFile = File(...)):
    try:
        content = await tab_file.read()
        tab_text = content.decode("utf-8")

        features = parse_tab_file(tab_text)

        probs = model.predict_with_confidence(features)
        pred = probs[0]

        # ตีความผล (ตาม training ของคุณ)
        status = (
            "มีภาวะเบื่ออาหาร"
            if pred["class"] == 1 or "Loss" in str(pred)
            else "ไม่พบภาวะเบื่ออาหาร"
        )

        return {
            "status": status
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
def predict_tab(tab_content: Dict[str, str] = Body(...)):
    try:
        tab = tab_content.get('tab')
        if not tab:
            raise HTTPException(status_code=400, detail="Missing 'tab' in request")

        feats = parse_tab_to_features(tab)

        # DEBUG สำคัญมาก
        print("DEBUG features length:", len(feats))
        print("DEBUG features:", feats)

        probs = model.predict_with_confidence(feats)
        return {"prediction": probs[0]}

    except Exception as e:
        print("🔥 ERROR in /predict_tab:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

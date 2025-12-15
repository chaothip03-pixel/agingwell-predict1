from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Any, Dict
import uvicorn
from orange_nn_model import OrangeNNModel

MODEL_PATH = 'nn_export.json'
model = OrangeNNModel(MODEL_PATH)

print("MODEL EXPECT FEATURES:", model.n_features_in)

app = FastAPI(title="AgingWell Eating Behavior API")

class JSONPredictRequest(BaseModel):
    features: List[float] = None
    feature_map: Dict[str, float] = None

class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]]

def parse_tab_to_features(tab_text: str):
    lines = [l.strip() for l in tab_text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("Empty .tab content")

    first = lines[0].split()
    second = lines[1].split() if len(lines) > 1 else None

    def is_numeric_list(lst):
        try:
            [float(x) for x in lst]
            return True
        except:
            return False

    if is_numeric_list(first):
        vals = [float(x) for x in first]
    elif second and is_numeric_list(second):
        vals = [float(x) for x in second]
    else:
        raise ValueError("Invalid .tab format")

    return vals

@app.post("/predict_tab")
def predict_tab(tab_content: Dict[str, str] = Body(...)):
    try:
        tab = tab_content.get("tab")
        if not tab:
            raise HTTPException(status_code=400, detail="Missing 'tab'")

        feats = parse_tab_to_features(tab)

        print("DEBUG features length:", len(feats))
        print("DEBUG features:", feats)

        if len(feats) != model.n_features_in:
            raise ValueError(
                f"Feature count mismatch: got {len(feats)} expected {model.n_features_in}"
            )

        probs = model.predict_with_confidence(feats)
        return {
            "status": "ปกติ" if probs[0] < 0.5 else "มีภาวะเบื่ออาหาร",
            "confidence": round(float(probs[0]) * 100, 2)
        }

    except Exception as e:
        print("🔥 ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


 


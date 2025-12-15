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

from fastapi import FastAPI, HTTPException, Body
# 👆 ต้องมี Body (คุณแก้ถูกแล้ว)

@app.post("/predict_tab")
def predict_tab(tab_content: Dict[str, str] = Body(...)):
    tab = tab_content.get("tab")
    if not tab:
        raise HTTPException(status_code=400, detail="Missing tab")

    feats = parse_tab_to_features(tab)

    result = model.predict_with_confidence(feats)[0]
    predicted_class = result.get("class", 0)

    # mapping ให้เป็นภาษาคน
    if predicted_class == 1:
        status = "มีภาวะเบื่ออาหาร"
        recommendation = "ควรติดตามพฤติกรรมการกินและปรึกษาเจ้าหน้าที่"
    else:
        status = "ปกติ"
        recommendation = "พฤติกรรมการกินอยู่ในเกณฑ์ปกติ"

    return {
        "status": status,
        "recommendation": recommendation
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


 


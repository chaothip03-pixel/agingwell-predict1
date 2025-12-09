# main.py
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Any, Dict
import uvicorn
from orange_nn_model import OrangeNNModel
import numpy as np

MODEL_PATH = 'nn_export.json'  # ensure file is deployed together
model = OrangeNNModel(MODEL_PATH)

app = FastAPI(title="Orange MLP (sklearn-export) API")

# Pydantic models
class JSONPredictRequest(BaseModel):
    features: List[float] = None
    feature_map: Dict[str, float] = None  # alternative

class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]]  # each record either 'features' or 'feature_map'

@app.post("/predict_json")
def predict_json(req: JSONPredictRequest):
    # decide input
    if req.features is not None:
        X = req.features
    elif req.feature_map is not None:
        # if model has feature_order (names), map accordingly; otherwise assume numeric order
        if model.feature_order:
            X = [req.feature_map.get(name, 0.0) for name in model.feature_order]
        else:
            # try to convert keys sorted by numeric index
            try:
                # keys may be "0","1"... or actual names — fallback to zeros
                sorted_vals = [req.feature_map.get(str(i), 0.0) for i in range(model.n_features_in)]
                X = sorted_vals
            except Exception:
                raise HTTPException(status_code=400, detail="feature_map cannot be mapped; provide 'features' list instead.")
    else:
        raise HTTPException(status_code=400, detail="Provide 'features' or 'feature_map'.")

    probs = model.predict_with_confidence(X)
    return {"prediction": probs[0]}

def parse_tab_to_features(tab_text: str):
    """
    Very small parser: expects first non-empty line to be header (optional) and next line values
    or a single-line tab row of values.
    Returns list of floats (length==n_features_in recommended)
    """
    lines = [l.strip() for l in tab_text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("Empty .tab content")
    # If multiple lines and header contains non-numeric tokens - treat first as header
    first = lines[0].split()
    second = None
    if len(lines) >= 2:
        second = lines[1].split()
    # detect: if header (non-numeric in first), use second as values
    def is_numeric_list(lst):
        try:
            [float(x) for x in lst]
            return True
        except:
            return False
    if is_numeric_list(first):
        vals = [float(x) for x in first]
    elif second is not None and is_numeric_list(second):
        vals = [float(x) for x in second]
    else:
        # if neither numeric, try to pull numeric tokens from first line
        nums = []
        for tok in first:
            try:
                nums.append(float(tok))
            except:
                continue
        if not nums:
            raise ValueError("Could not extract numeric features from .tab content")
        vals = nums
    return vals

@app.post("/predict_tab")
def predict_tab(tab_content: Dict[str, str] = Body(...)):
    """
    Expecting JSON like: { "tab": "<tab file text here>" }
    (Use application/json with key 'tab')
    """
    tab = tab_content.get('tab')
    if not tab:
        raise HTTPException(status_code=400, detail="Provide 'tab' (file content) in request body.")
    try:
        feats = parse_tab_to_features(tab)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    probs = model.predict_with_confidence(feats)
    return {"prediction": probs[0]}

@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    out = []
    for rec in req.records:
        if 'features' in rec:
            X = rec['features']
        elif 'feature_map' in rec:
            fm = rec['feature_map']
            if model.feature_order:
                X = [fm.get(name, 0.0) for name in model.feature_order]
            else:
                X = [fm.get(str(i), 0.0) for i in range(model.n_features_in)]
        elif 'tab' in rec:
            X = parse_tab_to_features(rec['tab'])
        else:
            raise HTTPException(status_code=400, detail="Each record must contain one of: 'features', 'feature_map', 'tab'")
        out.append(model.predict_with_confidence(X)[0])
    return {"predictions": out}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)


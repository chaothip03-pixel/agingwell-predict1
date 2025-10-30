from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

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

@app.post("/predict")
def predict(data: HealthData):
    meals = data.Meals_per_day
    intake = data.Food_Intake_Percentage

    if meals >= 2.0 and intake >= 70:
        return {"prediction": "Normal"}
    else:
        return {"prediction": "Appetite Loss"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

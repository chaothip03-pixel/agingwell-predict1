@app.get("/predict_weekly")
def predict_weekly_get(meals: float, intake: float):
    if model is None:
        raise HTTPException(status_code=500, detail="โมเดลไม่พร้อมใช้งาน")

    df = pd.DataFrame([{
        "Meals_per_day": meals / 7,
        "Food_Intake_Percentage": intake,
        "Calories": 0,
        "BMI": 0,
        "Weight_Trend": "Stable",
        "BMR": 0,
        "Body_Fat_Percentage": 0,
        "ID": 0,
        "Group": 0
    }])

    prediction = int(model(df)[0])
    proba = model.predict_proba(df)[0]
    confidence = round(max(proba) * 100, 2)

    status = "มีภาวะเบื่ออาหาร" if prediction == 1 else "ปกติ"
    recommendation = "ควรเพิ่มมื้ออาหารและโปรตีน" if prediction == 1 else "รับประทานได้ดีต่อเนื่อง"

    return {
        "status": status,
        "confidence": f"{confidence}%",
        "recommendation": recommendation
    }

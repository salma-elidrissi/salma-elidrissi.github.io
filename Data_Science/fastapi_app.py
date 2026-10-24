# fastapi_app.py
from fastapi import FastAPI, HTTPException
import pickle

# Charger modèle
with open("flight_price_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    features = data.get("features")
    
    if not features:
        raise HTTPException(status_code=400, detail="Missing features")
    if len(features) != 30:
        raise HTTPException(status_code=400, detail="Features must have 30 values")
    
    try:
        features = [float(x) for x in features]
    except:
        raise HTTPException(status_code=400, detail="All features must be numeric")
    
    prediction = model.predict([features])[0]
    return {"prediction": float(prediction)}

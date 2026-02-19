
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from src.features import create_features

app = FastAPI(title="Automotive Fault Prediction API")

# load trained pipeline
Model_path = joblib.load("models/fault_pipeline.pkl")

@app.get("/")
def root():
    return {"message": "Welcome to the Automotive Fault Prediction API. Use the /predict endpoint to get predictions."}

class MachineInput(BaseModel):
    Type: str
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: int

# Map your simple keys -> training column names
RENAME = {
    "Type": "Type",
    "Air_temperature": "Air temperature [K]",
    "Process_temperature": "Process temperature [K]",
    "Rotational_speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool_wear": "Tool wear [min]",
}
EXPECTED = list(RENAME.values())

@app.post("/predict")
def predict(input_data: MachineInput):

    df = pd.DataFrame([input_data.dict()]).rename(columns=RENAME)
    df = create_features(df)

    # Basic presence check
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    pred = Model_path.predict(df)[0]
    return {"prediction": int(pred), "status": "Failure" if pred == 1 else "No Failure"}

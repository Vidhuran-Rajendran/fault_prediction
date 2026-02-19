
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(
    title="Fault Prediction API",
    description="Simple API to test machine failure prediction.",
    version="1.0",
)

MODEL_PATH = Path(r"models\xgb.pkl")
MODEL = None
LOAD_ERR = None

@app.on_event("startup")
def load_model():
    """Load model on startup, but do not crash the app if it fails."""
    global MODEL, LOAD_ERR
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        MODEL = joblib.load(MODEL_PATH)
        LOAD_ERR = None
        print(f"[Startup] Model loaded from: {MODEL_PATH}")
    except Exception as e:
        MODEL = None
        LOAD_ERR = str(e)
        print(f"[Startup ERROR] {LOAD_ERR}")

@app.get("/")
def root():
    return {"message": "API running", "model_loaded": MODEL is not None, "error": LOAD_ERR}

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
    if MODEL is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {LOAD_ERR}")

    df = pd.DataFrame([input_data.dict()]).rename(columns=RENAME)

    # Basic presence check
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    pred = MODEL.predict(df)[0]
    return {"prediction": int(pred), "status": "Failure" if pred == 1 else "No Failure"}

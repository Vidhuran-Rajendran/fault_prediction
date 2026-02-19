import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for the model."""
    df = df.copy()
    # Example feature: Temperature difference
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["mechnanical_stress"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["wear_rate"] = df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1)  # Avoid division by zero
    return df
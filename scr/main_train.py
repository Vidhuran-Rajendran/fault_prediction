import pandas as pd
import os
from train import train
from evaluate import evaluate   

# Load data here to avoid global scope issues in features.py
data_path = r'E:\training\Fault _prediction\data\processed\processed_data.csv'
if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
    exit()  



if __name__ == "__main__":
    df = pd.read_csv(data_path)
    train(df)
    evaluate(df)
    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



data  = r'E:\training\Fault _prediction\data\raw\ai4i2020.csv'
raw_path = r'E:\training\Fault _prediction\data\raw'
processed_path = r'E:\training\Fault _prediction\data\processed'

def clean_data(data):
    df = pd.read_csv(data)
    df = df.ffill()
    print(df.head())
    df = df.drop(['UDI','Product ID','TWF','HDF','PWF','OSF','RNF'], axis=1)
    output_file = processed_path + r'\processed_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    return output_file



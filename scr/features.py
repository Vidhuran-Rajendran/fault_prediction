import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# It's better to pass data in rather than hardcode the path globally, 
# but for now I will keep the path here or move it to main.
# To avoid global execution issues, I'll remove the global df load.

def trans_pipe(df):
    X = df.drop('Machine failure', axis=1)
    
    # Identify columns
    cat = X.select_dtypes(include='object').columns
    num = X.select_dtypes(include='number').columns

    # Fix: Argument is 'transformers', not 'transformer'
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
    ])
    
    return preprocessor

def split_data(df):
    X = df.drop('Machine failure', axis=1)
    y = df['Machine failure']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from features import trans_pipe, split_data
from xgboost import XGBClassifier
import os

def train(df):
   
    # Get splits and preprocessor
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = trans_pipe(df)

    # Create pipelines
    logi = Pipeline(steps=[('preprocess', preprocessor), ('model', LogisticRegression())])
    xgb = Pipeline(steps=[('preprocess', preprocessor), ('model', XGBClassifier(n_estimators=250,
                                                                        learning_rate=0.05,
                                                                        max_depth=3,
                                                                        subsample=0.8,
                                                                        colsample_bytree=0.8,
                                                                        eval_metric='logloss',
                                                                        random_state=42))])

    # Train models
    print("Training Logistic Regression...")
    logi.fit(X_train, y_train)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(logi, 'models/logi.pkl')
    print("Saved models/logi.pkl")
    
    print("Training XGBoost...")
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, 'models/xgb.pkl')
    print("Saved models/xgb.pkl")
    return X_test, y_test



import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from features import create_features
from pipepline import build_pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from evaluate import evaluate
import os

df = pd.read_csv(r"E:\training\Fault _prediction\data\processed\processed_data.csv")

df = create_features(df)

TARGET = "Machine failure"
X = df.drop(columns=[TARGET])
y = df[TARGET]

num = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat = X.select_dtypes(include=["object"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

logi_pipe = build_pipeline(LogisticRegression(max_iter=1000),num,cat)
xgb_pipe = build_pipeline(XGBClassifier(n_estimators=250,
                                       learning_rate=0.05,
                                       max_depth=3,
                                       subsample=0.8,
                                       colsample_bytree=0.8,
                                       eval_metric='logloss',
                                       random_state=42),num,cat)

logi_pipe.fit(X_train, y_train)
xgb_pipe.fit(X_train, y_train)



logi_metrics = evaluate("Logistic Regression", logi_pipe, X_test, y_test)
xgb_metrics = evaluate("XGBoost", xgb_pipe, X_test, y_test)

best_model = logi_pipe if logi_metrics["F1-Score"] > xgb_metrics["F1-Score"] else xgb_pipe
best_name = "Logistic Regression" if logi_metrics["F1-Score"] > xgb_metrics["F1-Score"] else "XGBoost"
print(f"\nBest model: {best_name} with F1-Score: {max(logi_metrics['F1-Score'], xgb_metrics['F1-Score']):.4f}")

joblib.dump(best_model, "models/fault_pipeline.pkl")

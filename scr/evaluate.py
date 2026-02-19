from sklearn.metrics import (accuracy_score,
                            classification_report,
                            confusion_matrix,
                            f1_score,
                            recall_score,
                            precision_score,
                            roc_curve,
                            auc)
import joblib
from features import trans_pipe, split_data
import matplotlib.pyplot as plt
import pandas as pd
import os

def evaluate(df):


    # Get test data
    # We call split_data to get the same split as training
    # Note: ideally we should save the test set separately to guarantee consistency, 
    # but with a fixed random_state in split_data, it should be reproducible.
    _, X_test, _, y_test = split_data(df)
    
    models_paths = ['models/logi.pkl', 'models/xgb.pkl']
    results = []

    for model_path in models_paths:
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found.")
            continue
            
        model = joblib.load(model_path)
        print(f"\nEvaluating {model_path}...")
        
        y_pred = model.predict(X_test)
        
        # Check if model supports predict_proba
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': model_path,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"\n===== {model_path} =====")
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", classification_report(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_path}')
        plt.legend()
        plt.show() # Note: This might block execution if no UI is available, but user had it before.

    print("\n===== SIDE BY SIDE COMPARISON =====")
    print(f"{'Model':<25} {'F1 Score':<10} {'Accuracy':<10}")
    print("-" * 45)
    for row in results:
        print(f"{row['Model']:<25} {row['F1-Score']:<10.3f} {row['Accuracy']:<10.3f}")
    
    return results



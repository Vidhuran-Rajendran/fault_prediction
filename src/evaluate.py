from sklearn.metrics import (accuracy_score,
                            classification_report,
                            confusion_matrix,
                            f1_score,
                            recall_score,
                            precision_score,
                            roc_curve,
                            precision_recall_curve,
                            auc)
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os

def evaluate(name,pipeline,X_test,y_test):
    
    print(f"\n==================Evaluating {name}=======================")
    
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)    
    
    roc = roc_curve(y_test, probs)
    print(f"ROC AUC: {auc(roc[0], roc[1]):.4f}")
    
    precision,recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:\n", cm)
    
    print("Classification Report:\n", classification_report(y_test, preds)) 
  


    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.show() # Note: This might block execution if no UI is available, but user had it before.
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend()
    plt.show() # Note: This might block execution if no UI is available, but user had it before.
    return {
        "Model": name,
        "F1-Score": f1_score(y_test, preds),
        "Accuracy": accuracy_score(y_test, preds)
    }



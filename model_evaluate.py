from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def evaluate_model(model, X_test_tfidf, y_test):
    """Evaluates model performance."""
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

    print(f" Model Accuracy: {acc:.2f}%")
    print("Confusion Matrix:\n", cm)

    np.savetxt("results_summary.txt", cm, fmt='%d')
    print(" Results saved in results_summary.txt")
    return acc, cm

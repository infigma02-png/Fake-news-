from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

def train_model(X_train_tfidf, y_train):
    """Trains PassiveAggressiveClassifier on TF-IDF features."""
    model = PassiveAggressiveClassifier(max_iter=100)
    model.fit(X_train_tfidf, y_train)
    joblib.dump(model, 'saved_model.pkl')
    print("Model training completed and saved.")
    return model

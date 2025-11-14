# main.py — Fake News Detection Main Script
from load_data import load_dataset
from data_preprocess import preprocess_text
from vectorize_text import create_tfidf
from model_train import train_model
from model_evaluate import evaluate_model
from predict_sample import predict_text


# Step 1: Load Data
df = load_dataset("C:/Users/HP/Desktop/fake news/news.csv")

# Step 2: Preprocess
X_train, X_test, y_train, y_test = preprocess_text(df)

# Step 3: TF-IDF Vectorization
X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf(X_train, X_test)

# Step 4: Train Model
model = train_model(X_train_tfidf, y_train)

# Step 5: Evaluate Model
evaluate_model(model, X_test_tfidf, y_test)

# Step 6: Test a Sample Prediction
# Step 6: Test a Sample Prediction
sample_text = "President Donald Trump signed a new bill on economic reform during a press conference at the White House on Monday. The legislation focuses on boosting small business growth and improving employment rates. Officials stated that the bill will create thousands of new jobs across the country. The event was attended by several members of Congress and senior advisors. Experts described it as one of the administration’s key domestic achievements."


predict_text(model, vectorizer, sample_text)


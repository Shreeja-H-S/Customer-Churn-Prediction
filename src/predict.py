import joblib
import pandas as pd


pipeline = joblib.load("models/churn_pipeline.pkl")


def predict_customer(data):

    df = pd.DataFrame([data])

    prediction = pipeline.predict(df)

    return prediction[0]
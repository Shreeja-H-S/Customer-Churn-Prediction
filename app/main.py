from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_customer

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)


class Customer(BaseModel):

    age_group: str
    gender: str
    region: str
    customer_segment: str
    preferred_channel: str

    purchase_frequency: int
    avg_order_value: float
    total_spent: float
    recency_days: int
    website_visits: int

    discount_usage_rate: float
    email_open_rate: float
    cart_abandonment_rate: float

    loyalty_score: int
    engagement_score: int

    churn_risk: str


@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is Running!"
    }


@app.post("/predict")
def predict(customer: Customer):

    prediction = predict_customer(customer.model_dump())

    return {
        "prediction": int(prediction)
    }
# Customer Churn Prediction using XGBoost, FastAPI & Docker

## Overview

This project predicts whether a customer is likely to churn based on customer behavior and purchasing patterns.

The application is built using:

- Python
- XGBoost
- FastAPI
- Scikit-learn
- Docker

The trained model is exposed as a REST API using FastAPI and containerized with Docker.

---

## Features

- Data preprocessing
- Feature engineering
- XGBoost machine learning model
- FastAPI REST API
- Interactive Swagger UI
- Docker containerization
- Joblib model serialization

---

## Project Structure

```text
customer_churn_prediction/
│
├── app/
│   └── main.py
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── train_model.py
│   └── utils.py
│
├── models/
│   └── churn_pipeline.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── tests/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Joblib
- Docker
- Git

---

## Machine Learning Pipeline

1. Load customer dataset
2. Data preprocessing
3. Feature engineering
4. One-Hot Encoding
5. Train-Test Split
6. XGBoost Classifier
7. Model Evaluation
8. Save trained pipeline
9. Deploy using FastAPI
10. Containerize with Docker

---

## Model

Algorithm used:

**XGBoost Classifier**

Evaluation Metric:

- Accuracy: **94.60%**

---

## API Endpoints

### Home

```
GET /
```

Returns

```json
{
  "message": "Customer Churn Prediction API"
}
```

### Predict

```
POST /predict
```

Sample Request

```json
{
  "age_group": "25-34",
  "gender": "Male",
  "region": "North",
  "customer_segment": "VIP",
  "preferred_channel": "Online",
  "purchase_frequency": 8,
  "avg_order_value": 2500,
  "total_spent": 20000,
  "recency_days": 10,
  "website_visits": 25,
  "discount_usage_rate": 0.5,
  "email_open_rate": 0.7,
  "cart_abandonment_rate": 0.2,
  "loyalty_score": 90,
  "engagement_score": 85,
  "churn_risk": "Low"
}
```

Sample Response

```json
{
    "prediction": 0
}
```

---

## Running Locally

Clone the repository

```bash
git clone <repository-url>
```

Move into the project

```bash
cd customer_churn_prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the API

```bash
uvicorn app.main:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

---

## Docker

Build Image

```bash
docker build -t customer-churn-prediction .
```

Run Container

```bash
docker run -d -p 8000:8000 --name churn-api customer-churn-prediction
```

Swagger

```
http://localhost:8000/docs
```

---

## Future Improvements

- Hyperparameter tuning
- Cloud deployment
- CI/CD pipeline
- Model monitoring
- Authentication for API
- Batch prediction endpoint

---

## Author

**Shreeja H S**

GitHub:
https://github.com/Shreeja-H-S
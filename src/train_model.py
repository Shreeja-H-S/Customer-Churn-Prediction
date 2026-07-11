import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


def train_model():

    # Load dataset
    df = pd.read_csv("data/processed/cleaned_customer_churn.csv")

    # Features
    X = df.drop(["customer_id", "churn_flag"], axis=1)

    # Target
    y = df["churn_flag"]

    # Separate categorical and numerical columns
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # XGBoost Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    random_state=42,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    eval_metric="logloss"
                ),
            ),
        ]
    )

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train Model
    pipeline.fit(X_train, y_train)

    # Prediction
    predictions = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("=" * 50)
    print("Model Used : XGBoost")
    print(f"Accuracy   : {accuracy:.4f}")
    print("=" * 50)

    # Save Pipeline
    joblib.dump(
        pipeline,
        "models/churn_pipeline.pkl"
    )

    print("✅ XGBoost Pipeline saved successfully!")


if __name__ == "__main__":
    train_model()
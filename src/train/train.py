"""Train the churn model and save it to disk."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "internet_service"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"


def load_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"monthly_fee": "monthly_charges"})
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Data schema changed, missing columns: {missing}")
    return df


def build_model():
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train(data_path, model_path):
    df = load_data(data_path)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print("F1 :", round(f1_score(y_test, proba > 0.5), 3))
    print("AUC:", round(roc_auc_score(y_test, proba), 3))
    joblib.dump(model, model_path)
    print(f"saved {model_path}")
    return model


if __name__ == "__main__":
    train("data/churn_v2.csv", "model.joblib")
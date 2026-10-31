"""Make sure a trained model exists before tests import the app."""
import os

from src.train.train import train

if not os.path.exists("model.joblib"):
    train("data/churn_v2.csv", "model.joblib")
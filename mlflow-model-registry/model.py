import mlflow.pyfunc

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer


class SentimentDetector(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        self.vectorizer = None
        self.model = None

    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["model"])
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])

    def predict(self, context, input_df: pd.DataFrame) -> np.ndarray:
        inputs = self.vectorizer.transform(input_df["review"])
        return self.model.predict(inputs)


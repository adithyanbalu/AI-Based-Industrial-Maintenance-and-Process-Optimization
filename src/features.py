"""
AI-Based Industrial Maintenance and Process Optimization System
Industrial Physics Feature Engineering Transformer
Module Owner: Adithyan B (Team Lead)
Infosys Springboard Virtual Internship 7.0
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MODEL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


class IndustrialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Physics-informed domain feature engineering for industrial machinery.
    Derives key physical indicators without introducing target leakage:
    1. Temperature Difference (dT): Process Temp - Air Temp (drives Heat Dissipation Failure)
    2. Mechanical Power (W): Rotational speed * Torque * (2 * pi / 60) (drives Power Failure)
    3. Overstrain Index: Torque * Tool wear (drives Overstrain Failure)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=MODEL_FEATURES)

        # Temperature differential
        df["Temp_diff"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )

        # Mechanical power in Watts (Power = Torque * omega)
        df["Power_W"] = (
            df["Rotational speed [rpm]"]
            * df["Torque [Nm]"]
            * (2.0 * np.pi / 60.0)
        )

        # Overstrain metric (Torque * Tool wear)
        df["Strain"] = df["Torque [Nm]"] * df["Tool wear [min]"]

        return df

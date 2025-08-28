import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class Preprocessor:
    def __init__(self, df, scaler="robust", test_size=0.2, random_state=42):
        self.df = df.copy()
        self.scaler_type = scaler
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None

    def scale_features(self):
        """Scale Amount & Time using StandardScaler or RobustScaler"""
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()

        self.df[["Amount", "Time"]] = self.scaler.fit_transform(
            self.df[["Amount", "Time"]]
        )
        return self.df

    def train_test_split(self):
        """Stratified split into train and test sets"""
        X = self.df.drop("Class", axis=1)
        y = self.df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def handle_imbalance(self, X_train, y_train, strategy="baseline"):
        """Apply different imbalance strategies"""
        if strategy == "smote":
            sm = SMOTE(random_state=self.random_state)
            X_res, y_res = sm.fit_resample(X_train, y_train)

        elif strategy == "undersample":
            rus = RandomUnderSampler(random_state=self.random_state)
            X_res, y_res = rus.fit_resample(X_train, y_train)

        elif strategy == "baseline":
            X_res, y_res = X_train, y_train  # no resampling

        elif strategy == "class_weight":
            # handled later inside the model, not here
            X_res, y_res = X_train, y_train

        else:
            raise ValueError("Unknown strategy. Choose from baseline, smote, undersample, class_weight")

        return X_res, y_res

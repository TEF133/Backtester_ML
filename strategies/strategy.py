import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class BaseStrategy:
    """
    Base class for all strategies.
    Every strategy must implement fit() and predict().
    """

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class MLStrategy(BaseStrategy):
    """
    Machine Learning strategy for commodity futures.
    
    Takes a feature matrix (momentum, vol, returns) and learns
    to predict whether the next day return will be positive or negative.
    
    Signal output: +1 (long), -1 (short), 0 (flat)
    """

    def __init__(self, model_type="logistic", threshold=0.55):
        """
        model_type  : "logistic" or "random_forest"
        threshold   : minimum prediction probability to take a trade
                      0.55 means only trade when model is >55% confident
        """
        self.model_type = model_type
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(
                C=0.1,
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _make_labels(self, returns):
        """
        Convert returns into binary labels.
        +1 if next day return > 0
        -1 if next day return < 0
        """
        return np.where(returns > 0, 1, -1)

    def fit(self, X_train, y_train_returns):
        """
        Train the model.
        X_train         : feature matrix (DataFrame)
        y_train_returns : next day returns (Series)
        """
        y = self._make_labels(y_train_returns)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y)
        y_pred = self.model.predict(X_scaled)
        print(f"Training accuracy: {accuracy_score(y, y_pred):.2%}")
        return self

    def predict(self, X):
        """
        Generate trading signals.
        Returns Series of +1, -1, or 0
        """
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        signals = np.where(proba[:, 1] > self.threshold, 1,
                  np.where(proba[:, 0] > self.threshold, -1, 0))
        return pd.Series(signals, index=X.index, name="signal")

    def evaluate(self, X_test, y_test_returns):
        """
        Evaluate model on unseen test data.
        """
        y = self._make_labels(y_test_returns)
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        print("=== OUT-OF-SAMPLE RESULTS ===")
        print(f"Accuracy: {accuracy_score(y, y_pred):.2%}")
        print(classification_report(y, y_pred, target_names=["Short", "Long"]))
        return accuracy_score(y, y_pred)


# Quick test
if __name__ == "__main__":
    from data.data_loader import DataLoader
    from features.feature_engine import FeatureEngine

    # 1. Load data
    loader = DataLoader()
    df = loader.fetch(["CL=F"], start="2018-01-01", end="2024-01-01")

    # 2. Build features
    engine = FeatureEngine(df)
    features = engine.build_all()

    # 3. Flatten MultiIndex columns if present
    if isinstance(features.columns, pd.MultiIndex):
        features = features.xs("CL=F", axis=1, level=1)

    # 4. Build target — next day return
    close = df["Close"]
    if isinstance(features.columns, pd.MultiIndex):
        features = features.xs("CL=F", axis=1, level=1)
    features.columns = [str(c) for c in features.columns]


    close = close.squeeze()
    next_day_return = np.log(close / close.shift(1)).shift(-1)

    # 5. Align features and target
    common_idx = features.index.intersection(next_day_return.dropna().index)
    X = features.loc[common_idx]
    y = next_day_return.loc[common_idx].squeeze()

    # 6. Walk-forward split — 80% train, 20% test
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Training on {len(X_train)} days, testing on {len(X_test)} days\n")

    # 7. Train and evaluate
    strategy = MLStrategy(model_type="logistic", threshold=0.52)
    strategy.fit(X_train, y_train)
    strategy.evaluate(X_test, y_test)

    # 8. Generate signals
    signals = strategy.predict(X_test)
    print(f"\nSignal distribution:\n{signals.value_counts()}")
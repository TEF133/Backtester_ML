import pandas as pd
import numpy as np

class FeatureEngine:
    """
    Transforms raw OHLCV price data into trading signals / features.
    Designed for commodity futures — no lookahead bias.
    All features use .shift(1) so today's signal only uses yesterday's data.
    """

    def __init__(self, df):
        """
        df : raw OHLCV DataFrame from DataLoader
             expects a 'Close' column (MultiIndex or single ticker)
        """
        self.df = df

    def returns(self, periods=1):
        """Daily log returns — the base signal for most ML models."""
        close = self.df["Close"]
        return np.log(close / close.shift(periods)).shift(1)

    def momentum(self, windows=[5, 10, 20]):
        """
        Price momentum over multiple lookback windows.
        Classic signal in commodity trend following.
        e.g. window=20 → return over last 20 days
        """
        close = self.df["Close"]
        features = {}
        for w in windows:
            features[f"mom_{w}d"] = np.log(close / close.shift(w)).shift(1)
        return pd.concat(features, axis=1)

    def volatility(self, windows=[10, 20]):
        """
        Rolling realised volatility.
        Key input for options strategies and position sizing.
        """
        close = self.df["Close"]
        daily_ret = np.log(close / close.shift(1))
        features = {}
        for w in windows:
            features[f"vol_{w}d"] = daily_ret.rolling(w).std().shift(1) * np.sqrt(252)
        return pd.concat(features, axis=1)

    def roll_yield(self, front, back):
        """
        Roll yield = difference between front and back month futures.
        Critical signal for commodity strategies — measures contango/backwardation.
        front, back : price series (pandas Series)
        """
        ry = np.log(front / back).shift(1)
        ry.name = "roll_yield"
        return ry

    def zscore(self, series, window=20):
        """
        Normalise any signal into a z-score.
        Tells you how extreme a signal is vs its recent history.
        Used to generate mean-reversion entry signals.
        """
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return ((series - mean) / std).shift(1)

    def build_all(self, windows_mom=[5, 10, 20], windows_vol=[10, 20]):
        """
        Build the full feature matrix in one call.
        Returns a single DataFrame ready to feed into an ML model.
        """
        features = pd.concat([
            self.returns(),
            self.momentum(windows_mom),
            self.volatility(windows_vol)
        ], axis=1)
        features.dropna(inplace=True)
        return features


# Quick test
if __name__ == "__main__":
    from data.data_loader import DataLoader

    loader = DataLoader()
    df = loader.fetch(["CL=F", "BZ=F"], start="2020-01-01", end="2024-01-01")

    engine = FeatureEngine(df)
    features = engine.build_all()
    print(features.tail())
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Features: {features.columns.tolist()}")
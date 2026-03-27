import pandas as pd
import numpy as np


class FeatureEngine:
    """
    Transforms raw OHLCV price data into trading signals / features.
    Designed for commodity futures — no lookahead bias.
    All features use .shift(1) so today's signal only uses yesterday's data.
    """

    def __init__(self, df):
        self.df = df

    def _get_close(self, ticker=None):
        """Helper to extract close prices cleanly."""
        close = self.df["Close"]
        if isinstance(close.columns, pd.MultiIndex) and ticker:
            return close[ticker]
        if isinstance(close, pd.DataFrame) and len(close.columns) == 1:
            return close.squeeze()
        return close

    def returns(self, ticker=None, periods=1):
        """Daily log returns."""
        close = self._get_close(ticker)
        return np.log(close / close.shift(periods)).shift(1)

    def momentum(self, ticker=None, windows=[5, 10, 20]):
        """
        Price momentum over multiple lookback windows.
        Core CTA signal — trend following on commodities.
        """
        close = self._get_close(ticker)
        features = {}
        for w in windows:
            features[f"mom_{w}d"] = np.log(close / close.shift(w)).shift(1)
        return pd.DataFrame(features, index=close.index)

    def volatility(self, ticker=None, windows=[10, 20]):
        """
        Realised volatility — annualised.
        Used for position sizing and options strategies.
        """
        close = self._get_close(ticker)
        daily_ret = np.log(close / close.shift(1))
        features = {}
        for w in windows:
            features[f"vol_{w}d"] = daily_ret.rolling(w).std().shift(1) * np.sqrt(252)
        return pd.DataFrame(features, index=close.index)

    def roll_yield(self, front_ticker, back_ticker):
        """
        Roll yield = log(front / back month price).
        
        POSITIVE roll yield = BACKWARDATION → market expects prices to fall
        → Historically bullish signal — producers hedging, strong spot demand
        
        NEGATIVE roll yield = CONTANGO → market expects prices to rise
        → Bearish signal — storage costs, weak spot demand
        
        Critical signal for: Crude Oil, Natural Gas, Gold, Agricultural
        """
        front = self._get_close(front_ticker)
        back = self._get_close(back_ticker)
        ry = np.log(front / back).shift(1)
        return ry.rename("roll_yield")

    def zscore(self, series, window=20):
        """
        Normalise any signal into a z-score.
        Tells you how extreme the signal is vs recent history.
        Used for mean reversion entries.
        e.g. zscore > 2.0 → signal is 2 std devs above average → likely to revert
        """
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return ((series - mean) / std).shift(1)

    def atr(self, ticker=None, window=14):
        """
        Average True Range — measures market volatility including gaps.
        Better than simple volatility for futures because it captures
        overnight gaps (common in energy markets after OPEC news etc.)
        Used for: stop loss sizing, position sizing
        """
        if ticker and isinstance(self.df.columns, pd.MultiIndex):
            high  = self.df["High"][ticker]
            low   = self.df["Low"][ticker]
            close = self.df["Close"][ticker]
        else:
            high  = self.df["High"].squeeze()
            low   = self.df["Low"].squeeze()
            close = self.df["Close"].squeeze()

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(window).mean().shift(1).rename(f"atr_{window}d")

    def price_zscore(self, ticker=None, window=20):
        """
        Z-score of the price itself vs its rolling mean.
        Mean reversion signal — when price is too far from average,
        expect it to snap back.
        Common in spread trading (crack spread, calendar spreads).
        """
        close = self._get_close(ticker)
        return self.zscore(close, window).rename(f"price_zscore_{window}d")

    def vol_ratio(self, ticker=None):
        """
        Short-term vol / long-term vol ratio.
        > 1.0 → vol is spiking → potential trend starting or regime change
        < 1.0 → vol is compressing → potential breakout coming
        Useful for options strategies — tells you if vol is cheap or expensive.
        """
        close = self._get_close(ticker)
        daily_ret = np.log(close / close.shift(1))
        short_vol = daily_ret.rolling(5).std() * np.sqrt(252)
        long_vol  = daily_ret.rolling(20).std() * np.sqrt(252)
        return (short_vol / long_vol).shift(1).rename("vol_ratio")

    def build_all(self, ticker=None, windows_mom=[5, 10, 20], windows_vol=[10, 20]):
        """
        Build the full feature matrix in one call.
        Returns a single DataFrame ready to feed into an ML model.
        """
        features = pd.concat([
            self.returns(ticker).rename("return_1d"),
            self.momentum(ticker, windows_mom),
            self.volatility(ticker, windows_vol),
            self.price_zscore(ticker, window=20),
            self.vol_ratio(ticker),
            self.atr(ticker)
        ], axis=1)

        features.dropna(inplace=True)
        return features


# Quick test
if __name__ == "__main__":
    from data.data_loader import DataLoader

    loader = DataLoader()
    df = loader.fetch(["CL=F"], start="2018-01-01", end="2024-01-01")

    engine = FeatureEngine(df)
    features = engine.build_all()

    print(features.tail())
    print(f"\nFeature matrix shape : {features.shape}")
    print(f"Features             : {features.columns.tolist()}")
import numpy as np
import pandas as pd


class TrendFollowing:
    """
    Simple Moving Average Crossover strategy for commodity futures.
    
    Logic:
    - When fast MA crosses ABOVE slow MA → BUY (trend is up)
    - When fast MA crosses BELOW slow MA → SELL (trend is down)
    
    This is the core of every CTA fund (Man AHL, Winton, Campbell).
    Simple but robust — works because commodity trends persist.
    """

    def __init__(self, fast=20, slow=60):
        """
        fast : short lookback moving average (days)
        slow : long lookback moving average (days)
        Common pairs: 20/60, 50/200, 10/40
        """
        self.fast = fast
        self.slow = slow

    def fit(self, X_train, y_train=None):
        """No training needed — rule based strategy."""
        return self

    def predict(self, close):
        """
        Generate signals from close price series.
        Returns Series of +1 (long), -1 (short)
        """
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()

        # Signal: +1 when fast > slow, -1 when fast < slow
        signal = np.where(fast_ma > slow_ma, 1, -1)
        signal = pd.Series(signal, index=close.index, name="signal")

        # Shift by 1 — trade tomorrow based on today's signal
        return signal.shift(1)

    def predict_sized(self, close, vol_window=20):
        """
        Volatility-scaled position sizing.
        
        Instead of always trading 1 full unit:
        - Trade MORE when volatility is LOW (signal is cleaner)
        - Trade LESS when volatility is HIGH (risk is higher)
        
        This is how real CTAs size positions.
        Target annual vol = 15% (industry standard)
        """
        raw_signal = self.predict(close)

        # Annualised realised vol
        daily_ret = np.log(close / close.shift(1))
        realised_vol = daily_ret.rolling(vol_window).std() * np.sqrt(252)

        # Scale position: target_vol / realised_vol
        target_vol = 0.15
        position_size = (target_vol / realised_vol).clip(0.1, 3.0)

        # Sized signal = direction × size
        sized = raw_signal * position_size
        return sized.rename("signal")


class MultiAssetTrend:
    """
    Trend following across multiple commodity futures.
    Diversification is the only free lunch in finance.
    
    Trades CL (Crude), BZ (Brent), NG (Nat Gas) simultaneously.
    Combines signals with equal weighting.
    """

    def __init__(self, fast=20, slow=60):
        self.fast = fast
        self.slow = slow
        self.strategies = {}

    def fit(self, X_train=None, y_train=None):
        return self

    def predict(self, prices_df):
        """
        prices_df : DataFrame with one column per ticker (Close prices)
        Returns   : DataFrame of signals, one column per ticker
        """
        signals = {}
        for ticker in prices_df.columns:
            strat = TrendFollowing(self.fast, self.slow)
            signals[ticker] = strat.predict_sized(prices_df[ticker])
        return pd.DataFrame(signals)


# Quick test
if __name__ == "__main__":
    from data.data_loader import DataLoader

    loader = DataLoader()

    # Single asset test
    print("=== SINGLE ASSET: Crude Oil ===")
    df = loader.fetch(["CL=F"], start="2010-01-01", end="2024-01-01")
    close = df["Close"].squeeze()

    strat = TrendFollowing(fast=20, slow=60)
    signals = strat.predict_sized(close)

    print(f"Signal distribution:\n{signals.describe()}")
    print(f"Days long  : {(signals > 0).sum()}")
    print(f"Days short : {(signals < 0).sum()}")

    # Multi asset test
    print("\n=== MULTI ASSET: Energy Complex ===")
    df_multi = loader.fetch(
        ["CL=F", "BZ=F", "NG=F"],
        start="2010-01-01",
        end="2024-01-01"
    )
    close_multi = df_multi["Close"]
    close_multi.columns = close_multi.columns.droplevel(0) \
        if isinstance(close_multi.columns, pd.MultiIndex) else close_multi.columns

    multi = MultiAssetTrend(fast=20, slow=60)
    signals_multi = multi.predict(close_multi)
    print(f"\nSignals shape: {signals_multi.shape}")
    print(signals_multi.tail())
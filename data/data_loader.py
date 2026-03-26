import yfinance as yf
import pandas as pd
import os

class DataLoader:
    """
    Downloads and saves historical price data.
    Uses yfinance for now — Bloomberg API will replace this later.
    """

    def __init__(self, save_folder="data/raw"):
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)

    def fetch(self, tickers, start, end, interval="1d"):
        """
        Fetch OHLCV data for a list of tickers.
        
        tickers  : list of strings e.g. ["AAPL", "MSFT", "SPY"]
        start    : string e.g. "2020-01-01"
        end      : string e.g. "2024-01-01"
        interval : "1d" for daily, "1h" for hourly
        """
        print(f"Fetching data for: {tickers}")
        df = yf.download(tickers, start=start, end=end, interval=interval)
        print(f"Done. Shape: {df.shape}")
        return df

    def save(self, df, filename):
        """Save dataframe to parquet format (fast and efficient)."""
        path = os.path.join(self.save_folder, filename)
        df.to_parquet(path)
        print(f"Saved to {path}")

    def load(self, filename):
        """Load previously saved data."""
        path = os.path.join(self.save_folder, filename)
        df = pd.read_parquet(path)
        print(f"Loaded {filename} — shape: {df.shape}")
        return df


# Quick test — run this file directly to check everything works
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.fetch(["AAPL", "MSFT", "SPY"], start="2020-01-01", end="2024-01-01")
    print(df.tail())
    loader.save(df, "test_data.parquet")
    df2 = loader.load("test_data.parquet")
    print("Reload successful!")
    
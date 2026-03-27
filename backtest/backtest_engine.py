import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Simulates trading a strategy on historical data.
    Designed for commodity futures — includes transaction costs and position sizing.
    
    Inputs  : signals (+1, -1, 0) from MLStrategy
    Outputs : P&L, Sharpe ratio, drawdown, trade log
    """

    def __init__(self, signals, returns, transaction_cost=0.001, slippage=0.0005):
        """
        signals          : Series of +1, -1, 0 (from MLStrategy.predict)
        returns          : Series of actual next-day returns
        transaction_cost : cost per trade as fraction (0.001 = 0.1%)
        slippage         : market impact cost (0.0005 = 0.05%)
        """
        self.signals = signals
        self.returns = returns
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.results = None

    def run(self):
        """
        Run the backtest.
        Returns a DataFrame with daily P&L and portfolio statistics.
        """
        # Align signals and returns on same dates
        idx = self.signals.index.intersection(self.returns.index)
        signals = self.signals.loc[idx]
        returns = self.returns.loc[idx]

        # Calculate trade costs — charged every time position changes
        position_changes = signals.diff().abs()
        costs = position_changes * (self.transaction_cost + self.slippage)

        # Daily strategy return = signal × market return − costs
        strategy_returns = signals * returns - costs

        # Cumulative P&L (starting from 1.0 = $1 invested)
        cumulative = (1 + strategy_returns).cumprod()

        # Buy & hold benchmark (just hold long the whole time)
        benchmark = (1 + returns).cumprod()

        self.results = pd.DataFrame({
            "signal"           : signals,
            "market_return"    : returns,
            "strategy_return"  : strategy_returns,
            "cumulative_pnl"   : cumulative,
            "benchmark"        : benchmark,
            "costs"            : costs
        })

        return self.results

    def sharpe_ratio(self, periods_per_year=252):
        """
        Sharpe ratio = annualised return / annualised volatility.
        Industry standard risk-adjusted performance metric.
        > 1.0 = decent, > 2.0 = very good, > 3.0 = exceptional
        """
        if self.results is None:
            raise ValueError("Run backtest first.")
        r = self.results["strategy_return"]
        mean = r.mean() * periods_per_year
        std = r.std() * np.sqrt(periods_per_year)
        return mean / std if std != 0 else 0

    def max_drawdown(self):
        """
        Maximum peak-to-trough loss.
        e.g. -0.25 means the strategy lost 25% from its peak at some point.
        Critical metric for risk management.
        """
        if self.results is None:
            raise ValueError("Run backtest first.")
        cumulative = self.results["cumulative_pnl"]
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def summary(self):
        """Print a clean performance summary."""
        if self.results is None:
            raise ValueError("Run backtest first.")

        r = self.results["strategy_return"]
        total_return = self.results["cumulative_pnl"].iloc[-1] - 1
        benchmark_return = self.results["benchmark"].iloc[-1] - 1
        n_trades = (self.results["signal"].diff().abs() > 0).sum()

        print("=" * 40)
        print("       BACKTEST RESULTS SUMMARY")
        print("=" * 40)
        print(f"Period          : {self.results.index[0].date()} → {self.results.index[-1].date()}")
        print(f"Total return    : {total_return:.2%}")
        print(f"Benchmark       : {benchmark_return:.2%}")
        print(f"Sharpe ratio    : {self.sharpe_ratio():.2f}")
        print(f"Max drawdown    : {self.max_drawdown():.2%}")
        print(f"Annualised vol  : {r.std() * np.sqrt(252):.2%}")
        print(f"Win rate        : {(r > 0).mean():.2%}")
        print(f"Number of trades: {n_trades}")
        print(f"Total costs     : {self.results['costs'].sum():.4f}")
        print("=" * 40)


# Quick test
if __name__ == "__main__":
    from data.data_loader import DataLoader
    from features.feature_engine import FeatureEngine
    from strategies.strategy import MLStrategy

    # 1. Load data
    loader = DataLoader()
    df = loader.fetch(["CL=F"], start="2018-01-01", end="2024-01-01")

    # 2. Build features
    engine = FeatureEngine(df)
    features = engine.build_all()
    if isinstance(features.columns, pd.MultiIndex):
        features = features.xs("CL=F", axis=1, level=1)
    features.columns = [str(c) for c in features.columns]

    # 3. Build target
    close = df["Close"]
    if isinstance(close.columns, pd.MultiIndex):
        close = close.xs("CL=F", axis=1, level=1)
    close = close.squeeze()
    next_day_return = np.log(close / close.shift(1)).shift(-1)

    # 4. Align
    common_idx = features.index.intersection(next_day_return.dropna().index)
    X = features.loc[common_idx]
    y = next_day_return.loc[common_idx].squeeze()

    # 5. Walk-forward split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 6. Train strategy
    strategy = MLStrategy(model_type="logistic", threshold=0.52)
    strategy.fit(X_train, y_train)
    signals = strategy.predict(X_test)

    # 7. Run backtest
    bt = BacktestEngine(signals, y_test)
    results = bt.run()
    bt.summary()
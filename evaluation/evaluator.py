import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Evaluator:
    """
    Visualises and reports backtest performance.
    Produces P&L curve, drawdown chart, return distribution and signal analysis.
    """

    def __init__(self, results):
        """
        results : DataFrame from BacktestEngine.run()
        """
        self.results = results

    def plot_pnl(self, title="Strategy Performance", save_path=None):
        """
        Full performance dashboard:
        - Cumulative P&L vs benchmark
        - Drawdown chart
        - Daily return distribution
        - Signal over time
        """
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4)

        # --- Chart 1: Cumulative P&L vs Benchmark ---
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.results.index,
                 self.results["cumulative_pnl"],
                 label="Strategy", color="steelblue", linewidth=1.5)
        ax1.plot(self.results.index,
                 self.results["benchmark"],
                 label="Buy & Hold", color="orange",
                 linewidth=1.5, linestyle="--")
        ax1.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
        ax1.set_title("Cumulative P&L")
        ax1.set_ylabel("Portfolio Value (start = 1.0)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Chart 2: Drawdown ---
        ax2 = fig.add_subplot(gs[1, :])
        cumulative = self.results["cumulative_pnl"]
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        ax2.fill_between(self.results.index, drawdown, 0,
                         color="red", alpha=0.4, label="Drawdown")
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown %")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Chart 3: Return Distribution ---
        ax3 = fig.add_subplot(gs[2, 0])
        self.results["strategy_return"].hist(
            bins=40, ax=ax3, color="steelblue", edgecolor="white", alpha=0.8)
        ax3.axvline(0, color="red", linestyle="--", linewidth=1)
        ax3.set_title("Daily Return Distribution")
        ax3.set_xlabel("Daily Return")
        ax3.grid(True, alpha=0.3)

        # --- Chart 4: Signals over time ---
        ax4 = fig.add_subplot(gs[2, 1])
        signal_counts = self.results["signal"].value_counts()
        colors = {1: "green", 0: "grey", -1: "red"}
        bars = ax4.bar(
            ["Long (+1)", "Flat (0)", "Short (-1)"],
            [signal_counts.get(1, 0),
             signal_counts.get(0, 0),
             signal_counts.get(-1, 0)],
            color=["green", "grey", "red"],
            edgecolor="white"
        )
        ax4.set_title("Signal Distribution")
        ax4.set_ylabel("Number of Days")
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Chart saved to {save_path}")

        plt.show()

    def monthly_returns(self):
        """
        Monthly return heatmap table.
        Shows which months/years the strategy performed well.
        """
        r = self.results["strategy_return"].copy()
        r.index = pd.to_datetime(r.index)
        monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        table = monthly.groupby([monthly.index.year,
                                  monthly.index.month]).first().unstack()
        table.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]

        print("\n=== MONTHLY RETURNS ===")
        print(table.applymap(lambda x: f"{x:.1%}" if pd.notna(x) else "-"))
        return table

    def full_report(self, title="Strategy Report", save_path=None):
        """Run everything — chart + monthly returns table."""
        self.plot_pnl(title=title, save_path=save_path)
        self.monthly_returns()


# Quick test
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from data.data_loader import DataLoader
    from features.feature_engine import FeatureEngine
    from strategies.strategy import MLStrategy
    from backtest.backtest_engine import BacktestEngine

    # 1. Load data
    loader = DataLoader()
    df = loader.fetch(["CL=F"], start="2018-01-01", end="2024-01-01")

    # 2. Features
    engine = FeatureEngine(df)
    features = engine.build_all()
    if isinstance(features.columns, pd.MultiIndex):
        features = features.xs("CL=F", axis=1, level=1)
    features.columns = [str(c) for c in features.columns]

    # 3. Target
    close = df["Close"]
    if isinstance(close.columns, pd.MultiIndex):
        close = close.xs("CL=F", axis=1, level=1)
    close = close.squeeze()
    next_day_return = np.log(close / close.shift(1)).shift(-1)

    # 4. Align
    common_idx = features.index.intersection(next_day_return.dropna().index)
    X = features.loc[common_idx]
    y = next_day_return.loc[common_idx].squeeze()

    # 5. Split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 6. Train
    strategy = MLStrategy(model_type="logistic", threshold=0.52)
    strategy.fit(X_train, y_train)
    signals = strategy.predict(X_test)

    # 7. Backtest
    bt = BacktestEngine(signals, y_test)
    results = bt.run()
    bt.summary()

    # 8. Evaluate
    evaluator = Evaluator(results)
    evaluator.full_report(
        title="CL=F Logistic Regression Strategy",
        save_path="evaluation/strategy_report.png"
    )
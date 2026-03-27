import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Evaluator:
    """
    Visualises and reports backtest performance.
    """

    def __init__(self, results):
        self.results = results

    def sharpe_ratio(self, periods_per_year=252):
        r    = self.results["strategy_return"]
        mean = r.mean() * periods_per_year
        std  = r.std() * np.sqrt(periods_per_year)
        return mean / std if std != 0 else 0

    def max_drawdown(self):
        cumulative  = self.results["cumulative_pnl"]
        rolling_max = cumulative.cummax()
        drawdown    = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def var(self, confidence=0.95):
        return self.results["strategy_return"].quantile(1 - confidence)

    def cvar(self, confidence=0.95):
        var  = self.var(confidence)
        tail = self.results["strategy_return"][
            self.results["strategy_return"] <= var
        ]
        return tail.mean()

    def plot_pnl(self, title="Strategy Performance", save_path=None):
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45)

        sharpe    = self.sharpe_ratio()
        mdd       = self.max_drawdown()
        var95     = self.var(0.95)
        cvar95    = self.cvar(0.95)
        total_ret = self.results["cumulative_pnl"].iloc[-1] - 1
        bench_ret = self.results["benchmark"].iloc[-1] - 1

        # --- Chart 1: Cumulative P&L ---
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
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        metrics_text = (
            f"Sharpe: {sharpe:.2f}  |  "
            f"Max DD: {mdd:.1%}  |  "
            f"VaR 95%: {var95:.2%}  |  "
            f"CVaR 95%: {cvar95:.2%}  |  "
            f"Return: {total_ret:.1%}  |  "
            f"Benchmark: {bench_ret:.1%}"
        )
        ax1.text(0.01, 0.02, metrics_text,
                 transform=ax1.transAxes, fontsize=8.5,
                 bbox=dict(boxstyle="round",
                           facecolor="lightyellow", alpha=0.8))

        # --- Chart 2: Drawdown ---
        ax2 = fig.add_subplot(gs[1, :])
        cumulative  = self.results["cumulative_pnl"]
        rolling_max = cumulative.cummax()
        drawdown    = (cumulative - rolling_max) / rolling_max
        ax2.fill_between(self.results.index, drawdown, 0,
                         color="red", alpha=0.4, label="Drawdown")
        ax2.axhline(mdd, color="darkred", linestyle="--",
                    linewidth=1, label=f"Max DD: {mdd:.1%}")
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown %")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Chart 3: Return Distribution ---
        ax3 = fig.add_subplot(gs[2, 0])
        ret = self.results["strategy_return"]
        ax3.hist(ret, bins=60, color="steelblue",
                 edgecolor="white", alpha=0.8)
        ax3.axvline(0, color="black", linestyle="--",
                    linewidth=1, label="Zero")
        ax3.axvline(var95, color="red", linestyle="--",
                    linewidth=1.5, label=f"VaR 95%: {var95:.2%}")
        ax3.axvline(cvar95, color="darkred", linestyle="--",
                    linewidth=1.5, label=f"CVaR 95%: {cvar95:.2%}")
        ax3.set_title("Daily Return Distribution")
        ax3.set_xlabel("Daily Return")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # --- Chart 4: Signal Distribution ---
        ax4 = fig.add_subplot(gs[2, 1])
        signals = self.results["signal"]
        ax4.hist(signals, bins=40, color="steelblue",
                 edgecolor="white", alpha=0.8)
        ax4.axvline(0, color="red", linestyle="--", linewidth=1)
        ax4.axvline(signals.mean(), color="green", linestyle="--",
                    linewidth=1, label=f"Mean: {signals.mean():.2f}")
        ax4.set_title("Signal Distribution")
        ax4.set_xlabel("Signal Size")
        ax4.set_ylabel("Number of Days")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Chart saved to {save_path}")
        plt.show()

    def monthly_returns(self):
        r = self.results["strategy_return"].copy()
        r.index = pd.to_datetime(r.index)
        monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        table   = monthly.groupby(
            [monthly.index.year, monthly.index.month]
        ).first().unstack()
        table.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
        print("\n=== MONTHLY RETURNS ===")
        print(table.applymap(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"
        ))
        return table

    def full_report(self, title="Strategy Report", save_path=None):
        self.plot_pnl(title=title, save_path=save_path)
        self.monthly_returns()


# =============================================================================
# COMPARISON: Trend Following vs Triple Barrier ML
# =============================================================================

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from features.feature_engine import FeatureEngine
    from strategies.trend_following import TrendFollowing
    from strategies.triple_barrier import (
        TripleBarrierLabeler, TripleBarrierStrategy
    )
    from backtest.backtest_engine import BacktestEngine

    # 1. Load data
    loader = DataLoader()
    df     = loader.fetch(["CL=F"], start="2010-01-01", end="2024-01-01")
    close  = df["Close"].squeeze()
    next_day_return = np.log(close / close.shift(1)).shift(-1)

    # =========================================================
    # STRATEGY A — Trend Following (baseline)
    # =========================================================
    print("\n" + "=" * 60)
    print("STRATEGY A — Trend Following (20/60 MA)")
    print("=" * 60)

    tf_strat   = TrendFollowing(fast=20, slow=60)
    tf_signals = tf_strat.predict_sized(close)

    common_tf  = tf_signals.index.intersection(
                     next_day_return.dropna().index)
    bt_tf      = BacktestEngine(
                     tf_signals.loc[common_tf],
                     next_day_return.loc[common_tf]
                 )
    results_tf = bt_tf.run()
    bt_tf.summary()

    # =========================================================
    # STRATEGY B — Triple Barrier ML (walk-forward OOS)
    # =========================================================
    print("\n" + "=" * 60)
    print("STRATEGY B — Triple Barrier ML (Walk-Forward OOS)")
    print("=" * 60)

    # Labels
    labeler = TripleBarrierLabeler(
        upper_mult=1.5,
        lower_mult=1.5,
        max_days=10
    )
    labels = labeler.label(close)
    print(f"Label distribution:\n{labels['label'].value_counts()}\n")

    # Features
    engine   = FeatureEngine(df)
    features = engine.build_all()
    if isinstance(features.columns, pd.MultiIndex):
        features = features.xs("CL=F", axis=1, level=1)
    features.columns = [str(c) for c in features.columns]

    # Walk-forward signals
    tb_strat   = TripleBarrierStrategy(
                     model_type="random_forest",
                     train_years=3,
                     test_months=6
                 )
    tb_signals = tb_strat.walk_forward(features, labels, close.index)

    # Vol scale
    daily_ret    = np.log(close / close.shift(1))
    realised_vol = daily_ret.rolling(20).std() * np.sqrt(252)
    vol_scalar   = (0.15 / realised_vol).clip(0.1, 2.0)
    tb_signals_scaled = (tb_signals * vol_scalar).reindex(tb_signals.index)

    common_tb  = tb_signals_scaled.index.intersection(
                     next_day_return.dropna().index)
    bt_tb      = BacktestEngine(
                     tb_signals_scaled.loc[common_tb],
                     next_day_return.loc[common_tb]
                 )
    results_tb = bt_tb.run()
    bt_tb.summary()

    # =========================================================
    # COMPARISON CHART
    # =========================================================
    print("\nGenerating comparison chart...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Strategy Comparison: Trend Following vs Triple Barrier ML",
                 fontsize=14, fontweight="bold")

    # P&L comparison
    ax = axes[0]
    ax.plot(results_tf.index, results_tf["cumulative_pnl"],
            label=f"Trend Following "
                  f"(Sharpe: {bt_tf.sharpe_ratio():.2f})",
            color="steelblue", linewidth=1.5)
    ax.plot(results_tb.index, results_tb["cumulative_pnl"],
            label=f"Triple Barrier ML "
                  f"(Sharpe: {bt_tb.sharpe_ratio():.2f})",
            color="green", linewidth=1.5)
    ax.plot(results_tf.index, results_tf["benchmark"],
            label="Buy & Hold",
            color="orange", linewidth=1, linestyle="--")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title("Cumulative P&L")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Drawdown comparison
    ax2 = axes[1]
    for results, color, label in [
        (results_tf, "steelblue", "Trend Following"),
        (results_tb, "green",     "Triple Barrier ML")
    ]:
        cum = results["cumulative_pnl"]
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax2.fill_between(results.index, dd, 0,
                         alpha=0.3, color=color, label=label)
    ax2.set_title("Drawdown Comparison")
    ax2.set_ylabel("Drawdown %")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation/comparison_report.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Individual reports
    Evaluator(results_tf).full_report(
        title="Trend Following (20/60 MA)",
        save_path="evaluation/trend_report.png"
    )
    Evaluator(results_tb).full_report(
        title="Triple Barrier ML (Walk-Forward OOS)",
        save_path="evaluation/triple_barrier_report.png"
    )
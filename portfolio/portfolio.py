import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Portfolio:
    """
    Multi-asset portfolio manager for commodity futures.
    
    Combines signals from multiple assets with:
    - Volatility scaling per asset (equal risk contribution)
    - Optional regime-based asset rotation
    - Portfolio-level risk management
    """

    def __init__(self, target_vol=0.15, max_position=2.0):
        self.target_vol   = target_vol
        self.max_position = max_position

    def vol_scale(self, signals, returns, vol_window=20):
        """Scale each asset signal by its volatility — risk parity."""
        scaled = pd.DataFrame(index=signals.index)
        for col in signals.columns:
            if col not in returns.columns:
                continue
            ret    = returns[col]
            vol    = ret.rolling(vol_window).std() * np.sqrt(252)
            vol    = vol.reindex(signals.index).ffill()
            vol    = vol.replace(0, np.nan).fillna(vol.mean())
            scalar = (self.target_vol / vol).clip(0.1, self.max_position)
            scaled[col] = signals[col] * scalar
        return scaled

    def regime_rotate(self, scaled_signals, regime_labels,
                      regime_stats, returns):
        """
        Rotate asset weights based on current market regime.
        
        BULL TREND → overweight risk-on (CL, NG, HG)
        BEAR TREND → overweight safe haven (GC, SI)
        HIGH VOL   → reduce all positions by 60%
        SIDEWAYS   → equal weight (no change)
        """
        risk_on    = ["CL=F", "BZ=F", "NG=F", "HG=F", "ZC=F", "ZW=F"]
        safe_haven = ["GC=F", "SI=F"]

        adj = scaled_signals.copy()

        # Smooth regime — only act when stable for 5+ days
        stable_regime = regime_labels.rolling(
            5, min_periods=5
        ).apply(lambda x: x.iloc[-1] if len(set(x)) == 1 else -1)

        for date in adj.index:
            if date not in stable_regime.index:
                continue

            r = stable_regime.loc[date]

            # Skip NaN or unstable
            if pd.isna(r) or r == -1:
                continue

            r = int(r)

            if r not in regime_stats:
                continue

            label = regime_stats[r]["label"]

            for col in adj.columns:
                if label == "BULL TREND":
                    if col in risk_on:
                        adj.loc[date, col] *= 1.4
                    elif col in safe_haven:
                        adj.loc[date, col] *= 0.6

                elif label == "BEAR TREND":
                    if col in safe_haven:
                        adj.loc[date, col] *= 1.4
                    elif col in risk_on:
                        adj.loc[date, col] *= 0.6

                elif label == "HIGH VOL":
                    adj.loc[date, col] *= 0.4

                # SIDEWAYS → no change

        return adj

    def run_backtest(self, signals, returns,
                     regime_labels=None, regime_stats=None):
        """
        Run portfolio backtest.
        Optionally applies regime-based rotation.
        """
        scaled = self.vol_scale(signals, returns)

        if regime_labels is not None and regime_stats is not None:
            scaled = self.regime_rotate(
                scaled, regime_labels, regime_stats, returns
            )

        # Individual asset P&L
        asset_pnl = {}
        for col in scaled.columns:
            if col not in returns.columns:
                continue
            strat_ret      = scaled[col] * returns[col]
            asset_pnl[col] = (1 + strat_ret).cumprod()
        asset_pnl_df = pd.DataFrame(asset_pnl)

        # Combined portfolio P&L
        n             = scaled.shape[1]
        portfolio_ret = pd.Series(0.0, index=scaled.index)
        for col in scaled.columns:
            if col in returns.columns:
                portfolio_ret += scaled[col] * returns[col] / n

        portfolio_cum = (1 + portfolio_ret).cumprod()
        benchmark_ret = returns.mean(axis=1)
        benchmark_cum = (1 + benchmark_ret).cumprod()

        portfolio_results = pd.DataFrame({
            "strategy_return" : portfolio_ret,
            "cumulative_pnl"  : portfolio_cum,
            "benchmark"       : benchmark_cum,
            "signal"          : scaled.mean(axis=1)
        })

        return portfolio_results, asset_pnl_df

    def sharpe(self, returns_series, periods=252):
        mean = returns_series.mean() * periods
        std  = returns_series.std() * np.sqrt(periods)
        return mean / std if std != 0 else 0

    def max_drawdown(self, cumulative):
        rolling_max = cumulative.cummax()
        dd = (cumulative - rolling_max) / rolling_max
        return dd.min()

    def plot(self, portfolio_results, asset_pnl_df,
             title="Multi-Asset Portfolio", save_path=None):
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45)

        sharpe_val = self.sharpe(portfolio_results["strategy_return"])
        mdd        = self.max_drawdown(portfolio_results["cumulative_pnl"])
        total_ret  = portfolio_results["cumulative_pnl"].iloc[-1] - 1

        # --- Chart 1: Portfolio P&L ---
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(portfolio_results.index,
                 portfolio_results["cumulative_pnl"],
                 label="Portfolio", color="steelblue", linewidth=2)
        ax1.plot(portfolio_results.index,
                 portfolio_results["benchmark"],
                 label="Equal Weight B&H", color="orange",
                 linewidth=1.5, linestyle="--")
        ax1.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
        ax1.set_title("Portfolio Cumulative P&L")
        ax1.set_ylabel("Portfolio Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        metrics = (f"Sharpe: {sharpe_val:.2f}  |  "
                   f"Max DD: {mdd:.1%}  |  "
                   f"Total Return: {total_ret:.1%}")
        ax1.text(0.01, 0.02, metrics, transform=ax1.transAxes,
                 fontsize=9,
                 bbox=dict(boxstyle="round",
                           facecolor="lightyellow", alpha=0.8))

        # --- Chart 2: Individual Assets ---
        ax2    = fig.add_subplot(gs[1, 0])
        colors = ["steelblue", "darkblue", "green",
                  "gold", "silver", "orange", "brown", "pink"]
        for i, col in enumerate(asset_pnl_df.columns):
            ax2.plot(asset_pnl_df.index, asset_pnl_df[col],
                     label=col,
                     color=colors[i % len(colors)],
                     linewidth=1.2)
        ax2.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
        ax2.set_title("Individual Asset P&L")
        ax2.set_ylabel("Value")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # --- Chart 3: Drawdown ---
        ax3 = fig.add_subplot(gs[1, 1])
        cum = portfolio_results["cumulative_pnl"]
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax3.fill_between(portfolio_results.index, dd, 0,
                         color="red", alpha=0.4)
        ax3.axhline(mdd, color="darkred", linestyle="--",
                    linewidth=1, label=f"Max DD: {mdd:.1%}")
        ax3.set_title("Portfolio Drawdown")
        ax3.set_ylabel("Drawdown %")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # --- Chart 4: Rolling Sharpe ---
        ax4         = fig.add_subplot(gs[2, 0])
        rolling_ret = portfolio_results["strategy_return"]
        roll_sharpe = (
            rolling_ret.rolling(126).mean() * 252 /
            (rolling_ret.rolling(126).std() * np.sqrt(252))
        )
        ax4.plot(portfolio_results.index, roll_sharpe,
                 color="steelblue", linewidth=1.2)
        ax4.axhline(0, color="red", linestyle="--", linewidth=1)
        ax4.axhline(1, color="green", linestyle="--",
                    linewidth=1, label="Sharpe=1")
        ax4.set_title("Rolling 6M Sharpe Ratio")
        ax4.set_ylabel("Sharpe")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # --- Chart 5: Correlation heatmap ---
        ax5 = fig.add_subplot(gs[2, 1])
        if asset_pnl_df.shape[1] > 1:
            ret_df = asset_pnl_df.pct_change().dropna()
            corr   = ret_df.corr()
            im     = ax5.imshow(corr, cmap="RdYlGn",
                                vmin=-1, vmax=1, aspect="auto")
            plt.colorbar(im, ax=ax5)
            ax5.set_xticks(range(len(corr.columns)))
            ax5.set_yticks(range(len(corr.columns)))
            ax5.set_xticklabels(
                corr.columns, fontsize=7, rotation=45)
            ax5.set_yticklabels(corr.columns, fontsize=7)
            for i in range(len(corr)):
                for j in range(len(corr.columns)):
                    ax5.text(j, i, f"{corr.iloc[i,j]:.2f}",
                             ha="center", va="center", fontsize=7)
        ax5.set_title("Asset Return Correlation")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()

    def print_summary(self, portfolio_results, asset_pnl_df,
                      label="Portfolio"):
        r         = portfolio_results["strategy_return"]
        total_ret = portfolio_results["cumulative_pnl"].iloc[-1] - 1
        bench_ret = portfolio_results["benchmark"].iloc[-1] - 1

        print("=" * 55)
        print(f"     {label.upper()} SUMMARY")
        print("=" * 55)
        print(f"Period       : {portfolio_results.index[0].date()} "
              f"→ {portfolio_results.index[-1].date()}")
        print(f"Total return : {total_ret:.2%}")
        print(f"Benchmark    : {bench_ret:.2%}")
        print(f"Sharpe ratio : {self.sharpe(r):.2f}")
        print(f"Max drawdown : "
              f"{self.max_drawdown(portfolio_results['cumulative_pnl']):.2%}")
        print(f"Ann. vol     : {r.std() * np.sqrt(252):.2%}")
        print(f"Win rate     : {(r > 0).mean():.2%}")
        print()
        print("Individual Asset Performance:")
        for col in asset_pnl_df.columns:
            ret = asset_pnl_df[col].pct_change().dropna()
            print(f"  {col:8s} → "
                  f"Return: {asset_pnl_df[col].iloc[-1]-1:.2%} | "
                  f"Sharpe: {self.sharpe(ret):.2f}")
        print("=" * 55)


# =============================================================================
# MAIN — Compare plain vs regime-aware multi-asset portfolio
# =============================================================================

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from strategies.trend_following import TrendFollowing
    from features.regime_detector import RegimeDetector

    # 1. Load data
    loader  = DataLoader()
    tickers = ["CL=F", "BZ=F", "NG=F", "GC=F", "SI=F", "HG=F"]
    df      = loader.fetch(tickers, start="2010-01-01", end="2024-01-01")

    close = df["Close"]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.droplevel(0)
    close = close.dropna(thresh=int(len(close) * 0.8), axis=1)
    print(f"Assets: {close.columns.tolist()}")

    # 2. Build returns and signals
    returns = np.log(close / close.shift(1)).shift(-1)
    signals = pd.DataFrame(index=close.index)
    for ticker in close.columns:
        strat = TrendFollowing(fast=20, slow=60)
        signals[ticker] = strat.predict(close[ticker])

    # 3. Fit regime detector on Crude Oil
    print("\nFitting regime detector on CL=F...")
    cl_close = close["CL=F"] if "CL=F" in close.columns \
               else close.iloc[:, 0]
    detector = RegimeDetector(n_regimes=4)
    detector.fit(cl_close)
    regime_labels, regime_probs = detector.predict(cl_close)
    regime_stats = detector.label_regimes(cl_close, regime_labels)
    detector.print_summary()

    # 4. Plain portfolio
    print("\n--- PLAIN PORTFOLIO ---")
    portfolio = Portfolio(target_vol=0.15)
    res_plain, pnl_plain = portfolio.run_backtest(signals, returns)
    portfolio.print_summary(res_plain, pnl_plain, "Plain Portfolio")

    # 5. Regime-aware portfolio
    print("\n--- REGIME-AWARE PORTFOLIO ---")
    res_regime, pnl_regime = portfolio.run_backtest(
        signals, returns,
        regime_labels = regime_labels,
        regime_stats  = regime_stats
    )
    portfolio.print_summary(
        res_regime, pnl_regime, "Regime-Aware Portfolio"
    )

    # 6. Comparison chart
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        "Plain vs Regime-Aware Multi-Asset Portfolio",
        fontsize=14, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(res_plain.index, res_plain["cumulative_pnl"],
            label=f"Plain "
                  f"(Sharpe: "
                  f"{portfolio.sharpe(res_plain['strategy_return']):.2f})",
            color="steelblue", linewidth=1.5)
    ax.plot(res_regime.index, res_regime["cumulative_pnl"],
            label=f"Regime-Aware "
                  f"(Sharpe: "
                  f"{portfolio.sharpe(res_regime['strategy_return']):.2f})",
            color="green", linewidth=1.5)
    ax.plot(res_plain.index, res_plain["benchmark"],
            label="Buy & Hold", color="orange", linestyle="--")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title("Cumulative P&L")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for res, color, label in [
        (res_plain,  "steelblue", "Plain"),
        (res_regime, "green",     "Regime-Aware")
    ]:
        cum = res["cumulative_pnl"]
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax2.fill_between(res.index, dd, 0,
                         alpha=0.3, color=color, label=label)
    ax2.set_title("Drawdown Comparison")
    ax2.set_ylabel("Drawdown %")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation/regime_portfolio_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # 7. Individual full reports
    portfolio.plot(
        res_plain, pnl_plain,
        title     = "Plain Multi-Asset Portfolio",
        save_path = "evaluation/plain_portfolio.png"
    )
    portfolio.plot(
        res_regime, pnl_regime,
        title     = "Regime-Aware Multi-Asset Portfolio",
        save_path = "evaluation/regime_portfolio.png"
    )
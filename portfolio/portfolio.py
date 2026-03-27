import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Portfolio:
    """
    Multi-asset portfolio manager for commodity futures.
    
    Combines signals from multiple assets with:
    - Volatility scaling per asset (each contributes equal risk)
    - Correlation-adjusted position sizing
    - Portfolio-level risk management
    
    Assets: Crude Oil (CL=F), Natural Gas (NG=F), Gold (GC=F)
    """

    def __init__(self, target_vol=0.15, max_position=2.0):
        """
        target_vol   : target annualised portfolio volatility (15%)
        max_position : maximum position size per asset
        """
        self.target_vol   = target_vol
        self.max_position = max_position

    def vol_scale(self, signals, returns, vol_window=20):
        """
        Scale each asset's signal by its volatility.
        
        Each asset contributes EQUAL RISK to the portfolio.
        e.g. NG is 3x more volatile than Gold →
             NG position is 3x smaller than Gold position
        
        This is called 'risk parity' — used by Bridgewater, AQR etc.
        """
        scaled = pd.DataFrame(index=signals.index)

        for col in signals.columns:
            if col not in returns.columns:
                continue
            ret = returns[col]
            vol = ret.rolling(vol_window).std() * np.sqrt(252)
            vol = vol.reindex(signals.index).fillna(method="ffill")
            vol = vol.replace(0, np.nan).fillna(vol.mean())

            # Scale: target_vol / asset_vol
            scalar = (self.target_vol / vol).clip(0.1, self.max_position)
            scaled[col] = signals[col] * scalar

        return scaled

    def correlation_matrix(self, returns, window=60):
        """
        Rolling correlation between assets.
        Shows how diversified the portfolio actually is.
        Low correlation = better diversification = higher Sharpe.
        """
        return returns.rolling(window).corr()

    def combine(self, signals, returns, method="equal_weight"):
        """
        Combine multi-asset signals into portfolio weights.
        
        method options:
        - equal_weight   : each asset gets 1/N weight
        - vol_weighted   : smaller weight for more volatile assets
        - signal_weighted: weight by signal strength
        """
        scaled = self.vol_scale(signals, returns)

        if method == "equal_weight":
            n = scaled.shape[1]
            portfolio_signal = scaled.sum(axis=1) / n

        elif method == "vol_weighted":
            # Inverse vol weighting
            vols = {}
            for col in returns.columns:
                if col in scaled.columns:
                    vols[col] = returns[col].rolling(20).std().reindex(
                        scaled.index).fillna(method="ffill")
            vol_df   = pd.DataFrame(vols)
            inv_vol  = 1 / vol_df
            weights  = inv_vol.div(inv_vol.sum(axis=1), axis=0)
            portfolio_signal = (scaled * weights).sum(axis=1)

        elif method == "signal_weighted":
            # Weight by absolute signal strength
            abs_signals = scaled.abs()
            weights     = abs_signals.div(
                abs_signals.sum(axis=1), axis=0
            ).fillna(1 / scaled.shape[1])
            portfolio_signal = (scaled * weights).sum(axis=1)

        return portfolio_signal.rename("portfolio_signal")

    def run_backtest(self, signals, returns):
        """
        Run portfolio backtest across all assets.
        
        Returns:
        - portfolio_results : combined portfolio P&L
        - asset_results     : individual asset P&L
        """
        scaled = self.vol_scale(signals, returns)

        # Individual asset P&L
        asset_pnl = {}
        for col in scaled.columns:
            if col not in returns.columns:
                continue
            strat_ret = scaled[col] * returns[col]
            asset_pnl[col] = (1 + strat_ret).cumprod()

        asset_pnl_df = pd.DataFrame(asset_pnl)

        # Combined portfolio P&L
        n = scaled.shape[1]
        portfolio_ret = pd.Series(0.0, index=scaled.index)
        for col in scaled.columns:
            if col in returns.columns:
                portfolio_ret += scaled[col] * returns[col] / n

        portfolio_cumulative = (1 + portfolio_ret).cumprod()

        # Benchmark — equal weight buy & hold
        benchmark_ret = returns.mean(axis=1)
        benchmark_cum = (1 + benchmark_ret).cumprod()

        portfolio_results = pd.DataFrame({
            "strategy_return" : portfolio_ret,
            "cumulative_pnl"  : portfolio_cumulative,
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
        """
        Full portfolio dashboard:
        - Portfolio vs benchmark
        - Individual asset P&L
        - Drawdown
        - Asset correlation heatmap
        """
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
                 label="Portfolio", color="steelblue",
                 linewidth=2)
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
        ax2   = fig.add_subplot(gs[1, 0])
        colors = ["steelblue", "green", "gold"]
        for i, col in enumerate(asset_pnl_df.columns):
            ax2.plot(asset_pnl_df.index, asset_pnl_df[col],
                     label=col, color=colors[i % len(colors)],
                     linewidth=1.2)
        ax2.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
        ax2.set_title("Individual Asset P&L")
        ax2.set_ylabel("Value")
        ax2.legend(fontsize=8)
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
        ax4 = fig.add_subplot(gs[2, 0])
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

        # --- Chart 5: Asset correlation ---
        ax5 = fig.add_subplot(gs[2, 1])
        if asset_pnl_df.shape[1] > 1:
            ret_df = asset_pnl_df.pct_change().dropna()
            corr   = ret_df.corr()
            im     = ax5.imshow(corr, cmap="RdYlGn",
                                vmin=-1, vmax=1, aspect="auto")
            plt.colorbar(im, ax=ax5)
            ax5.set_xticks(range(len(corr.columns)))
            ax5.set_yticks(range(len(corr.columns)))
            ax5.set_xticklabels(corr.columns, fontsize=8)
            ax5.set_yticklabels(corr.columns, fontsize=8)
            for i in range(len(corr)):
                for j in range(len(corr.columns)):
                    ax5.text(j, i, f"{corr.iloc[i,j]:.2f}",
                             ha="center", va="center", fontsize=9)
        ax5.set_title("Asset Return Correlation")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()

    def print_summary(self, portfolio_results, asset_pnl_df):
        r         = portfolio_results["strategy_return"]
        total_ret = portfolio_results["cumulative_pnl"].iloc[-1] - 1
        bench_ret = portfolio_results["benchmark"].iloc[-1] - 1

        print("=" * 50)
        print("     MULTI-ASSET PORTFOLIO SUMMARY")
        print("=" * 50)
        print(f"Period       : {portfolio_results.index[0].date()} "
              f"→ {portfolio_results.index[-1].date()}")
        print(f"Total return : {total_ret:.2%}")
        print(f"Benchmark    : {bench_ret:.2%}")
        print(f"Sharpe ratio : {self.sharpe(r):.2f}")
        print(f"Max drawdown : {self.max_drawdown(portfolio_results['cumulative_pnl']):.2%}")
        print(f"Ann. vol     : {r.std() * np.sqrt(252):.2%}")
        print(f"Win rate     : {(r > 0).mean():.2%}")
        print()
        print("Individual Asset Performance:")
        for col in asset_pnl_df.columns:
            ret = asset_pnl_df[col].pct_change().dropna()
            print(f"  {col:8s} → "
                  f"Return: {asset_pnl_df[col].iloc[-1]-1:.2%} | "
                  f"Sharpe: {self.sharpe(ret):.2f}")
        print("=" * 50)


# Quick test
if __name__ == "__main__":
    from data.data_loader import DataLoader
    from strategies.trend_following import TrendFollowing

    # 1. Load multi-asset data
    loader = DataLoader()
    tickers = ["CL=F", "NG=F", "GC=F"]
    df = loader.fetch(tickers, start="2010-01-01", end="2024-01-01")

    close   = df["Close"]
    close.columns = close.columns.droplevel(0) \
        if isinstance(close.columns, pd.MultiIndex) else close.columns

    # 2. Build signals for each asset
    signals = pd.DataFrame(index=close.index)
    for ticker in tickers:
        strat = TrendFollowing(fast=20, slow=60)
        signals[ticker] = strat.predict(close[ticker])

    # 3. Build returns
    returns = np.log(close / close.shift(1)).shift(-1)

    # 4. Run portfolio
    portfolio = Portfolio(target_vol=0.15)
    portfolio_results, asset_pnl = portfolio.run_backtest(signals, returns)
    portfolio.print_summary(portfolio_results, asset_pnl)

    # 5. Plot
    portfolio.plot(
        portfolio_results, asset_pnl,
        title="Multi-Asset Trend Following: CL + NG + Gold",
        save_path="evaluation/portfolio_report.png"
    )
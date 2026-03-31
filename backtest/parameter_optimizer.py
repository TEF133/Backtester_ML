import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from strategies.trend_following import TrendFollowing
from backtest.backtest_engine import BacktestEngine


# =============================================================================
# PART 1 — PARAMETER OPTIMIZER
# =============================================================================

class ParameterOptimizer:
    """
    Walk-forward parameter optimizer.
    
    Finds the best strategy parameters using walk-forward
    validation to prevent overfitting.
    
    Process:
    1. Split data into train/test windows
    2. For each train window: try all parameter combinations
    3. Pick best parameters on TRAIN data only
    4. Apply those parameters to TEST data (out-of-sample)
    5. Roll forward and repeat
    """

    def __init__(self,
                 train_years = 3,
                 test_months = 6,
                 metric      = "sharpe",
                 n_jobs      = 1):
        self.train_years  = train_years
        self.test_months  = test_months
        self.metric       = metric
        self.n_jobs       = n_jobs
        self.fold_results = []
        self.all_oos      = []

    def _evaluate(self, signals, returns):
        """Evaluate signals using chosen metric."""
        bt  = BacktestEngine(signals, returns)
        res = bt.run()
        r   = res["strategy_return"]

        if self.metric == "sharpe":
            mean = r.mean() * 252
            std  = r.std() * np.sqrt(252)
            return mean / std if std > 0 else -999

        elif self.metric == "return":
            return (1 + r).prod() - 1

        elif self.metric == "calmar":
            ann_ret = r.mean() * 252
            cum     = (1 + r).cumprod()
            dd      = (
                (cum - cum.cummax()) / cum.cummax()
            ).min()
            return ann_ret / abs(dd) if dd != 0 else -999

        return 0.0

    def optimize_trend_following(self, close, returns,
                                  fast_range = None,
                                  slow_range = None):
        """
        Optimize MA crossover parameters using walk-forward.
        """
        if fast_range is None:
            fast_range = [5, 10, 15, 20, 30]
        if slow_range is None:
            slow_range = [30, 40, 60, 80, 100, 120]

        param_grid = [
            (f, s) for f, s in product(fast_range, slow_range)
            if f < s
        ]

        print(f"Optimizing MA Crossover")
        print(f"Grid: {len(param_grid)} combinations | "
              f"Train: {self.train_years}yr | "
              f"Test: {self.test_months}m | "
              f"Metric: {self.metric}")
        print("=" * 65)

        train_size = pd.DateOffset(years=self.train_years)
        test_size  = pd.DateOffset(months=self.test_months)
        start_date = close.index[0]
        end_date   = close.index[-1]
        test_start = start_date + train_size

        all_signals = []
        fold        = 0

        while test_start < end_date:
            test_end = min(test_start + test_size, end_date)

            close_train = close[close.index <  test_start]
            ret_train   = returns[returns.index < test_start]
            close_test  = close[
                (close.index >= test_start) &
                (close.index <  test_end)
            ]
            ret_test    = returns[
                (returns.index >= test_start) &
                (returns.index <  test_end)
            ]

            if len(close_train) < 100 or len(close_test) < 10:
                test_start = test_end
                continue

            # Optimize on train
            best_score  = -999
            best_params = param_grid[0]

            for fast, slow in param_grid:
                strat  = TrendFollowing(fast=fast, slow=slow)
                sig    = strat.predict_sized(close_train)
                common = sig.index.intersection(ret_train.index)
                if len(common) < 20:
                    continue
                score = self._evaluate(
                    sig.loc[common], ret_train.loc[common]
                )
                if score > best_score:
                    best_score  = score
                    best_params = (fast, slow)

            # Apply best params to test
            best_fast, best_slow = best_params
            strat_oos = TrendFollowing(
                fast=best_fast, slow=best_slow
            )
            sig_oos    = strat_oos.predict_sized(close_test)
            common_oos = sig_oos.index.intersection(
                ret_test.index
            )

            if len(common_oos) > 0:
                oos_score = self._evaluate(
                    sig_oos.loc[common_oos],
                    ret_test.loc[common_oos]
                )
                all_signals.append(sig_oos.loc[common_oos])

                self.fold_results.append({
                    "fold"       : fold,
                    "test_start" : test_start,
                    "test_end"   : test_end,
                    "best_fast"  : best_fast,
                    "best_slow"  : best_slow,
                    "train_score": best_score,
                    "oos_score"  : oos_score,
                    "n_params"   : len(param_grid)
                })

                print(f"Fold {fold+1}: "
                      f"[{test_start.date()} -> "
                      f"{test_end.date()}] | "
                      f"Best: MA({best_fast}/{best_slow}) | "
                      f"Train: {best_score:.2f} | "
                      f"OOS: {oos_score:.2f}")

            test_start = test_end
            fold      += 1

        print("=" * 65)
        if self.fold_results:
            avg_train = np.mean(
                [f["train_score"] for f in self.fold_results]
            )
            avg_oos = np.mean(
                [f["oos_score"] for f in self.fold_results]
            )
            print(f"Avg Train {self.metric}: {avg_train:.2f}")
            print(f"Avg OOS   {self.metric}: {avg_oos:.2f}")
            ratio = avg_oos / avg_train \
                    if avg_train != 0 else 0
            print(f"OOS/Train ratio: {ratio:.2f} "
                  f"(1.0=perfect, <0.5=overfit)")

        return pd.concat(all_signals) if all_signals \
               else pd.Series(dtype=float)

    def optimize_options(self, close, returns,
                          otm_range   = None,
                          tenor_range = None,
                          strategy    = "short_strangle"):
        """Optimize options strategy parameters."""
        from backtest.options_backtester import OptionsBacktester

        if otm_range is None:
            otm_range = [0.03, 0.05, 0.07, 0.10]
        if tenor_range is None:
            tenor_range = [20, 30, 45, 60]

        param_grid = list(product(otm_range, tenor_range))

        print(f"\nOptimizing Options: {strategy}")
        print(f"Grid: {len(param_grid)} combinations")
        print("=" * 65)

        train_size = pd.DateOffset(years=self.train_years)
        test_size  = pd.DateOffset(months=self.test_months)
        start_date = close.index[0]
        end_date   = close.index[-1]
        test_start = start_date + train_size

        all_results = []
        fold        = 0

        while test_start < end_date:
            test_end    = min(test_start + test_size, end_date)
            close_train = close[close.index <  test_start]
            close_test  = close[
                (close.index >= test_start) &
                (close.index <  test_end)
            ]

            if len(close_train) < 100 or len(close_test) < 10:
                test_start = test_end
                continue

            best_score  = -999
            best_params = param_grid[0]

            for otm, tenor in param_grid:
                try:
                    bt = OptionsBacktester(
                        close        = close_train,
                        strategy     = strategy,
                        option_tenor = tenor,
                        otm_pct      = otm,
                        delta_hedge  = True
                    )
                    bt.run()
                    score = bt.sharpe()
                    if score > best_score:
                        best_score  = score
                        best_params = (otm, tenor)
                except:
                    continue

            best_otm, best_tenor = best_params
            try:
                bt_oos = OptionsBacktester(
                    close        = close_test,
                    strategy     = strategy,
                    option_tenor = best_tenor,
                    otm_pct      = best_otm,
                    delta_hedge  = True
                )
                bt_oos.run()
                oos_score = bt_oos.sharpe()

                all_results.append({
                    "fold"       : fold,
                    "test_start" : test_start,
                    "test_end"   : test_end,
                    "best_otm"   : best_otm,
                    "best_tenor" : best_tenor,
                    "train_score": best_score,
                    "oos_score"  : oos_score
                })

                print(f"Fold {fold+1}: "
                      f"[{test_start.date()} -> "
                      f"{test_end.date()}] | "
                      f"OTM={best_otm:.0%} "
                      f"Tenor={best_tenor}d | "
                      f"Train: {best_score:.2f} | "
                      f"OOS: {oos_score:.2f}")
            except:
                pass

            test_start = test_end
            fold      += 1

        return pd.DataFrame(all_results)

    def plot_results(self, close, returns,
                     signals_fixed     = None,
                     signals_optimized = None,
                     save_path         = None):
        """Compare fixed vs optimized parameters."""
        if not self.fold_results:
            print("No fold results.")
            return

        fold_df = pd.DataFrame(self.fold_results)

        fig = plt.figure(figsize=(16, 18))
        fig.suptitle(
            "Walk-Forward Parameter Optimization Analysis",
            fontsize=14, fontweight="bold"
        )
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45)

        # --- Chart 1: P&L comparison ---
        ax1 = fig.add_subplot(gs[0, :])
        if signals_optimized is not None and \
           len(signals_optimized) > 0:
            common = signals_optimized.index.intersection(
                returns.index
            )
            bt_opt  = BacktestEngine(
                signals_optimized.loc[common],
                returns.loc[common]
            )
            res_opt = bt_opt.run()
            ax1.plot(
                res_opt.index,
                res_opt["cumulative_pnl"],
                label=f"Optimized "
                      f"(Sharpe: {bt_opt.sharpe_ratio():.2f})",
                color="green", linewidth=1.5
            )

        if signals_fixed is not None:
            common_f = signals_fixed.index.intersection(
                returns.index
            )
            bt_fix  = BacktestEngine(
                signals_fixed.loc[common_f],
                returns.loc[common_f]
            )
            res_fix = bt_fix.run()
            ax1.plot(
                res_fix.index,
                res_fix["cumulative_pnl"],
                label=f"Fixed MA(20/60) "
                      f"(Sharpe: {bt_fix.sharpe_ratio():.2f})",
                color="steelblue", linewidth=1.5
            )
            ax1.plot(
                res_fix.index,
                res_fix["benchmark"],
                label="Buy & Hold",
                color="orange", linestyle="--"
            )

        ax1.axhline(1.0, color="grey", linestyle=":",
                    linewidth=0.8)
        ax1.set_title("Fixed vs Optimized Parameters: P&L")
        ax1.set_ylabel("Portfolio Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Chart 2: Train vs OOS ---
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(fold_df["fold"],
                 fold_df["train_score"],
                 label=f"Train {self.metric}",
                 color="steelblue", linewidth=1.5,
                 marker="o", markersize=4)
        ax2.plot(fold_df["fold"],
                 fold_df["oos_score"],
                 label=f"OOS {self.metric}",
                 color="green", linewidth=1.5,
                 marker="s", markersize=4)
        ax2.axhline(0, color="red", linestyle="--",
                    linewidth=1)
        ax2.set_title(
            f"Train vs OOS {self.metric.title()} per Fold"
        )
        ax2.set_xlabel("Fold")
        ax2.set_ylabel(self.metric.title())
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Chart 3: Best fast MA ---
        ax3 = fig.add_subplot(gs[2, 0])
        if "best_fast" in fold_df.columns:
            ax3.bar(fold_df["fold"], fold_df["best_fast"],
                    color="steelblue", alpha=0.8)
            ax3.set_title("Best Fast MA Window per Fold")
            ax3.set_xlabel("Fold")
            ax3.set_ylabel("Fast MA (days)")
            ax3.grid(True, alpha=0.3, axis="y")

        # --- Chart 4: Best slow MA ---
        ax4 = fig.add_subplot(gs[2, 1])
        if "best_slow" in fold_df.columns:
            ax4.bar(fold_df["fold"], fold_df["best_slow"],
                    color="orange", alpha=0.8)
            ax4.set_title("Best Slow MA Window per Fold")
            ax4.set_xlabel("Fold")
            ax4.set_ylabel("Slow MA (days)")
            ax4.grid(True, alpha=0.3, axis="y")

        # --- Chart 5: Parameter heatmap ---
        ax5 = fig.add_subplot(gs[3, :])
        if "best_fast" in fold_df.columns and \
           "best_slow" in fold_df.columns:
            param_counts = fold_df.groupby(
                ["best_fast", "best_slow"]
            ).size().reset_index(name="count")

            scatter = ax5.scatter(
                param_counts["best_fast"],
                param_counts["best_slow"],
                s          = param_counts["count"] * 200,
                c          = param_counts["count"],
                cmap       = "YlOrRd",
                alpha      = 0.8,
                edgecolors = "black"
            )
            plt.colorbar(scatter, ax=ax5,
                         label="Times Selected")
            ax5.set_title(
                "Parameter Selection Frequency"
            )
            ax5.set_xlabel("Fast MA Window")
            ax5.set_ylabel("Slow MA Window")
            ax5.grid(True, alpha=0.3)

            for _, row in param_counts.iterrows():
                ax5.annotate(
                    f"{int(row['count'])}x",
                    (row["best_fast"], row["best_slow"]),
                    ha="center", va="center",
                    fontsize=8, fontweight="bold"
                )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150,
                        bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()

    def summary(self):
        if not self.fold_results:
            print("No results yet.")
            return

        fold_df   = pd.DataFrame(self.fold_results)
        avg_train = fold_df["train_score"].mean()
        avg_oos   = fold_df["oos_score"].mean()
        ratio     = avg_oos / avg_train if avg_train != 0 else 0

        print("=" * 60)
        print("     WALK-FORWARD OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Total folds      : {len(fold_df)}")
        print(f"Metric           : {self.metric}")
        print(f"Avg train score  : {avg_train:.3f}")
        print(f"Avg OOS score    : {avg_oos:.3f}")
        print(f"OOS/Train ratio  : {ratio:.2f}")

        if ratio > 0.7:
            print("Assessment       : Low overfitting risk")
        elif ratio > 0.4:
            print("Assessment       : Moderate overfitting")
        else:
            print("Assessment       : High overfitting risk")

        if "best_fast" in fold_df.columns:
            print()
            print("Most selected parameters:")
            top = fold_df.groupby(
                ["best_fast", "best_slow"]
            ).size().sort_values(ascending=False).head(3)
            for (f, s), count in top.items():
                print(f"  MA({f}/{s}) selected {count}x")
        print("=" * 60)


# =============================================================================
# MAIN — Multi-asset portfolio optimization
# =============================================================================

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from portfolio.portfolio import Portfolio

    # 1. Load multi-asset data
    loader  = DataLoader()
    tickers = ["CL=F", "BZ=F", "NG=F", "GC=F", "SI=F", "HG=F"]
    df      = loader.fetch(
        tickers, start="2010-01-01", end="2024-01-01"
    )

    close = df["Close"]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.droplevel(0)
    close   = close.dropna(thresh=int(len(close) * 0.8), axis=1)
    close   = close[close.min(axis=1) > 0]
    tickers = close.columns.tolist()
    print(f"Assets: {tickers}")

    returns = np.log(close / close.shift(1)).shift(-1)

    # 2. Optimize per asset
    print("\n" + "="*65)
    print("OPTIMIZING MA PARAMETERS PER ASSET")
    print("="*65)

    best_params_per_asset = {}
    optimized_signals     = pd.DataFrame(index=close.index)
    fixed_signals         = pd.DataFrame(index=close.index)

    for ticker in tickers:
        print(f"\n--- Optimizing {ticker} ---")
        cl  = close[ticker].dropna()
        cl  = cl[cl > 0]
        ret = returns[ticker].reindex(cl.index).dropna()
        cl  = cl.reindex(ret.index)

        # Fixed baseline
        tf_fix              = TrendFollowing(fast=20, slow=60)
        sig_fix             = tf_fix.predict(cl)
        fixed_signals[ticker] = sig_fix

        # Walk-forward optimized
        opt     = ParameterOptimizer(
            train_years = 3,
            test_months = 6,
            metric      = "sharpe"
        )
        sig_opt = opt.optimize_trend_following(
            close      = cl,
            returns    = ret,
            fast_range = [5, 10, 20, 30],
            slow_range = [40, 60, 80, 120]
        )
        optimized_signals[ticker] = sig_opt.reindex(
            close.index
        )

        # Store most selected params
        if opt.fold_results:
            fold_df = pd.DataFrame(opt.fold_results)
            top     = fold_df.groupby(
                ["best_fast", "best_slow"]
            ).size().idxmax()
            best_params_per_asset[ticker] = {
                "fast": top[0],
                "slow": top[1]
            }
            oos_avg = fold_df["oos_score"].mean()
            print(f"  Most selected: MA({top[0]}/{top[1]}) | "
                  f"Avg OOS Sharpe: {oos_avg:.2f}")

    print("\n" + "="*65)
    print("BEST PARAMETERS PER ASSET:")
    print("="*65)
    for ticker, params in best_params_per_asset.items():
        print(f"  {ticker:8s} -> MA({params['fast']}"
              f"/{params['slow']})")

    # 3. Portfolio backtest — fixed vs optimized
    print("\nRunning portfolio backtests...")
    portfolio = Portfolio(target_vol=0.15)

    res_fix, pnl_fix = portfolio.run_backtest(
        fixed_signals.dropna(how="all"), returns
    )

    opt_sig_clean    = optimized_signals.fillna(0)
    res_opt, pnl_opt = portfolio.run_backtest(
        opt_sig_clean, returns
    )

    sharpe_fix = portfolio.sharpe(res_fix["strategy_return"])
    sharpe_opt = portfolio.sharpe(res_opt["strategy_return"])
    mdd_fix    = portfolio.max_drawdown(res_fix["cumulative_pnl"])
    mdd_opt    = portfolio.max_drawdown(res_opt["cumulative_pnl"])

    print(f"\nFixed Portfolio     : "
          f"Sharpe={sharpe_fix:.2f} | "
          f"Max DD={mdd_fix:.1%}")
    print(f"Optimized Portfolio : "
          f"Sharpe={sharpe_opt:.2f} | "
          f"Max DD={mdd_opt:.1%}")

    # 4. Comparison chart
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        "Fixed MA(20/60) vs Per-Asset Optimized: "
        "Multi-Asset Portfolio",
        fontsize=14, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(res_fix.index, res_fix["cumulative_pnl"],
            label=f"Fixed MA(20/60) all assets "
                  f"(Sharpe: {sharpe_fix:.2f})",
            color="steelblue", linewidth=1.5)
    ax.plot(res_opt.index, res_opt["cumulative_pnl"],
            label=f"Per-Asset Optimized "
                  f"(Sharpe: {sharpe_opt:.2f})",
            color="green", linewidth=1.5)
    ax.plot(res_fix.index, res_fix["benchmark"],
            label="Buy & Hold",
            color="orange", linestyle="--")
    ax.axhline(1.0, color="grey", linestyle=":",
               linewidth=0.8)
    ax.set_title("Cumulative P&L")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for res, color, label in [
        (res_fix, "steelblue", "Fixed"),
        (res_opt, "green",     "Optimized")
    ]:
        cum = res["cumulative_pnl"]
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax2.fill_between(res.index, dd, 0,
                         alpha=0.3, color=color,
                         label=label)
    ax2.set_title("Drawdown Comparison")
    ax2.set_ylabel("Drawdown %")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation/portfolio_optimization.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Done!")
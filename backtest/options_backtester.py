import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from strategies.options_engine import BlackScholes, VolSurface


# =============================================================================
# PART 1 — OPTIONS BACKTEST ENGINE
# =============================================================================

class OptionsBacktester:
    """
    Backtests options strategies on historical futures data.
    
    Features:
    - Rolling option positions (new contract each month)
    - Delta hedging (optional)
    - Additive dollar P&L
    - IV capped at realistic levels
    - Negative price filtering
    - Vol regime filter (optional)
    """

    def __init__(self,
                 close,
                 strategy         = "short_strangle",
                 option_tenor     = 30,
                 roll_days_before = 5,
                 vol_window       = 20,
                 r                = 0.05,
                 otm_pct          = 0.05,
                 delta_hedge      = True,
                 option_cost      = 0.002,
                 futures_cost     = 0.001,
                 vol_regime_filter= False,
                 vrp_threshold    = 0.5):

        self.close = close[close > 5].copy()

        self.strategy          = strategy
        self.option_tenor      = option_tenor
        self.roll_days_before  = roll_days_before
        self.vol_window        = vol_window
        self.r                 = r
        self.otm_pct           = otm_pct
        self.delta_hedge       = delta_hedge
        self.option_cost       = option_cost
        self.futures_cost      = futures_cost
        self.vol_regime_filter = vol_regime_filter
        self.vrp_threshold     = vrp_threshold

        ret     = np.log(self.close / self.close.shift(1))
        self.iv = (
            ret.rolling(vol_window).std() * np.sqrt(252)
        ).clip(0.05, 1.5)

        iv_proxy   = ret.rolling(vol_window * 2).std() * np.sqrt(252)
        vrp        = iv_proxy - self.iv
        vrp_z      = (
            vrp - vrp.rolling(60).mean()
        ) / (vrp.rolling(60).std() + 1e-8)
        self.vrp_z = vrp_z.shift(1)

        self.results      = None
        self.position_log = None

    def _regime_allows_trade(self, date, strategy):
        if not self.vol_regime_filter:
            return True
        if date not in self.vrp_z.index:
            return True
        z = self.vrp_z.loc[date]
        if pd.isna(z):
            return True
        if strategy in ["short_strangle", "short_straddle"]:
            return z > self.vrp_threshold
        elif strategy == "long_straddle":
            return z < -self.vrp_threshold
        return True

    def _get_strikes(self, F, strategy):
        if strategy == "short_strangle":
            return {
                "call_strike": F * (1 + self.otm_pct),
                "put_strike" : F * (1 - self.otm_pct)
            }
        elif strategy in ["short_straddle", "long_straddle"]:
            return {"call_strike": F, "put_strike": F}
        elif strategy == "bull_call_spread":
            return {
                "long_strike" : F,
                "short_strike": F * (1 + self.otm_pct * 2)
            }
        else:
            return {"call_strike": F, "put_strike": F}

    def _option_value(self, strategy, strikes, F, T, sigma):
        r = self.r
        if strategy in ["short_strangle", "short_straddle"]:
            return -(
                BlackScholes.price(
                    F, strikes["call_strike"], T, r, sigma, "call"
                ) +
                BlackScholes.price(
                    F, strikes["put_strike"], T, r, sigma, "put"
                )
            )
        elif strategy == "long_straddle":
            return (
                BlackScholes.price(
                    F, strikes["call_strike"], T, r, sigma, "call"
                ) +
                BlackScholes.price(
                    F, strikes["put_strike"], T, r, sigma, "put"
                )
            )
        elif strategy == "bull_call_spread":
            return (
                BlackScholes.price(
                    F, strikes["long_strike"], T, r, sigma, "call"
                ) -
                BlackScholes.price(
                    F, strikes["short_strike"], T, r, sigma, "call"
                )
            )
        return 0.0

    def _portfolio_delta(self, strategy, strikes, F, T, sigma):
        r = self.r
        if strategy in ["short_strangle", "short_straddle"]:
            return -(
                BlackScholes.delta(
                    F, strikes["call_strike"], T, r, sigma, "call"
                ) +
                BlackScholes.delta(
                    F, strikes["put_strike"], T, r, sigma, "put"
                )
            )
        elif strategy == "long_straddle":
            return (
                BlackScholes.delta(
                    F, strikes["call_strike"], T, r, sigma, "call"
                ) +
                BlackScholes.delta(
                    F, strikes["put_strike"], T, r, sigma, "put"
                )
            )
        return 0.0

    def run(self):
        dates       = self.close.index
        n           = len(dates)
        tenor_years = self.option_tenor / 365

        daily_pnl      = pd.Series(0.0, index=dates)
        position_log   = []
        current_pos    = None
        hedge_position = 0.0
        prev_option_val= 0.0

        entry_dates = set()
        i = self.vol_window
        while i < n - self.option_tenor:
            entry_dates.add(dates[i])
            i += self.option_tenor

        print(f"Strategy      : {self.strategy}")
        print(f"Tenor         : {self.option_tenor} days")
        print(f"Delta hedge   : {self.delta_hedge}")
        print(f"OTM distance  : {self.otm_pct:.0%}")
        print(f"Vol filter    : {self.vol_regime_filter}")
        print(f"Total entries : {len(entry_dates)}")
        print("-" * 50)

        for idx in range(self.vol_window, n - 1):
            date  = dates[idx]
            F     = self.close.iloc[idx]
            sigma = self.iv.iloc[idx]

            if pd.isna(F) or pd.isna(sigma) or sigma <= 0:
                continue

            if date in entry_dates:
                if current_pos is not None:
                    position_log[-1]["exit_date"] = date
                    position_log[-1]["exit_F"]    = F

                if self._regime_allows_trade(date, self.strategy):
                    strikes      = self._get_strikes(F, self.strategy)
                    entry_val    = self._option_value(
                        self.strategy, strikes, F, tenor_years, sigma
                    )
                    current_pos  = {
                        "entry_date"  : date,
                        "entry_F"     : F,
                        "entry_sigma" : sigma,
                        "strikes"     : strikes,
                        "entry_T"     : tenor_years,
                        "entry_val"   : entry_val,
                        "days_held"   : 0
                    }
                    position_log.append(current_pos)
                    prev_option_val = entry_val
                    hedge_position  = 0.0
                else:
                    current_pos     = None
                    prev_option_val = 0.0
                    hedge_position  = 0.0

            if current_pos is not None:
                days_held   = current_pos["days_held"]
                remaining_T = max(
                    tenor_years - days_held / 365, 1/365
                )
                strikes = current_pos["strikes"]

                curr_val    = self._option_value(
                    self.strategy, strikes, F, remaining_T, sigma
                )
                options_pnl     = curr_val - prev_option_val
                prev_option_val = curr_val

                hedge_pnl = 0.0
                if self.delta_hedge:
                    port_delta  = self._portfolio_delta(
                        self.strategy, strikes, F, remaining_T, sigma
                    )
                    hedge_trade = port_delta - hedge_position
                    hedge_cost  = abs(hedge_trade) * self.futures_cost
                    F_prev      = self.close.iloc[idx - 1]
                    hedge_pnl   = (
                        hedge_position * (F - F_prev) - hedge_cost
                    )
                    hedge_position = port_delta

                daily_pnl.iloc[idx] = options_pnl + hedge_pnl
                current_pos["days_held"] += 1

                if days_held >= self.option_tenor - 1:
                    trade_pnl = curr_val - current_pos["entry_val"]
                    position_log[-1]["trade_pnl"] = trade_pnl
                    position_log[-1]["premium"]   = abs(
                        current_pos["entry_val"]
                    )
                    current_pos     = None
                    hedge_position  = 0.0
                    prev_option_val = 0.0

        cumulative = daily_pnl.cumsum()
        ret        = np.log(self.close / self.close.shift(1)).fillna(0)
        benchmark  = (1 + ret).cumprod()

        self.results = pd.DataFrame({
            "daily_pnl"    : daily_pnl,
            "cumulative"   : cumulative,
            "benchmark"    : benchmark,
            "futures_price": self.close,
            "iv"           : self.iv,
            "vrp_z"        : self.vrp_z
        })
        self.position_log = pd.DataFrame(position_log)
        return self.results

    def sharpe(self, periods=252):
        r    = self.results["daily_pnl"]
        mean = r.mean() * periods
        std  = r.std() * np.sqrt(periods)
        return mean / std if std != 0 else 0

    def max_drawdown(self):
        cum = self.results["cumulative"]
        dd  = cum - cum.cummax()
        return dd.min()

    def win_rate(self):
        if self.position_log is None or self.position_log.empty:
            return np.nan
        if "trade_pnl" not in self.position_log.columns:
            return np.nan
        completed = self.position_log["trade_pnl"].dropna()
        return (completed > 0).mean() if len(completed) > 0 else np.nan

    def summary(self):
        r         = self.results["daily_pnl"]
        total_pnl = self.results["cumulative"].iloc[-1]

        print("=" * 55)
        print(f"   OPTIONS BACKTEST: {self.strategy.upper()}")
        print("=" * 55)
        print(f"Period        : "
              f"{self.results.index[0].date()} -> "
              f"{self.results.index[-1].date()}")
        print(f"Total PnL     : {total_pnl:+.2f} USD/contract")
        print(f"Sharpe ratio  : {self.sharpe():.2f}")
        print(f"Max drawdown  : {self.max_drawdown():+.2f} USD")
        print(f"Daily vol     : {r.std() * np.sqrt(252):.4f}")
        print(f"Win rate      : {self.win_rate():.2%}")
        print(f"Avg IV        : {self.iv.mean():.1%}")
        if self.position_log is not None and \
           not self.position_log.empty and \
           "trade_pnl" in self.position_log.columns:
            completed = self.position_log["trade_pnl"].dropna()
            print(f"Total trades  : {len(completed)}")
            if len(completed) > 0:
                print(f"Avg trade PnL : {completed.mean():+.4f}")
                print(f"Best trade    : {completed.max():+.4f}")
                print(f"Worst trade   : {completed.min():+.4f}")
        print("=" * 55)

    def plot(self, title=None, save_path=None):
        if title is None:
            title = (
                f"Options Backtest: "
                f"{self.strategy.replace('_', ' ').title()} | "
                f"Sharpe: {self.sharpe():.2f} | "
                f"Max DD: {self.max_drawdown():.2f} USD"
            )

        fig = plt.figure(figsize=(16, 20))
        fig.suptitle(title, fontsize=13, fontweight="bold")
        gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45)

        # --- Chart 1: Cumulative P&L ---
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.results.index,
                 self.results["cumulative"],
                 label="Options Strategy",
                 color="steelblue", linewidth=1.5)
        ax1.axhline(0, color="grey", linestyle=":", linewidth=0.8)
        ax1.set_title("Cumulative Dollar P&L")
        ax1.set_ylabel("P&L (USD/contract)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        metrics = (
            f"Sharpe: {self.sharpe():.2f}  |  "
            f"Max DD: {self.max_drawdown():.2f} USD  |  "
            f"Win Rate: {self.win_rate():.1%}  |  "
            f"Total PnL: {self.results['cumulative'].iloc[-1]:+.2f} USD"
        )
        ax1.text(0.01, 0.02, metrics,
                 transform=ax1.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round",
                           facecolor="lightyellow", alpha=0.8))

        # --- Chart 2: Dollar drawdown ---
        ax2 = fig.add_subplot(gs[1, :])
        cum = self.results["cumulative"]
        dd  = cum - cum.cummax()
        ax2.fill_between(self.results.index, dd, 0,
                         color="red", alpha=0.4)
        ax2.axhline(self.max_drawdown(), color="darkred",
                    linestyle="--", linewidth=1,
                    label=f"Max DD: {self.max_drawdown():.2f} USD")
        ax2.set_title("Dollar Drawdown")
        ax2.set_ylabel("Drawdown (USD)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Chart 3: Futures + IV ---
        ax3      = fig.add_subplot(gs[2, 0])
        ax3_twin = ax3.twinx()
        ax3.plot(self.results.index,
                 self.results["futures_price"],
                 color="black", linewidth=1, label="Futures")
        ax3_twin.plot(self.results.index,
                      self.results["iv"],
                      color="red", linewidth=1,
                      alpha=0.7, label="IV")
        ax3.set_title("Futures Price + IV")
        ax3.set_ylabel("Futures Price")
        ax3_twin.set_ylabel("Implied Vol", color="red")
        ax3.legend(loc="upper left", fontsize=8)
        ax3_twin.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # --- Chart 4: Daily P&L distribution ---
        ax4 = fig.add_subplot(gs[2, 1])
        pnl = self.results["daily_pnl"]
        ax4.hist(pnl[pnl != 0], bins=50,
                 color="steelblue", edgecolor="white", alpha=0.8)
        ax4.axvline(0, color="red", linestyle="--", linewidth=1)
        ax4.axvline(pnl.mean(), color="green", linestyle="--",
                    linewidth=1,
                    label=f"Mean: {pnl.mean():.4f}")
        ax4.set_title("Daily P&L Distribution")
        ax4.set_xlabel("Daily P&L (USD)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # --- Chart 5: VRP Z-Score ---
        ax5 = fig.add_subplot(gs[3, :])
        ax5.plot(self.results.index,
                 self.results["vrp_z"],
                 color="purple", linewidth=1)
        ax5.axhline(self.vrp_threshold, color="red",
                    linestyle="--", linewidth=1,
                    label=f"Sell vol ({self.vrp_threshold})")
        ax5.axhline(-self.vrp_threshold, color="green",
                    linestyle="--", linewidth=1,
                    label=f"Buy vol (-{self.vrp_threshold})")
        ax5.axhline(0, color="black", linewidth=0.5)
        ax5.set_title("Vol Risk Premium Z-Score")
        ax5.set_ylabel("VRP Z-Score")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # --- Chart 6: P&L per trade ---
        ax6 = fig.add_subplot(gs[4, :])
        if self.position_log is not None and \
           not self.position_log.empty and \
           "trade_pnl" in self.position_log.columns:
            completed = self.position_log.dropna(
                subset=["trade_pnl"]
            )
            if not completed.empty:
                colors_bar = [
                    "green" if p > 0 else "red"
                    for p in completed["trade_pnl"]
                ]
                ax6.bar(range(len(completed)),
                        completed["trade_pnl"],
                        color=colors_bar, alpha=0.8)
                ax6.axhline(0, color="black", linewidth=0.8)
                ax6.set_title("P&L per Trade (USD)")
                ax6.set_xlabel("Trade Number")
                ax6.set_ylabel("P&L (USD)")
                ax6.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from data.data_loader import DataLoader

    loader = DataLoader()
    df     = loader.fetch(
        ["CL=F"], start="2015-01-01", end="2024-01-01"
    )
    close  = df["Close"].squeeze().dropna()
    close  = close[close > 5]

    strategies = [
        ("short_strangle", True,  False,
         "Short Strangle Hedged No Filter"),
        ("short_strangle", True,  True,
         "Short Strangle Hedged Vol Filter"),
        ("long_straddle",  True,  False,
         "Long Straddle Hedged No Filter"),
        ("long_straddle",  True,  True,
         "Long Straddle Hedged Vol Filter"),
    ]

    results_all = {}

    for strat, hedge, vf, label in strategies:
        print(f"\n{'='*55}")
        print(f"Running: {label}")
        print(f"{'='*55}")

        bt = OptionsBacktester(
            close             = close,
            strategy          = strat,
            option_tenor      = 30,
            vol_window        = 20,
            r                 = 0.05,
            otm_pct           = 0.05,
            delta_hedge       = hedge,
            option_cost       = 0.002,
            futures_cost      = 0.001,
            vol_regime_filter = vf,
            vrp_threshold     = 0.5
        )
        results = bt.run()
        bt.summary()
        bt.plot(
            title     = label,
            save_path = f"evaluation/options_"
                        f"{strat}_{'filtered' if vf else 'raw'}.png"
        )
        results_all[label] = {
            "results": results,
            "sharpe" : bt.sharpe(),
            "mdd"    : bt.max_drawdown(),
            "bt"     : bt
        }

    # Comparison chart
    print("\nGenerating comparison chart...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        "Options Strategies: Impact of Vol Regime Filter",
        fontsize=14, fontweight="bold"
    )

    colors = ["steelblue", "darkblue", "green", "darkgreen"]
    for i, (label, data) in enumerate(results_all.items()):
        axes[0].plot(
            data["results"].index,
            data["results"]["cumulative"],
            label=f"{label} (Sharpe: {data['sharpe']:.2f})",
            color=colors[i], linewidth=1.5
        )
        cum = data["results"]["cumulative"]
        dd  = cum - cum.cummax()
        axes[1].fill_between(
            data["results"].index, dd, 0,
            alpha=0.25, color=colors[i], label=label
        )

    axes[0].axhline(0, color="grey", linestyle=":", linewidth=0.8)
    axes[0].set_title("Cumulative Dollar P&L")
    axes[0].set_ylabel("P&L (USD)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Dollar Drawdown Comparison")
    axes[1].set_ylabel("Drawdown (USD)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation/options_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Done!")
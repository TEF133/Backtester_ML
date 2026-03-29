import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# Import our options engine
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
    
    Handles:
    - Rolling option positions (new contract each month)
    - Delta hedging (optional)
    - Realistic P&L using Black-76/BAW pricing
    - Transaction costs for options and futures
    
    Strategies supported:
    - Short strangle    : sell OTM call + OTM put
    - Short straddle    : sell ATM call + ATM put  
    - Long straddle     : buy ATM call + ATM put
    - Bull call spread  : long lower call, short upper call
    - Calendar spread   : long far expiry, short near expiry
    - Delta hedged vol  : trade vol, hedge delta daily
    """

    def __init__(self,
                 close,
                 strategy="short_strangle",
                 option_tenor=30,
                 roll_days_before=5,
                 vol_window=20,
                 r=0.05,
                 otm_pct=0.05,
                 delta_hedge=True,
                 option_cost=0.002,
                 futures_cost=0.001):
        """
        close            : futures price series (pd.Series)
        strategy         : options strategy to backtest
        option_tenor     : days to expiry when entering (e.g. 30)
        roll_days_before : roll position N days before expiry
        vol_window       : lookback for realised vol (IV proxy)
        r                : risk-free rate
        otm_pct          : OTM distance for strangle
                           0.05 = 5% OTM
        delta_hedge      : if True, delta hedge daily with futures
        option_cost      : transaction cost for options
        futures_cost     : transaction cost for futures hedge
        """
        self.close            = close
        self.strategy         = strategy
        self.option_tenor     = option_tenor
        self.roll_days_before = roll_days_before
        self.vol_window       = vol_window
        self.r                = r
        self.otm_pct          = otm_pct
        self.delta_hedge      = delta_hedge
        self.option_cost      = option_cost
        self.futures_cost     = futures_cost

        # Calculate rolling IV proxy
        ret      = np.log(close / close.shift(1))
        self.iv  = ret.rolling(vol_window).std() * np.sqrt(252)
        self.results = None

    def _get_strikes(self, F, strategy):
        """Calculate strikes based on strategy and current price."""
        if strategy == "short_strangle":
            return {
                "call_strike" : F * (1 + self.otm_pct),
                "put_strike"  : F * (1 - self.otm_pct)
            }
        elif strategy == "short_straddle":
            return {"call_strike": F, "put_strike": F}
        elif strategy == "long_straddle":
            return {"call_strike": F, "put_strike": F}
        elif strategy == "bull_call_spread":
            return {
                "long_strike" : F,
                "short_strike": F * (1 + self.otm_pct * 2)
            }
        else:
            return {"call_strike": F, "put_strike": F}

    def _option_pnl_at_expiry(self, strategy, strikes,
                               entry_F, expiry_F, sigma,
                               entry_T):
        """
        Calculate option P&L at expiry.
        Premium received/paid at entry, payoff at expiry.
        """
        r = self.r

        if strategy in ["short_strangle", "short_straddle"]:
            # SELL call + SELL put
            # Receive premium upfront, pay payoff at expiry
            call_K = strikes["call_strike"]
            put_K  = strikes["put_strike"]

            call_premium = BlackScholes.price(
                entry_F, call_K, entry_T, r, sigma, "call"
            )
            put_premium = BlackScholes.price(
                entry_F, put_K, entry_T, r, sigma, "put"
            )

            call_payoff = max(expiry_F - call_K, 0)
            put_payoff  = max(put_K - expiry_F, 0)

            pnl = (call_premium + put_premium) - \
                  (call_payoff + put_payoff)
            premium_received = call_premium + put_premium

        elif strategy == "long_straddle":
            # BUY call + BUY put
            call_K = strikes["call_strike"]
            put_K  = strikes["put_strike"]

            call_premium = BlackScholes.price(
                entry_F, call_K, entry_T, r, sigma, "call"
            )
            put_premium = BlackScholes.price(
                entry_F, put_K, entry_T, r, sigma, "put"
            )

            call_payoff = max(expiry_F - call_K, 0)
            put_payoff  = max(put_K - expiry_F, 0)

            pnl = (call_payoff + put_payoff) - \
                  (call_premium + put_premium)
            premium_received = -(call_premium + put_premium)

        elif strategy == "bull_call_spread":
            long_K  = strikes["long_strike"]
            short_K = strikes["short_strike"]

            long_premium  = BlackScholes.price(
                entry_F, long_K, entry_T, r, sigma, "call"
            )
            short_premium = BlackScholes.price(
                entry_F, short_K, entry_T, r, sigma, "call"
            )

            long_payoff  = max(expiry_F - long_K, 0)
            short_payoff = max(expiry_F - short_K, 0)

            pnl = (long_payoff - short_payoff) - \
                  (long_premium - short_premium)
            premium_received = short_premium - long_premium

        else:
            pnl = 0
            premium_received = 0

        # Transaction costs
        cost = self.option_cost * abs(premium_received)
        return pnl - cost, premium_received

    def _portfolio_delta(self, strategy, strikes, F, T, sigma):
        """Calculate total portfolio delta for hedging."""
        r = self.r

        if strategy in ["short_strangle", "short_straddle"]:
            call_delta = -BlackScholes.delta(
                F, strikes["call_strike"], T, r, sigma, "call"
            )
            put_delta = -BlackScholes.delta(
                F, strikes["put_strike"], T, r, sigma, "put"
            )
            return call_delta + put_delta

        elif strategy == "long_straddle":
            call_delta = BlackScholes.delta(
                F, strikes["call_strike"], T, r, sigma, "call"
            )
            put_delta = BlackScholes.delta(
                F, strikes["put_strike"], T, r, sigma, "put"
            )
            return call_delta + put_delta

        return 0.0

    def run(self):
        """
        Run the options backtest.
        
        Logic:
        1. Every month, enter new options position
        2. Daily: mark-to-market options + delta hedge if enabled
        3. At expiry: close position and take P&L
        4. Roll into next month's options
        """
        dates       = self.close.index
        n           = len(dates)
        tenor_years = self.option_tenor / 365

        daily_pnl      = pd.Series(0.0, index=dates)
        position_log   = []
        current_pos    = None
        hedge_position = 0.0  # current futures hedge

        # Entry frequency: every option_tenor days
        entry_dates = set()
        i = self.vol_window
        while i < n - self.option_tenor:
            entry_dates.add(dates[i])
            i += self.option_tenor

        print(f"Strategy      : {self.strategy}")
        print(f"Tenor         : {self.option_tenor} days")
        print(f"Delta hedge   : {self.delta_hedge}")
        print(f"OTM distance  : {self.otm_pct:.0%}")
        print(f"Total entries : {len(entry_dates)}")
        print("-" * 50)

        for idx in range(self.vol_window, n - 1):
            date    = dates[idx]
            F       = self.close.iloc[idx]
            sigma   = self.iv.iloc[idx]

            if pd.isna(F) or pd.isna(sigma) or sigma <= 0:
                continue

            # --- ENTRY ---
            if date in entry_dates:
                # Close existing position first
                if current_pos is not None:
                    position_log[-1]["exit_date"] = date
                    position_log[-1]["exit_F"]    = F

                # Enter new position
                strikes      = self._get_strikes(F, self.strategy)
                current_pos  = {
                    "entry_date" : date,
                    "entry_F"    : F,
                    "entry_sigma": sigma,
                    "strikes"    : strikes,
                    "entry_T"    : tenor_years,
                    "days_held"  : 0
                }
                position_log.append(current_pos)
                hedge_position = 0.0

            # --- DAILY MARK TO MARKET ---
            if current_pos is not None:
                days_held = current_pos["days_held"]
                remaining_T = max(
                    tenor_years - days_held / 365, 1/365
                )
                strikes = current_pos["strikes"]

                # Mark options to market
                entry_F     = current_pos["entry_F"]
                entry_sigma = current_pos["entry_sigma"]
                entry_T     = current_pos["entry_T"]

                # Yesterday's option value
                prev_T = max(remaining_T + 1/365, 1/365)
                prev_F = self.close.iloc[idx - 1]

                # Today's vs yesterday's option portfolio value
                if self.strategy in ["short_strangle",
                                      "short_straddle"]:
                    call_K = strikes.get("call_strike", F)
                    put_K  = strikes.get("put_strike",  F)

                    today_val = -(
                        BlackScholes.price(
                            F, call_K, remaining_T,
                            self.r, sigma, "call"
                        ) +
                        BlackScholes.price(
                            F, put_K, remaining_T,
                            self.r, sigma, "put"
                        )
                    )
                    yest_val = -(
                        BlackScholes.price(
                            prev_F, call_K, prev_T,
                            self.r, entry_sigma, "call"
                        ) +
                        BlackScholes.price(
                            prev_F, put_K, prev_T,
                            self.r, entry_sigma, "put"
                        )
                    )

                elif self.strategy == "long_straddle":
                    call_K = strikes.get("call_strike", F)
                    put_K  = strikes.get("put_strike",  F)

                    today_val = (
                        BlackScholes.price(
                            F, call_K, remaining_T,
                            self.r, sigma, "call"
                        ) +
                        BlackScholes.price(
                            F, put_K, remaining_T,
                            self.r, sigma, "put"
                        )
                    )
                    yest_val = (
                        BlackScholes.price(
                            prev_F, call_K, prev_T,
                            self.r, entry_sigma, "call"
                        ) +
                        BlackScholes.price(
                            prev_F, put_K, prev_T,
                            self.r, entry_sigma, "put"
                        )
                    )
                else:
                    today_val = 0
                    yest_val  = 0

                # Options daily P&L
                options_pnl = today_val - yest_val

                # Delta hedge P&L
                hedge_pnl = 0.0
                if self.delta_hedge:
                    port_delta  = self._portfolio_delta(
                        self.strategy, strikes, F,
                        remaining_T, sigma
                    )
                    hedge_trade = port_delta - hedge_position
                    hedge_cost  = (abs(hedge_trade) *
                                   self.futures_cost)
                    F_prev      = self.close.iloc[idx - 1]
                    hedge_pnl   = (hedge_position *
                                   (F - F_prev) - hedge_cost)
                    hedge_position = port_delta

                daily_pnl.iloc[idx] = options_pnl + hedge_pnl
                current_pos["days_held"] += 1

                # --- EXPIRY ---
                if days_held >= self.option_tenor - 1:
                    expiry_pnl, premium = self._option_pnl_at_expiry(
                        self.strategy, strikes,
                        entry_F, F, entry_sigma, entry_T
                    )
                    position_log[-1]["trade_pnl"] = expiry_pnl
                    position_log[-1]["premium"]   = premium
                    current_pos = None
                    hedge_position = 0.0

        # Build results
        cumulative = (1 + daily_pnl).cumprod()
        ret        = np.log(self.close / self.close.shift(1)).shift(-1)
        benchmark  = (1 + ret.fillna(0)).cumprod()

        self.results = pd.DataFrame({
            "daily_pnl"    : daily_pnl,
            "cumulative"   : cumulative,
            "benchmark"    : benchmark,
            "futures_price": self.close,
            "iv"           : self.iv
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
        dd  = (cum - cum.cummax()) / cum.cummax()
        return dd.min()

    def win_rate(self):
        if self.position_log.empty or \
           "trade_pnl" not in self.position_log.columns:
            return np.nan
        completed = self.position_log["trade_pnl"].dropna()
        return (completed > 0).mean()

    def summary(self):
        r         = self.results["daily_pnl"]
        total_ret = self.results["cumulative"].iloc[-1] - 1

        print("=" * 55)
        print(f"   OPTIONS BACKTEST: {self.strategy.upper()}")
        print("=" * 55)
        print(f"Period       : "
              f"{self.results.index[0].date()} → "
              f"{self.results.index[-1].date()}")
        print(f"Total return : {total_ret:.2%}")
        print(f"Sharpe ratio : {self.sharpe():.2f}")
        print(f"Max drawdown : {self.max_drawdown():.2%}")
        print(f"Ann. vol     : {r.std() * np.sqrt(252):.2%}")
        print(f"Win rate     : {self.win_rate():.2%}")
        print(f"Avg IV       : {self.iv.mean():.1%}")
        if not self.position_log.empty and \
           "trade_pnl" in self.position_log.columns:
            completed = self.position_log["trade_pnl"].dropna()
            print(f"Total trades : {len(completed)}")
            print(f"Avg trade PnL: {completed.mean():.4f}")
        print("=" * 55)

    def plot(self, title=None, save_path=None):
        """Full options backtest dashboard."""
        if title is None:
            title = (f"Options Backtest: "
                     f"{self.strategy.replace('_', ' ').title()}"
                     f" | Sharpe: {self.sharpe():.2f} | "
                     f"Max DD: {self.max_drawdown():.1%}")

        fig = plt.figure(figsize=(16, 18))
        fig.suptitle(title, fontsize=13, fontweight="bold")
        gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45)

        # --- Chart 1: Cumulative P&L ---
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.results.index,
                 self.results["cumulative"],
                 label="Options Strategy",
                 color="steelblue", linewidth=1.5)
        ax1.plot(self.results.index,
                 self.results["benchmark"],
                 label="Buy & Hold Futures",
                 color="orange", linewidth=1.5, linestyle="--")
        ax1.axhline(1.0, color="grey", linestyle=":",
                    linewidth=0.8)
        ax1.set_title("Cumulative P&L")
        ax1.set_ylabel("Portfolio Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        metrics = (f"Sharpe: {self.sharpe():.2f}  |  "
                   f"Max DD: {self.max_drawdown():.1%}  |  "
                   f"Win Rate: {self.win_rate():.1%}")
        ax1.text(0.01, 0.02, metrics,
                 transform=ax1.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round",
                           facecolor="lightyellow", alpha=0.8))

        # --- Chart 2: Drawdown ---
        ax2 = fig.add_subplot(gs[1, :])
        cum = self.results["cumulative"]
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax2.fill_between(self.results.index, dd, 0,
                         color="red", alpha=0.4)
        ax2.axhline(self.max_drawdown(), color="darkred",
                    linestyle="--", linewidth=1,
                    label=f"Max DD: {self.max_drawdown():.1%}")
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown %")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Chart 3: Futures price + IV ---
        ax3 = fig.add_subplot(gs[2, 0])
        ax3_twin = ax3.twinx()
        ax3.plot(self.results.index,
                 self.results["futures_price"],
                 color="black", linewidth=1, label="Futures")
        ax3_twin.plot(self.results.index,
                      self.results["iv"],
                      color="red", linewidth=1,
                      alpha=0.7, label="IV (Realised)")
        ax3.set_title("Futures Price + Implied Vol")
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
        ax4.set_xlabel("Daily P&L")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # --- Chart 5: Trade P&L per trade ---
        ax5 = fig.add_subplot(gs[3, :])
        if not self.position_log.empty and \
           "trade_pnl" in self.position_log.columns:
            completed = self.position_log.dropna(
                subset=["trade_pnl"]
            )
            if not completed.empty:
                colors_bar = [
                    "green" if p > 0 else "red"
                    for p in completed["trade_pnl"]
                ]
                ax5.bar(range(len(completed)),
                        completed["trade_pnl"],
                        color=colors_bar, alpha=0.8)
                ax5.axhline(0, color="black", linewidth=0.8)
                ax5.set_title("P&L per Trade")
                ax5.set_xlabel("Trade Number")
                ax5.set_ylabel("P&L")
                ax5.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()


# =============================================================================
# MAIN — Compare all strategies
# =============================================================================

if __name__ == "__main__":
    from data.data_loader import DataLoader

    # Load CL data
    loader = DataLoader()
    df     = loader.fetch(
        ["CL=F"], start="2015-01-01", end="2024-01-01"
    )
    close  = df["Close"].squeeze().dropna()

    strategies = [
        ("short_strangle", True,  "Short Strangle (Delta Hedged)"),
        ("short_strangle", False, "Short Strangle (No Hedge)"),
        ("short_straddle", True,  "Short Straddle (Delta Hedged)"),
        ("long_straddle",  True,  "Long Straddle  (Delta Hedged)"),
    ]

    results_all = {}

    for strat, hedge, label in strategies:
        print(f"\n{'='*55}")
        print(f"Running: {label}")
        print(f"{'='*55}")

        bt = OptionsBacktester(
            close        = close,
            strategy     = strat,
            option_tenor = 30,
            vol_window   = 20,
            r            = 0.05,
            otm_pct      = 0.05,
            delta_hedge  = hedge,
            option_cost  = 0.002,
            futures_cost = 0.001
        )
        results = bt.run()
        bt.summary()
        bt.plot(
            title     = label,
            save_path = f"evaluation/options_{strat}_"
                        f"{'hedged' if hedge else 'raw'}.png"
        )
        results_all[label] = {
            "results" : results,
            "sharpe"  : bt.sharpe(),
            "mdd"     : bt.max_drawdown(),
            "bt"      : bt
        }

    # Comparison chart
    print("\nGenerating comparison chart...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        "Options Strategies Comparison: CL=F 2015-2024",
        fontsize=14, fontweight="bold"
    )

    colors = ["steelblue", "lightblue", "green", "red"]
    for i, (label, data) in enumerate(results_all.items()):
        axes[0].plot(
            data["results"].index,
            data["results"]["cumulative"],
            label=f"{label} (Sharpe: {data['sharpe']:.2f})",
            color=colors[i], linewidth=1.5
        )
        cum = data["results"]["cumulative"]
        dd  = (cum - cum.cummax()) / cum.cummax()
        axes[1].fill_between(
            data["results"].index, dd, 0,
            alpha=0.25, color=colors[i], label=label
        )

    axes[0].axhline(1.0, color="grey", linestyle=":",
                    linewidth=0.8)
    axes[0].set_title("Cumulative P&L")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Drawdown Comparison")
    axes[1].set_ylabel("Drawdown %")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation/options_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Done!")
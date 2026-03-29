import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# PART 1 — BLACK-76 + BARONE-ADESI WHALEY ENGINE
# =============================================================================

class BlackScholes:
    """
    Options pricing engine for commodity futures.
    
    Two models:
    - Black-76          : European futures options (analytical)
    - Barone-Adesi Whaley (BAW) : American futures options
                          Industry standard for CL, NG, GC options
    
    Why American matters for CL:
    - Early exercise premium can be significant
    - Especially for deep ITM options near expiry
    - BAW gives ~same price as binomial tree but 1000x faster
    """

    # ----------------------------------------------------------------
    # BLACK-76 (European)
    # ----------------------------------------------------------------

    @staticmethod
    def d1(F, K, T, r, sigma):
        return (np.log(F / K) + 0.5 * sigma**2 * T) / (
            sigma * np.sqrt(T) + 1e-10
        )

    @staticmethod
    def d2(F, K, T, r, sigma):
        return BlackScholes.d1(F, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def european_price(F, K, T, r, sigma, option_type="call"):
        """Black-76 European futures option price."""
        if T <= 0:
            return max(F - K, 0) if option_type == "call" \
                   else max(K - F, 0)

        d1 = BlackScholes.d1(F, K, T, r, sigma)
        d2 = BlackScholes.d2(F, K, T, r, sigma)
        df = np.exp(-r * T)

        if option_type == "call":
            return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    # ----------------------------------------------------------------
    # BARONE-ADESI WHALEY (American)
    # ----------------------------------------------------------------

    @staticmethod
    def american_price(F, K, T, r, sigma, option_type="call"):
        """
        Barone-Adesi Whaley approximation for American futures option.
        
        Standard model used by:
        - CME for CL, NG options
        - Bloomberg OVML
        - All major commodity options desks
        
        Returns American price >= European price
        Difference = early exercise premium
        """
        if T <= 0:
            return max(F - K, 0) if option_type == "call" \
                   else max(K - F, 0)

        euro = BlackScholes.european_price(
            F, K, T, r, sigma, option_type
        )

        # BAW parameters
        M  = 2 * r / (sigma**2)
        q2 = (-(M - 1) + np.sqrt((M - 1)**2 + 4 * M)) / 2

        if option_type == "call":
            # Critical futures price
            def _objective(F_star):
                euro_star = BlackScholes.european_price(
                    F_star, K, T, r, sigma, "call"
                )
                d1_star = BlackScholes.d1(F_star, K, T, r, sigma)
                return (F_star - K) - euro_star - \
                       (F_star / q2) * (1 - norm.cdf(d1_star))

            try:
                F_star = brentq(_objective, K * 0.01, K * 10,
                                maxiter=100)
            except:
                return euro

            if F >= F_star:
                return F - K  # early exercise optimal

            d1 = BlackScholes.d1(F, K, T, r, sigma)
            A2 = (F_star / q2) * (1 - norm.cdf(
                BlackScholes.d1(F_star, K, T, r, sigma)
            ))
            return euro + A2 * (F / F_star) ** q2

        else:  # put
            q1 = (-(M - 1) - np.sqrt((M - 1)**2 + 4 * M)) / 2

            def _objective(F_star):
                euro_star = BlackScholes.european_price(
                    F_star, K, T, r, sigma, "put"
                )
                d1_star = BlackScholes.d1(F_star, K, T, r, sigma)
                return (K - F_star) - euro_star + \
                       (F_star / q1) * (1 - norm.cdf(-d1_star))

            try:
                F_star = brentq(_objective, K * 0.01, K * 5,
                                maxiter=100)
            except:
                return euro

            if F <= F_star:
                return K - F  # early exercise optimal

            d1 = BlackScholes.d1(F, K, T, r, sigma)
            A1 = -(F_star / q1) * (1 - norm.cdf(
                -BlackScholes.d1(F_star, K, T, r, sigma)
            ))
            return euro + A1 * (F / F_star) ** q1

    @staticmethod
    def price(F, K, T, r, sigma, option_type="call",
              american=True):
        """
        Price option — American (BAW) by default for CL.
        Set american=False for European (theoretical).
        """
        if american:
            return BlackScholes.american_price(
                F, K, T, r, sigma, option_type
            )
        else:
            return BlackScholes.european_price(
                F, K, T, r, sigma, option_type
            )

    @staticmethod
    def delta(F, K, T, r, sigma, option_type="call"):
        """Delta via finite difference (works for both American/European)."""
        if T <= 0:
            if option_type == "call":
                return 1.0 if F > K else 0.0
            else:
                return -1.0 if F < K else 0.0

        dF   = F * 0.001
        p_up = BlackScholes.price(
            F + dF, K, T, r, sigma, option_type
        )
        p_dn = BlackScholes.price(
            F - dF, K, T, r, sigma, option_type
        )
        return (p_up - p_dn) / (2 * dF)

    @staticmethod
    def gamma(F, K, T, r, sigma, option_type="call"):
        """Gamma via finite difference."""
        if T <= 0:
            return 0.0
        dF   = F * 0.001
        p_up = BlackScholes.price(F + dF, K, T, r, sigma, option_type)
        p_mid= BlackScholes.price(F,      K, T, r, sigma, option_type)
        p_dn = BlackScholes.price(F - dF, K, T, r, sigma, option_type)
        return (p_up - 2 * p_mid + p_dn) / (dF**2)

    @staticmethod
    def vega(F, K, T, r, sigma, option_type="call"):
        """Vega via finite difference."""
        if T <= 0:
            return 0.0
        ds   = 0.001
        p_up = BlackScholes.price(F, K, T, r, sigma + ds, option_type)
        p_dn = BlackScholes.price(F, K, T, r, sigma - ds, option_type)
        return (p_up - p_dn) / (2 * ds)

    @staticmethod
    def theta(F, K, T, r, sigma, option_type="call"):
        """Theta per day via finite difference."""
        if T <= 1/365:
            return 0.0
        dt   = 1/365
        p_now  = BlackScholes.price(F, K, T,      r, sigma, option_type)
        p_next = BlackScholes.price(F, K, T - dt, r, sigma, option_type)
        return p_next - p_now

    @staticmethod
    def implied_vol(market_price, F, K, T, r,
                    option_type="call", american=True):
        """Implied vol from market price."""
        if T <= 0:
            return np.nan
        intrinsic = max(F - K, 0) if option_type == "call" \
                    else max(K - F, 0)
        if market_price <= intrinsic:
            return np.nan
        try:
            def objective(sigma):
                return BlackScholes.price(
                    F, K, T, r, sigma, option_type, american
                ) - market_price
            return brentq(objective, 0.001, 5.0, maxiter=100)
        except:
            return np.nan

    @staticmethod
    def all_greeks(F, K, T, r, sigma, option_type="call"):
        """All Greeks as dictionary."""
        return {
            "price" : BlackScholes.price(F, K, T, r, sigma, option_type),
            "delta" : BlackScholes.delta(F, K, T, r, sigma, option_type),
            "gamma" : BlackScholes.gamma(F, K, T, r, sigma, option_type),
            "vega"  : BlackScholes.vega(F, K, T, r, sigma, option_type),
            "theta" : BlackScholes.theta(F, K, T, r, sigma, option_type)
        }

    @staticmethod
    def early_exercise_premium(F, K, T, r, sigma, option_type="call"):
        """
        Early exercise premium = American - European price.
        Shows value of American feature.
        """
        american = BlackScholes.american_price(F, K, T, r, sigma, option_type)
        european = BlackScholes.european_price(F, K, T, r, sigma, option_type)
        return american - european


# =============================================================================
# PART 2 — VOL SURFACE
# =============================================================================

class VolSurface:
    """
    Implied vol surface for commodity futures options.
    Parametric model — replace with Bloomberg OVML in production.
    """

    def __init__(self, atm_vol, skew=-0.15,
                 smile=0.05, term_slope=-0.02):
        self.atm_vol    = atm_vol
        self.skew       = skew
        self.smile      = smile
        self.term_slope = term_slope

    def get_vol(self, F, K, T):
        moneyness = np.log(K / F)
        vol_adj   = self.skew * moneyness + self.smile * moneyness**2
        term_adj  = self.term_slope * T
        return max(self.atm_vol + vol_adj + term_adj, 0.01)

    def build_surface(self, F, strikes, expiries):
        surface = pd.DataFrame(
            index   = [f"{k:.0f}" for k in strikes],
            columns = [f"{int(t*365)}d" for t in expiries]
        )
        for K in strikes:
            for T in expiries:
                surface.loc[f"{K:.0f}", f"{int(T*365)}d"] = \
                    round(self.get_vol(F, K, T), 4)
        return surface.astype(float)


# =============================================================================
# PART 3 — OPTIONS PORTFOLIO
# =============================================================================

class OptionsPortfolio:
    """Options portfolio with Greeks and P&L analysis."""

    def __init__(self, F, r=0.05):
        self.F         = F
        self.r         = r
        self.positions = []

    def add_position(self, K, T, sigma, option_type,
                     quantity, name=""):
        greeks   = BlackScholes.all_greeks(
            self.F, K, T, self.r, sigma, option_type
        )
        position = {
            "name"        : name or f"{option_type} K={K:.0f} "
                            f"T={int(T*365)}d",
            "K"           : K,
            "T"           : T,
            "sigma"       : sigma,
            "option_type" : option_type,
            "quantity"    : quantity,
            **{k: v * quantity for k, v in greeks.items()}
        }
        self.positions.append(position)
        return greeks

    def portfolio_greeks(self):
        total = {
            "price": 0, "delta": 0,
            "gamma": 0, "vega" : 0, "theta": 0
        }
        for pos in self.positions:
            for greek in total:
                total[greek] += pos[greek]
        return total

    def delta_hedge_size(self):
        return -self.portfolio_greeks().get("delta", 0)

    def print_summary(self):
        print("=" * 70)
        print("            OPTIONS PORTFOLIO SUMMARY (AMERICAN BAW)")
        print("=" * 70)
        print(f"Underlying futures price: ${self.F:.2f}")
        print()
        print(f"{'Position':<28} {'Price':>8} {'Delta':>8} "
              f"{'Gamma':>8} {'Vega':>8} {'Theta':>8}")
        print("-" * 70)
        for pos in self.positions:
            print(f"{pos['name']:<28} "
                  f"{pos['price']:>8.3f} "
                  f"{pos['delta']:>8.3f} "
                  f"{pos['gamma']:>8.4f} "
                  f"{pos['vega']:>8.3f} "
                  f"{pos['theta']:>8.4f}")
        print("-" * 70)
        total = self.portfolio_greeks()
        print(f"{'TOTAL':<28} "
              f"{total['price']:>8.3f} "
              f"{total['delta']:>8.3f} "
              f"{total['gamma']:>8.4f} "
              f"{total['vega']:>8.3f} "
              f"{total['theta']:>8.4f}")
        print(f"\nDelta hedge: {self.delta_hedge_size():+.3f} futures")
        print("=" * 70)


# =============================================================================
# PART 4 — HISTORICAL GREEKS
# =============================================================================

class HistoricalGreeks:
    """Rolling Greeks on historical data using BAW model."""

    def __init__(self, r=0.05, T=30/365, vol_window=20):
        self.r          = r
        self.T          = T
        self.vol_window = vol_window

    def calculate(self, close, strike_pct=1.0, option_type="call"):
        ret   = np.log(close / close.shift(1))
        sigma = ret.rolling(self.vol_window).std() * np.sqrt(252)
        results = []
        for date, F in close.items():
            if pd.isna(F):
                continue
            s = sigma.get(date, np.nan)
            if pd.isna(s) or s <= 0:
                continue
            K      = F * strike_pct
            greeks = BlackScholes.all_greeks(
                F, K, self.T, self.r, s, option_type
            )
            greeks.update({"date": date, "F": F, "K": K, "sigma": s})
            results.append(greeks)
        return pd.DataFrame(results).set_index("date")


# =============================================================================
# PART 5 — VOL REGIME SIGNAL
# =============================================================================

class VolRegimeSignal:
    """Vol risk premium signal: IV - RV spread."""

    def __init__(self, rv_window=20):
        self.rv_window = rv_window

    def calculate(self, close):
        ret      = np.log(close / close.shift(1))
        rv       = ret.rolling(self.rv_window).std() * np.sqrt(252)
        iv_proxy = ret.rolling(self.rv_window * 2).std() * np.sqrt(252)
        vrp      = iv_proxy - rv
        vrp_z    = (vrp - vrp.rolling(60).mean()) / \
                   (vrp.rolling(60).std() + 1e-8)
        return pd.DataFrame({
            "rv": rv, "iv_proxy": iv_proxy,
            "vrp": vrp, "vrp_zscore": vrp_z
        })

    def signal(self, close, threshold=1.0):
        vrp_df = self.calculate(close)
        z      = vrp_df["vrp_zscore"]
        sig    = pd.Series(0, index=z.index, name="vol_signal")
        sig[z >  threshold] =  1
        sig[z < -threshold] = -1
        return sig.shift(1)


# =============================================================================
# MAIN — SINGLE SCROLLING WINDOW
# =============================================================================

if __name__ == "__main__":
    from data.data_loader import DataLoader

    # Load real CL data
    loader  = DataLoader()
    df_hist = loader.fetch(
        ["CL=F"], start="2020-01-01", end="2024-12-31"
    )
    close  = df_hist["Close"].squeeze().dropna()
    ret    = np.log(close / close.shift(1))
    F      = float(close.iloc[-1])
    sigma  = float(ret.tail(20).std() * np.sqrt(252))
    r      = 0.05

    # --- Print pricing comparison ---
    print("=" * 65)
    print("BLACK-76 vs BAW (American) PRICING — CL=F")
    print("=" * 65)
    print(f"F=${F:.2f} | ATM | T=30d | Vol={sigma:.1%}\n")
    print(f"{'':6} {'European':>12} {'American':>12} "
          f"{'Early Ex Premium':>18}")
    print("-" * 50)
    for opt in ["call", "put"]:
        euro = BlackScholes.european_price(F, F, 30/365, r, sigma, opt)
        amer = BlackScholes.american_price(F, F, 30/365, r, sigma, opt)
        prem = amer - euro
        print(f"{opt.upper():6} ${euro:>10.3f}   ${amer:>10.3f}   "
              f"${prem:>16.4f}")

    print("\nGreeks (American BAW):")
    for opt in ["call", "put"]:
        g = BlackScholes.all_greeks(F, F, 30/365, r, sigma, opt)
        print(f"{opt.upper():4}: "
              f"Price=${g['price']:.3f} | "
              f"Delta={g['delta']:+.3f} | "
              f"Gamma={g['gamma']:.4f} | "
              f"Vega={g['vega']:.3f} | "
              f"Theta={g['theta']:.4f}")

    # --- Build all data ---
    vol_surface = VolSurface(
        atm_vol=sigma, skew=-0.15,
        smile=0.05, term_slope=-0.02
    )
    strikes  = np.arange(
        round(F * 0.75 / 5) * 5,
        round(F * 1.25 / 5) * 5 + 5, 5
    ).astype(float)
    expiries = [7/365, 14/365, 30/365, 60/365, 90/365]
    surface  = vol_surface.build_surface(F, strikes, expiries)

    # Portfolio — short strangle
    otm_call = round(F * 1.10 / 5) * 5
    otm_put  = round(F * 0.90 / 5) * 5
    portfolio = OptionsPortfolio(F=F, r=r)
    portfolio.add_position(
        otm_call, 30/365,
        vol_surface.get_vol(F, otm_call, 30/365),
        "call", -1, f"Short Call K={otm_call:.0f}"
    )
    portfolio.add_position(
        otm_put, 30/365,
        vol_surface.get_vol(F, otm_put, 30/365),
        "put", -1, f"Short Put K={otm_put:.0f}"
    )
    portfolio.print_summary()

    # Historical Greeks
    hist   = HistoricalGreeks(r=r, T=30/365, vol_window=20)
    gdf    = hist.calculate(close, strike_pct=1.0, option_type="call")

    # Vol signal
    vrs    = VolRegimeSignal(rv_window=20)
    vrp_df = vrs.calculate(close)
    sig    = vrs.signal(close)

    # P&L profile data
    spot_range = np.linspace(F * 0.7, F * 1.3, 200)
    total_pnl  = np.zeros(len(spot_range))
    for pos in portfolio.positions:
        K, otype, qty = pos["K"], pos["option_type"], pos["quantity"]
        cost = abs(pos["price"] / qty) if qty != 0 else 0
        for i, S in enumerate(spot_range):
            payoff = max(S - K, 0) if otype == "call" else max(K - S, 0)
            total_pnl[i] += qty * (payoff - cost)

    deltas_profile = []
    for S in spot_range:
        total_delta = sum(
            BlackScholes.delta(
                S, pos["K"], pos["T"], r, pos["sigma"],
                pos["option_type"]
            ) * pos["quantity"]
            for pos in portfolio.positions
        )
        deltas_profile.append(total_delta)

    # =========================================================
    # SINGLE SCROLLING FIGURE — all charts in one window
    # =========================================================
    fig = plt.figure(figsize=(16, 52))
    fig.suptitle(
        f"Crude Oil Options Analysis — CL=F  "
        f"F=${F:.0f}  ATM Vol={sigma:.1%}  (American BAW)",
        fontsize=14, fontweight="bold", y=0.995
    )
    gs = gridspec.GridSpec(
        9, 2, figure=fig, hspace=0.5, wspace=0.35
    )

    colors_map = plt.cm.viridis(np.linspace(0, 1, len(expiries)))

    # --- Row 0: Vol smile ---
    ax0 = fig.add_subplot(gs[0, :])
    for i, T in enumerate(expiries):
        vols = [vol_surface.get_vol(F, K, T) for K in strikes]
        ax0.plot(strikes, vols, label=f"{int(T*365)}d",
                 color=colors_map[i], linewidth=1.5)
    ax0.axvline(F, color="red", linestyle="--",
                linewidth=1, label=f"ATM (F={F:.0f})")
    ax0.set_title("Vol Smile / Skew by Expiry")
    ax0.set_xlabel("Strike")
    ax0.set_ylabel("Implied Vol")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # --- Row 1: ATM term structure + Heatmap ---
    ax1a = fig.add_subplot(gs[1, 0])
    atm_vols = [vol_surface.get_vol(F, F, T) for T in expiries]
    ax1a.plot([int(T*365) for T in expiries], atm_vols,
              color="steelblue", linewidth=2, marker="o")
    ax1a.set_title("ATM Vol Term Structure")
    ax1a.set_xlabel("Days to Expiry")
    ax1a.set_ylabel("ATM Implied Vol")
    ax1a.grid(True, alpha=0.3)

    ax1b = fig.add_subplot(gs[1, 1])
    im   = ax1b.imshow(surface.values, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax1b)
    ax1b.set_xticks(range(len(expiries)))
    ax1b.set_xticklabels([f"{int(T*365)}d" for T in expiries], fontsize=7)
    ax1b.set_yticks(range(len(strikes)))
    ax1b.set_yticklabels([f"{K:.0f}" for K in strikes], fontsize=7)
    ax1b.set_title("Vol Surface Heatmap")

    # --- Row 2: P&L profile + Delta profile ---
    ax2a = fig.add_subplot(gs[2, 0])
    ax2a.plot(spot_range, total_pnl, color="steelblue", linewidth=2)
    ax2a.axhline(0, color="black", linewidth=0.8)
    ax2a.axvline(F, color="red", linestyle="--",
                 linewidth=1, label=f"F=${F:.0f}")
    ax2a.fill_between(spot_range, total_pnl, 0,
                      where=(total_pnl > 0),
                      alpha=0.3, color="green", label="Profit")
    ax2a.fill_between(spot_range, total_pnl, 0,
                      where=(total_pnl < 0),
                      alpha=0.3, color="red", label="Loss")
    ax2a.set_title("Short Strangle P&L at Expiry")
    ax2a.set_xlabel("Futures Price")
    ax2a.set_ylabel("P&L ($)")
    ax2a.legend(fontsize=8)
    ax2a.grid(True, alpha=0.3)

    ax2b = fig.add_subplot(gs[2, 1])
    ax2b.plot(spot_range, deltas_profile, color="orange", linewidth=2)
    ax2b.axhline(0, color="black", linewidth=0.8)
    ax2b.axvline(F, color="red", linestyle="--",
                 linewidth=1, label=f"F=${F:.0f}")
    ax2b.set_title("Short Strangle Delta Profile")
    ax2b.set_xlabel("Futures Price")
    ax2b.set_ylabel("Portfolio Delta")
    ax2b.legend(fontsize=8)
    ax2b.grid(True, alpha=0.3)

    # --- Rows 3-5: Historical Greeks ---
    greek_plots = [
        ("sigma", "Realised Vol (IV Proxy)", "steelblue", gs[3, 0]),
        ("price", "ATM Call Price",           "green",     gs[3, 1]),
        ("delta", "Delta",                    "orange",    gs[4, 0]),
        ("gamma", "Gamma",                    "red",       gs[4, 1]),
        ("vega",  "Vega",                     "purple",    gs[5, 0]),
        ("theta", "Theta (Daily Decay)",       "brown",     gs[5, 1]),
    ]
    for col, label, color, loc in greek_plots:
        ax = fig.add_subplot(loc)
        ax.plot(gdf.index, gdf[col], color=color, linewidth=1)
        ax.set_title(f"Historical {label}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # --- Row 6-7: Vol regime signal ---
    ax6 = fig.add_subplot(gs[6, :])
    ax6.plot(vrp_df.index, vrp_df["rv"],
             label="Realised Vol", color="steelblue")
    ax6.plot(vrp_df.index, vrp_df["iv_proxy"],
             label="IV Proxy", color="orange")
    ax6.set_title("Realised vs Implied Vol")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[7, :])
    ax7.plot(vrp_df.index, vrp_df["vrp_zscore"],
             color="purple", linewidth=1)
    ax7.axhline(1.0, color="red", linestyle="--",
                label="Sell vol")
    ax7.axhline(-1.0, color="green", linestyle="--",
                label="Buy vol")
    ax7.axhline(0, color="black", linewidth=0.5)
    ax7.set_title("Vol Risk Premium Z-Score")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[8, :])
    colors_sig = ["red" if s == 1 else
                  "green" if s == -1 else "grey"
                  for s in sig.values]
    ax8.bar(sig.index, sig, color=colors_sig, alpha=0.7, width=1)
    ax8.set_title("Vol Signal (+1=Sell Vol, -1=Buy Vol, 0=Neutral)")
    ax8.set_ylabel("Signal")
    ax8.grid(True, alpha=0.3)

    plt.savefig("evaluation/options_full_report.png",
                dpi=120, bbox_inches="tight")
    print("\nSaved to evaluation/options_full_report.png")
    plt.show()

    print(f"\nVol signal: Sell={( sig==1).sum()} | "
          f"Buy={(sig==-1).sum()} | "
          f"Neutral={(sig==0).sum()}")
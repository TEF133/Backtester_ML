import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


class RegimeDetector:
    """
    Detects market regimes using Gaussian Mixture Models (GMM).
    
    For commodities, typical regimes are:
    BULL TREND : positive returns, low vol, trending up
    BEAR TREND : negative returns, low vol, trending down
    HIGH VOL   : spike in volatility, mean reverting
    SIDEWAYS   : low momentum, low vol, choppy
    """

    def __init__(self, n_regimes=4, lookback=252):
        self.n_regimes    = n_regimes
        self.lookback     = lookback
        self.model        = None
        self.scaler       = StandardScaler()
        self.regime_stats = None

    def _build_regime_features(self, close):
        ret      = np.log(close / close.shift(1))
        features = pd.DataFrame(index=close.index)

        features["trend_20d"] = np.log(
            close / close.shift(20)
        ) / (ret.rolling(20).std() * np.sqrt(20) + 1e-8)

        features["trend_60d"] = np.log(
            close / close.shift(60)
        ) / (ret.rolling(60).std() * np.sqrt(60) + 1e-8)

        features["vol_20d"]   = ret.rolling(20).std() * np.sqrt(252)
        features["vol_60d"]   = ret.rolling(60).std() * np.sqrt(252)

        features["vol_ratio"] = (
            ret.rolling(10).std() /
            (ret.rolling(40).std() + 1e-8)
        )

        roll_mean             = close.rolling(20).mean()
        roll_std              = close.rolling(20).std()
        features["zscore"]    = (close - roll_mean) / (roll_std + 1e-8)

        features["mom_5d"]    = np.log(close / close.shift(5))
        features["mom_20d"]   = np.log(close / close.shift(20))

        features["mom_consistency"] = ret.rolling(20).apply(
            lambda x: (x > 0).mean()
        )

        return features.dropna()

    def fit(self, close):
        features   = self._build_regime_features(close)
        X          = self.scaler.fit_transform(features)
        self.model = GaussianMixture(
            n_components    = self.n_regimes,
            covariance_type = "full",
            n_init          = 10,
            random_state    = 42
        )
        self.model.fit(X)
        self.feature_index = features.index
        return self

    def predict(self, close):
        features   = self._build_regime_features(close)
        common     = features.index
        X          = self.scaler.transform(features)
        raw_labels = self.model.predict(X)
        probs      = self.model.predict_proba(X)

        # Smooth labels to avoid daily flickering
        labels = pd.Series(raw_labels, index=common)
        labels = labels.rolling(5, min_periods=1).apply(
            lambda x: pd.Series(x).mode()[0]
        ).astype(int)

        regime_labels = labels.rename("regime")
        regime_probs  = pd.DataFrame(
            probs,
            index   = common,
            columns = [f"prob_regime_{i}"
                       for i in range(self.n_regimes)]
        )
        return regime_labels, regime_probs

    def label_regimes(self, close, regime_labels):
        ret   = np.log(close / close.shift(1))
        stats = {}

        for r in range(self.n_regimes):
            mask   = regime_labels == r
            common = close.index.intersection(mask.index)
            mask   = mask.loc[common]
            r_ret  = ret.loc[common][mask.values]

            ann_ret = r_ret.mean() * 252
            ann_vol = r_ret.std() * np.sqrt(252)
            n_days  = mask.sum()

            if ann_ret > 0.02:
                label = "BULL TREND"
                color = "green"
            elif ann_ret < -0.02:
                label = "BEAR TREND"
                color = "red"
            elif ann_vol > 0.30:
                label = "HIGH VOL"
                color = "orange"
            else:
                label = "SIDEWAYS"
                color = "grey"

            stats[r] = {
                "label"   : label,
                "color"   : color,
                "ann_ret" : ann_ret,
                "ann_vol" : ann_vol,
                "n_days"  : n_days,
                "sharpe"  : ann_ret / ann_vol if ann_vol > 0 else 0
            }

        self.regime_stats = stats
        return stats

    def plot(self, close, regime_labels, regime_probs,
             title="Market Regime Detection", save_path=None):
        if self.regime_stats is None:
            self.label_regimes(close, regime_labels)

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45)
        colors = [self.regime_stats[r]["color"]
                  for r in range(self.n_regimes)]

        # --- Chart 1: Price with regime background ---
        ax1        = fig.add_subplot(gs[0, :])
        common_idx = close.index.intersection(regime_labels.index)
        close_plot = close.loc[common_idx]
        labels_plot= regime_labels.loc[common_idx]

        ax1.plot(close_plot.index, close_plot.values,
                 color="black", linewidth=1, zorder=5)

        prev_regime = labels_plot.iloc[0]
        start_idx   = labels_plot.index[0]
        for date, regime in labels_plot.items():
            if regime != prev_regime:
                ax1.axvspan(start_idx, date,
                            alpha=0.25, color=colors[prev_regime])
                start_idx   = date
                prev_regime = regime
        ax1.axvspan(start_idx, labels_plot.index[-1],
                    alpha=0.25, color=colors[prev_regime])

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.regime_stats[r]["color"],
                  alpha=0.5,
                  label=(f"R{r}: {self.regime_stats[r]['label']} "
                         f"({self.regime_stats[r]['n_days']}d)"))
            for r in range(self.n_regimes)
        ]
        ax1.legend(handles=legend_elements, fontsize=8, loc="upper left")
        ax1.set_title("Price with Regime Background")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.2)

        # --- Chart 2: Regime probabilities ---
        ax2        = fig.add_subplot(gs[1, :])
        common_prob= close.index.intersection(regime_probs.index)
        rp         = regime_probs.loc[common_prob]
        for r in range(self.n_regimes):
            col   = f"prob_regime_{r}"
            label = self.regime_stats[r]["label"]
            ax2.plot(rp.index, rp[col].rolling(5).mean(),
                     label=f"R{r}: {label}",
                     color=colors[r], linewidth=1)
        ax2.set_title("Regime Probabilities (5d smoothed)")
        ax2.set_ylabel("Probability")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # --- Chart 3: Ann return per regime ---
        ax3          = fig.add_subplot(gs[2, 0])
        regime_names = [f"R{r}: {self.regime_stats[r]['label']}"
                        for r in range(self.n_regimes)]
        ann_rets     = [self.regime_stats[r]["ann_ret"]
                        for r in range(self.n_regimes)]
        bar_colors   = [self.regime_stats[r]["color"]
                        for r in range(self.n_regimes)]
        ax3.bar(regime_names, ann_rets,
                color=bar_colors, edgecolor="white", alpha=0.8)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_title("Annualised Return per Regime")
        ax3.set_ylabel("Ann. Return")
        ax3.tick_params(axis="x", labelsize=7)
        ax3.grid(True, alpha=0.3, axis="y")

        # --- Chart 4: Time in each regime ---
        ax4         = fig.add_subplot(gs[2, 1])
        regime_days = [self.regime_stats[r]["n_days"]
                       for r in range(self.n_regimes)]
        ax4.pie(regime_days, labels=regime_names,
                colors=bar_colors, autopct="%1.0f%%", startangle=90)
        ax4.set_title("Time Spent in Each Regime")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()

    def print_summary(self):
        if self.regime_stats is None:
            print("Run label_regimes() first.")
            return
        print("=" * 60)
        print("           REGIME DETECTION SUMMARY")
        print("=" * 60)
        for r, stats in self.regime_stats.items():
            print(f"Regime {r} — {stats['label']:12s} | "
                  f"Days: {stats['n_days']:4d} | "
                  f"Ann Ret: {stats['ann_ret']:+.1%} | "
                  f"Vol: {stats['ann_vol']:.1%} | "
                  f"Sharpe: {stats['sharpe']:+.2f}")
        print("=" * 60)


class RegimeAwareStrategy:
    """
    Adapts position sizing based on current regime.
    BULL/BEAR TREND → 1.5x position
    HIGH VOL        → 0.3x position
    SIDEWAYS        → 0.7x position
    Only acts on regimes stable for 5+ consecutive days.
    """

    def __init__(self, regime_detector):
        self.detector = regime_detector

    def get_multiplier(self, regime_probs_row):
        """Weighted average multiplier based on regime probabilities."""
        stats = self.detector.regime_stats
        total = 0.0
        for r in range(self.detector.n_regimes):
            label = stats[r]["label"]
            prob  = regime_probs_row[f"prob_regime_{r}"]
            if label in ["BULL TREND", "BEAR TREND"]:
                mult = 1.5
            elif label == "HIGH VOL":
                mult = 0.3
            else:
                mult = 0.7
            total += prob * mult
        return total

    def apply(self, signals, regime_labels, regime_probs):
        """
        Scale signals by regime multiplier.
        Only adjusts on stable regimes (5+ consecutive days same regime).
        This prevents excessive trading from regime flickering.
        """
        adj_signals = signals.copy()
        common      = signals.index.intersection(regime_probs.index)

        # Build stable regime — only act when regime held for 5+ days
        stable = regime_labels.rolling(5, min_periods=5).apply(
            lambda x: x.iloc[-1] if len(set(x)) == 1 else -1
        )

        for date in common:
            if date not in stable.index:
                continue
            if stable.loc[date] == -1:
                continue  # unstable — keep original signal
            mult = self.get_multiplier(regime_probs.loc[date])
            adj_signals.loc[date] *= mult

        return adj_signals


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from strategies.trend_following import TrendFollowing
    from backtest.backtest_engine import BacktestEngine

    # 1. Load data
    loader = DataLoader()
    df     = loader.fetch(["CL=F"], start="2010-01-01", end="2024-01-01")
    close  = df["Close"].squeeze()

    # 2. Detect regimes
    print("Fitting regime detector...")
    detector = RegimeDetector(n_regimes=4)
    detector.fit(close)
    regime_labels, regime_probs = detector.predict(close)
    stats = detector.label_regimes(close, regime_labels)
    detector.print_summary()

    # 3. Plot
    detector.plot(
        close, regime_labels, regime_probs,
        title     = "Crude Oil Regime Detection (GMM)",
        save_path = "evaluation/regime_detection.png"
    )

    # 4. Compare plain vs regime-aware
    print("\nComparing strategies...")
    next_ret = np.log(close / close.shift(1)).shift(-1)

    # Plain trend
    tf        = TrendFollowing(fast=20, slow=60)
    sig_tf    = tf.predict_sized(close)
    common    = sig_tf.index.intersection(next_ret.dropna().index)
    bt_plain  = BacktestEngine(sig_tf.loc[common], next_ret.loc[common])
    res_plain = bt_plain.run()

    # Regime-aware trend
    reg_strat  = RegimeAwareStrategy(detector)
    sig_regime = reg_strat.apply(sig_tf, regime_labels, regime_probs)
    common2    = sig_regime.index.intersection(next_ret.dropna().index)
    bt_regime  = BacktestEngine(
        sig_regime.loc[common2], next_ret.loc[common2]
    )
    res_regime = bt_regime.run()

    print("\nPlain Trend Following:")
    bt_plain.summary()
    print("\nRegime-Aware Trend Following:")
    bt_regime.summary()

    # 5. Comparison chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(res_plain.index, res_plain["cumulative_pnl"],
            label=f"Plain Trend "
                  f"(Sharpe: {bt_plain.sharpe_ratio():.2f})",
            color="steelblue", linewidth=1.5)
    ax.plot(res_regime.index, res_regime["cumulative_pnl"],
            label=f"Regime-Aware "
                  f"(Sharpe: {bt_regime.sharpe_ratio():.2f})",
            color="green", linewidth=1.5)
    ax.plot(res_plain.index, res_plain["benchmark"],
            label="Buy & Hold",
            color="orange", linestyle="--")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title("Plain Trend vs Regime-Aware: Crude Oil")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("evaluation/regime_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.show()
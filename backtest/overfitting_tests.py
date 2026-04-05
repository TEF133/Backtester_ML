import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations, product
from scipy.stats import norm
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
# PART 1 — DEFLATED SHARPE RATIO (DSR)
# =============================================================================

class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014)

    Corrects for multiple testing: if you test N strategies,
    the best one has an inflated Sharpe purely by luck.

    n_trials = 1  → fixed strategy (no search)
    n_trials = 20 → 20 combinations tested

    DSR p > 0.95 → GENUINE EDGE
    DSR p > 0.50 → BORDERLINE
    DSR p < 0.50 → LIKELY LUCK
    """

    def __init__(self, n_trials=1, skewness=0.0, kurtosis=3.0):
        self.n_trials = max(1, n_trials)
        self.skewness = skewness
        self.kurtosis = kurtosis

    def expected_max_sharpe(self):
        if self.n_trials <= 1:
            return 0.0
        e_max = (
            (1 - 0.5772) / np.log(self.n_trials) ** 0.5 +
            1 / (2 * np.log(self.n_trials)) ** 0.5
        )
        return e_max

    def compute(self, sharpe_obs, n_obs):
        sharpe_benchmark = self.expected_max_sharpe()
        sr_daily = sharpe_obs       / np.sqrt(252)
        sr_bench = sharpe_benchmark / np.sqrt(252)

        var_sr = (
            1 +
            0.5 * sr_daily ** 2 -
            self.skewness * sr_daily +
            (self.kurtosis - 3) / 4 * sr_daily ** 2
        ) / max(n_obs, 1)

        std_sr  = np.sqrt(var_sr)
        dsr     = (sr_daily - sr_bench) / (std_sr + 1e-10)
        p_value = float(norm.cdf(dsr))

        verdict = "GENUINE EDGE" if p_value > 0.95 else \
                  "BORDERLINE"   if p_value > 0.50 else \
                  "LIKELY LUCK"

        return {
            "dsr"             : dsr,
            "p_value"         : p_value,
            "verdict"         : verdict,
            "sharpe_obs"      : sharpe_obs,
            "sharpe_benchmark": sharpe_benchmark,
            "n_trials"        : self.n_trials,
            "n_obs"           : n_obs
        }

    def print_result(self, r):
        print("=" * 55)
        print("      DEFLATED SHARPE RATIO (DSR)")
        print("=" * 55)
        print(f"Observed Sharpe   : {r['sharpe_obs']:.3f}")
        print(f"Benchmark (luck)  : {r['sharpe_benchmark']:.3f}")
        print(f"Strategies tested : {r['n_trials']}")
        print(f"Observations      : {r['n_obs']}")
        print(f"DSR (z-score)     : {r['dsr']:.3f}")
        print(f"P-value           : {r['p_value']:.3f}")
        print(f"Verdict           : {r['verdict']}")
        print("=" * 55)


# =============================================================================
# PART 2 — PROBABILITY OF BACKTEST OVERFITTING (PBO)
# =============================================================================

class ProbabilityOfOverfitting:
    """
    Probability of Backtest Overfitting (PBO)
    — Lopez de Prado & Bailey (2014)

    For every possible train/test split combination:
    1. Find best strategy in-sample
    2. Check if it beats median OOS

    PBO < 0.10 → LOW overfitting
    PBO < 0.30 → MODERATE
    PBO < 0.50 → HIGH
    PBO > 0.50 → VERY HIGH
    """

    def __init__(self, n_splits=8):
        self.n_splits = n_splits
        self.pbo      = None
        self.sr_pairs = []

    def _sharpe(self, returns):
        if len(returns) < 2:
            return 0.0
        mean = returns.mean() * 252
        std  = returns.std()  * np.sqrt(252)
        return mean / std if std > 0 else 0.0

    def compute(self, signals_matrix, returns):
        n_obs      = len(returns)
        block_size = max(1, n_obs // self.n_splits)

        blocks = []
        for i in range(self.n_splits):
            start = i * block_size
            end   = start + block_size \
                    if i < self.n_splits - 1 else n_obs
            blocks.append(list(range(start, end)))

        n_train    = self.n_splits // 2
        all_combos = list(combinations(
            range(self.n_splits), n_train
        ))

        print(f"PBO Computation:")
        print(f"Strategies   : {signals_matrix.shape[1]}")
        print(f"Observations : {n_obs}")
        print(f"Splits       : {self.n_splits}")
        print(f"Combinations : {len(all_combos)}")
        print("-" * 40)

        overfit_count = 0
        self.sr_pairs = []

        for combo in all_combos:
            train_idx = []
            for b in combo:
                train_idx.extend(blocks[b])
            test_idx = []
            for b in range(self.n_splits):
                if b not in combo:
                    test_idx.extend(blocks[b])

            train_idx = sorted(train_idx)
            test_idx  = sorted(test_idx)
            ret_train = returns.iloc[train_idx]
            ret_test  = returns.iloc[test_idx]

            is_sharpes  = []
            oos_sharpes = []

            for col in signals_matrix.columns:
                sig    = signals_matrix[col]
                sig_tr = sig.iloc[train_idx]
                sig_te = sig.iloc[test_idx]

                ctr = sig_tr.index.intersection(ret_train.index)
                cte = sig_te.index.intersection(ret_test.index)

                if len(ctr) < 5 or len(cte) < 5:
                    is_sharpes.append(0.0)
                    oos_sharpes.append(0.0)
                    continue

                is_sharpes.append(self._sharpe(
                    sig_tr.loc[ctr] * ret_train.loc[ctr]
                ))
                oos_sharpes.append(self._sharpe(
                    sig_te.loc[cte] * ret_test.loc[cte]
                ))

            if not is_sharpes:
                continue

            best_idx    = int(np.argmax(is_sharpes))
            best_oos_sr = oos_sharpes[best_idx]
            median_oos  = np.median(oos_sharpes)

            if best_oos_sr < median_oos:
                overfit_count += 1

            self.sr_pairs.append({
                "best_is_sr" : is_sharpes[best_idx],
                "best_oos_sr": best_oos_sr,
                "median_oos" : median_oos,
                "overfit"    : best_oos_sr < median_oos
            })

        n_combos = len(self.sr_pairs)
        self.pbo = overfit_count / n_combos \
                   if n_combos > 0 else 0.5

        verdict = "LOW OVERFITTING"  if self.pbo < 0.10 else \
                  "MODERATE"         if self.pbo < 0.30 else \
                  "HIGH OVERFITTING" if self.pbo < 0.50 else \
                  "VERY HIGH"

        result = {
            "pbo"          : self.pbo,
            "verdict"      : verdict,
            "overfit_count": overfit_count,
            "total_combos" : n_combos,
            "n_strategies" : signals_matrix.shape[1]
        }
        print(f"PBO      : {self.pbo:.3f}")
        print(f"Verdict  : {verdict}")
        return result


# =============================================================================
# PART 3 — DIAGNOSTICS
# =============================================================================

class OverfittingDiagnostics:
    """
    1. Haircut Sharpe      : full-period OOS / IS ratio
    2. Parameter Stability : CV across nearby params (±5 days)
    3. Walk-Forward Consistency : positive fold rate
    """

    def haircut_sharpe(self, is_sharpe, oos_sharpe):
        """
        Haircut = OOS / IS (full period Sharpes).
        Negative haircut = difficult OOS period, not necessarily overfit.
        """
        if abs(is_sharpe) < 1e-6:
            return {
                "haircut"   : 0.0,
                "verdict"   : "N/A (IS Sharpe near zero)",
                "is_sharpe" : is_sharpe,
                "oos_sharpe": oos_sharpe
            }
        hc = oos_sharpe / is_sharpe

        if hc < 0:
            verdict = "NEGATIVE — OOS PERIOD DIFFICULT"
        elif hc > 0.75:
            verdict = "EXCELLENT"
        elif hc > 0.50:
            verdict = "ACCEPTABLE"
        elif hc > 0.25:
            verdict = "MODERATE HAIRCUT"
        else:
            verdict = "SEVERE OVERFITTING"

        return {
            "haircut"   : hc,
            "verdict"   : verdict,
            "is_sharpe" : is_sharpe,
            "oos_sharpe": oos_sharpe
        }

    def parameter_stability(self, close, returns,
                             center_fast=20,
                             center_slow=60,
                             perturbation=5):
        """
        Sharpe stability across nearby parameters (±5 days).
        Low CV  → STABLE   (robust)
        High CV → UNSTABLE (fragile)

        Note: CV is unreliable when mean Sharpe is near zero.
        Use absolute std as secondary metric.
        """
        fast_vals = [
            max(2, center_fast - perturbation),
            center_fast,
            center_fast + perturbation
        ]
        slow_vals = [
            max(center_fast + 5,
                center_slow - perturbation),
            center_slow,
            center_slow + perturbation
        ]

        results = []
        for fast in fast_vals:
            for slow in slow_vals:
                if fast >= slow:
                    continue
                strat  = TrendFollowing(fast=fast, slow=slow)
                sig    = strat.predict_sized(close)
                common = sig.index.intersection(returns.index)
                bt     = BacktestEngine(
                    sig.loc[common], returns.loc[common]
                )
                bt.run()
                results.append({
                    "fast"  : fast,
                    "slow"  : slow,
                    "sharpe": bt.sharpe_ratio(),
                    "max_dd": bt.max_drawdown(),
                })

        df          = pd.DataFrame(results)
        sharpe_std  = df["sharpe"].std()
        sharpe_mean = df["sharpe"].mean()

        # Use absolute std instead of CV when mean is near zero
        if abs(sharpe_mean) > 0.10:
            cv = sharpe_std / abs(sharpe_mean)
        else:
            # Mean near zero — use std directly as measure
            cv = sharpe_std * 5

        verdict = "STABLE"                         if cv < 0.20 else \
                  "MODERATE STABILITY"             if cv < 0.50 else \
                  "UNSTABLE — possible overfitting"

        return {
            "results"    : df,
            "sharpe_mean": sharpe_mean,
            "sharpe_std" : sharpe_std,
            "cv"         : cv,
            "verdict"    : verdict
        }

    def walk_forward_consistency(self, fold_results):
        """
        OOS Sharpe consistency across folds.
        Adjusted thresholds for commodity futures.

        > 60% positive AND CV < 0.5 → CONSISTENT
        > 45% positive              → MIXED (acceptable)
        < 45% positive              → INCONSISTENT
        """
        if not fold_results:
            return {}

        df          = pd.DataFrame(fold_results)
        oos         = df["oos_score"].values
        pos_folds   = (oos > 0).mean()
        sharpe_mean = oos.mean()
        sharpe_std  = oos.std()
        cv          = sharpe_std / abs(sharpe_mean) \
                      if abs(sharpe_mean) > 1e-6 else 999.0

        verdict = "CONSISTENT" \
                  if pos_folds > 0.60 and cv < 0.5 else \
                  "MIXED — acceptable for commodities" \
                  if pos_folds > 0.45 else \
                  "INCONSISTENT — check strategy"

        return {
            "positive_fold_rate": pos_folds,
            "oos_sharpe_mean"   : sharpe_mean,
            "oos_sharpe_std"    : sharpe_std,
            "coefficient_of_var": cv,
            "n_folds"           : len(oos),
            "verdict"           : verdict
        }


# =============================================================================
# PART 4 — FULL REPORT
# =============================================================================

class OverfittingReport:
    """
    Unified overfitting report — 5 tests.

    n_trials=1  → fixed strategy (MA(20/60), no search)
    n_trials=20 → best of 20 combinations
    """

    def __init__(self, n_trials=1, n_splits=8):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.results  = {}

    def run(self, close, returns,
            strategy_signals,
            full_is_sharpe,
            full_oos_sharpe,
            fold_results=None,
            fast=20, slow=60):

        print("\n" + "="*60)
        print("     COMPREHENSIVE OVERFITTING ANALYSIS")
        print(f"     Strategy: MA({fast}/{slow}) | "
              f"n_trials={self.n_trials}")
        print("="*60)

        diag = OverfittingDiagnostics()

        # 1. DSR
        print("\n[ 1 / 5 ]  Deflated Sharpe Ratio...")
        dsr_calc   = DeflatedSharpeRatio(
            n_trials = self.n_trials,
            skewness = float(returns.skew()),
            kurtosis = float(returns.kurtosis() + 3)
        )
        dsr_result = dsr_calc.compute(
            full_oos_sharpe, len(returns)
        )
        dsr_calc.print_result(dsr_result)
        self.results["dsr"] = dsr_result

        # 2. Haircut
        print("\n[ 2 / 5 ]  Haircut Sharpe...")
        hc = diag.haircut_sharpe(full_is_sharpe, full_oos_sharpe)
        print(f"IS Sharpe  (full) : {hc['is_sharpe']:.3f}")
        print(f"OOS Sharpe (full) : {hc['oos_sharpe']:.3f}")
        print(f"Haircut           : {hc['haircut']:.3f}")
        print(f"Verdict           : {hc['verdict']}")
        self.results["haircut"] = hc

        # 3. Parameter Stability
        print("\n[ 3 / 5 ]  Parameter Stability (±5 days)...")
        stab = diag.parameter_stability(
            close, returns,
            center_fast  = fast,
            center_slow  = slow,
            perturbation = 5
        )
        print(f"Sharpe mean : {stab['sharpe_mean']:.3f}")
        print(f"Sharpe std  : {stab['sharpe_std']:.3f}")
        print(f"CV          : {stab['cv']:.3f}")
        print(f"Verdict     : {stab['verdict']}")
        self.results["stability"] = stab

        # 4. Walk-Forward Consistency
        print("\n[ 4 / 5 ]  Walk-Forward Consistency...")
        if fold_results:
            wf = diag.walk_forward_consistency(fold_results)
            print(f"Positive folds  : "
                  f"{wf['positive_fold_rate']:.1%}")
            print(f"OOS mean Sharpe : "
                  f"{wf['oos_sharpe_mean']:.3f}")
            print(f"OOS std Sharpe  : "
                  f"{wf['oos_sharpe_std']:.3f}")
            print(f"CV              : "
                  f"{wf['coefficient_of_var']:.3f}")
            print(f"Verdict         : {wf['verdict']}")
            self.results["walk_forward"] = wf
        else:
            print("No fold results — skipping.")

        # 5. PBO
        print("\n[ 5 / 5 ]  Probability of Backtest Overfitting...")
        fast_range = [
            max(2, fast - 10), max(2, fast - 5),
            fast, fast + 5, fast + 10
        ]
        slow_range = [
            max(fast + 5, slow - 10),
            max(fast + 5, slow - 5),
            slow, slow + 5, slow + 10
        ]

        sig_matrix = {}
        for f, s in product(fast_range, slow_range):
            if f >= s:
                continue
            strat  = TrendFollowing(fast=f, slow=s)
            sig    = strat.predict_sized(close)
            common = sig.index.intersection(returns.index)
            sig_matrix[f"MA({f}/{s})"] = sig.loc[common]

        sig_df  = pd.DataFrame(sig_matrix)
        ret_aln = returns.reindex(sig_df.index).dropna()
        sig_df  = sig_df.reindex(ret_aln.index)

        pbo_calc   = ProbabilityOfOverfitting(
            n_splits=self.n_splits
        )
        pbo_result = pbo_calc.compute(sig_df, ret_aln)
        self.results["pbo"] = pbo_result

        self._print_summary()
        return self.results, pbo_calc, stab

    def _print_summary(self):
        print("\n" + "="*65)
        print("          OVERFITTING ANALYSIS SUMMARY")
        print("="*65)

        checks = []

        dsr     = self.results.get("dsr", {})
        verdict = dsr.get("verdict", "N/A")
        status  = "PASS" if "GENUINE" in verdict else \
                  "WARN" if "BORDER"  in verdict else "FAIL"
        checks.append(("Deflated Sharpe (DSR)",
                        f"p={dsr.get('p_value',0):.3f}",
                        verdict, status))

        hc      = self.results.get("haircut", {})
        verdict = hc.get("verdict", "N/A")
        status  = "PASS" if "EXCEL"    in verdict or \
                            "ACCEP"    in verdict else \
                  "WARN" if "MODER"    in verdict or \
                            "NEGATIVE" in verdict else "FAIL"
        checks.append(("Haircut Sharpe",
                        f"{hc.get('haircut',0):.3f}",
                        verdict, status))

        st      = self.results.get("stability", {})
        verdict = st.get("verdict", "N/A")
        status  = "PASS" if verdict == "STABLE" else \
                  "WARN" if "MODERATE" in verdict else "FAIL"
        checks.append(("Parameter Stability",
                        f"CV={st.get('cv',0):.3f}",
                        verdict, status))

        wf = self.results.get("walk_forward", {})
        if wf:
            verdict = wf.get("verdict", "N/A")
            status  = "PASS" if "CONSISTENT" == verdict else \
                      "WARN" if "MIXED"       in verdict else "FAIL"
            checks.append(("Walk-Forward Consistency",
                            f"{wf.get('positive_fold_rate',0):.0%} pos",
                            verdict, status))

        pbo     = self.results.get("pbo", {})
        verdict = pbo.get("verdict", "N/A")
        pbo_val = pbo.get("pbo", 0)
        status  = "PASS" if pbo_val < 0.10 else \
                  "WARN" if pbo_val < 0.30 else "FAIL"
        checks.append(("Prob. Overfitting (PBO)",
                        f"{pbo_val:.3f}",
                        verdict, status))

        print(f"{'Test':<30} {'Value':<12} "
              f"{'Verdict':<35} Status")
        print("-" * 85)
        for name, value, verdict, status in checks:
            icon = "✓" if status == "PASS" else \
                   "~" if status == "WARN" else "✗"
            print(f"{icon} {name:<28} {value:<12} "
                  f"{verdict:<35} {status}")

        passes = sum(1 for _,_,_,s in checks if s == "PASS")
        warns  = sum(1 for _,_,_,s in checks if s == "WARN")
        fails  = sum(1 for _,_,_,s in checks if s == "FAIL")
        print("-" * 85)
        print(f"Score: {passes} PASS  {warns} WARN  "
              f"{fails} FAIL  ({passes}/{len(checks)} passed)")

        if passes + warns >= 4:
            print("OVERALL: LOW-MODERATE OVERFITTING RISK")
        elif passes + warns >= 2:
            print("OVERALL: MODERATE RISK — use with caution")
        else:
            print("OVERALL: HIGH RISK — review strategy")
        print("="*65)

    def plot(self, pbo_calc, stab, save_path=None):
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(
            "Overfitting Analysis Dashboard",
            fontsize=14, fontweight="bold"
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45)

        if pbo_calc.sr_pairs:
            df_pbo = pd.DataFrame(pbo_calc.sr_pairs)
            ax1    = fig.add_subplot(gs[0, 0])
            colors = ["red" if o else "green"
                      for o in df_pbo["overfit"]]
            ax1.scatter(df_pbo["best_is_sr"],
                        df_pbo["best_oos_sr"],
                        c=colors, alpha=0.5, s=20)
            lim = max(
                abs(df_pbo["best_is_sr"]).max(),
                abs(df_pbo["best_oos_sr"]).max()
            ) * 1.1
            ax1.plot([-lim, lim], [-lim, lim],
                     "b--", linewidth=1, alpha=0.5)
            ax1.axhline(0, color="black", linewidth=0.5)
            ax1.axvline(0, color="black", linewidth=0.5)
            ax1.set_title(
                f"PBO = {pbo_calc.pbo:.3f}  "
                f"({self.results['pbo']['verdict']})"
            )
            ax1.set_xlabel("Best IS Sharpe")
            ax1.set_ylabel("Best OOS Sharpe")
            ax1.grid(True, alpha=0.3)

        if pbo_calc.sr_pairs:
            df_pbo = pd.DataFrame(pbo_calc.sr_pairs)
            ax2    = fig.add_subplot(gs[0, 1])
            oos    = df_pbo["best_oos_sr"].values
            ax2.hist(oos, bins=25, color="steelblue",
                     edgecolor="white", alpha=0.8)
            ax2.axvline(0, color="red", linestyle="--",
                        linewidth=1.5)
            ax2.axvline(np.mean(oos), color="green",
                        linestyle="--", linewidth=1.5,
                        label=f"Mean: {np.mean(oos):.3f}")
            ax2.set_title("OOS Sharpe Distribution")
            ax2.set_xlabel("OOS Sharpe")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

        if stab and "results" in stab:
            df_st = stab["results"]
            ax3   = fig.add_subplot(gs[1, 0])
            try:
                pivot = df_st.pivot(
                    index="fast", columns="slow",
                    values="sharpe"
                )
                im = ax3.imshow(
                    pivot.values, cmap="RdYlGn",
                    aspect="auto", vmin=-0.5, vmax=1.5
                )
                plt.colorbar(im, ax=ax3)
                ax3.set_xticks(range(len(pivot.columns)))
                ax3.set_xticklabels(
                    pivot.columns, fontsize=7
                )
                ax3.set_yticks(range(len(pivot.index)))
                ax3.set_yticklabels(pivot.index, fontsize=7)
                ax3.set_title(
                    f"Parameter Stability "
                    f"(CV={stab['cv']:.2f} — "
                    f"{stab['verdict']})"
                )
                ax3.set_xlabel("Slow MA")
                ax3.set_ylabel("Fast MA")
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        v = pivot.values[i, j]
                        if not np.isnan(v):
                            ax3.text(
                                j, i, f"{v:.2f}",
                                ha="center", va="center",
                                fontsize=7
                            )
            except Exception:
                ax3.set_title("Parameter Stability (error)")

        if stab and "results" in stab:
            df_st  = stab["results"]
            ax4    = fig.add_subplot(gs[1, 1])
            labels = [f"MA({r['fast']}/{r['slow']})"
                      for _, r in df_st.iterrows()]
            colors = ["green" if s > 0 else "red"
                      for s in df_st["sharpe"]]
            ax4.bar(range(len(df_st)), df_st["sharpe"],
                    color=colors, alpha=0.8)
            ax4.axhline(0, color="black", linewidth=0.8)
            ax4.axhline(
                stab["sharpe_mean"],
                color="blue", linestyle="--",
                linewidth=1.5,
                label=f"Mean: {stab['sharpe_mean']:.2f}"
            )
            ax4.set_xticks(range(len(df_st)))
            ax4.set_xticklabels(
                labels, rotation=45,
                ha="right", fontsize=7
            )
            ax4.set_title("Sharpe per Nearby Parameter")
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3, axis="y")

        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")
        rows = []
        dsr  = self.results.get("dsr",  {})
        hc   = self.results.get("haircut", {})
        st   = self.results.get("stability", {})
        wf   = self.results.get("walk_forward", {})
        pbo  = self.results.get("pbo",  {})

        rows.append(["Deflated Sharpe (DSR)",
                     f"p={dsr.get('p_value',0):.3f}",
                     dsr.get("verdict","N/A")])
        rows.append(["Haircut Sharpe",
                     f"{hc.get('haircut',0):.3f}",
                     hc.get("verdict","N/A")])
        rows.append(["Parameter Stability",
                     f"CV={st.get('cv',0):.3f}",
                     st.get("verdict","N/A")])
        if wf:
            rows.append([
                "Walk-Forward Consistency",
                f"{wf.get('positive_fold_rate',0):.0%} positive",
                wf.get("verdict","N/A")
            ])
        rows.append(["Prob. of Overfitting (PBO)",
                     f"{pbo.get('pbo',0):.3f}",
                     pbo.get("verdict","N/A")])

        t = ax5.table(
            cellText  = rows,
            colLabels = ["Test", "Value", "Verdict"],
            cellLoc   = "center",
            loc       = "center",
            colWidths = [0.32, 0.22, 0.46]
        )
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.8)

        color_map = {
            "GENUINE" : "#c8f7c5",
            "EXCEL"   : "#c8f7c5",
            "ACCEP"   : "#c8f7c5",
            "STABLE"  : "#c8f7c5",
            "CONSIST" : "#c8f7c5",
            "LOW OVE" : "#c8f7c5",
            "BORDER"  : "#ffeaa7",
            "MIXED"   : "#ffeaa7",
            "MODER"   : "#ffeaa7",
            "NEGATIV" : "#ffeaa7",
            "LIKELY"  : "#fab1a0",
            "HIGH"    : "#fab1a0",
            "SEVERE"  : "#fab1a0",
            "INCONS"  : "#fab1a0",
            "VERY HI" : "#fab1a0",
            "UNSTAB"  : "#fab1a0",
        }
        for i, row in enumerate(rows):
            verdict = row[2].upper()
            color   = "#ffffff"
            for key, col in color_map.items():
                if key in verdict:
                    color = col
                    break
            for j in range(3):
                t[i+1, j].set_facecolor(color)

        ax5.set_title(
            "Overfitting Test Scorecard",
            fontweight="bold", pad=10
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150,
                        bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()


# =============================================================================
# MAIN — Test MULTI-ASSET PORTFOLIO (our actual best strategy)
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

    # 2. Build portfolio signals
    signals = pd.DataFrame(index=close.index)
    for ticker in tickers:
        strat = TrendFollowing(fast=20, slow=60)
        signals[ticker] = strat.predict(close[ticker])

    portfolio = Portfolio(target_vol=0.15)

    # 3. Full IS backtest (2010-2024)
    print("\nComputing full IS Sharpe (2010-2024)...")
    res_full, _ = portfolio.run_backtest(signals, returns)
    full_is_sharpe = portfolio.sharpe(
        res_full["strategy_return"]
    )
    print(f"Full IS Sharpe: {full_is_sharpe:.3f}")

    # 4. Full OOS Sharpe (last 30% = ~2019-2024)
    split           = int(len(res_full) * 0.70)
    ret_oos         = res_full["strategy_return"].iloc[split:]
    full_oos_sharpe = portfolio.sharpe(ret_oos)
    print(f"Full OOS Sharpe (last 30%): {full_oos_sharpe:.3f}")

    # 5. Walk-forward fold results
    print("\nBuilding walk-forward fold results...")
    train_size   = pd.DateOffset(years=3)
    test_size    = pd.DateOffset(months=6)
    test_start   = close.index[0] + train_size
    end_date     = close.index[-1]
    fold_results = []
    oos_signals  = []

    while test_start < end_date:
        test_end  = min(test_start + test_size, end_date)

        sig_train = signals[signals.index <  test_start]
        sig_test  = signals[
            (signals.index >= test_start) &
            (signals.index <  test_end)
        ]
        ret_train = returns[returns.index <  test_start]
        ret_test  = returns[
            (returns.index >= test_start) &
            (returns.index <  test_end)
        ]

        if len(sig_train) < 100 or len(sig_test) < 10:
            test_start = test_end
            continue

        res_is,  _ = portfolio.run_backtest(
            sig_train, ret_train
        )
        res_oos, _ = portfolio.run_backtest(
            sig_test, ret_test
        )

        is_sh  = portfolio.sharpe(res_is["strategy_return"])
        oos_sh = portfolio.sharpe(res_oos["strategy_return"])

        fold_results.append({
            "fold"       : len(fold_results),
            "test_start" : test_start,
            "oos_score"  : oos_sh,
            "train_score": is_sh
        })
        oos_signals.append(sig_test.mean(axis=1))
        test_start = test_end

    sig_combined = pd.concat(oos_signals)
    avg_oos = np.mean([f["oos_score"]   for f in fold_results])
    avg_is  = np.mean([f["train_score"] for f in fold_results])
    print(f"Walk-forward avg IS  Sharpe: {avg_is:.3f}")
    print(f"Walk-forward avg OOS Sharpe: {avg_oos:.3f}")
    print(f"Folds: {len(fold_results)}")

    # 6. Use CL as representative for stability/PBO tests
    cl_close   = close["CL=F"] if "CL=F" in close.columns \
                 else close.iloc[:, 0]
    cl_returns = returns[cl_close.name].reindex(
        cl_close.index
    ).dropna()
    cl_close   = cl_close.reindex(cl_returns.index)

    # 7. Run full overfitting report
    # n_trials=1 — MA(20/60) was NOT searched, it was fixed
    report = OverfittingReport(n_trials=1, n_splits=8)
    results, pbo_calc, stab = report.run(
        close           = cl_close,
        returns         = cl_returns,
        strategy_signals= sig_combined,
        full_is_sharpe  = full_is_sharpe,
        full_oos_sharpe = full_oos_sharpe,
        fold_results    = fold_results,
        fast            = 20,
        slow            = 60
    )

    # 8. Plot dashboard
    report.plot(
        pbo_calc  = pbo_calc,
        stab      = stab,
        save_path = "evaluation/overfitting_analysis.png"
    )
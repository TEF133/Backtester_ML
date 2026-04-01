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

    DSR > 0 (p > 0.95) → genuine edge
    DSR < 0 (p < 0.50) → likely lucky
    """

    def __init__(self, n_trials, skewness=0.0, kurtosis=3.0):
        self.n_trials = n_trials
        self.skewness = skewness
        self.kurtosis = kurtosis

    def expected_max_sharpe(self, n_obs):
        if self.n_trials <= 1:
            return 0.0
        e_max = (
            (1 - 0.5772) / np.log(self.n_trials) ** 0.5 +
            1 / (2 * np.log(self.n_trials)) ** 0.5
        )
        return e_max

    def compute(self, sharpe_obs, n_obs, sharpe_benchmark=None):
        if sharpe_benchmark is None:
            sharpe_benchmark = self.expected_max_sharpe(n_obs)

        sr_daily = sharpe_obs / np.sqrt(252)
        sr_bench = sharpe_benchmark / np.sqrt(252)

        var_sr = (
            1 +
            0.5 * sr_daily**2 -
            self.skewness * sr_daily +
            (self.kurtosis - 3) / 4 * sr_daily**2
        ) / max(n_obs, 1)

        std_sr  = np.sqrt(var_sr)
        dsr     = (sr_daily - sr_bench) / (std_sr + 1e-10)
        p_value = norm.cdf(dsr)

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

    def print_result(self, result):
        print("=" * 55)
        print("      DEFLATED SHARPE RATIO (DSR)")
        print("=" * 55)
        print(f"Observed Sharpe   : {result['sharpe_obs']:.3f}")
        print(f"Benchmark (luck)  : {result['sharpe_benchmark']:.3f}")
        print(f"Strategies tested : {result['n_trials']}")
        print(f"Observations      : {result['n_obs']}")
        print(f"DSR (z-score)     : {result['dsr']:.3f}")
        print(f"P-value           : {result['p_value']:.3f}")
        print(f"Verdict           : {result['verdict']}")
        print("=" * 55)


# =============================================================================
# PART 2 — PROBABILITY OF BACKTEST OVERFITTING (PBO)
# =============================================================================

class ProbabilityOfOverfitting:
    """
    Probability of Backtest Overfitting (PBO)
    — Lopez de Prado & Bailey (2014)

    Uses Combinatorial Cross-Validation:
    For every possible train/test split combination:
    1. Find best strategy in-sample
    2. Check if it beats median OOS

    PBO < 0.10 → low overfitting
    PBO > 0.50 → very likely overfit
    """

    def __init__(self, n_splits=8):
        self.n_splits = n_splits
        self.pbo      = None
        self.sr_pairs = []

    def _sharpe(self, returns):
        if len(returns) < 2:
            return 0.0
        mean = returns.mean() * 252
        std  = returns.std() * np.sqrt(252)
        return mean / std if std > 0 else 0.0

    def compute(self, signals_matrix, returns):
        n_obs      = len(returns)
        block_size = n_obs // self.n_splits

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
        print(f"Strategies      : {signals_matrix.shape[1]}")
        print(f"Observations    : {n_obs}")
        print(f"Splits          : {self.n_splits}")
        print(f"Combinations    : {len(all_combos)}")
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
                sig      = signals_matrix[col]
                sig_tr   = sig.iloc[train_idx]
                sig_te   = sig.iloc[test_idx]

                common_tr = sig_tr.index.intersection(
                    ret_train.index
                )
                common_te = sig_te.index.intersection(
                    ret_test.index
                )

                if len(common_tr) < 5 or len(common_te) < 5:
                    is_sharpes.append(0.0)
                    oos_sharpes.append(0.0)
                    continue

                ret_is  = sig_tr.loc[common_tr] * \
                          ret_train.loc[common_tr]
                ret_oos = sig_te.loc[common_te] * \
                          ret_test.loc[common_te]

                is_sharpes.append(self._sharpe(ret_is))
                oos_sharpes.append(self._sharpe(ret_oos))

            if not is_sharpes:
                continue

            best_is_idx = int(np.argmax(is_sharpes))
            best_oos_sr = oos_sharpes[best_is_idx]
            median_oos  = np.median(oos_sharpes)

            if best_oos_sr < median_oos:
                overfit_count += 1

            self.sr_pairs.append({
                "best_is_sr" : is_sharpes[best_is_idx],
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
                  "VERY HIGH — AVOID"

        result = {
            "pbo"          : self.pbo,
            "verdict"      : verdict,
            "overfit_count": overfit_count,
            "total_combos" : n_combos,
            "n_strategies" : signals_matrix.shape[1]
        }

        print(f"PBO             : {self.pbo:.3f}")
        print(f"Verdict         : {verdict}")
        return result


# =============================================================================
# PART 3 — DIAGNOSTICS
# =============================================================================

class OverfittingDiagnostics:
    """
    1. Haircut Sharpe  : OOS / IS ratio. Target > 0.50
    2. Parameter Stability : CV of Sharpe across nearby params
    3. Walk-Forward Consistency : positive fold rate
    """

    def haircut_sharpe(self, is_sharpe, oos_sharpe):
        if is_sharpe == 0:
            return {"haircut": 0.0, "verdict": "N/A",
                    "is_sharpe": is_sharpe,
                    "oos_sharpe": oos_sharpe}
        hc      = oos_sharpe / is_sharpe
        verdict = "EXCELLENT"         if hc > 0.75 else \
                  "ACCEPTABLE"        if hc > 0.50 else \
                  "MODERATE HAIRCUT"  if hc > 0.25 else \
                  "SEVERE OVERFITTING"
        return {
            "haircut"   : hc,
            "verdict"   : verdict,
            "is_sharpe" : is_sharpe,
            "oos_sharpe": oos_sharpe
        }

    def parameter_stability(self, close, returns,
                             center_fast=20, center_slow=60,
                             perturbation=10):
        """
        Tests Sharpe stability across nearby parameters.
        Robust strategy: similar Sharpe for MA(10/50), MA(20/60), MA(30/70)
        Overfit strategy: performance collapses with small param change
        """
        fast_range = [
            max(2, center_fast - perturbation),
            center_fast,
            center_fast + perturbation
        ]
        slow_range = [
            max(center_fast + 5,
                center_slow - perturbation),
            center_slow,
            center_slow + perturbation
        ]

        results = []
        for fast in fast_range:
            for slow in slow_range:
                if fast >= slow:
                    continue
                strat  = TrendFollowing(fast=fast, slow=slow)
                sig    = strat.predict_sized(close)
                common = sig.index.intersection(returns.index)
                bt     = BacktestEngine(
                    sig.loc[common], returns.loc[common]
                )
                res    = bt.run()
                r      = res["strategy_return"]
                results.append({
                    "fast"  : fast,
                    "slow"  : slow,
                    "sharpe": bt.sharpe_ratio(),
                    "max_dd": bt.max_drawdown(),
                    "return": (1 + r).prod() - 1
                })

        df          = pd.DataFrame(results)
        sharpe_std  = df["sharpe"].std()
        sharpe_mean = df["sharpe"].mean()
        cv          = sharpe_std / abs(sharpe_mean) \
                      if sharpe_mean != 0 else 999

        # Correct verdict thresholds
        verdict = "STABLE"                       if cv < 0.15 else \
                  "MODERATE STABILITY"           if cv < 0.30 else \
                  "UNSTABLE — possible overfitting"

        return {
            "results"    : df,
            "sharpe_mean": sharpe_mean,
            "sharpe_std" : sharpe_std,
            "cv"         : cv,
            "verdict"    : verdict
        }

    def walk_forward_consistency(self, fold_results):
        if not fold_results:
            return {}

        df          = pd.DataFrame(fold_results)
        oos_sharpes = df["oos_score"].values
        pos_folds   = (oos_sharpes > 0).mean()
        sharpe_std  = oos_sharpes.std()
        sharpe_mean = oos_sharpes.mean()
        cv          = sharpe_std / abs(sharpe_mean) \
                      if sharpe_mean != 0 else 999

        verdict = "CONSISTENT"   if pos_folds > 0.6 and cv < 0.5 else \
                  "MIXED"        if pos_folds > 0.4 else \
                  "INCONSISTENT — possible overfitting"

        return {
            "positive_fold_rate": pos_folds,
            "oos_sharpe_mean"   : sharpe_mean,
            "oos_sharpe_std"    : sharpe_std,
            "coefficient_of_var": cv,
            "n_folds"           : len(oos_sharpes),
            "verdict"           : verdict
        }


# =============================================================================
# PART 4 — FULL REPORT
# =============================================================================

class OverfittingReport:
    """
    Runs all 5 overfitting tests and produces a unified report.
    """

    def __init__(self, n_trials=1, n_splits=8):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.results  = {}

    def run(self, close, returns,
            strategy_signals, oos_sharpe,
            is_sharpe, fold_results=None,
            fast=20, slow=60):

        print("\n" + "="*60)
        print("     COMPREHENSIVE OVERFITTING ANALYSIS")
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
            oos_sharpe, len(returns)
        )
        dsr_calc.print_result(dsr_result)
        self.results["dsr"] = dsr_result

        # 2. Haircut Sharpe
        print("\n[ 2 / 5 ]  Haircut Sharpe...")
        hc = diag.haircut_sharpe(is_sharpe, oos_sharpe)
        print(f"IS Sharpe  : {hc['is_sharpe']:.3f}")
        print(f"OOS Sharpe : {hc['oos_sharpe']:.3f}")
        print(f"Haircut    : {hc['haircut']:.3f}")
        print(f"Verdict    : {hc['verdict']}")
        self.results["haircut"] = hc

        # 3. Parameter Stability
        print("\n[ 3 / 5 ]  Parameter Stability...")
        stab = diag.parameter_stability(
            close, returns,
            center_fast  = fast,
            center_slow  = slow,
            perturbation = 10
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
            print(f"Positive folds  : {wf['positive_fold_rate']:.1%}")
            print(f"OOS Sharpe mean : {wf['oos_sharpe_mean']:.3f}")
            print(f"OOS Sharpe std  : {wf['oos_sharpe_std']:.3f}")
            print(f"CV              : {wf['coefficient_of_var']:.3f}")
            print(f"Verdict         : {wf['verdict']}")
            self.results["walk_forward"] = wf
        else:
            print("No fold results — skipping.")

        # 5. PBO
        print("\n[ 5 / 5 ]  Probability of Backtest Overfitting...")
        fast_range = [10, 15, 20, 25, 30]
        slow_range = [40, 50, 60, 70, 80]
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
        print("\n" + "="*60)
        print("     OVERFITTING ANALYSIS SUMMARY")
        print("="*60)

        checks = []

        # DSR
        dsr     = self.results.get("dsr", {})
        verdict = dsr.get("verdict", "N/A")
        status  = "PASS" if "GENUINE"  in verdict else \
                  "WARN" if "BORDER"   in verdict else "FAIL"
        checks.append(("Deflated Sharpe (DSR)",
                        f"p={dsr.get('p_value',0):.3f}",
                        verdict, status))

        # Haircut
        hc      = self.results.get("haircut", {})
        verdict = hc.get("verdict", "N/A")
        status  = "PASS" if "EXCEL"  in verdict or \
                            "ACCEP"  in verdict else \
                  "WARN" if "MODER"  in verdict else "FAIL"
        checks.append(("Haircut Sharpe",
                        f"{hc.get('haircut',0):.3f}",
                        verdict, status))

        # Stability — fixed verdict check
        st      = self.results.get("stability", {})
        verdict = st.get("verdict", "N/A")
        status  = "PASS" if verdict == "STABLE" else \
                  "WARN" if "MODERATE" in verdict else "FAIL"
        checks.append(("Parameter Stability",
                        f"CV={st.get('cv',0):.3f}",
                        verdict, status))

        # Walk-forward
        wf = self.results.get("walk_forward", {})
        if wf:
            verdict = wf.get("verdict", "N/A")
            status  = "PASS" if "CONSISTENT"  in verdict else \
                      "WARN" if "MIXED"        in verdict else "FAIL"
            checks.append(("Walk-Forward Consistency",
                            f"{wf.get('positive_fold_rate',0):.0%} pos folds",
                            verdict, status))

        # PBO
        pbo     = self.results.get("pbo", {})
        verdict = pbo.get("verdict", "N/A")
        pbo_val = pbo.get("pbo", 0)
        status  = "PASS" if pbo_val < 0.10 else \
                  "WARN" if pbo_val < 0.30 else "FAIL"
        checks.append(("Prob. of Overfitting (PBO)",
                        f"PBO={pbo_val:.3f}",
                        verdict, status))

        print(f"{'Test':<30} {'Value':<20} "
              f"{'Verdict':<30} {'Status'}")
        print("-" * 85)
        for name, value, verdict, status in checks:
            icon = "✓" if status == "PASS" else \
                   "~" if status == "WARN" else "✗"
            print(f"{icon} {name:<28} {value:<20} "
                  f"{verdict:<30} {status}")

        passes = sum(1 for _,_,_,s in checks if s == "PASS")
        warns  = sum(1 for _,_,_,s in checks if s == "WARN")
        fails  = sum(1 for _,_,_,s in checks if s == "FAIL")
        print("-" * 85)
        print(f"Score: {passes} PASS  {warns} WARN  "
              f"{fails} FAIL  ({passes}/{len(checks)} passed)")

        if passes >= 4:
            print("OVERALL: LOW OVERFITTING RISK — strategy appears genuine")
        elif passes >= 2:
            print("OVERALL: MODERATE RISK — use with caution")
        else:
            print("OVERALL: HIGH OVERFITTING RISK — do not trade live")
        print("="*60)

    def plot(self, pbo_calc, stab, save_path=None):
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(
            "Overfitting Analysis Dashboard",
            fontsize=14, fontweight="bold"
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45)

        # PBO scatter
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

        # PBO OOS distribution
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

        # Parameter stability heatmap
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
                ax3.set_yticklabels(
                    pivot.index, fontsize=7
                )
                ax3.set_title(
                    f"Parameter Stability (CV={stab['cv']:.2f})"
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

        # Sharpe bar per param combo
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
                color="blue", linestyle="--", linewidth=1.5,
                label=f"Mean: {stab['sharpe_mean']:.2f}"
            )
            ax4.set_xticks(range(len(df_st)))
            ax4.set_xticklabels(
                labels, rotation=45, ha="right", fontsize=7
            )
            ax4.set_title("Sharpe per Parameter")
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3, axis="y")

        # Scorecard table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")
        rows = []
        dsr  = self.results.get("dsr", {})
        hc   = self.results.get("haircut", {})
        st   = self.results.get("stability", {})
        wf   = self.results.get("walk_forward", {})
        pbo  = self.results.get("pbo", {})

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
            colWidths = [0.35, 0.25, 0.4]
        )
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.8)

        colors_map = {
            "GENUINE" : "#c8f7c5",
            "EXCEL"   : "#c8f7c5",
            "ACCEP"   : "#c8f7c5",
            "STABLE"  : "#c8f7c5",
            "CONSIST" : "#c8f7c5",
            "LOW OVE" : "#c8f7c5",
            "BORDER"  : "#ffeaa7",
            "MIXED"   : "#ffeaa7",
            "MODER"   : "#ffeaa7",
            "LIKELY"  : "#fab1a0",
            "HIGH"    : "#fab1a0",
            "SEVERE"  : "#fab1a0",
            "INCONS"  : "#fab1a0",
            "VERY HI" : "#fab1a0",
            "UNSTAB"  : "#fab1a0",
        }
        for i, row in enumerate(rows):
            verdict = row[2]
            color   = "#ffffff"
            for key, col in colors_map.items():
                if key in verdict.upper():
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
# MAIN — Test FIXED MA(20/60) properly with n_trials=1
# =============================================================================

if __name__ == "__main__":
    from data.data_loader import DataLoader

    # 1. Load data
    loader  = DataLoader()
    df      = loader.fetch(
        ["CL=F"], start="2010-01-01", end="2024-01-01"
    )
    close   = df["Close"].squeeze().dropna()
    close   = close[close > 5]
    returns = np.log(close / close.shift(1)).shift(-1).dropna()
    close   = close.reindex(returns.index)

    # 2. Walk-forward IS and OOS Sharpes for fixed MA(20/60)
    print("Running walk-forward for fixed MA(20/60)...")
    train_size      = pd.DateOffset(years=3)
    test_size       = pd.DateOffset(months=6)
    start_date      = close.index[0]
    end_date        = close.index[-1]
    test_start      = start_date + train_size

    all_oos_signals = []
    all_is_sharpes  = []
    all_oos_sharpes = []
    fold_results    = []

    while test_start < end_date:
        test_end    = min(test_start + test_size, end_date)
        close_train = close[close.index <  test_start]
        close_test  = close[
            (close.index >= test_start) &
            (close.index <  test_end)
        ]
        ret_train   = returns[returns.index <  test_start]
        ret_test    = returns[
            (returns.index >= test_start) &
            (returns.index <  test_end)
        ]

        if len(close_train) < 100 or len(close_test) < 10:
            test_start = test_end
            continue

        strat = TrendFollowing(fast=20, slow=60)

        # IS
        sig_is    = strat.predict_sized(close_train)
        common_is = sig_is.index.intersection(ret_train.index)
        bt_is     = BacktestEngine(
            sig_is.loc[common_is], ret_train.loc[common_is]
        )
        bt_is.run()
        is_sh = bt_is.sharpe_ratio()

        # OOS
        sig_oos    = strat.predict_sized(close_test)
        common_oos = sig_oos.index.intersection(ret_test.index)
        bt_oos     = BacktestEngine(
            sig_oos.loc[common_oos], ret_test.loc[common_oos]
        )
        bt_oos.run()
        oos_sh = bt_oos.sharpe_ratio()

        all_oos_signals.append(sig_oos.loc[common_oos])
        all_is_sharpes.append(is_sh)
        all_oos_sharpes.append(oos_sh)
        fold_results.append({
            "fold"       : len(fold_results),
            "test_start" : test_start,
            "oos_score"  : oos_sh,
            "train_score": is_sh
        })

        test_start = test_end

    sig_combined = pd.concat(all_oos_signals)
    is_sharpe    = np.mean(all_is_sharpes)
    oos_sharpe   = np.mean(all_oos_sharpes)

    print(f"Avg IS  Sharpe (per fold): {is_sharpe:.3f}")
    print(f"Avg OOS Sharpe (per fold): {oos_sharpe:.3f}")
    print(f"Haircut                  : "
          f"{oos_sharpe/is_sharpe:.3f}" if is_sharpe != 0
          else "Haircut: N/A")

    # 3. Run full overfitting report
    # n_trials=1 because MA(20/60) was NOT searched
    report = OverfittingReport(n_trials=1, n_splits=8)
    results, pbo_calc, stab = report.run(
        close            = close,
        returns          = returns,
        strategy_signals = sig_combined,
        oos_sharpe       = oos_sharpe,
        is_sharpe        = is_sharpe,
        fold_results     = fold_results,
        fast             = 20,
        slow             = 60
    )

    # 4. Plot dashboard
    report.plot(
        pbo_calc  = pbo_calc,
        stab      = stab,
        save_path = "evaluation/overfitting_analysis.png"
    )
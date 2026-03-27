import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# PART 1 — TRIPLE BARRIER LABELING
# =============================================================================

class TripleBarrierLabeler:
    """
    Labels each trade using 3 barriers:
    Upper barrier  → take profit → label +1
    Lower barrier  → stop loss   → label -1
    Time barrier   → time exit   → label  0
    """

    def __init__(self,
                 upper_mult=1.5,
                 lower_mult=1.5,
                 max_days=10,
                 vol_window=20):
        self.upper_mult = upper_mult
        self.lower_mult = lower_mult
        self.max_days   = max_days
        self.vol_window = vol_window

    def _daily_vol(self, close):
        returns = np.log(close / close.shift(1))
        return returns.rolling(self.vol_window).std()

    def label(self, close):
        """
        Apply triple barrier labeling.
        Returns DataFrame with label, ret, barrier_hit, days_held,
        entry_date, exit_date.
        """
        daily_vol = self._daily_vol(close)
        labels = []

        for i in range(self.vol_window, len(close) - self.max_days):
            t0    = close.index[i]
            price = close.iloc[i]
            vol   = daily_vol.iloc[i]

            upper = price * (1 + self.upper_mult * vol)
            lower = price * (1 - self.lower_mult * vol)

            future = close.iloc[i+1 : i+1+self.max_days]

            label       = 0
            barrier_hit = "time"
            days_held   = self.max_days
            exit_date   = future.index[-1]

            for j, (date, fut_price) in enumerate(future.items()):
                if fut_price >= upper:
                    label       = 1
                    barrier_hit = "upper"
                    days_held   = j + 1
                    exit_date   = date
                    break
                elif fut_price <= lower:
                    label       = -1
                    barrier_hit = "lower"
                    days_held   = j + 1
                    exit_date   = date
                    break

            # Actual return of the trade
            exit_price = close.loc[exit_date]
            ret        = (exit_price - price) / price

            labels.append({
                "date"        : t0,
                "exit_date"   : exit_date,
                "label"       : label,
                "ret"         : ret,
                "barrier_hit" : barrier_hit,
                "days_held"   : days_held,
                "upper"       : upper,
                "lower"       : lower
            })

        return pd.DataFrame(labels).set_index("date")


# =============================================================================
# PART 2 — SIGNAL EXPANDER
# Converts trade labels into daily signals by holding for full duration
# =============================================================================

class SignalExpander:
    """
    Converts entry signals into daily position signals.

    Example:
    Entry on 2020-01-05, label=+1, days_held=8
    → Daily signal = +1 for 8 consecutive days

    This is how real trading works — you hold until your exit condition is met.
    """

    def expand(self, signals_df, close_index):
        """
        signals_df : DataFrame with index=entry_date,
                     columns=[signal, exit_date, days_held]
        close_index: full date index of price series

        Returns daily signal Series
        """
        daily = pd.Series(0.0, index=close_index)

        for entry_date, row in signals_df.iterrows():
            if entry_date not in close_index:
                continue
            signal    = row["signal"]
            exit_date = row["exit_date"]

            # Fill signal from entry to exit
            mask = (close_index >= entry_date) & (close_index <= exit_date)
            daily[mask] = signal

        return daily.rename("signal")


# =============================================================================
# PART 3 — ML STRATEGY WITH WALK-FORWARD VALIDATION
# =============================================================================

class TripleBarrierStrategy:
    """
    ML strategy trained on triple barrier labels.
    Uses expanding window walk-forward validation.
    Holds each signal for its full barrier duration.
    """

    def __init__(self,
                 model_type="random_forest",
                 train_years=3,
                 test_months=6):
        self.model_type   = model_type
        self.train_years  = train_years
        self.test_months  = test_months
        self.fold_results = []

    def _build_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )

    def walk_forward(self, features, labels, close_index):
        """
        Walk-forward validation.
        Returns daily signal Series (expanded for full hold duration).
        """
        common = features.index.intersection(labels.index)
        X = features.loc[common]
        y = labels.loc[common, "label"]
        label_meta = labels.loc[common, ["exit_date", "days_held"]]

        train_size = pd.DateOffset(years=self.train_years)
        test_size  = pd.DateOffset(months=self.test_months)
        start_date = X.index[0]
        end_date   = X.index[-1]

        all_trade_signals = []  # list of (entry, exit, signal)
        test_start = start_date + train_size
        fold = 0

        print("Walk-Forward Validation")
        print(f"Train: {self.train_years} years | Test: {self.test_months} months")
        print("=" * 60)

        while test_start < end_date:
            test_end = min(test_start + test_size, end_date)

            X_train = X[X.index <  test_start]
            y_train = y[y.index <  test_start]
            X_test  = X[(X.index >= test_start) & (X.index < test_end)]
            y_test  = y[(y.index >= test_start) & (y.index < test_end)]
            meta_test = label_meta[
                (label_meta.index >= test_start) &
                (label_meta.index <  test_end)
            ]

            if len(X_train) < 100 or len(X_test) < 10:
                test_start = test_end
                continue

            # Scale
            scaler  = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_te_sc = scaler.transform(X_test)

            # Train
            model = self._build_model()
            model.fit(X_tr_sc, y_train)

            # Predict
            y_pred = model.predict(X_te_sc)
            y_prob = model.predict_proba(X_te_sc)

            # Confidence filter — only trade when model is confident
            max_prob = y_prob.max(axis=1)
            y_pred_filtered = np.where(max_prob > 0.45, y_pred, 0)

            # Store trade signals with entry/exit dates
            for i, (entry_date, pred) in enumerate(
                zip(X_test.index, y_pred_filtered)
            ):
                if pred != 0 and entry_date in meta_test.index:
                    exit_date = meta_test.loc[entry_date, "exit_date"]
                    all_trade_signals.append({
                        "entry_date" : entry_date,
                        "exit_date"  : exit_date,
                        "signal"     : pred,
                        "days_held"  : meta_test.loc[entry_date, "days_held"]
                    })

            fold_acc = accuracy_score(y_test, y_pred)
            self.fold_results.append({
                "fold"      : fold,
                "test_start": test_start,
                "test_end"  : test_end,
                "accuracy"  : fold_acc
            })

            print(f"Fold {fold+1}: "
                  f"[{X_train.index[0].date()} → "
                  f"{X_train.index[-1].date()}] | "
                  f"Test [{test_start.date()} → {test_end.date()}] | "
                  f"Acc: {fold_acc:.1%} | "
                  f"Trades: {(y_pred_filtered != 0).sum()}")

            test_start = test_end
            fold += 1

        print("=" * 60)
        avg_acc = np.mean([f["accuracy"] for f in self.fold_results])
        print(f"Total folds     : {fold}")
        print(f"Average accuracy: {avg_acc:.1%}")
        print(f"Total trades    : {len(all_trade_signals)}")

        # Expand trade signals into daily signals
        if not all_trade_signals:
            print("WARNING: No trades generated!")
            return pd.Series(0.0, index=close_index, name="signal")

        trades_df = pd.DataFrame(all_trade_signals).set_index("entry_date")
        expander  = SignalExpander()
        daily_signals = expander.expand(trades_df, close_index)

        print(f"\nDaily signal distribution:")
        print(f"  Long  (+1): {(daily_signals > 0).sum()} days")
        print(f"  Flat   (0): {(daily_signals == 0).sum()} days")
        print(f"  Short (-1): {(daily_signals < 0).sum()} days")

        return daily_signals

    def feature_importance(self, features, labels):
        common = features.index.intersection(labels.index)
        X      = features.loc[common]
        y      = labels.loc[common, "label"]
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        model  = self._build_model()
        model.fit(X_sc, y)

        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        print("\n=== FEATURE IMPORTANCE ===")
        print(importance.round(4))
        return importance


# Quick test
if __name__ == "__main__":
    from data.data_loader import DataLoader
    from features.feature_engine import FeatureEngine

    loader = DataLoader()
    df     = loader.fetch(["CL=F"], start="2010-01-01", end="2024-01-01")
    close  = df["Close"].squeeze()

    print("Building triple barrier labels...")
    labeler = TripleBarrierLabeler(
        upper_mult=1.5,
        lower_mult=1.5,
        max_days=10
    )
    labels = labeler.label(close)
    print(f"Label distribution:\n{labels['label'].value_counts()}")
    print(f"Avg days held: {labels['days_held'].mean():.1f}")

    print("\nBuilding features...")
    engine   = FeatureEngine(df)
    features = engine.build_all()
    if isinstance(features.columns, pd.MultiIndex):
        features = features.xs("CL=F", axis=1, level=1)
    features.columns = [str(c) for c in features.columns]

    print("\nRunning walk-forward...")
    strategy = TripleBarrierStrategy(
        model_type="random_forest",
        train_years=3,
        test_months=6
    )
    signals = strategy.walk_forward(features, labels, close.index)
    strategy.feature_importance(features, labels)
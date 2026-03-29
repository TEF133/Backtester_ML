import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# PART 1 — IMPROVED TRADING ENVIRONMENT
# =============================================================================

class CommodityTradingEnv(gym.Env):
    """
    Improved trading environment with:
    - Sharpe-inspired reward shaping
    - Drawdown penalty
    - Transaction cost penalty
    - Regime awareness
    - Proper feature normalization
    """

    def __init__(self, features, returns,
                 transaction_cost=0.001,
                 reward_scaling=100.0,
                 drawdown_penalty=0.1,
                 trade_penalty=0.5):
        super().__init__()

        # Normalize features to [-1, 1] range
        self.raw_features     = features.values.astype(np.float32)
        self.features         = self._normalize(self.raw_features)
        self.returns          = returns.values.astype(np.float32)
        self.dates            = features.index
        self.transaction_cost = transaction_cost
        self.reward_scaling   = reward_scaling
        self.drawdown_penalty = drawdown_penalty
        self.trade_penalty    = trade_penalty
        self.n_features       = features.shape[1]

        # Action: 0=short, 1=flat, 2=long
        self.action_space = spaces.Discrete(3)

        # Observation: features + position + drawdown + vol
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  = np.inf,
            shape = (self.n_features + 3,),
            dtype = np.float32
        )

        self.reset()

    def _normalize(self, X):
        """Clip extreme values and normalize."""
        X = np.clip(X, -10, 10)
        mean = np.nanmean(X, axis=0)
        std  = np.nanstd(X, axis=0) + 1e-8
        return ((X - mean) / std).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step  = 0
        self.position      = 0
        self.portfolio_val = 1.0
        self.peak_val      = 1.0
        self.returns_log   = []
        self.trade_log     = []
        return self._get_observation(), {}

    def _get_observation(self):
        features = self.features[self.current_step]
        position = np.array([self.position], dtype=np.float32)

        # Current drawdown
        drawdown = np.array(
            [(self.portfolio_val - self.peak_val) / (self.peak_val + 1e-8)],
            dtype=np.float32
        )

        # Recent volatility of strategy returns
        if len(self.returns_log) > 5:
            recent_vol = np.array(
                [np.std(self.returns_log[-20:]) * np.sqrt(252)],
                dtype=np.float32
            )
        else:
            recent_vol = np.array([0.15], dtype=np.float32)

        return np.concatenate([features, position, drawdown, recent_vol])

    def step(self, action):
        new_position     = action - 1  # 0→-1, 1→0, 2→+1
        position_changed = int(new_position != self.position)
        cost             = position_changed * self.transaction_cost
        market_ret       = self.returns[self.current_step]
        strategy_ret     = new_position * market_ret - cost

        # Update portfolio
        self.portfolio_val *= (1 + strategy_ret)
        self.peak_val       = max(self.peak_val, self.portfolio_val)
        self.returns_log.append(strategy_ret)

        # Current drawdown
        drawdown = (self.portfolio_val - self.peak_val) / (
            self.peak_val + 1e-8
        )

        # REWARD SHAPING
        # Base reward: strategy return
        base_reward = strategy_ret * self.reward_scaling

        # Drawdown penalty: punish being in drawdown
        dd_penalty = self.drawdown_penalty * abs(min(drawdown, 0)) * \
                     self.reward_scaling

        # Trade penalty: punish excessive trading
        trade_penalty = self.trade_penalty * position_changed

        # Sharpe bonus: reward consistent positive returns
        if len(self.returns_log) > 20:
            recent   = np.array(self.returns_log[-20:])
            mean_ret = recent.mean() * 252
            vol_ret  = recent.std() * np.sqrt(252) + 1e-8
            sharpe_bonus = 0.1 * (mean_ret / vol_ret)
        else:
            sharpe_bonus = 0.0

        reward = float(
            base_reward - dd_penalty - trade_penalty + sharpe_bonus
        )

        self.trade_log.append({
            "step"         : self.current_step,
            "date"         : self.dates[self.current_step],
            "position"     : new_position,
            "market_ret"   : market_ret,
            "strategy_ret" : strategy_ret,
            "portfolio_val": self.portfolio_val,
            "drawdown"     : drawdown
        })

        self.position     = new_position
        self.current_step += 1

        done = self.current_step >= len(self.features) - 1
        obs  = self._get_observation() if not done else \
               np.zeros(self.n_features + 3, dtype=np.float32)

        return obs, reward, done, False, {"portfolio_val": self.portfolio_val}

    def get_results(self):
        return pd.DataFrame(self.trade_log).set_index("date")


# =============================================================================
# PART 2 — TRAINING CALLBACK
# =============================================================================

class TrainingCallback(BaseCallback):
    """Monitor training progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        return True


# =============================================================================
# PART 3 — REGIME-AWARE RL TRADER
# =============================================================================

class RLTrader:
    """
    Regime-aware RL trader using PPO.
    
    Improvements over basic version:
    1. Better reward shaping (Sharpe + drawdown penalty)
    2. Regime labels as additional features
    3. More training timesteps
    4. Feature normalization
    5. Walk-forward validation
    """

    def __init__(self, train_years=3, test_months=6,
                 total_timesteps=200000):
        self.train_years     = train_years
        self.test_months     = test_months
        self.total_timesteps = total_timesteps
        self.fold_results    = []

    def _add_regime_features(self, features, regime_labels,
                              regime_probs):
        """Add regime information as additional features."""
        if regime_labels is None:
            return features

        common = features.index.intersection(regime_labels.index)
        feat   = features.loc[common].copy()

        # Add regime label (normalized)
        rl = regime_labels.loc[common].astype(float)
        feat["regime"] = (rl - rl.mean()) / (rl.std() + 1e-8)

        # Add regime probabilities
        if regime_probs is not None:
            rp = regime_probs.loc[
                regime_probs.index.intersection(common)
            ]
            for col in rp.columns:
                feat[col] = rp[col].reindex(common).fillna(0.25)

        return feat

    def _make_env(self, features, returns):
        def _init():
            return CommodityTradingEnv(features, returns)
        return _init

    def walk_forward(self, features, returns,
                     regime_labels=None, regime_probs=None):
        """Walk-forward RL training with regime awareness."""

        # Add regime features if provided
        if regime_labels is not None:
            features = self._add_regime_features(
                features, regime_labels, regime_probs
            )
            common   = features.index.intersection(returns.index)
            features = features.loc[common]
            returns  = returns.loc[common]

        train_size = pd.DateOffset(years=self.train_years)
        test_size  = pd.DateOffset(months=self.test_months)
        start_date = features.index[0]
        end_date   = features.index[-1]
        test_start = start_date + train_size

        all_signals = []
        fold = 0

        print("Regime-Aware RL Walk-Forward Training")
        print(f"Train: {self.train_years}yr | "
              f"Test: {self.test_months}m | "
              f"Timesteps: {self.total_timesteps:,}")
        print(f"Features: {features.shape[1]} "
              f"(incl. regime: {regime_labels is not None})")
        print("=" * 65)

        while test_start < end_date:
            test_end = min(test_start + test_size, end_date)

            X_train = features[features.index <  test_start]
            y_train = returns[returns.index   <  test_start]
            X_test  = features[(features.index >= test_start) &
                                (features.index <  test_end)]
            y_test  = returns[(returns.index   >= test_start) &
                               (returns.index  <  test_end)]

            if len(X_train) < 200 or len(X_test) < 10:
                test_start = test_end
                continue

            print(f"Fold {fold+1}: "
                  f"[{X_train.index[0].date()} → "
                  f"{X_train.index[-1].date()}] | "
                  f"Test: [{test_start.date()} → "
                  f"{test_end.date()}]")

            # Train PPO
            train_env = DummyVecEnv([self._make_env(X_train, y_train)])
            model     = PPO(
                "MlpPolicy",
                train_env,
                learning_rate = 1e-4,    # slower = more stable
                n_steps       = 1024,
                batch_size    = 128,
                n_epochs      = 15,
                gamma         = 0.95,
                gae_lambda    = 0.95,
                clip_range    = 0.2,
                ent_coef      = 0.01,    # exploration bonus
                verbose       = 0,
                policy_kwargs = dict(
                    net_arch = [256, 256, 128]  # deeper network
                )
            )
            model.learn(total_timesteps=self.total_timesteps)

            # Test OOS
            test_env = CommodityTradingEnv(X_test, y_test)
            obs, _   = test_env.reset()
            signals  = []

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = test_env.step(int(action))
                signals.append(int(action) - 1)
                if done:
                    break

            results   = test_env.get_results()
            oos_ret   = results["strategy_ret"]
            sharpe    = (oos_ret.mean() * 252 /
                        (oos_ret.std() * np.sqrt(252) + 1e-8))
            final_val = test_env.portfolio_val

            long_pct  = signals.count(1)  / len(signals) * 100
            flat_pct  = signals.count(0)  / len(signals) * 100
            short_pct = signals.count(-1) / len(signals) * 100

            print(f"  OOS: Val={final_val:.3f} | "
                  f"Sharpe={sharpe:.2f} | "
                  f"L:{long_pct:.0f}% F:{flat_pct:.0f}% "
                  f"S:{short_pct:.0f}%")

            self.fold_results.append({
                "fold"      : fold,
                "test_start": test_start,
                "test_end"  : test_end,
                "sharpe"    : sharpe,
                "final_val" : final_val
            })

            signal_series = pd.Series(
                signals[:len(X_test)],
                index = X_test.index[:len(signals)],
                name  = "signal"
            )
            all_signals.append(signal_series)

            test_start = test_end
            fold += 1

        print("=" * 65)
        avg_sharpe = np.mean([f["sharpe"] for f in self.fold_results])
        avg_val    = np.mean([f["final_val"] for f in self.fold_results])
        print(f"Folds: {fold} | "
              f"Avg Sharpe: {avg_sharpe:.2f} | "
              f"Avg OOS Val: {avg_val:.3f}")

        return pd.concat(all_signals)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from features.feature_engine import FeatureEngine
    from features.regime_detector import RegimeDetector
    from backtest.backtest_engine import BacktestEngine
    from strategies.trend_following import TrendFollowing

    # 1. Load data
    loader = DataLoader()
    df     = loader.fetch(["CL=F"], start="2010-01-01", end="2024-01-01")
    close  = df["Close"].squeeze()

    # 2. Build features
    print("Building features...")
    engine   = FeatureEngine(df)
    features = engine.build_all()
    if isinstance(features.columns, pd.MultiIndex):
        features = features.xs("CL=F", axis=1, level=1)
    features.columns = [str(c) for c in features.columns]

    # 3. Build returns
    next_ret = np.log(close / close.shift(1)).shift(-1)
    common   = features.index.intersection(next_ret.dropna().index)
    X        = features.loc[common]
    y        = next_ret.loc[common]

    # 4. Detect regimes
    print("Detecting regimes...")
    detector = RegimeDetector(n_regimes=4)
    detector.fit(close)
    regime_labels, regime_probs = detector.predict(close)
    detector.label_regimes(close, regime_labels)
    detector.print_summary()

    # 5. Train regime-aware RL agent
    print("\nTraining regime-aware RL agent...")
    rl_trader = RLTrader(
        train_years     = 3,
        test_months     = 6,
        total_timesteps = 200000
    )
    rl_signals = rl_trader.walk_forward(
        X, y,
        regime_labels = regime_labels,
        regime_probs  = regime_probs
    )

    # 6. Backtest RL
    common2 = rl_signals.index.intersection(y.dropna().index)
    bt_rl   = BacktestEngine(rl_signals.loc[common2], y.loc[common2])
    res_rl  = bt_rl.run()
    print("\n=== RL AGENT (REGIME-AWARE) ===")
    bt_rl.summary()

    # 7. Baseline trend following
    tf      = TrendFollowing(fast=20, slow=60)
    sig_tf  = tf.predict_sized(close)
    common3 = sig_tf.index.intersection(y.dropna().index)
    bt_tf   = BacktestEngine(sig_tf.loc[common3], y.loc[common3])
    res_tf  = bt_tf.run()
    print("\n=== TREND FOLLOWING BASELINE ===")
    bt_tf.summary()

    # 8. Comparison chart
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "Regime-Aware RL Agent vs Trend Following: Crude Oil",
        fontsize=14, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(res_rl.index, res_rl["cumulative_pnl"],
            label=f"RL Agent (Regime-Aware) "
                  f"(Sharpe: {bt_rl.sharpe_ratio():.2f})",
            color="green", linewidth=1.5)
    ax.plot(res_tf.index, res_tf["cumulative_pnl"],
            label=f"Trend Following "
                  f"(Sharpe: {bt_tf.sharpe_ratio():.2f})",
            color="steelblue", linewidth=1.5)
    ax.plot(res_tf.index, res_tf["benchmark"],
            label="Buy & Hold",
            color="orange", linestyle="--")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title("Cumulative P&L")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for res, color, label in [
        (res_rl, "green",     "RL Agent"),
        (res_tf, "steelblue", "Trend Following")
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
    plt.savefig("evaluation/rl_regime_vs_trend.png",
                dpi=150, bbox_inches="tight")
    plt.show()
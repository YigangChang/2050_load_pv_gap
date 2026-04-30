"""
02_load_regression.py
=====================
需求端 Panel OLS 模型估計 + 日群集 Bootstrap。

模型：
    Load_t = α + Σ μ_h·Hour_h + Σ β_h·[HotDeg_t × I(hour=h)]
           + Σ δ_w·Weekday_w + ρ·Trend_t + ε_t

關鍵輸出：
    β_h (24個)：每小時每升溫1°C的負荷增量（MW/°C）

執行後輸出：
    output/results/regression_results.txt    ← OLS摘要報告
    output/results/beta_point.npy            ← β_h 點估計 (24,)
    output/results/beta_bootstrap.npy        ← Bootstrap矩陣 (B, 24)
    output/results/ols_model.pkl             ← OLS模型物件（選存）
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import pickle

import config

# ─── 可調整設定 ───────────────────────────────────────────────────────────────
B_BOOTSTRAP = config.B_BOOTSTRAP   # Bootstrap 次數
SEED        = config.RANDOM_SEED


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    建構OLS設計矩陣（X）與依變數（y）。

    特徵：
        - 小時虛擬變數 (hour_1 ~ hour_23，hour_0為參考組)
        - 星期虛擬變數 (weekday_1 ~ weekday_6，weekday_0=週一為參考組)
        - 趨勢項 trend
        - 逐小時溫度交叉項：hotdeg_h0 ~ hotdeg_h23（β_h 的來源）
        - 截距（statsmodels 的 add_constant）
    """
    df = df.copy()

    # 逐小時溫度交叉項：β_h = 溫度對第h小時負荷的邊際效果
    for h in range(24):
        df[f"hotdeg_h{h}"] = df["hotdeg"] * (df["hour"] == h).astype(float)

    # 小時虛擬（hour_0 為參考組）
    hour_dummies = pd.get_dummies(df["hour"], prefix="hour", drop_first=True).astype(float)

    # 星期虛擬（weekday_0=週一 為參考組）
    wd_dummies = pd.get_dummies(df["weekday"], prefix="wd", drop_first=True).astype(float)

    # 組合特徵
    hotdeg_cols = [f"hotdeg_h{h}" for h in range(24)]
    X = pd.concat([
        df[hotdeg_cols],
        hour_dummies,
        wd_dummies,
        df[["trend"]],
    ], axis=1)
    X = sm.add_constant(X, has_constant="add")

    y = df["load_mw"]
    return X, y


def fit_ols(X: pd.DataFrame, y: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """使用 statsmodels OLS 估計模型。"""
    model = sm.OLS(y, X)
    result = model.fit()
    return result


def extract_beta(result, n_hours: int = 24) -> np.ndarray:
    """
    從OLS結果中抽取 β_h（逐小時溫度係數）。
    係數命名為 hotdeg_h0 ~ hotdeg_h23。
    """
    beta = np.array([result.params[f"hotdeg_h{h}"] for h in range(n_hours)])
    return beta


def bootstrap_beta(df: pd.DataFrame, B: int = B_BOOTSTRAP,
                   seed: int = SEED) -> np.ndarray:
    """
    日群集 Bootstrap：以「日期」為重抽單位，保留日內觀測的時間相關結構。

    回傳：
        beta_boot : np.ndarray, shape (B, 24)
    """
    rng   = np.random.default_rng(seed)
    dates = df["datetime"].dt.date.unique()   # 所有唯一日期
    n_days = len(dates)
    beta_boot = np.zeros((B, 24))

    print(f"  Bootstrap 開始（B={B}，群集單位：{n_days} 天）...")

    for b in range(B):
        if (b + 1) % 100 == 0:
            print(f"    {b+1}/{B}", end="\r")

        # 有放回抽取日期
        sampled_dates = rng.choice(dates, size=n_days, replace=True)

        # 建構 bootstrap 樣本
        frames = []
        for d in sampled_dates:
            frames.append(df[df["datetime"].dt.date == d])
        df_b = pd.concat(frames, ignore_index=True)

        # 重新建趨勢（保持相同長度）
        df_b = df_b.reset_index(drop=True)
        df_b["trend"] = np.arange(len(df_b)) / max(len(df_b) - 1, 1)

        X_b, y_b = build_design_matrix(df_b)
        try:
            res_b = fit_ols(X_b, y_b)
            beta_boot[b] = extract_beta(res_b)
        except Exception as e:
            # 萬一某次 bootstrap 奇異矩陣，保留上一次結果
            print(f"\n  ⚠ bootstrap {b} 失敗: {e}，使用前次結果")
            if b > 0:
                beta_boot[b] = beta_boot[b - 1]

    print(f"\n  ✓ Bootstrap 完成")
    return beta_boot


def main():
    print("=" * 60)
    print("02 需求端回歸模型（OLS + Bootstrap）")
    print("=" * 60)

    # ── 1. 載入訓練集 ────────────────────────────────────────────────────────
    print("\n[1/5] 載入訓練集（2018-2021）...")
    panel_train = pd.read_parquet(config.RESULT_DIR / "panel_train.parquet")
    print(f"      {len(panel_train):,} 筆，欄位：{list(panel_train.columns)}")

    # ── 2. 建構設計矩陣 ───────────────────────────────────────────────────────
    print("\n[2/5] 建構設計矩陣...")
    X_train, y_train = build_design_matrix(panel_train)
    print(f"      X shape: {X_train.shape}")

    # ── 3. OLS 估計 ───────────────────────────────────────────────────────────
    print("\n[3/5] OLS 估計...")
    result = fit_ols(X_train, y_train)
    beta_point = extract_beta(result)

    print(f"\n  R²     = {result.rsquared:.4f}")
    print(f"  Adj.R² = {result.rsquared_adj:.4f}")
    print(f"  AIC    = {result.aic:.1f}")
    print(f"  Obs.   = {int(result.nobs):,}")

    print("\n  逐小時溫度係數 β_h（MW/°C）：")
    print(f"  {'Hour':>4}  {'β_h':>8}  {'SE':>8}  {'t':>7}  {'p':>6}")
    for h in range(24):
        p = result.pvalues[f"hotdeg_h{h}"]
        t = result.tvalues[f"hotdeg_h{h}"]
        se = result.bse[f"hotdeg_h{h}"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {h:>4}  {beta_point[h]:>8.2f}  {se:>8.2f}  {t:>7.2f}  {p:>6.4f} {sig}")

    # ── 4. Bootstrap β_h ─────────────────────────────────────────────────────
    print("\n[4/5] 日群集 Bootstrap...")
    beta_boot = bootstrap_beta(panel_train)

    # Bootstrap 統計摘要
    beta_lo = np.percentile(beta_boot, 2.5, axis=0)
    beta_hi = np.percentile(beta_boot, 97.5, axis=0)
    print("\n  Bootstrap 95% CI（β_h）：")
    print(f"  {'Hour':>4}  {'點估計':>8}  {'2.5%':>8}  {'97.5%':>8}")
    for h in range(24):
        print(f"  {h:>4}  {beta_point[h]:>8.2f}  {beta_lo[h]:>8.2f}  {beta_hi[h]:>8.2f}")

    # ── 5. 儲存結果 ───────────────────────────────────────────────────────────
    print("\n[5/5] 儲存結果...")

    # OLS 摘要文字
    with open(config.RESULT_DIR / "regression_results.txt", "w", encoding="utf-8") as f:
        f.write(result.summary().as_text())
    print("  ✓ OLS 摘要 → regression_results.txt")

    # β_h 陣列
    np.save(config.RESULT_DIR / "beta_point.npy", beta_point)
    print(f"  ✓ β_h 點估計 → beta_point.npy  (shape: {beta_point.shape})")

    np.save(config.RESULT_DIR / "beta_bootstrap.npy", beta_boot)
    print(f"  ✓ Bootstrap矩陣 → beta_bootstrap.npy  (shape: {beta_boot.shape})")

    # OLS 模型物件（備用）
    with open(config.RESULT_DIR / "ols_model.pkl", "wb") as f:
        pickle.dump(result, f)
    print("  ✓ OLS 模型 → ols_model.pkl")

    print("\n✓ 02 回歸完成。")


if __name__ == "__main__":
    main()

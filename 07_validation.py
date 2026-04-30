"""
07_validation.py
================
模型驗證（兩個目標）：

A. 需求端回歸模型驗證（Hold-out 2022）
   - 以 2018-2021 訓練的模型預測 2022 年負荷
   - 評估指標：RMSE, MAE, R²（整體 + 逐小時）

B. 合成氣候品質驗證（1995-2014）
   - ERA5 實測（1995-2014）vs 5個氣候模型的合成氣候同期
   - 評估溫度與輻射的分布重現能力
   - 指標：KS 統計量（分布差異）、均值偏差、標準差偏差

執行後輸出：
    output/results/validation_load.csv         ← 逐小時 RMSE/MAE/R²
    output/results/validation_climate.csv      ← 各模型 KS 統計量
    output/figures/validation_load_scatter.png
    output/figures/validation_load_hourly.png
    output/figures/validation_climate_temp.png
    output/figures/validation_climate_ssrd.png
"""

import numpy as np
import pandas as pd
from scipy import stats
import pickle

import config
import utils


# ─────────────────────────────────────────────────────────────────────────────
# A. 負荷模型驗證（Hold-out 2022）
# ─────────────────────────────────────────────────────────────────────────────

def validate_load_model():
    """
    用 2022 hold-out 驗證集評估負荷回歸模型。
    """
    print("\n── A. 負荷模型驗證（Hold-out 2022）──")

    # 載入 hold-out 資料
    panel_holdout = pd.read_parquet(config.RESULT_DIR / "panel_holdout.parquet")

    # 載入 OLS 模型
    with open(config.RESULT_DIR / "ols_model.pkl", "rb") as f:
        ols_result = pickle.load(f)

    # 重建設計矩陣（與訓練時一致）
    import importlib
    _reg = importlib.import_module("02_load_regression")
    build_design_matrix = _reg.build_design_matrix

    X_test, y_test = build_design_matrix(panel_holdout)

    # 對齊欄位（確保測試集有訓練集的所有虛擬變數）
    train_cols = ols_result.model.exog_names
    for c in train_cols:
        if c not in X_test.columns:
            X_test[c] = 0.0
    X_test = X_test[train_cols]

    # 預測
    y_pred = ols_result.predict(X_test)

    # 整體指標
    residuals = y_test.values - y_pred.values
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae  = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_test.values - y_test.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot

    print(f"\n  整體指標（Hold-out 2022）：")
    print(f"    RMSE = {rmse:>8,.1f} MW")
    print(f"    MAE  = {mae:>8,.1f} MW")
    print(f"    R²   = {r2:>8.4f}")

    # 逐小時指標
    panel_holdout["y_pred"] = y_pred.values
    panel_holdout["residual"] = residuals

    hourly_metrics = []
    for h in range(24):
        sub = panel_holdout[panel_holdout["hour"] == h]
        res = sub["residual"].values
        h_rmse = np.sqrt(np.mean(res ** 2))
        h_mae  = np.mean(np.abs(res))
        h_r2   = 1 - np.sum(res**2) / np.sum((sub["load_mw"].values - sub["load_mw"].mean())**2)
        hourly_metrics.append({"hour": h, "rmse_mw": h_rmse, "mae_mw": h_mae, "r2": h_r2})

    df_hourly = pd.DataFrame(hourly_metrics)
    df_hourly.to_csv(config.RESULT_DIR / "validation_load.csv", index=False)
    print(f"\n  逐小時 RMSE（MW）：")
    for _, row in df_hourly.iterrows():
        bar = "█" * int(row["rmse_mw"] / 50)
        print(f"    {int(row['hour']):02d}:00  RMSE={row['rmse_mw']:>6,.0f} MW  R²={row['r2']:.3f}  {bar}")

    # 繪圖
    _plot_load_validation(y_test.values, y_pred.values, panel_holdout)

    return df_hourly


# ─────────────────────────────────────────────────────────────────────────────
# B. 合成氣候品質驗證（1995-2014）
# ─────────────────────────────────────────────────────────────────────────────

def validate_climate_models():
    """
    ERA5 實測（1995-2014）vs 5個氣候模型合成氣候同期。
    """
    print("\n── B. 合成氣候品質驗證（1995-2014）──")

    # 載入 ERA5 實測（1995-2014）
    era5_valid = utils.load_era5_validation()
    era5_t = era5_valid["t2m_c"].values
    era5_s = era5_valid["ssrd"].values

    print(f"  ERA5（1995-2014）: "
          f"T mean={era5_t.mean():.2f}°C, std={era5_t.std():.2f}°C | "
          f"SSRD mean={era5_s.mean():.1f} W/m²")

    rows = []
    for model in config.CLIMATE_MODELS:
        df_syn = utils.load_synthetic_climate_valid(model)
        syn_t = df_syn["t2m_c"].values
        syn_s = df_syn["ssrd"].values

        # KS 檢定（溫度）
        ks_t, pval_t = stats.ks_2samp(era5_t, syn_t)
        # KS 檢定（輻射）
        ks_s, pval_s = stats.ks_2samp(era5_s, syn_s)

        # 均值偏差
        bias_t = syn_t.mean() - era5_t.mean()
        bias_s = syn_s.mean() - era5_s.mean()

        # 標準差比
        std_ratio_t = syn_t.std() / era5_t.std()
        std_ratio_s = syn_s.std() / era5_s.std()

        rows.append({
            "model":        model,
            "t_mean_era5":  era5_t.mean(),
            "t_mean_syn":   syn_t.mean(),
            "t_bias":       bias_t,
            "t_std_ratio":  std_ratio_t,
            "t_ks_stat":    ks_t,
            "t_ks_pval":    pval_t,
            "s_mean_era5":  era5_s.mean(),
            "s_mean_syn":   syn_s.mean(),
            "s_bias":       bias_s,
            "s_std_ratio":  std_ratio_s,
            "s_ks_stat":    ks_s,
            "s_ks_pval":    pval_s,
        })

        print(f"\n  {model}：")
        print(f"    溫度  bias={bias_t:+.3f}°C  std_ratio={std_ratio_t:.3f}  "
              f"KS={ks_t:.4f} (p={pval_t:.4f})")
        print(f"    輻射  bias={bias_s:+.1f} W/m²  std_ratio={std_ratio_s:.3f}  "
              f"KS={ks_s:.4f} (p={pval_s:.4f})")

    df_climate = pd.DataFrame(rows)
    df_climate.to_csv(config.RESULT_DIR / "validation_climate.csv", index=False)
    print(f"\n  ✓ validation_climate.csv")

    _plot_climate_validation(era5_valid)
    return df_climate


# ─────────────────────────────────────────────────────────────────────────────
# 繪圖函數
# ─────────────────────────────────────────────────────────────────────────────

def _plot_load_validation(y_true, y_pred, panel):
    """Plot load model validation (scatter + hourly RMSE)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  ⚠ matplotlib not installed, skipping plot")
        return

    plt.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False})

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # Left: predicted vs actual scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.3, s=5, color="#2196F3")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lim, lim, "r--", linewidth=1, label="Perfect Fit")
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    ax1.set_xlabel("Observed Load (MW)", fontsize=10)
    ax1.set_ylabel("Predicted Load (MW)", fontsize=10)
    ax1.set_title(f"Hold-out 2022\nRMSE = {rmse:,.0f} MW,  R² = {r2:.4f}", fontsize=11)
    ax1.legend(fontsize=8)

    # Right: hourly RMSE bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    hourly_rmse = panel.groupby("hour").apply(
        lambda x: np.sqrt(np.mean(x["residual"]**2))
    )
    ax2.bar(hourly_rmse.index, hourly_rmse.values, color="#FF5722", alpha=0.8)
    ax2.set_xlabel("Hour", fontsize=10)
    ax2.set_ylabel("RMSE (MW)", fontsize=10)
    ax2.set_title("Hourly RMSE — Hold-out 2022", fontsize=11)
    ax2.set_xticks(range(0, 24, 3))

    plt.tight_layout()
    fpath = config.FIGURE_DIR / "validation_load.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fpath}")


def _plot_climate_validation(era5_valid: pd.DataFrame):
    """Plot synthetic climate validation (distribution comparison)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not installed, skipping plot")
        return

    plt.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_specs = [
        (axes[0], "t2m_c", "t2m_c", "Temperature", "°C"),
        (axes[1], "ssrd",  "ssrd",  "Solar Radiation", "W/m²"),
    ]

    for ax, var, col_era5, label, unit in plot_specs:
        era5_vals = era5_valid[col_era5].values

        # Daytime filter for radiation (SSRD > 0)
        if var == "ssrd":
            era5_vals = era5_vals[era5_vals > 0]

        bins = np.linspace(era5_vals.min(), era5_vals.max(), 50)
        ax.hist(era5_vals, bins=bins, density=True, alpha=0.6,
                color="black", label="ERA5 (Observed)", zorder=5)

        colors_model = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
        for model, color in zip(config.CLIMATE_MODELS, colors_model):
            df_syn   = utils.load_synthetic_climate_valid(model)
            syn_vals = df_syn[var].values
            if var == "ssrd":
                syn_vals = syn_vals[syn_vals > 0]
            ax.hist(syn_vals, bins=bins, density=True, alpha=0.4,
                    histtype="step", linewidth=1.5, color=color, label=model)

        ax.set_xlabel(f"{label} ({unit})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Synthetic Climate Validation: {label} Distribution (1995–2014)",
                     fontsize=11)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fpath = config.FIGURE_DIR / "validation_climate.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("07 模型驗證")
    print("=" * 60)

    df_load    = validate_load_model()
    df_climate = validate_climate_models()

    print("\n" + "=" * 60)
    print("驗證摘要")
    print("=" * 60)
    print(f"\n負荷模型（整體）：")
    total_rmse = df_load["rmse_mw"].mean()
    total_mae  = df_load["mae_mw"].mean()
    print(f"  逐小時平均 RMSE = {total_rmse:,.1f} MW")
    print(f"  逐小時平均 MAE  = {total_mae:,.1f} MW")

    print(f"\n合成氣候（溫度 KS 統計量，越小越好）：")
    for _, row in df_climate.iterrows():
        grade = "✓" if row["t_ks_stat"] < 0.05 else "△" if row["t_ks_stat"] < 0.10 else "✗"
        print(f"  {grade} {row['model']:<20}  KS={row['t_ks_stat']:.4f}  "
              f"bias={row['t_bias']:+.3f}°C")

    print("\n✓ 07 驗證完成。")


if __name__ == "__main__":
    main()

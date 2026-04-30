"""
06_sobol_analysis.py
====================
Sobol 不確定性分解（使用 SALib 函式庫）。

四個不確定性來源：
    X1：K       ~ Uniform(1.58, 2.11)   結構性負荷調整係數
    X2：beta_u  ~ Uniform(0, 1)          → 對應 bootstrap 樣本索引（連續化處理）
    X3：clim_u  ~ Uniform(0, 1)          → 對應氣候年索引（連續化處理）
    X4：rad_u   ~ Uniform(0.95, 1.05)   → 輻射縮放因子（TCCIP SSRD ±5% 不確定性）

背景說明（輻射不確定性）：
    TCCIP 合成氣候的 SSRD 是從 ERA5 歷史資料重建，並非氣候模式真正的未來輻射投影。
    以乘數 rad_factor = rad_u 施加於未來 PV 的 SSRD；PV 基準使用 ERA5 實測值，不受影響。

輸出（以 July P99 DailyMaxGapIncrease 為目標變數）：
    S1  : 一階 Sobol 指數（各自獨立貢獻）
    ST  : 全階 Sobol 指數（含交互作用）

依賴套件：
    pip install SALib

執行後輸出：
    output/results/sobol_{C}gw.csv      ← S1, ST 及信賴區間
    output/figures/sobol_bar_{C}gw.png  ← 長條圖
"""

import numpy as np
import pandas as pd
import time

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_OK = True
except ImportError:
    SALIB_OK = False
    print("⚠ SALib 未安裝。請執行：pip install SALib")
    print("  本模組將改用手動方差分解（近似）。")

import config
import utils
import importlib
_pv = importlib.import_module("03_pv_model")
compute_pv_output = _pv.compute_pv_output


# ─────────────────────────────────────────────────────────────────────────────
# 模型函數：給定四個輸入，回傳 July P99
# ─────────────────────────────────────────────────────────────────────────────

def model_july_p99(
    X:              np.ndarray,     # (N_samples, 4)：[K, beta_u, clim_u, rad_u]
    all_years:      list,
    load_baseline:  np.ndarray,
    hotdeg_baseline:np.ndarray,
    pv_baseline:    np.ndarray,
    beta_bootstrap: np.ndarray,
    pv_capacity_gw: float,
) -> np.ndarray:
    """
    批次計算模型輸出（July P99 DailyMaxGapIncrease）。

    Parameters
    ----------
    X : shape (N, 4)，每列為 [K, beta_u, clim_u, rad_u]
        rad_u 直接用作輻射縮放因子（已在 Uniform(RAD_LOW, RAD_HIGH) 空間抽樣）

    Returns
    -------
    Y : shape (N,)，每個樣本的 July P99（MW）
    """
    N       = X.shape[0]
    n_clim  = len(all_years)
    B       = beta_bootstrap.shape[0]
    Y       = np.zeros(N)

    t0 = time.time()
    for i in range(N):
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{N}  ({elapsed:.0f}s)", end="\r")

        K          = float(X[i, 0])
        beta_idx   = int(X[i, 1] * B) % B         # Uniform(0,1) → integer [0, B-1]
        clim_idx   = int(X[i, 2] * n_clim) % n_clim
        rad_factor = float(X[i, 3])                # Uniform(RAD_LOW, RAD_HIGH)

        climate_year  = all_years[clim_idx]
        beta_h        = beta_bootstrap[beta_idx]   # (24,)

        # 計算 31 天的 DailyMaxGapIncrease
        t2m           = climate_year["t2m"]           # (31, 24)
        ssrd          = climate_year["ssrd"]          # (31, 24)，中央值
        hotdeg_future = climate_year["hotdeg"]        # (31, 24)

        # ── 輻射縮放（只對未來 PV 計算施加）────────────────────────────────
        ssrd_adjusted = ssrd * rad_factor             # (31, 24)

        delta_hotdeg = hotdeg_future - hotdeg_baseline[np.newaxis, :]
        load_gap     = (load_baseline[np.newaxis, :] * (K - 1.0)
                        + beta_h[np.newaxis, :] * delta_hotdeg)

        pv_future = compute_pv_output(
            t2m.ravel(), ssrd_adjusted.ravel(), pv_capacity_gw
        ).reshape(31, 24)

        # pv_baseline 使用歷史實測輻射，不受 rad_factor 影響
        pv_gap    = pv_baseline[np.newaxis, :] - pv_future
        total_gap = load_gap + pv_gap
        daily_max = total_gap.max(axis=1)

        Y[i] = np.percentile(daily_max, 99)

    return Y


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("06 Sobol 不確定性分解")
    print("=" * 60)

    if not SALIB_OK:
        print("SALib 未安裝，改執行手動方差分解（近似版本）。")
        run_manual_variance_decomposition()
        return

    # ── 1. 載入前置結果 ───────────────────────────────────────────────────────
    print("\n[1/4] 載入資料...")
    load_baseline    = np.load(config.RESULT_DIR / "load_baseline.npy")
    hotdeg_baseline  = np.load(config.RESULT_DIR / "hotdeg_baseline.npy")
    beta_bootstrap   = np.load(config.RESULT_DIR / "beta_bootstrap.npy")

    print("  載入氣候年...")
    all_years = _load_climate_years()

    # ── 2. 定義 SALib 問題（四個不確定性來源）───────────────────────────────
    problem = {
        "num_vars": 4,
        "names": ["K", "beta_u", "clim_u", "rad_u"],
        "bounds": [
            [config.K_LOW,  config.K_HIGH],    # K：結構性負荷係數
            [0.0, 1.0],                         # beta_u → bootstrap 索引
            [0.0, 1.0],                         # clim_u → 氣候年索引
            [config.RAD_LOW, config.RAD_HIGH],  # rad_u → 輻射縮放因子
        ],
    }

    print(f"\n[2/4] Saltelli 採樣（N_SOBOL={config.N_SOBOL}，D=4）...")
    print(f"  實際評估次數 = {config.N_SOBOL} × (2×4+2) = {config.N_SOBOL * 10:,}")
    param_values = saltelli.sample(
        problem, config.N_SOBOL, calc_second_order=False
    )
    print(f"  實際取得樣本數: {len(param_values):,}")

    # ── 3. 對三個容量情境各執行 Sobol 分析 ──────────────────────────────────
    print("\n[3/4] 執行模型評估（三個容量情境）...")

    all_results = []

    for cap in config.PV_CAPACITIES_GW:
        print(f"\n  ── PV 容量: {cap} GW ──")
        pv_baseline = np.load(config.RESULT_DIR / f"pv_baseline_{cap}gw.npy")

        Y = model_july_p99(
            X               = param_values,
            all_years       = all_years,
            load_baseline   = load_baseline,
            hotdeg_baseline = hotdeg_baseline,
            pv_baseline     = pv_baseline,
            beta_bootstrap  = beta_bootstrap,
            pv_capacity_gw  = cap,
        )
        print(f"\n  Y 統計: mean={Y.mean():.1f}, P95={np.percentile(Y,95):.1f} MW")

        # Sobol 分析
        Si = sobol.analyze(
            problem, Y, calc_second_order=False, print_to_console=False
        )

        labels = ["K（結構性負荷）", "β_h（回歸統計）", "氣候情境", "輻射縮放（±5%）"]
        print(f"\n  Sobol 指數（PV={cap} GW）：")
        print(f"  {'來源':<20}  {'S1':>8}  {'S1_conf':>8}  {'ST':>8}  {'ST_conf':>8}")
        for j, label in enumerate(labels):
            print(f"  {label:<20}  {Si['S1'][j]:>8.4f}  {Si['S1_conf'][j]:>8.4f}"
                  f"  {Si['ST'][j]:>8.4f}  {Si['ST_conf'][j]:>8.4f}")

        # 儲存
        result_df = pd.DataFrame({
            "source":  ["K_structural", "beta_statistical", "climate", "radiation"],
            "label":   labels,
            "S1":      Si["S1"],
            "S1_conf": Si["S1_conf"],
            "ST":      Si["ST"],
            "ST_conf": Si["ST_conf"],
            "capacity_gw": cap,
        })
        result_df.to_csv(config.RESULT_DIR / f"sobol_{cap}gw.csv", index=False)
        all_results.append(result_df)
        print(f"  ✓ sobol_{cap}gw.csv")

    # 合併所有結果
    pd.concat(all_results).to_csv(
        config.RESULT_DIR / "sobol_all.csv", index=False
    )

    print("\n[4/4] 繪製 Sobol 長條圖...")
    _plot_sobol(all_results)
    print("\n✓ 06 Sobol 分析完成。")


def _load_climate_years() -> list:
    """從 utils 載入氣候年（複用 05_monte_carlo 的邏輯）。"""
    all_years = []
    for model in config.CLIMATE_MODELS:
        for ssp in config.SSPS:
            df = utils.load_synthetic_climate(model, ssp)
            for yr in sorted(df["year"].unique()):
                df_yr = df[df["year"] == yr].sort_values(["day", "hour"])
                t2m   = df_yr["t2m_c"].values.reshape(31, 24)
                ssrd  = df_yr["ssrd"].values.reshape(31, 24)   # 中央值
                hotdeg = np.maximum(0.0, t2m - config.T_THRESHOLD)
                all_years.append({
                    "model":  model, "ssp": ssp, "year": yr,
                    "t2m":    t2m,   "ssrd": ssrd, "hotdeg": hotdeg,
                })
    print(f"  ✓ 共載入 {len(all_years)} 個氣候年")
    return all_years


def run_manual_variance_decomposition():
    """
    SALib 未安裝時的備用方案：
    手動計算四個來源的方差貢獻（近似一階 Sobol 指數）。

    方法：
        固定三個來源的抽樣，只對第四個來源取變異數，
        與全方差比較得到各來源的解釋比例。
    """
    print("\n[備用] 手動方差分解（四個不確定性來源）...")
    rng = np.random.default_rng(config.RANDOM_SEED)

    load_baseline    = np.load(config.RESULT_DIR / "load_baseline.npy")
    hotdeg_baseline  = np.load(config.RESULT_DIR / "hotdeg_baseline.npy")
    beta_bootstrap   = np.load(config.RESULT_DIR / "beta_bootstrap.npy")
    all_years        = _load_climate_years()

    N   = 2000    # 每次抽樣數量
    B   = beta_bootstrap.shape[0]
    NC  = len(all_years)
    CAP = 60      # 以 60 GW 為代表

    pv_baseline = np.load(config.RESULT_DIR / f"pv_baseline_{CAP}gw.npy")

    def eval_july_p99(K_arr, beta_idx_arr, clim_idx_arr, rad_arr):
        results = []
        for i in range(len(K_arr)):
            cy         = all_years[clim_idx_arr[i]]
            bh         = beta_bootstrap[beta_idx_arr[i]]
            K          = K_arr[i]
            rad_factor = rad_arr[i]
            t2m        = cy["t2m"]
            ssrd_adj   = cy["ssrd"] * rad_factor         # 輻射縮放
            delta_hd   = cy["hotdeg"] - hotdeg_baseline[np.newaxis, :]
            load_gap   = load_baseline[np.newaxis,:] * (K-1) + bh[np.newaxis,:] * delta_hd
            pv_future  = compute_pv_output(
                t2m.ravel(), ssrd_adj.ravel(), CAP
            ).reshape(31, 24)
            pv_gap     = pv_baseline[np.newaxis,:] - pv_future
            dm         = (load_gap + pv_gap).max(axis=1)
            results.append(np.percentile(dm, 99))
        return np.array(results)

    K_mid  = (config.K_LOW + config.K_HIGH) / 2
    bi_mid = B // 2
    ci_mid = NC // 2
    rm_mid = (config.RAD_LOW + config.RAD_HIGH) / 2

    # 全方差（所有來源均隨機）
    K_all  = rng.uniform(config.K_LOW,  config.K_HIGH,  N)
    bi_all = rng.integers(0, B,  N)
    ci_all = rng.integers(0, NC, N)
    ra_all = rng.uniform(config.RAD_LOW, config.RAD_HIGH, N)
    Y_all  = eval_july_p99(K_all, bi_all, ci_all, ra_all)
    var_total = Y_all.var()

    # 固定其他三個，只隨機 K
    Y_K_only = eval_july_p99(
        rng.uniform(config.K_LOW, config.K_HIGH, N),
        np.full(N, bi_mid), np.full(N, ci_mid), np.full(N, rm_mid)
    )

    # 固定其他三個，只隨機 β
    Y_b_only = eval_july_p99(
        np.full(N, K_mid),
        rng.integers(0, B, N), np.full(N, ci_mid), np.full(N, rm_mid)
    )

    # 固定其他三個，只隨機氣候
    Y_c_only = eval_july_p99(
        np.full(N, K_mid), np.full(N, bi_mid),
        rng.integers(0, NC, N), np.full(N, rm_mid)
    )

    # 固定其他三個，只隨機輻射
    Y_r_only = eval_july_p99(
        np.full(N, K_mid), np.full(N, bi_mid), np.full(N, ci_mid),
        rng.uniform(config.RAD_LOW, config.RAD_HIGH, N)
    )

    v_K  = Y_K_only.var()
    v_b  = Y_b_only.var()
    v_c  = Y_c_only.var()
    v_r  = Y_r_only.var()
    v_sum = v_K + v_b + v_c + v_r

    print(f"\n  近似 Sobol 一階指數（PV={CAP} GW）：")
    print(f"  K（結構性負荷）  : {v_K/var_total:.3f}  ({v_K/v_sum*100:.1f}% 相對比例)")
    print(f"  β_h（回歸統計）  : {v_b/var_total:.3f}  ({v_b/v_sum*100:.1f}% 相對比例)")
    print(f"  氣候情境         : {v_c/var_total:.3f}  ({v_c/v_sum*100:.1f}% 相對比例)")
    print(f"  輻射縮放（±5%）  : {v_r/var_total:.3f}  ({v_r/v_sum*100:.1f}% 相對比例)")
    print(f"  交互作用+殘差    : {1-(v_K+v_b+v_c+v_r)/var_total:.3f}")

    pd.DataFrame([
        {"source": "K_structural",    "approx_S1": v_K/var_total},
        {"source": "beta_statistical", "approx_S1": v_b/var_total},
        {"source": "climate",          "approx_S1": v_c/var_total},
        {"source": "radiation",        "approx_S1": v_r/var_total},
        {"source": "interaction",      "approx_S1": 1-(v_K+v_b+v_c+v_r)/var_total},
    ]).to_csv(config.RESULT_DIR / f"sobol_manual_{CAP}gw.csv", index=False)
    print(f"  ✓ sobol_manual_{CAP}gw.csv")


def _plot_sobol(all_results: list):
    """繪製 Sobol 指數長條圖（需 matplotlib）。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        print("  ⚠ matplotlib 未安裝，跳過繪圖")
        return

    plt.rcParams.update({
        "font.family":        "sans-serif",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(1, len(config.PV_CAPACITIES_GW),
                             figsize=(16, 5), sharey=True)
    colors       = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    labels_short = [
        "K\n(Structural\nLoad)",
        "$\\beta_h$\n(Statistical)",
        "Climate\nScenario",
        "Radiation\n(±5%)",
    ]

    for ax, df in zip(axes, all_results):
        cap = df["capacity_gw"].iloc[0]
        x   = np.arange(len(df))
        ax.bar(x - 0.2, df["S1"], 0.35, label="S1 (First-order)",
               color=colors, alpha=0.85)
        ax.bar(x + 0.2, df["ST"], 0.35, label="ST (Total-order)",
               color=colors, alpha=0.45, edgecolor="gray")
        ax.errorbar(x - 0.2, df["S1"], yerr=df["S1_conf"], fmt="none",
                    color="black", capsize=4, linewidth=1.2)
        ax.errorbar(x + 0.2, df["ST"], yerr=df["ST_conf"], fmt="none",
                    color="black", capsize=4, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, fontsize=9)
        ax.set_title(f"PV = {cap} GW", fontsize=11)
        ax.set_ylabel("Sobol Index", fontsize=10)
        ax.set_ylim(0, 1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Sobol Sensitivity Analysis (4 Sources): July P99 DailyMaxGapIncrease",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = config.FIGURE_DIR / "sobol_bar.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fpath}")


if __name__ == "__main__":
    main()

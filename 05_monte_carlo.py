"""
05_monte_carlo.py
=================
四層不確定性蒙地卡羅主程式。

不確定性架構（四個來源）：
    1. 氣候情境  ：5模型 × 2SSP × 20年 = 200組 July（每組31天）
    2. β_h 統計  ：Bootstrap 1000組回歸係數
    3. K 結構性  ：K ~ Uniform(1.58, 2.11)（結構性負荷調整係數）
    4. 輻射不確定：rad_factor ~ Uniform(0.95, 1.05)

背景說明（輻射不確定性）：
    TCCIP 合成氣候的 SSRD 是從 ERA5 歷史資料重建，
    並非氣候模式真正的未來輻射投影（全球雲量變化難以在區域尺度準確預測）。
    資料已提供 ±5% 的不確定性區間（ssrd_wm2_rad_minus_5 / ssrd_wm2_rad_plus_5），
    以乘數 rad_factor ~ Uniform(0.95, 1.05) 納入蒙地卡羅。
    ※ 僅對 PV_future 的 SSRD 施加此縮放；PV_baseline 使用 ERA5 實測值，不變動。

核心計算（每個 July，31天 × 24小時）：
    ssrd_adj(d,h)    = ssrd_future(d,h) × rad_factor
    pv_future(d,h)   = PVWatts(t2m(d,h), ssrd_adj(d,h), cap)
    gap(d, h)        = load_baseline[h] × (K−1)
                     + β_h[h] × ΔHotDeg(d, h)
                     + (PV_baseline[h] − pv_future(d,h))
    DailyMaxGapIncrease(d) = max_h gap(d, h)
    July_P99               = percentile(DailyMaxGapIncrease, 99)

執行後輸出：
    output/results/mc_daily_max_gap_{C}gw.npy   ← shape (N_MC, 31), MW
    output/results/mc_july_p99_{C}gw.npy        ← shape (N_MC,), MW
    output/results/mc_meta.npy                  ← shape (N_MC, 4)，每次模擬的
                                                    [clim_idx, beta_idx, K, rad_factor]
    output/results/mc_summary.csv               ← 各容量情境統計摘要
"""

import numpy as np
import pandas as pd
import time

import config
import utils
import importlib
_pv = importlib.import_module("03_pv_model")
compute_pv_output = _pv.compute_pv_output


# ─────────────────────────────────────────────────────────────────────────────
# 輔助：載入所有未來氣候年（按索引存取）
# ─────────────────────────────────────────────────────────────────────────────

def load_all_climate_years() -> list[dict]:
    """
    載入所有200個氣候年（10情境 × 20年），
    每個氣候年為一個 dict：
        {model, ssp, year, t2m, ssrd, hotdeg}
        shape 均為 (31, 24)

    注意：此處 ssrd 儲存中央值（ssrd_wm2），
    rad_factor 在模擬時才乘入，保持中央值與不確定性分離。
    """
    print("  載入所有未來氣候年（200個 July）...")
    all_years = []

    for model in config.CLIMATE_MODELS:
        for ssp in config.SSPS:
            df = utils.load_synthetic_climate(model, ssp)

            for yr in sorted(df["year"].unique()):
                df_yr = df[df["year"] == yr].copy()
                df_yr = df_yr.sort_values(["day", "hour"])

                assert len(df_yr) == 744, \
                    f"氣候年 {model}/{ssp}/{yr} 資料筆數: {len(df_yr)}（預期744）"

                t2m    = df_yr["t2m_c"].values.reshape(31, 24)
                ssrd   = df_yr["ssrd"].values.reshape(31, 24)       # 中央值
                hotdeg = np.maximum(0.0, t2m - config.T_THRESHOLD)

                all_years.append({
                    "model":  model,
                    "ssp":    ssp,
                    "year":   yr,
                    "t2m":    t2m,
                    "ssrd":   ssrd,   # 中央值，模擬時乘以 rad_factor
                    "hotdeg": hotdeg,
                })

    print(f"  ✓ 共載入 {len(all_years)} 個氣候年")
    return all_years


# ─────────────────────────────────────────────────────────────────────────────
# 核心：單次模擬計算 DailyMaxGapIncrease（31天）
# ─────────────────────────────────────────────────────────────────────────────

def compute_daily_max_gap(
    climate_year:    dict,
    beta_h:          np.ndarray,   # (24,)
    K:               float,
    rad_factor:      float,        # ← 新增：輻射縮放因子
    load_baseline:   np.ndarray,   # (24,)
    hotdeg_baseline: np.ndarray,   # (24,)
    pv_baseline:     np.ndarray,   # (24,) MW，使用 ERA5 實測值，不受 rad_factor 影響
    pv_capacity_gw:  float,
) -> np.ndarray:
    """
    計算單個 July 的 DailyMaxGapIncrease（共31天）。

    rad_factor 只作用於 ssrd_future（未來輻射不確定性），
    pv_baseline 使用 ERA5 歷史實測輻射，不做縮放。

    Returns
    -------
    daily_max : np.ndarray, shape (31,), 單位 MW
    """
    t2m          = climate_year["t2m"]     # (31, 24)
    ssrd_central = climate_year["ssrd"]    # (31, 24)，中央值
    hotdeg_future = climate_year["hotdeg"] # (31, 24)

    # ── 輻射縮放（只對未來 PV 計算施加）──────────────────────────────────────
    ssrd_adjusted = ssrd_central * rad_factor   # (31, 24)

    # ── ΔHotDeg ──────────────────────────────────────────────────────────────
    delta_hotdeg = hotdeg_future - hotdeg_baseline[np.newaxis, :]   # (31, 24)

    # ── 需求端缺口 ────────────────────────────────────────────────────────────
    load_gap = (load_baseline[np.newaxis, :] * (K - 1.0)
                + beta_h[np.newaxis, :] * delta_hotdeg)             # (31, 24)

    # ── 供給端缺口（使用輻射調整後的 SSRD）─────────────────────────────────
    pv_future = compute_pv_output(
        temp_air_c     = t2m.ravel(),
        ssrd_wm2       = ssrd_adjusted.ravel(),   # ← 套用 rad_factor
        pv_capacity_gw = pv_capacity_gw,
    ).reshape(31, 24)

    # pv_baseline 使用歷史實測輻射，不受 rad_factor 影響
    pv_gap = pv_baseline[np.newaxis, :] - pv_future                 # (31, 24)

    # ── 合計並取日最大 ────────────────────────────────────────────────────────
    total_gap = load_gap + pv_gap                                    # (31, 24)
    daily_max = total_gap.max(axis=1)                                # (31,)
    return daily_max


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    pv_capacity_gw:  float,
    all_years:       list[dict],
    load_baseline:   np.ndarray,
    hotdeg_baseline: np.ndarray,
    pv_baseline:     np.ndarray,
    beta_bootstrap:  np.ndarray,   # (B, 24)
    n_mc:            int = config.N_MC,
    seed:            int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    執行蒙地卡羅模擬（四個不確定性來源）。

    每次模擬從四個來源各抽一個樣本：
        1. clim_idx  ← 氣候年索引（0~199）
        2. K         ← Uniform(K_LOW, K_HIGH)
        3. beta_idx  ← Bootstrap樣本索引（0~999）
        4. rad_factor← Uniform(RAD_LOW, RAD_HIGH)

    Returns
    -------
    daily_max_all : np.ndarray, shape (N_MC, 31), MW
    july_p99      : np.ndarray, shape (N_MC,), MW
    mc_meta       : np.ndarray, shape (N_MC, 4)
                    欄：[clim_idx, beta_idx, K, rad_factor]
    """
    rng    = np.random.default_rng(seed)
    n_clim = len(all_years)
    B      = beta_bootstrap.shape[0]

    daily_max_all = np.zeros((n_mc, 31))
    july_p99      = np.zeros(n_mc)
    mc_meta       = np.zeros((n_mc, 4))   # [clim_idx, beta_idx, K, rad_factor]

    t0 = time.time()
    for i in range(n_mc):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n_mc - i - 1)
            print(f"    {i+1:5d}/{n_mc}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  end="\r")

        # ── 從四個不確定性來源各抽一個樣本 ──
        clim_idx   = int(rng.integers(0, n_clim))
        beta_idx   = int(rng.integers(0, B))
        K          = float(rng.uniform(config.K_LOW, config.K_HIGH))
        rad_factor = float(rng.uniform(config.RAD_LOW, config.RAD_HIGH))

        climate_year = all_years[clim_idx]
        beta_h       = beta_bootstrap[beta_idx]

        # ── 計算 DailyMaxGapIncrease（31天）──
        daily_max = compute_daily_max_gap(
            climate_year    = climate_year,
            beta_h          = beta_h,
            K               = K,
            rad_factor      = rad_factor,
            load_baseline   = load_baseline,
            hotdeg_baseline = hotdeg_baseline,
            pv_baseline     = pv_baseline,
            pv_capacity_gw  = pv_capacity_gw,
        )

        daily_max_all[i]  = daily_max
        july_p99[i]       = np.percentile(daily_max, 99)
        mc_meta[i]        = [clim_idx, beta_idx, K, rad_factor]

    print(f"\n  ✓ {n_mc} 次模擬完成，耗時 {time.time()-t0:.1f}s")
    return daily_max_all, july_p99, mc_meta


def main():
    print("=" * 60)
    print("05 蒙地卡羅模擬（四個不確定性來源）")
    print("=" * 60)
    print(f"  不確定性來源：")
    print(f"    1. 氣候情境  ：5模型 × 2SSP × 20年 = 200組")
    print(f"    2. β_h 統計  ：Bootstrap {config.B_BOOTSTRAP} 組")
    print(f"    3. K 結構性  ：Uniform({config.K_LOW}, {config.K_HIGH})")
    print(f"    4. 輻射縮放  ：Uniform({config.RAD_LOW}, {config.RAD_HIGH})  [±5% SSRD]")

    # ── 1. 載入前置結果 ───────────────────────────────────────────────────────
    print("\n[1/4] 載入基準值與 Bootstrap 結果...")
    load_baseline    = np.load(config.RESULT_DIR / "load_baseline.npy")
    hotdeg_baseline  = np.load(config.RESULT_DIR / "hotdeg_baseline.npy")
    beta_bootstrap   = np.load(config.RESULT_DIR / "beta_bootstrap.npy")
    print(f"  load_baseline  : {load_baseline.shape}")
    print(f"  hotdeg_baseline: {hotdeg_baseline.shape}")
    print(f"  beta_bootstrap : {beta_bootstrap.shape}")

    # ── 2. 載入所有未來氣候年 ─────────────────────────────────────────────────
    print("\n[2/4] 載入氣候年資料...")
    all_years = load_all_climate_years()

    # ── 3. 對三個 PV 容量情境各跑蒙地卡羅 ─────────────────────────────────────
    print(f"\n[3/4] 執行蒙地卡羅（N_MC={config.N_MC:,}）...")
    summary_rows = []

    for cap in config.PV_CAPACITIES_GW:
        print(f"\n  ── PV 容量: {cap} GW ──")
        pv_baseline = np.load(config.RESULT_DIR / f"pv_baseline_{cap}gw.npy")

        daily_max_all, july_p99, mc_meta = run_monte_carlo(
            pv_capacity_gw  = cap,
            all_years       = all_years,
            load_baseline   = load_baseline,
            hotdeg_baseline = hotdeg_baseline,
            pv_baseline     = pv_baseline,
            beta_bootstrap  = beta_bootstrap,
            n_mc            = config.N_MC,
        )

        # 儲存原始結果
        np.save(config.RESULT_DIR / f"mc_daily_max_gap_{cap}gw.npy", daily_max_all)
        np.save(config.RESULT_DIR / f"mc_july_p99_{cap}gw.npy",      july_p99)
        np.save(config.RESULT_DIR / f"mc_meta_{cap}gw.npy",          mc_meta)

        # 統計摘要
        pooled = daily_max_all.ravel()
        row = {"capacity_gw": cap}
        for pct in config.PERCENTILES:
            row[f"pooled_p{pct}_mw"]    = np.percentile(pooled, pct)
            row[f"july_p99_p{pct}_mw"]  = np.percentile(july_p99, pct)
        row["pooled_mean_mw"]   = pooled.mean()
        row["july_p99_mean_mw"] = july_p99.mean()
        summary_rows.append(row)

        # 輻射不確定性對結果的影響（邊際分析）
        rad_vals = mc_meta[:, 3]
        corr = np.corrcoef(rad_vals, july_p99)[0, 1]

        print(f"  DailyMaxGapIncrease（所有模擬日子 pooled）：")
        print(f"    mean={pooled.mean():>9,.1f} MW")
        for pct in [50, 90, 95, 99]:
            print(f"    P{pct:<2} ={np.percentile(pooled, pct):>9,.1f} MW")
        print(f"  July-P99 分布（{config.N_MC}次模擬）：")
        print(f"    mean={july_p99.mean():>9,.1f} MW  "
              f"P95={np.percentile(july_p99,95):>9,.1f} MW")
        print(f"  rad_factor vs July-P99 相關係數: {corr:.4f}")
        print(f"  （負相關表示輻射增加→PV增加→缺口縮小，符合預期）")

    # ── 4. 儲存摘要 ───────────────────────────────────────────────────────────
    print("\n[4/4] 儲存摘要...")
    pd.DataFrame(summary_rows).to_csv(
        config.RESULT_DIR / "mc_summary.csv", index=False
    )
    print("  ✓ mc_summary.csv")
    print("\n✓ 05 蒙地卡羅完成。")


if __name__ == "__main__":
    main()

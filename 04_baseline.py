"""
04_baseline.py
==============
計算歷史基準值（2018-2022）：
    1. Load^baseline(h)  ← 逐小時平均負荷（24×1 向量）
    2. PV^baseline(h)    ← 2050裝置容量 × ERA5-2018-2022氣候下的PV產量（24×1 向量）
    3. hotdeg_baseline(h)← 逐小時平均 HotDegree（24×1 向量，供蒙地卡羅計算 ΔHotDeg）

說明：
    PV^baseline 與 PV^future 均使用 2050 年裝置容量，
    只改變氣候輸入，方可正確隔離氣候效應而非容量擴張效應。

執行後輸出：
    output/results/load_baseline.npy        ← shape (24,), MW
    output/results/hotdeg_baseline.npy      ← shape (24,), °C
    output/results/pv_baseline_{C}gw.npy   ← shape (24,), MW，C ∈ {40, 60, 80}
    output/results/baseline_summary.csv    ← 可讀性摘要
"""

import numpy as np
import pandas as pd

import importlib
import config
import utils

_pv = importlib.import_module("03_pv_model")
compute_pv_output = _pv.compute_pv_output


def compute_load_baseline(panel: pd.DataFrame) -> np.ndarray:
    """
    計算逐小時平均負荷（2018-2022）。

    Parameters
    ----------
    panel : pd.DataFrame，含 hour, load_mw 欄位

    Returns
    -------
    load_baseline : np.ndarray, shape (24,), 單位 MW
    """
    baseline = panel.groupby("hour")["load_mw"].mean().sort_index().values
    assert baseline.shape == (24,), f"預期 shape (24,)，得到 {baseline.shape}"
    return baseline


def compute_hotdeg_baseline(panel: pd.DataFrame) -> np.ndarray:
    """
    計算逐小時平均 HotDegree（2018-2022）。

    用於計算未來情境的 ΔHotDeg：
        ΔHotDeg(h, d) = HotDeg_future(h, d) - hotdeg_baseline(h)

    Returns
    -------
    hotdeg_baseline : np.ndarray, shape (24,)
    """
    baseline = panel.groupby("hour")["hotdeg"].mean().sort_index().values
    return baseline


def compute_pv_baseline(
    era5_df:        pd.DataFrame,
    pv_capacity_gw: float,
) -> np.ndarray:
    """
    計算 PV^baseline：以 ERA5-2018-2022 氣候計算，再取逐小時平均。

    Parameters
    ----------
    era5_df        : pd.DataFrame，含 hour, t2m_c, ssrd 欄位（2018-2022）
    pv_capacity_gw : float，2050年PV裝置容量（GW）

    Returns
    -------
    pv_baseline : np.ndarray, shape (24,), 單位 MW
    """
    df = era5_df.copy()
    df["pv_mw"] = compute_pv_output(
        temp_air_c     = df["t2m_c"].values,
        ssrd_wm2       = df["ssrd"].values,
        pv_capacity_gw = pv_capacity_gw,
    )
    pv_base = df.groupby("hour")["pv_mw"].mean().sort_index().values
    assert pv_base.shape == (24,)
    return pv_base


def main():
    print("=" * 60)
    print("04 基準值計算")
    print("=" * 60)

    # ── 1. 載入訓練資料（2018-2021）及ERA5全資料（2018-2022）───────────────
    print("\n[1/4] 載入資料...")
    panel_train = pd.read_parquet(config.RESULT_DIR / "panel_train.parquet")
    panel_all   = pd.read_parquet(config.RESULT_DIR / "panel_all.parquet")

    # ERA5 全期（2018-2022）用於 PV baseline
    era5_all = utils.load_era5_land_all(config.ALL_YEARS)
    print(f"      訓練集: {len(panel_train):,} 筆，ERA5全期: {len(era5_all):,} 筆")

    # ── 2. Load^baseline ─────────────────────────────────────────────────────
    print("\n[2/4] 計算 Load^baseline（逐小時平均，訓練集 2018-2021）...")
    load_baseline = compute_load_baseline(panel_train)
    np.save(config.RESULT_DIR / "load_baseline.npy", load_baseline)
    print(f"  ✓ shape: {load_baseline.shape}")
    for h, v in enumerate(load_baseline):
        print(f"     {h:02d}:00  {v:9,.1f} MW")

    # ── 3. HotDeg^baseline ───────────────────────────────────────────────────
    print("\n[3/4] 計算 HotDeg^baseline（逐小時平均 HotDegree）...")
    hotdeg_baseline = compute_hotdeg_baseline(panel_train)
    np.save(config.RESULT_DIR / "hotdeg_baseline.npy", hotdeg_baseline)
    print(f"  ✓ shape: {hotdeg_baseline.shape}")
    print(f"  日間（08-18時）平均 HotDeg: "
          f"{hotdeg_baseline[8:19].mean():.2f} °C")

    # ── 4. PV^baseline（三個容量情境）────────────────────────────────────────
    print("\n[4/4] 計算 PV^baseline（三個容量情境：40/60/80 GW）...")
    summary_rows = []

    for cap in config.PV_CAPACITIES_GW:
        pv_base = compute_pv_baseline(era5_all, cap)
        fname = config.RESULT_DIR / f"pv_baseline_{cap}gw.npy"
        np.save(fname, pv_base)
        daily_gen = pv_base.sum()  # MWh/day（逐小時加總）
        cf = daily_gen / (cap * 1000 * 24) * 100  # 容量因子 %

        print(f"\n  PV @ {cap} GW：")
        print(f"    峰值輸出: {pv_base.max():>8,.1f} MW  "
              f"（@{pv_base.argmax():02d}:00）")
        print(f"    日均發電: {daily_gen:>8,.1f} MWh/day")
        print(f"    日均容量因子: {cf:.1f}%")

        for h in range(24):
            summary_rows.append({
                "capacity_gw": cap,
                "hour":        h,
                "pv_baseline_mw": pv_base[h],
                "load_baseline_mw": load_baseline[h],
                "hotdeg_baseline": hotdeg_baseline[h],
            })

    # ── 儲存可讀性摘要 ────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.RESULT_DIR / "baseline_summary.csv", index=False)
    print("\n  ✓ 摘要 → baseline_summary.csv")
    print("\n✓ 04 基準值計算完成。")


if __name__ == "__main__":
    main()

"""
01_data_preprocessing.py
========================
載入所有歷史原始資料，合併為Panel格式，計算HotDegree，
並輸出標準化的 panel 資料集供後續回歸使用。

執行後輸出：
    output/results/panel_all.parquet     ← 2018-2022全部資料（含hold-out 2022）
    output/results/panel_train.parquet   ← 2018-2021訓練集
    output/results/panel_holdout.parquet ← 2022 hold-out 驗證集
    output/results/data_summary.csv      ← 描述性統計
"""

import pandas as pd
import numpy as np

import config
import utils


def main():
    print("=" * 60)
    print("01 資料前處理")
    print("=" * 60)

    # ── 1. 建構完整 Panel（2018-2022）────────────────────────────────────────
    print("\n[1/4] 載入 ERA5-Land 與台電負荷，合併 Panel...")
    panel = utils.build_panel(years=config.ALL_YEARS)

    print(f"      Panel shape: {panel.shape}")
    print(f"      時間範圍: {panel['datetime'].min()} → {panel['datetime'].max()}")
    print(f"      缺失值：\n{panel.isnull().sum()}")

    # ── 2. 資料品質檢查 ───────────────────────────────────────────────────────
    print("\n[2/4] 資料品質檢查...")
    _quality_check(panel)

    # ── 3. 切分訓練集 / Hold-out ──────────────────────────────────────────────
    print("\n[3/4] 切分訓練集（2018-2021）與 Hold-out（2022）...")
    panel_train   = panel[panel["datetime"].dt.year.isin(config.TRAIN_YEARS)].reset_index(drop=True)
    panel_holdout = panel[panel["datetime"].dt.year == config.HOLDOUT_YEAR].reset_index(drop=True)

    print(f"      訓練集: {len(panel_train):,} 筆 ({panel_train['datetime'].dt.year.unique().tolist()})")
    print(f"      Hold-out: {len(panel_holdout):,} 筆 ({config.HOLDOUT_YEAR})")

    # ── 4. 儲存 ───────────────────────────────────────────────────────────────
    print("\n[4/4] 儲存結果至 output/results/...")
    panel.to_parquet(config.RESULT_DIR / "panel_all.parquet", index=False)
    panel_train.to_parquet(config.RESULT_DIR / "panel_train.parquet", index=False)
    panel_holdout.to_parquet(config.RESULT_DIR / "panel_holdout.parquet", index=False)

    # 描述性統計
    summary = panel.describe().T
    summary.to_csv(config.RESULT_DIR / "data_summary.csv")

    print("\n✓ 前處理完成。")
    _print_stats(panel)


def _quality_check(panel: pd.DataFrame):
    """基本資料品質確認。"""
    # 溫度範圍（台灣7月合理範圍）
    t_min, t_max = panel["t2m_c"].min(), panel["t2m_c"].max()
    assert 15 < t_min < 40, f"氣溫最小值異常: {t_min:.2f}°C"
    assert 20 < t_max < 40, f"氣溫最大值異常: {t_max:.2f}°C"

    # 負荷範圍（台灣7月合理範圍，MW）
    load_min, load_max = panel["load_mw"].min(), panel["load_mw"].max()
    assert 15000 < load_min, f"負荷最小值異常: {load_min:.0f} MW"
    assert load_max < 50000, f"負荷最大值異常: {load_max:.0f} MW"

    # 每年應有 24×31 = 744 筆
    n_per_year = panel.groupby(panel["datetime"].dt.year).size()
    for yr, n in n_per_year.items():
        if n != 744:
            print(f"  ⚠ {yr} 年資料筆數: {n}（預期744）")
        else:
            print(f"  ✓ {yr}: {n} 筆（24小時×31天）")

    # HotDegree應均為非負
    assert (panel["hotdeg"] >= 0).all(), "HotDegree 出現負值"
    pct_hot = (panel["hotdeg"] > 0).mean() * 100
    print(f"  ✓ 超過21°C的小時占比: {pct_hot:.1f}%")


def _print_stats(panel: pd.DataFrame):
    """印出關鍵描述性統計。"""
    print("\n─── 關鍵統計摘要 ───")
    print(f"  氣溫 (°C)   : mean={panel['t2m_c'].mean():.2f}, "
          f"min={panel['t2m_c'].min():.2f}, max={panel['t2m_c'].max():.2f}")
    print(f"  HotDegree    : mean={panel['hotdeg'].mean():.2f}, "
          f"max={panel['hotdeg'].max():.2f}")
    print(f"  負荷 (MW)   : mean={panel['load_mw'].mean():,.0f}, "
          f"min={panel['load_mw'].min():,.0f}, max={panel['load_mw'].max():,.0f}")

    print("\n─── 逐小時平均負荷（驗證日內形態）───")
    hourly_mean = panel.groupby("hour")["load_mw"].mean()
    for h, v in hourly_mean.items():
        bar = "█" * int(v / 1000)
        print(f"  {h:02d}:00  {v:7,.0f} MW  {bar}")


if __name__ == "__main__":
    main()

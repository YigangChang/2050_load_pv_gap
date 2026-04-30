"""
03_pv_model.py
==============
太陽能產量物理模型（PVWatts 風格）。

本模組為純函數庫，不直接執行。
供 04_baseline.py 與 05_monte_carlo.py 呼叫。

物理公式：
    T_module = T_air + (NOCT - 20) / 800 × SSRD
    P_output = PV_cap_MW × η × (SSRD / G_STC) × max(0, 1 + γ_p × (T_module - T_STC))

其中：
    NOCT  = 45°C  （標準操作模組溫度，隱含標準風速條件）
    γ_p   = -0.0045 /°C  （多晶矽溫度係數）
    η     = 0.80  （系統效率，含逆變器損失）
    G_STC = 1000 W/m²  （標準測試條件輻射）
    T_STC = 25°C  （標準測試條件溫度）

說明：
    (SSRD / G_STC) 項使 PV 在夜間（SSRD=0）自動輸出 0，
    並將日間輸出對應實際輻射強度。
"""

import numpy as np
import config


# ─────────────────────────────────────────────────────────────────────────────
# 核心物理函數
# ─────────────────────────────────────────────────────────────────────────────

def compute_t_module(
    temp_air_c: np.ndarray,
    ssrd_wm2:   np.ndarray,
    noct:       float = config.NOCT,
) -> np.ndarray:
    """
    計算 PV 模組溫度（°C）。

    Parameters
    ----------
    temp_air_c : array-like, 環境氣溫（°C）
    ssrd_wm2   : array-like, 短波向下輻射（W/m²）
    noct       : float, 標準操作模組溫度（°C），預設45°C

    Returns
    -------
    T_module : np.ndarray, 模組溫度（°C）
    """
    temp_air_c = np.asarray(temp_air_c, dtype=float)
    ssrd_wm2   = np.asarray(ssrd_wm2,   dtype=float)
    t_module   = temp_air_c + (noct - 20.0) / 800.0 * ssrd_wm2
    return t_module


def compute_pv_output(
    temp_air_c:     np.ndarray,
    ssrd_wm2:       np.ndarray,
    pv_capacity_gw: float,
    eta:            float = config.ETA_SYSTEM,
    gamma_p:        float = config.GAMMA_P,
    noct:           float = config.NOCT,
    g_stc:          float = config.G_STC,
    t_stc:          float = config.T_STC,
) -> np.ndarray:
    """
    計算 PV 逐時輸出（MW）。

    Parameters
    ----------
    temp_air_c     : array-like, 環境氣溫（°C）
    ssrd_wm2       : array-like, 短波向下輻射（W/m²）
    pv_capacity_gw : float, PV 裝置容量（GW），自動轉為MW
    eta            : float, 系統整體效率
    gamma_p        : float, 溫度係數（/°C），負值
    noct           : float, 標準操作模組溫度（°C）
    g_stc          : float, STC 標準輻射（W/m²）
    t_stc          : float, STC 標準溫度（°C）

    Returns
    -------
    pv_mw : np.ndarray, PV 輸出（MW）
    """
    temp_air_c = np.asarray(temp_air_c, dtype=float)
    ssrd_wm2   = np.asarray(ssrd_wm2,   dtype=float)

    pv_capacity_mw = pv_capacity_gw * 1000.0   # GW → MW

    t_module = compute_t_module(temp_air_c, ssrd_wm2, noct)

    # 溫度效率因子（不可為負）
    temp_factor = 1.0 + gamma_p * (t_module - t_stc)
    temp_factor = np.maximum(temp_factor, 0.0)

    # 輻射比例因子（夜間=0，日間依輻射強度縮放）
    irr_factor = ssrd_wm2 / g_stc
    irr_factor = np.maximum(irr_factor, 0.0)   # 防止負值（數值誤差）

    pv_mw = pv_capacity_mw * eta * irr_factor * temp_factor
    return pv_mw


# ─────────────────────────────────────────────────────────────────────────────
# 批次計算（多容量情境，向量化）
# ─────────────────────────────────────────────────────────────────────────────

def compute_pv_output_multi_capacity(
    temp_air_c:      np.ndarray,
    ssrd_wm2:        np.ndarray,
    capacities_gw:   list = None,
    **kwargs,
) -> np.ndarray:
    """
    同時計算多個裝置容量情境的 PV 輸出。

    Returns
    -------
    pv_mw : np.ndarray, shape (..., n_capacities)
        最後一個維度對應各容量情境（40/60/80 GW）
    """
    if capacities_gw is None:
        capacities_gw = config.PV_CAPACITIES_GW

    # 先計算 1 GW 情境，再乘以各容量（因為輸出與容量線性關係）
    pv_per_gw = compute_pv_output(temp_air_c, ssrd_wm2, pv_capacity_gw=1.0, **kwargs)

    # stack: (..., n_cap)
    pv_mw = np.stack(
        [pv_per_gw * cap for cap in capacities_gw],
        axis=-1,
    )
    return pv_mw


# ─────────────────────────────────────────────────────────────────────────────
# 快速測試（run_all.py 步驟3 呼叫，或直接執行本模組時）
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """PVWatts 模型功能測試（供 run_all.py 呼叫）。"""
    print("=" * 60)
    print("03 PV 模型函數測試")
    print("=" * 60)

    # 典型測試值
    test_cases = [
        ("夜間 (SSRD=0)",       25.0,   0.0),
        ("陰天 (SSRD=200)",     30.0, 200.0),
        ("正午 (SSRD=900)",     35.0, 900.0),
        ("高溫正午 (SSRD=950)", 38.0, 950.0),
    ]

    print("\n  典型情境測試：")
    for label, t_air, ssrd in test_cases:
        t_mod   = compute_t_module(t_air, ssrd)
        pv_1gw  = compute_pv_output(t_air, ssrd, pv_capacity_gw=1.0)
        pv_40gw = compute_pv_output(t_air, ssrd, pv_capacity_gw=40.0)
        print(f"  [{label}]")
        print(f"    T_air={t_air}°C, SSRD={ssrd} W/m²")
        print(f"    T_module={t_mod:.2f}°C")
        print(f"    PV @ 1 GW  = {pv_1gw:>8.2f} MW  (CF: {pv_1gw/1000:.3f})")
        print(f"    PV @ 40 GW = {pv_40gw:>8.2f} MW  (CF: {pv_40gw/40000:.3f})")
        print()

    # 溫度效應示範：相同輻射，不同氣溫
    print("  溫度效應（SSRD=800 W/m²，PV=60 GW）：")
    for t in [25, 30, 35, 38, 42]:
        pv = compute_pv_output(t, 800.0, 60.0)
        print(f"    T_air={t}°C → {pv:,.1f} MW")

    print("\n✓ 03 PV 模型測試完成。")


if __name__ == "__main__":
    main()

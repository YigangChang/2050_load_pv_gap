"""
utils.py
========
共用I/O與資料轉換工具函數。
所有原始CSV的讀取邏輯集中於此，其他模組呼叫這些函數即可。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. ERA5-Land 歷史氣候（2018-2022）
# ─────────────────────────────────────────────────────────────────────────────

def load_era5_land(year: int) -> pd.DataFrame:
    """
    載入單一年份的ERA5-Land 7月逐時資料。

    原始欄位：
        taipei_time        : YYYY-MM-DD HH:MM:SS（字串）
        averaged_t2m       : 2公尺氣溫，Kelvin
        averaged_ssrd_w_m2 : 地表向下短波輻射，W/m²

    回傳欄位（標準化）：
        datetime : pd.Timestamp（台灣時間）
        t2m_c    : 氣溫，°C
        ssrd     : 短波輻射，W/m²
        year, month, day, hour, weekday, trend_norm
    """
    fpath = config.ERA5_DIR / f"era5land_tw_{year}_07.csv"
    df = pd.read_csv(fpath, parse_dates=["taipei_time"])
    df = df.rename(columns={
        "taipei_time":        "datetime",
        "averaged_t2m":       "t2m_k",
        "averaged_ssrd_w_m2": "ssrd",
    })
    df["t2m_c"] = df["t2m_k"] - 273.15      # Kelvin → Celsius
    df = df.drop(columns=["t2m_k"])
    df = _add_time_features(df, "datetime")
    return df[["datetime", "t2m_c", "ssrd", "year", "month", "day", "hour", "weekday"]]


def load_era5_land_all(years: List[int] = None) -> pd.DataFrame:
    """載入並合併多年ERA5-Land資料（預設：ALL_YEARS）。"""
    if years is None:
        years = config.ALL_YEARS
    frames = [load_era5_land(y) for y in years]
    return pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 台電逐時負荷（2018-2022，橫向格式）
# ─────────────────────────────────────────────────────────────────────────────

def load_load_data(year: int) -> pd.DataFrame:
    """
    載入單一年份的台電7月逐時負荷（橫向格式）。

    原始格式：
        - 第一欄（無標頭）：小時字串 "00:00", "01:00", ..., "23:00"
        - 其餘欄：日期字串 "7/1/YYYY", "7/2/YYYY", ...

    回傳欄位：
        datetime : pd.Timestamp
        load_mw  : 負荷，MW
        year, month, day, hour, weekday
    """
    fpath = config.LOAD_DIR / f"{year}07_hourly.csv"
    df_wide = pd.read_csv(fpath, index_col=0)
    # index = "00:00"..."23:00"，欄 = "7/1/2018"..."7/31/2018"
    df_wide.index.name = "hour_str"
    df_wide.columns.name = "date_str"

    # 轉長格式
    df_long = df_wide.stack().reset_index()
    df_long.columns = ["hour_str", "date_str", "load_mw"]

    # 組合 datetime
    df_long["date"]     = pd.to_datetime(df_long["date_str"])
    df_long["hour"]     = df_long["hour_str"].str[:2].astype(int)
    df_long["datetime"] = df_long["date"] + pd.to_timedelta(df_long["hour"], unit="h")
    df_long["load_mw"]  = pd.to_numeric(df_long["load_mw"], errors="coerce")

    df_long = _add_time_features(df_long, "datetime")
    return df_long[["datetime", "load_mw", "year", "month", "day", "hour", "weekday"]]


def load_load_data_all(years: List[int] = None) -> pd.DataFrame:
    """載入並合併多年負荷資料（預設：ALL_YEARS）。"""
    if years is None:
        years = config.ALL_YEARS
    frames = [load_load_data(y) for y in years]
    return pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TCCIP合成氣候（未來 2041-2060 / 驗證 1995-2014）
# ─────────────────────────────────────────────────────────────────────────────

def load_synthetic_climate(model: str, ssp: str) -> pd.DataFrame:
    """
    載入TCCIP未來投影合成氣候（2041-2060）。

    原始欄位：time, temp_air_c, ssrd_wm2, ssrd_wm2_rad_minus_5, ssrd_wm2_rad_plus_5

    回傳欄位：
        datetime : pd.Timestamp
        t2m_c    : 氣溫（°C，已為攝氏，直接使用）
        ssrd     : 短波輻射（W/m²）
        model, ssp, year, month, day, hour
    """
    fpath = config.FUTURE_DIR / model / ssp / "July_hourly_synthetic_weather.csv"
    df = pd.read_csv(fpath, parse_dates=["time"])
    df = df.rename(columns={
        "time":       "datetime",
        "temp_air_c": "t2m_c",
        "ssrd_wm2":   "ssrd",
    })
    df = _add_time_features(df, "datetime")
    df["model"] = model
    df["ssp"]   = ssp
    return df[["datetime", "t2m_c", "ssrd", "model", "ssp", "year", "month", "day", "hour"]]


def load_all_future_climate() -> pd.DataFrame:
    """載入所有模型 × SSP 的未來氣候，合併為單一DataFrame。"""
    frames = []
    for model in config.CLIMATE_MODELS:
        for ssp in config.SSPS:
            frames.append(load_synthetic_climate(model, ssp))
    return pd.concat(frames, ignore_index=True)


def load_synthetic_climate_valid(model: str) -> pd.DataFrame:
    """
    載入驗證期合成氣候（1995-2014）。
    注意：MRI-ESM2-0 在資料夾中命名為 AR6_MRI-ESM2-0。
    """
    folder_name = config.VALID_MODEL_MAP[model]
    fpath = config.SYNTH_VALID_DIR / folder_name / "July_hourly_synthetic_weather.csv"
    df = pd.read_csv(fpath, parse_dates=["time"])
    df = df.rename(columns={
        "time":       "datetime",
        "temp_air_c": "t2m_c",
        "ssrd_wm2":   "ssrd",
    })
    df = _add_time_features(df, "datetime")
    df["model"] = model
    return df[["datetime", "t2m_c", "ssrd", "model", "year", "month", "day", "hour"]]


# ─────────────────────────────────────────────────────────────────────────────
# 4. ERA5 驗證資料（1995-2014，已為°C）
# ─────────────────────────────────────────────────────────────────────────────

def load_era5_validation() -> pd.DataFrame:
    """
    載入ERA5 1995-2014年7月實測資料（用於驗證合成氣候品質）。

    原始欄位：valid_time_tw, t2m_c（°C）, ssrd_wm2（W/m²）
    """
    df = pd.read_csv(config.ERA5_VALID_FILE, parse_dates=["valid_time_tw"])
    df = df.rename(columns={
        "valid_time_tw": "datetime",
        "ssrd_wm2":      "ssrd",
    })
    df = _add_time_features(df, "datetime")
    return df[["datetime", "t2m_c", "ssrd", "year", "month", "day", "hour"]]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Panel 資料集建構（ERA5 + 負荷 合併）
# ─────────────────────────────────────────────────────────────────────────────

def build_panel(years: List[int] = None) -> pd.DataFrame:
    """
    合併ERA5-Land與台電負荷，計算HotDegree與趨勢變數。

    回傳欄位：
        datetime, t2m_c, ssrd, load_mw,
        hotdeg,       ← max(0, t2m_c - T_THRESHOLD)
        hour, weekday,
        trend         ← 以 2018-07-01 00:00 為起點的小時序號（正規化至[0,1]）
    """
    if years is None:
        years = config.ALL_YEARS

    era5 = load_era5_land_all(years)
    load = load_load_data_all(years)

    # 合併（以 datetime 為 key）
    panel = pd.merge(era5, load[["datetime", "load_mw"]], on="datetime", how="inner")
    panel = panel.dropna(subset=["load_mw", "t2m_c"]).reset_index(drop=True)

    # HotDegree（Cooling Degree Hour，基準21°C）
    panel["hotdeg"] = np.maximum(0.0, panel["t2m_c"] - config.T_THRESHOLD)

    # 趨勢變數：以整個訓練期的小時序號正規化至 [0, 1]
    panel = panel.sort_values("datetime").reset_index(drop=True)
    n = len(panel)
    panel["trend"] = np.arange(n) / max(n - 1, 1)

    cols = ["datetime", "t2m_c", "ssrd", "load_mw",
            "hotdeg", "hour", "weekday", "trend"]
    return panel[cols]


# ─────────────────────────────────────────────────────────────────────────────
# 6. 內部工具函數
# ─────────────────────────────────────────────────────────────────────────────

def _add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """從 datetime 欄位抽取 year, month, day, hour, weekday。"""
    dt = df[dt_col]
    df["year"]    = dt.dt.year
    df["month"]   = dt.dt.month
    df["day"]     = dt.dt.day
    df["hour"]    = dt.dt.hour
    df["weekday"] = dt.dt.weekday   # 0=Monday, 6=Sunday
    return df


def hotdeg_from_temp(t2m_c: np.ndarray, threshold: float = None) -> np.ndarray:
    """計算 HotDegree = max(0, T - threshold)。"""
    if threshold is None:
        threshold = config.T_THRESHOLD
    return np.maximum(0.0, t2m_c - threshold)

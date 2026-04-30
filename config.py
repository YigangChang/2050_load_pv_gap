"""
config.py
=========
集中管理所有路徑、物理參數、統計參數。
所有模組皆從此檔案匯入常數，避免硬編碼散落在各處。
"""

from pathlib import Path

# ── 專案根目錄 ──────────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\Yifang\Desktop\Load_temp_function")

# ── 輸入資料路徑 ─────────────────────────────────────────────────────────────
ERA5_DIR        = BASE_DIR / "clean_era5land_July_2018_2022"
LOAD_DIR        = BASE_DIR / "cleaned_hourly_load_July"
FUTURE_DIR      = BASE_DIR / "2041_2060_tccip_ar6_temperature_July"
ERA5_VALID_FILE = BASE_DIR / "ERA5_1995_2014_July.csv"
SYNTH_VALID_DIR = BASE_DIR / "hourly_temp_ssdr_1995_2014_validation_July"

# ── 輸出路徑 ──────────────────────────────────────────────────────────────────
OUTPUT_DIR  = BASE_DIR / "output"
RESULT_DIR  = OUTPUT_DIR / "results"
FIGURE_DIR  = OUTPUT_DIR / "figures"

# 自動建立輸出資料夾
for _d in [RESULT_DIR, FIGURE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── 氣候模型設定 ──────────────────────────────────────────────────────────────
CLIMATE_MODELS = [
    "GFDL-ESM4",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "TaiESM1",
]
SSPS = ["ssp245", "ssp585"]

# 驗證期合成氣候的模型資料夾名稱（MRI 命名不同）
VALID_MODEL_MAP = {
    "GFDL-ESM4":     "GFDL-ESM4",
    "MIROC6":        "MIROC6",
    "MPI-ESM1-2-HR": "MPI-ESM1-2-HR",
    "MRI-ESM2-0":    "AR6_MRI-ESM2-0",
    "TaiESM1":       "TaiESM1",
}

# ── 訓練 / 持出年份 ───────────────────────────────────────────────────────────
TRAIN_YEARS   = [2018, 2019, 2020, 2021]   # 用於回歸訓練
HOLDOUT_YEAR  = 2022                        # 用於模型驗證（hold-out）
ALL_YEARS     = TRAIN_YEARS + [HOLDOUT_YEAR]

# ── 物理模型參數（PVWatts）────────────────────────────────────────────────────
NOCT        = 45.0    # °C，標準操作模組溫度
GAMMA_P     = -0.0045 # /°C，多晶矽溫度係數（-0.45%/°C）
ETA_SYSTEM  = 0.80    # 系統整體效率（含逆變器損失）
G_STC       = 1000.0  # W/m²，標準測試條件輻射強度
T_STC       = 25.0    # °C，標準測試條件溫度

# ── PV裝置容量情境（GW）──────────────────────────────────────────────────────
PV_CAPACITIES_GW = [40, 60, 80]   # 國發會2050淨零路徑目標區間

# ── 需求端模型參數 ────────────────────────────────────────────────────────────
T_THRESHOLD   = 21.0  # °C，HotDegree計算基準溫度
B_BOOTSTRAP   = 1000  # Bootstrap 重抽樣次數
RANDOM_SEED   = 42    # 亂數種子（重現性）

# ── 蒙地卡羅參數 ──────────────────────────────────────────────────────────────
N_MC          = 5000  # Monte Carlo 模擬次數（一般模擬用）
K_LOW         = 1.58  # 結構性負荷調整係數下界（國發會低需求帶）
K_HIGH        = 2.11  # 結構性負荷調整係數上界（國發會高需求帶）

# ── 輻射不確定性參數 ──────────────────────────────────────────────────────────
# TCCIP合成氣候的SSRD是從ERA5歷史資料重建，非氣候模式真正的未來輻射投影。
# 資料已提供 ±5% 的不確定性區間（ssrd_wm2_rad_minus_5 / ssrd_wm2_rad_plus_5），
# 以乘數形式表達：ssrd_future = ssrd_central × rad_factor
RAD_LOW       = 0.95  # 輻射縮放因子下界（-5%）
RAD_HIGH      = 1.05  # 輻射縮放因子上界（+5%）

# ── Sobol 分析參數 ────────────────────────────────────────────────────────────
# 四個不確定性來源：K, β_h, 氣候年, 輻射縮放因子
# 實際評估次數 = N_SOBOL × (2×D + 2) = N_SOBOL × 10
N_SOBOL       = 1024  # Saltelli 基礎樣本數；D=4，共 N_SOBOL × 10 次評估

# ── 輸出分位數設定 ────────────────────────────────────────────────────────────
PERCENTILES   = [50, 75, 90, 95, 99]

import os
import pandas as pd

src = os.path.join(os.path.dirname(__file__), "ERA5_1995_2014_hourly_TW_OneValue.csv")
dst = os.path.join(os.path.dirname(__file__), "ERA5_1995_2014_July_hourly_TW_OneValue_converted.csv")

# --- Load & sort by UTC time ---
df = pd.read_csv(src, parse_dates=["valid_time"])
df = df.sort_values("valid_time").reset_index(drop=True)

# --- Temperature: K -> C ---
df["t2m_c"] = df["t2m"] - 273.15

# --- Convert UTC -> Taipei time (UTC+8) first, as requested ---
df["valid_time_tw"] = df["valid_time"] + pd.Timedelta(hours=8)

# --- SSRD: accumulated J/m² -> W/m² ---
# ERA5 SSRD accumulates within each forecast cycle and resets at the start of a
# new cycle. Resets show up as a sharp *negative* drop in the raw ssrd series.
# Auto-detect resets via negative diffs — no hardcoded hours needed.
#   - At a reset row: the raw value is the first hour's accumulation → divide by 3600
#   - At all other rows: hourly increment = diff → divide by 3600
#   - Clip to 0 to remove tiny negative rounding artefacts at night

ssrd_diff = df["ssrd"].diff()                        # NaN for row 0
# Real forecast resets drop the accumulated value by ~1.5e7–2e7 J/m².
# Floating-point artefacts on plateau rows may produce tiny negative diffs
# (e.g. -0.001). Use a 1e6 threshold to distinguish real resets from artefacts.
is_reset  = ssrd_diff.isna() | (ssrd_diff < -1e6)   # first row OR real reset

df["ssrd_wm2"] = ssrd_diff / 3600
df.loc[is_reset, "ssrd_wm2"] = df.loc[is_reset, "ssrd"] / 3600
df["ssrd_wm2"] = df["ssrd_wm2"].clip(lower=0)

# --- Filter July (by Taipei time) ---
july = df[df["valid_time_tw"].dt.month == 7].copy()

# --- Select output columns ---
out = july[["valid_time_tw", "t2m_c", "ssrd_wm2"]].reset_index(drop=True)

# --- Save ---
out.to_csv(dst, index=False)

print(f"Rows written : {len(out)}")
print(f"Output file  : {dst}")

print("\nSample — nighttime rows (Taipei 20:00-23:00, should be 0 W/m²):")
night = out[out["valid_time_tw"].dt.hour.between(20, 23)].head(8)
print(night.to_string())

print("\nSample — peak daytime rows (Taipei 10:00-14:00):")
peak = out[out["valid_time_tw"].dt.hour.between(10, 14)].head(10)
print(peak.to_string())

print("\nSample — early morning (Taipei 06:00-09:00, rising solar):")
morn = out[out["valid_time_tw"].dt.hour.between(6, 9)].head(8)
print(morn.to_string())

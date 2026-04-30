"""
08_visualization.py
===================
論文圖表輸出（全部圖表集中在此模組）。

圖表清單：
    Fig 1. β_h 逐小時溫度係數（含Bootstrap 95% CI）
    Fig 2. 缺口增量分布圖（三個PV容量情境）
    Fig 3. SSP245 vs SSP585 P99 對比（箱型圖）
    Fig 4. PV容量敏感度（40/60/80 GW 的 P95/P99）
    Fig 5. 逐氣候模型的 July P99 分布
    Fig 6. 未來溫度分布右移（熱浪增頻說明圖）
    Fig 7. 極端缺口日的氣溫分布（敘事一致性檢查）
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import config
import utils

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family":        "sans-serif",
        "axes.unicode_minus": False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":         120,
    })
    MPL_OK = True
except ImportError:
    MPL_OK = False
    print("⚠ matplotlib not installed: pip install matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Hourly temperature-load coefficients β_h
# ─────────────────────────────────────────────────────────────────────────────

def fig1_beta_hourly():
    beta_point = np.load(config.RESULT_DIR / "beta_point.npy")
    beta_boot  = np.load(config.RESULT_DIR / "beta_bootstrap.npy")

    lo = np.percentile(beta_boot, 2.5,  axis=0)
    hi = np.percentile(beta_boot, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    hours = np.arange(24)

    ax.fill_between(hours, lo, hi, alpha=0.25, color="#2196F3",
                    label="Bootstrap 95% CI")
    ax.plot(hours, beta_point, "o-", color="#1565C0", linewidth=2,
            markersize=5, label="Point estimate $\\hat{\\beta}_h$")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Hour (Taiwan Standard Time)", fontsize=11)
    ax.set_ylabel("Temperature-Load Coefficient $\\hat{\\beta}_h$ (MW/°C)", fontsize=11)
    ax.set_title("Fig 1: Hourly Temperature-Load Coefficients (2018–2021 Panel OLS)",
                 fontsize=12)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=30)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save("fig1_beta_hourly.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: DailyMaxGapIncrease distribution (three PV capacities)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_gap_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for ax, cap, color in zip(axes, config.PV_CAPACITIES_GW, colors):
        dm     = np.load(config.RESULT_DIR / f"mc_daily_max_gap_{cap}gw.npy")
        pooled = dm.ravel()

        ax.hist(pooled, bins=60, density=True, color=color,
                alpha=0.7, edgecolor="white")
        for pct, ls in [(95, "--"), (99, "-")]:
            val = np.percentile(pooled, pct)
            ax.axvline(val, color="black", linestyle=ls, linewidth=1.5,
                       label=f"P{pct} = {val:,.0f} MW")

        mean_val = pooled.mean()
        ax.axvline(mean_val, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Mean = {mean_val:,.0f} MW")

        ax.set_title(f"PV = {cap} GW", fontsize=11)
        ax.set_xlabel("DailyMaxGapIncrease (MW)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set_ylabel("Density", fontsize=10)
    fig.suptitle("Fig 2: DailyMaxGapIncrease Distribution (All Simulations)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("fig2_gap_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: SSP245 vs SSP585 tail-risk comparison (box plot)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_ssp_comparison():
    beta_boot = np.load(config.RESULT_DIR / "beta_bootstrap.npy")
    load_base = np.load(config.RESULT_DIR / "load_baseline.npy")
    hotd_base = np.load(config.RESULT_DIR / "hotdeg_baseline.npy")

    import importlib
    _pv = importlib.import_module("03_pv_model")
    compute_pv_output = _pv.compute_pv_output

    rng          = np.random.default_rng(config.RANDOM_SEED + 1)
    N_SUBSAMPLE  = 1000
    CAP          = 60
    pv_base      = np.load(config.RESULT_DIR / f"pv_baseline_{CAP}gw.npy")

    data_by_ssp = {ssp: [] for ssp in config.SSPS}

    for ssp in config.SSPS:
        years_ssp = []
        for model in config.CLIMATE_MODELS:
            df = utils.load_synthetic_climate(model, ssp)
            for yr in sorted(df["year"].unique()):
                df_yr  = df[df["year"] == yr].sort_values(["day", "hour"])
                t2m    = df_yr["t2m_c"].values.reshape(31, 24)
                ssrd   = df_yr["ssrd"].values.reshape(31, 24)
                hotdeg = np.maximum(0.0, t2m - config.T_THRESHOLD)
                years_ssp.append({"t2m": t2m, "ssrd": ssrd, "hotdeg": hotdeg})

        K_samples = rng.uniform(config.K_LOW, config.K_HIGH, N_SUBSAMPLE)
        for i in range(N_SUBSAMPLE):
            cy  = years_ssp[rng.integers(0, len(years_ssp))]
            bh  = beta_boot[rng.integers(0, len(beta_boot))]
            K   = K_samples[i]
            dhd = cy["hotdeg"] - hotd_base[np.newaxis, :]
            lgap = load_base[np.newaxis, :] * (K - 1) + bh[np.newaxis, :] * dhd
            pf  = compute_pv_output(
                cy["t2m"].ravel(), cy["ssrd"].ravel(), CAP
            ).reshape(31, 24)
            pg  = pv_base[np.newaxis, :] - pf
            dm  = (lgap + pg).max(axis=1)
            data_by_ssp[ssp].append(np.percentile(dm, 99))

    fig, ax = plt.subplots(figsize=(7, 5))
    bp_data = [data_by_ssp[ssp] for ssp in config.SSPS]
    bp = ax.boxplot(bp_data, patch_artist=True, widths=0.4,
                    labels=["SSP2-4.5", "SSP5-8.5"],
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], ["#64B5F6", "#EF5350"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("July P99 DailyMaxGapIncrease (MW)", fontsize=10)
    ax.set_title(
        f"Fig 3: Tail Risk Comparison — SSP2-4.5 vs SSP5-8.5 (PV = {CAP} GW)",
        fontsize=11)
    plt.tight_layout()
    _save("fig3_ssp_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: PV capacity sensitivity (40 / 60 / 80 GW)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_pv_sensitivity():
    fig, ax = plt.subplots(figsize=(8, 5))

    pct_labels = [50, 90, 95, 99]
    x          = np.arange(len(pct_labels))
    bar_width  = 0.25
    colors     = ["#90CAF9", "#2196F3", "#1565C0"]
    offsets    = [-bar_width, 0, bar_width]

    for cap, color, offset in zip(config.PV_CAPACITIES_GW, colors, offsets):
        pooled = np.load(
            config.RESULT_DIR / f"mc_daily_max_gap_{cap}gw.npy"
        ).ravel()
        vals = [np.percentile(pooled, p) for p in pct_labels]
        ax.bar(x + offset, vals, bar_width, label=f"{cap} GW",
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in pct_labels], fontsize=10)
    ax.set_ylabel("DailyMaxGapIncrease (MW)", fontsize=10)
    ax.set_title("Fig 4: PV Capacity Sensitivity (40 / 60 / 80 GW)", fontsize=11)
    ax.legend(title="PV Capacity", fontsize=9)
    plt.tight_layout()
    _save("fig4_pv_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: July P99 distribution by climate model (violin plot)
# ─────────────────────────────────────────────────────────────────────────────

def fig5_model_comparison():
    beta_boot = np.load(config.RESULT_DIR / "beta_bootstrap.npy")
    load_base = np.load(config.RESULT_DIR / "load_baseline.npy")
    hotd_base = np.load(config.RESULT_DIR / "hotdeg_baseline.npy")

    import importlib
    _pv = importlib.import_module("03_pv_model")
    compute_pv_output = _pv.compute_pv_output

    CAP      = 60
    N_SUB    = 300
    pv_base  = np.load(config.RESULT_DIR / f"pv_baseline_{CAP}gw.npy")
    rng      = np.random.default_rng(config.RANDOM_SEED + 2)

    model_p99 = {m: {ssp: [] for ssp in config.SSPS}
                 for m in config.CLIMATE_MODELS}

    for model in config.CLIMATE_MODELS:
        for ssp in config.SSPS:
            df       = utils.load_synthetic_climate(model, ssp)
            years_ms = []
            for yr in sorted(df["year"].unique()):
                df_yr  = df[df["year"] == yr].sort_values(["day", "hour"])
                t2m    = df_yr["t2m_c"].values.reshape(31, 24)
                ssrd   = df_yr["ssrd"].values.reshape(31, 24)
                hotdeg = np.maximum(0.0, t2m - config.T_THRESHOLD)
                years_ms.append({"t2m": t2m, "ssrd": ssrd, "hotdeg": hotdeg})
            for _ in range(N_SUB):
                cy   = years_ms[rng.integers(0, len(years_ms))]
                bh   = beta_boot[rng.integers(0, len(beta_boot))]
                K    = rng.uniform(config.K_LOW, config.K_HIGH)
                dhd  = cy["hotdeg"] - hotd_base[np.newaxis, :]
                lgap = load_base[np.newaxis, :] * (K - 1) + bh[np.newaxis, :] * dhd
                pf   = compute_pv_output(
                    cy["t2m"].ravel(), cy["ssrd"].ravel(), CAP
                ).reshape(31, 24)
                pg   = pv_base[np.newaxis, :] - pf
                dm   = (lgap + pg).max(axis=1)
                model_p99[model][ssp].append(np.percentile(dm, 99))

    fig, ax  = plt.subplots(figsize=(12, 5))
    positions   = np.arange(len(config.CLIMATE_MODELS))
    ssp_colors  = {"ssp245": "#64B5F6", "ssp585": "#EF5350"}
    ssp_offset  = {"ssp245": -0.2, "ssp585": 0.2}
    ssp_labels  = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}

    for ssp in config.SSPS:
        data = [model_p99[m][ssp] for m in config.CLIMATE_MODELS]
        vp   = ax.violinplot(data, positions=positions + ssp_offset[ssp],
                             widths=0.35, showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(ssp_colors[ssp])
            body.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(config.CLIMATE_MODELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("July P99 DailyMaxGapIncrease (MW)", fontsize=10)
    ax.set_title(f"Fig 5: July P99 Distribution by Climate Model (PV = {CAP} GW)",
                 fontsize=11)
    legend_elems = [
        mpatches.Patch(facecolor=ssp_colors[s], alpha=0.6, label=ssp_labels[s])
        for s in config.SSPS
    ]
    ax.legend(handles=legend_elems, fontsize=9)
    plt.tight_layout()
    _save("fig5_model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Temperature distribution shift (heatwave frequency increase)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_temperature_shift():
    era5_hist = utils.load_era5_land_all(config.ALL_YEARS)
    t_hist    = era5_hist["t2m_c"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    bins    = np.linspace(16, 36, 60)

    ax.hist(t_hist, bins=bins, density=True, color="steelblue", alpha=0.6,
            label="Historical (ERA5 2018–2022)")

    colors_future = plt.cm.Reds(np.linspace(0.4, 0.9, len(config.CLIMATE_MODELS)))
    for model, color in zip(config.CLIMATE_MODELS, colors_future):
        t_fut = []
        for ssp in config.SSPS:
            df = utils.load_synthetic_climate(model, ssp)
            t_fut.append(df["t2m_c"].values)
        t_fut = np.concatenate(t_fut)
        ax.hist(t_fut, bins=bins, density=True, histtype="step",
                linewidth=1.2, color=color, alpha=0.8, label=model)

    t_p95 = np.percentile(t_hist, 95)
    ax.axvline(t_p95, color="black", linestyle="--", linewidth=1.5,
               label=f"Historical P95 = {t_p95:.1f}°C")

    ax.set_xlabel("Hourly Temperature (°C)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Fig 6: Taiwan July Temperature Distribution Shift (Historical vs 2041–2060)",
        fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    _save("fig6_temperature_shift.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Temperature on extreme gap days (narrative consistency check)
# ─────────────────────────────────────────────────────────────────────────────

def fig7_gap_vs_temperature():
    dm_all   = np.load(config.RESULT_DIR / "mc_daily_max_gap_60gw.npy")  # (N_MC, 31)
    era5_all = utils.load_era5_land_all(config.ALL_YEARS)

    daily_temp    = era5_all.groupby(["year", "day"])["t2m_c"].mean().values
    daily_max_gap = dm_all.ravel()

    n             = min(len(daily_temp), len(daily_max_gap))
    daily_temp    = daily_temp[:n]
    daily_max_gap = daily_max_gap[:n]

    threshold_gap = np.percentile(daily_max_gap, 95)
    extreme_mask  = daily_max_gap >= threshold_gap
    normal_mask   = ~extreme_mask

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(daily_temp[normal_mask],  daily_max_gap[normal_mask],
               alpha=0.2, s=5, color="#2196F3", label="Normal Days (< P95)")
    ax.scatter(daily_temp[extreme_mask], daily_max_gap[extreme_mask],
               alpha=0.6, s=15, color="#F44336", label="Extreme Days (≥ P95)")
    ax.axhline(threshold_gap, color="black", linestyle="--", linewidth=1,
               label=f"P95 Threshold = {threshold_gap:,.0f} MW")

    ax.set_xlabel("Daily Mean Temperature (°C)", fontsize=10)
    ax.set_ylabel("DailyMaxGapIncrease (MW)", fontsize=10)
    ax.set_title(
        "Fig 7: Temperature Distribution on Extreme Gap Days (Narrative Consistency Check)",
        fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("fig7_gap_vs_temp.png")


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _save(filename: str):
    fpath = config.FIGURE_DIR / filename
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("08 Figure Output")
    print("=" * 60)

    if not MPL_OK:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    figs = [
        ("Fig 1: β_h Temperature Coefficients",  fig1_beta_hourly),
        ("Fig 2: Gap Increase Distribution",      fig2_gap_distribution),
        ("Fig 3: SSP Comparison",                 fig3_ssp_comparison),
        ("Fig 4: PV Capacity Sensitivity",        fig4_pv_sensitivity),
        ("Fig 5: Climate Model Comparison",       fig5_model_comparison),
        ("Fig 6: Temperature Distribution Shift", fig6_temperature_shift),
        ("Fig 7: Gap vs Temperature Consistency", fig7_gap_vs_temperature),
    ]

    for title, func in figs:
        print(f"\n[Plotting] {title}...")
        try:
            func()
        except FileNotFoundError as e:
            print(f"  ⚠ Skipped (prerequisite result not found): {e}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n✓ Figure output complete.")
    print(f"   Output directory: {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()

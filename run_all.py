"""
run_all.py
==========
依序執行全部分析流程。

使用方式：
    python run_all.py           # 執行全部步驟
    python run_all.py --step 2  # 只執行步驟2（回歸）
    python run_all.py --from 3  # 從步驟3開始執行

需要安裝的套件：
    pip install pandas numpy statsmodels scipy matplotlib SALib pyarrow
"""

import argparse
import sys
import time
import importlib
import traceback


STEPS = [
    (1, "資料前處理",         "01_data_preprocessing"),
    (2, "需求端回歸+Bootstrap","02_load_regression"),
    (3, "PV模型測試",          "03_pv_model"),
    (4, "基準值計算",          "04_baseline"),
    (5, "蒙地卡羅模擬",        "05_monte_carlo"),
    (6, "Sobol不確定性分解",   "06_sobol_analysis"),
    (7, "模型驗證",            "07_validation"),
    (8, "圖表輸出",            "08_visualization"),
]


def run_step(step_num: int, step_name: str, module_name: str) -> bool:
    """執行單一步驟，回傳是否成功。"""
    print("\n" + "=" * 70)
    print(f"步驟 {step_num}/8：{step_name}")
    print("=" * 70)
    t0 = time.time()

    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - t0
        print(f"\n✓ 步驟 {step_num} 完成（耗時 {elapsed:.1f}s）")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n✗ 步驟 {step_num} 失敗（耗時 {elapsed:.1f}s）")
        print(f"  錯誤：{type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="執行熱浪-再生能源尾端風險分析流程")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=int, help="只執行指定步驟（1-8）")
    group.add_argument("--from", dest="from_step", type=int,
                       help="從指定步驟開始執行（1-8）")
    group.add_argument("--skip", type=int, nargs="+",
                       help="跳過指定步驟（例如 --skip 6 7）")
    args = parser.parse_args()

    # 決定要執行哪些步驟
    if args.step:
        steps_to_run = [s for s in STEPS if s[0] == args.step]
    elif args.from_step:
        steps_to_run = [s for s in STEPS if s[0] >= args.from_step]
    elif args.skip:
        steps_to_run = [s for s in STEPS if s[0] not in args.skip]
    else:
        steps_to_run = STEPS

    print("=" * 70)
    print("熱浪增頻下台灣2050年再生能源系統可靠度尾端風險")
    print("分析流程啟動")
    print("=" * 70)
    print(f"\n將執行以下步驟：{[s[0] for s in steps_to_run]}")

    t_total = time.time()
    results = {}

    for step_num, step_name, module_name in steps_to_run:
        success = run_step(step_num, step_name, module_name)
        results[step_num] = success

        if not success:
            print(f"\n⚠ 步驟 {step_num} 失敗。後續步驟可能無法正確執行。")
            ans = input("是否繼續執行後續步驟？(y/n): ").strip().lower()
            if ans != "y":
                print("中止執行。")
                break

    # 最終摘要
    elapsed_total = time.time() - t_total
    print("\n" + "=" * 70)
    print("執行摘要")
    print("=" * 70)
    for step_num, step_name, _ in steps_to_run:
        status = "✓" if results.get(step_num) else "✗"
        print(f"  {status} 步驟 {step_num}：{step_name}")

    print(f"\n總耗時：{elapsed_total/60:.1f} 分鐘")

    if all(results.values()):
        print("\n✓ 全部步驟完成！")
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\n⚠ 以下步驟失敗：{failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()

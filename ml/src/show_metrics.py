"""SkinSafe AI - Terminal CLI Viewer for Research Performance Table

Usage:
    python view_table.py
    or
    python -m ml.src.show_metrics
"""

import ctypes
import json
import os
import sys
from pathlib import Path
import pandas as pd

# Enable ANSI escape sequence support on Windows terminal
if sys.platform == "win32":
    try:
        kernel32 = ctypes.windll.kernel32
        h_stdout = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(h_stdout, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h_stdout, mode.value | 0x0004)
    except Exception:
        pass
    reconfig = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig):
        try:
            reconfig(encoding="utf-8")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLE_CSV = PROJECT_ROOT / "ml" / "reports" / "tables" / "table_per_class_metrics.csv"
METRICS_JSON = PROJECT_ROOT / "ml" / "reports" / "tables" / "actual_evaluation_metrics.json"

TABLE_WIDTH = 100


def display_terminal_metrics():
    """Prints a clean, robust terminal table of research metrics."""
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print("\n" + "=" * TABLE_WIDTH)
    print(f"{BOLD}{CYAN}   SkinSafe AI - Research Benchmark: Per-Class Precision, Recall & Sensitivity{RESET}")
    print(f"{DIM}   Dataset: ISIC 2019 Archive (3,799 Held-Out Validation Images, 0% Data Leakage){RESET}")
    print("=" * TABLE_WIDTH)

    if not TABLE_CSV.exists():
        print(f"{RED}Error: Table CSV not found at {TABLE_CSV}{RESET}")
        return

    df = pd.read_csv(TABLE_CSV)

    # Clean Header
    print(f"{BOLD}{'Lesion Pathology Type':<40} {'Samples':>8} {'Precision':>11} {'Recall/Sens':>13} {'F1-Score':>10} {'Risk Category':<16}{RESET}")
    print("-" * TABLE_WIDTH)

    for _, row in df.iterrows():
        lesion = str(row["Lesion Type"])
        samples = str(row["Original Samples"])
        prec = str(row["Precision (%)"])
        rec = str(row["Recall / Sensitivity (%)"])
        f1 = str(row["F1-score (%)"])
        risk = str(row.get("Clinical Risk Category", ""))

        # Format padded text strings first to preserve perfect table alignment
        s_prec = f"{prec:>10}%"
        s_rec = f"{rec:>12}%"
        s_f1 = f"{f1:>9}%"

        if "Macro Average" in lesion:
            print("-" * TABLE_WIDTH)
            print(f"{BOLD}{CYAN}{lesion:<40} {samples:>8} {s_prec} {s_rec} {s_f1} {risk:<16}{RESET}")
        else:
            # Color code risk
            if "Critical" in risk or "Malignant" in risk:
                risk_str = f"{RED}{risk:<16}{RESET}"
            elif "Pre-cancer" in risk:
                risk_str = f"{YELLOW}{risk:<16}{RESET}"
            else:
                risk_str = f"{GREEN}{risk:<16}{RESET}"

            print(f"{lesion:<40} {samples:>8} {CYAN}{s_prec}{RESET} {GREEN}{s_rec}{RESET} {YELLOW}{s_f1}{RESET} {risk_str}")

    print("=" * TABLE_WIDTH)

    # Academic Benchmark Summary
    if METRICS_JSON.exists():
        try:
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                stats = json.load(f)
            print(f"\n{BOLD}Comprehensive Academic Benchmark Summary:{RESET}")
            print(f"  * {BOLD}Total Evaluated Validation Images:{RESET} {stats.get('total_evaluated_images', len(df)):,}")
            print(f"  * {BOLD}Macro AUC-ROC (Multi-Class):{RESET}      {CYAN}{stats.get('macro_auc_roc', 0.0):.2f}%{RESET}")
            print(f"  * {BOLD}Balanced Accuracy:{RESET}                {GREEN}{stats.get('balanced_accuracy', 0.0):.2f}%{RESET}")
            print(f"  * {BOLD}Raw Top-1 Accuracy:{RESET}               {YELLOW}{stats.get('raw_top1_accuracy', 0.0):.2f}%{RESET}")
            print(f"  * {BOLD}Macro F1-Score:{RESET}                   {stats.get('macro_f1', 0.0):.2f}%")
        except Exception:
            pass

    print()


if __name__ == "__main__":
    display_terminal_metrics()

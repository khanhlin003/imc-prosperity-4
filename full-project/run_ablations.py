"""Quick ablation runner for Evelyn_r3.py.

Patches a few configuration constants in a copy of the strategy file,
backtests on round 3 (all 3 days, merged) and prints a per-variant
PnL breakdown.  Restores the original file when done.
"""
import os, re, shutil, subprocess, sys, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SRC  = ROOT / "Evelyn_r3.py"
TMP  = ROOT / "Evelyn_r3_variant.py"
PYBT = ROOT / ".venv" / "bin" / "prosperity3bt"

_BEST = {"IV_SCALPING_THR": 0.05, "THR_OPEN": 0.2,
         "VEV_SCALPING_STRIKES": "['VEV_5100', 'VEV_5200', 'VEV_5400']"}

VARIANTS = {
    "drop35_open02":           dict(_BEST),
    "drop35_open015":          {**_BEST, "THR_OPEN": 0.15},
    "drop35_open01":           {**_BEST, "THR_OPEN": 0.10},
    "drop35_open005":          {**_BEST, "THR_OPEN": 0.05},
    "drop35_open01_close+":    {**_BEST, "THR_OPEN": 0.10, "THR_CLOSE": 0.2},
    "drop35_open01_close+02_act0": {**_BEST, "THR_OPEN": 0.10, "THR_CLOSE": 0.2, "IV_SCALPING_THR": 0.0},

    # try with 5500 in (it didn't trade much before but maybe at lower threshold)
    "drop3_only_open01":       {"IV_SCALPING_THR": 0.05, "THR_OPEN": 0.10,
                                "VEV_SCALPING_STRIKES":
                                  "['VEV_5100', 'VEV_5200', 'VEV_5400', 'VEV_5500']"},

    # try without 5400 (small loss)
    "drop345_open01":          {"IV_SCALPING_THR": 0.05, "THR_OPEN": 0.10,
                                "VEV_SCALPING_STRIKES":
                                  "['VEV_5100', 'VEV_5200']"},

    # try just the two big winners 5100 + 5200
    "only_5100_5200":          {"IV_SCALPING_THR": 0.05, "THR_OPEN": 0.10, "THR_CLOSE": 0.2,
                                "VEV_SCALPING_STRIKES":
                                  "['VEV_5100', 'VEV_5200']"},
}


def patch(text: str, overrides: dict[str, object]) -> str:
    for k, v in overrides.items():
        # match the whole assignment, including continuation lines whose
        # body is indented (e.g. multi-line list literal for VEV_ACTIVE_STRIKES).
        # We grab from the start of the assignment up to (but not including)
        # the next line that starts in column 0 with non-space.
        rx = re.compile(rf"^{k}\s*=.*?(?=\n[^ \t#])", re.M | re.S)
        if isinstance(v, str):
            repl = f"{k} = {v}"
        else:
            repl = f"{k} = {v!r}"
        new_text, n = rx.subn(repl, text, count=1)
        if n == 0:
            # fallback: single-line
            rx = re.compile(rf"^{k}\s*=.*$", re.M)
            new_text, n = rx.subn(repl, text, count=1)
            if n == 0:
                raise RuntimeError(f"could not patch {k}")
        text = new_text
    return text


def parse_summary(stdout: str):
    out = {}
    cur = None
    in_summary = False
    for line in stdout.splitlines():
        if line.startswith("Profit summary"):
            in_summary = True
            cur = None
            continue
        if in_summary:
            continue
        m = re.match(r"^Backtesting .* on round (\d+) day (\d+)", line)
        if m:
            cur = f"day{m.group(2)}"
            out[cur] = {}
            continue
        m = re.match(r"^([A-Z_0-9]+):\s*(-?[\d,]+)$", line)
        if m and cur is not None:
            out[cur][m.group(1)] = int(m.group(2).replace(",", ""))
        m = re.match(r"^Total profit:\s*(-?[\d,]+)", line)
        if m and cur is not None:
            out[cur]["TOTAL"] = int(m.group(1).replace(",", ""))
    return out


def run_one(name: str, overrides: dict[str, object]):
    base = SRC.read_text()
    patched = patch(base, overrides) if overrides else base
    TMP.write_text(patched)

    cmd = [str(PYBT), str(TMP), "3", "--data", "./data",
           "--no-progress", "--merge-pnl", "--no-out"]
    print(f"\n=== {name}  overrides={overrides} ===", flush=True)
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    stdout = res.stdout
    if res.returncode != 0:
        print("BACKTESTER FAILED", res.returncode)
        print(stdout[-2000:])
        print(res.stderr[-1000:])
        return None
    parsed = parse_summary(stdout)
    return parsed


def main():
    results = {}
    for name, ov in VARIANTS.items():
        results[name] = run_one(name, ov)

    if TMP.exists():
        TMP.unlink()

    # Per-variant per-product table
    products = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT",
                "VEV_4000", "VEV_4500",
                "VEV_5000", "VEV_5100", "VEV_5200",
                "VEV_5300", "VEV_5400", "VEV_5500"]

    print("\n\n=========  PER-DAY TOTALS  =========")
    print(f"{'variant':<22} {'d0':>10} {'d1':>10} {'d2':>10} {'sum':>10}")
    for name, days in results.items():
        if not days: continue
        d0 = days.get("day0", {}).get("TOTAL", 0)
        d1 = days.get("day1", {}).get("TOTAL", 0)
        d2 = days.get("day2", {}).get("TOTAL", 0)
        s  = d0 + d1 + d2
        print(f"{name:<22} {d0:>10,} {d1:>10,} {d2:>10,} {s:>10,}")

    print("\n\n=========  PER-PRODUCT (3-day sum)  =========")
    header = f"{'variant':<22} " + " ".join(f"{p[:8]:>9}" for p in products) + f" {'TOTAL':>10}"
    print(header)
    for name, days in results.items():
        if not days: continue
        agg = {p: 0 for p in products}
        total = 0
        for d in days.values():
            for p in products:
                agg[p] += d.get(p, 0)
            total += d.get("TOTAL", 0)
        line = f"{name:<22} " + " ".join(f"{agg[p]:>9,}" for p in products) + f" {total:>10,}"
        print(line)


if __name__ == "__main__":
    main()

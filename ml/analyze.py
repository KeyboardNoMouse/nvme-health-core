"""Failure-pattern analysis: rank observed failure modes, profile each, and quantify relationships.
Writes web/public/model/patterns.json."""
import json, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("../data/NVMe_Drive_Failure_Dataset.csv")
OUT = Path("../web/public/model/patterns.json")
COLS = ["Power_On_Hours", "Percent_Life_Used", "Total_TBW_TB", "Temperature_C", "Media_Errors",
        "Unsafe_Shutdowns", "CRC_Errors", "Read_Error_Rate", "Write_Error_Rate"]
LABEL = {"Power_On_Hours": "Power-on hours", "Percent_Life_Used": "Life used %", "Total_TBW_TB": "TB written",
         "Temperature_C": "Temperature °C", "Media_Errors": "Media errors", "Unsafe_Shutdowns": "Unsafe shutdowns",
         "CRC_Errors": "CRC errors", "Read_Error_Rate": "Read error rate", "Write_Error_Rate": "Write error rate"}
MODES = {1: ("Wear-out", "Flash nearing end of life: very high TBW and life used."),
         2: ("Thermal", "Sustained high temperature (typically above 70°C) causing error spikes."),
         3: ("Power-related", "Repeated unsafe shutdowns leading to corruption or CRC errors."),
         4: ("Controller / firmware", "Failures tied to specific firmware versions or controller instability."),
         5: ("Early-life", "High error rates in the first 3,000 hours, likely manufacturing defects.")}

healthy = df[df.Failure_Mode == 0]
fails = df[df.Failure_Flag == 1]
h_mean = healthy[COLS].mean()

patterns, missing = [], []
for mode, (name, desc) in MODES.items():
    g = df[df.Failure_Mode == mode]
    if len(g) == 0:
        missing.append(dict(mode=mode, name=name, desc=desc)); continue
    m = g[COLS].mean()
    distinct = [dict(metric=LABEL[c], value=round(float(m[c]), 1), healthy=round(float(h_mean[c]), 1),
                     ratio=round(float(m[c] / h_mean[c]), 2)) for c in COLS
                if m[c] / h_mean[c] >= 1.4 or m[c] / h_mean[c] <= 0.6]
    distinct.sort(key=lambda d: -abs(np.log(d["ratio"])))
    fw = g.Firmware_Version.value_counts(normalize=True).round(3).to_dict()
    patterns.append(dict(mode=mode, name=name, desc=desc, count=int(len(g)), share=round(len(g) / len(fails), 3),
        means={c: round(float(m[c]), 1) for c in COLS}, distinct=distinct,
        firmware={k: fw.get(k, 0.0) for k in sorted(df.Firmware_Version.unique())},
        vendors=g.Vendor.value_counts().to_dict()))
patterns.sort(key=lambda p: -p["count"])
for i, p in enumerate(patterns): p["rank"] = i + 1

def rate_bins(col, edges, labels):
    b = pd.cut(df[col], edges, labels=labels)
    t = df.groupby(b, observed=True).Failure_Flag.agg(["size", "sum"])
    return [dict(label=str(k), n=int(r["size"]), failed=int(r["sum"]), rate=round(float(r["sum"] / r["size"]), 4)) for k, r in t.iterrows()]

bins = dict(
    temperature=rate_bins("Temperature_C", [0, 35, 40, 45, 50, 55, 60, 100], ["≤35", "35–40", "40–45", "45–50", "50–55", "55–60", ">60"]),
    hours=rate_bins("Power_On_Hours", [0, 3000, 10000, 20000, 30000, 40000, 1e9], ["≤3k", "3–10k", "10–20k", "20–30k", "30–40k", ">40k"]),
    life=rate_bins("Percent_Life_Used", [-1, 20, 40, 60, 80, 95, 101], ["≤20", "20–40", "40–60", "60–80", "80–95", ">95"]),
    read_err=rate_bins("Read_Error_Rate", [-1, 5, 8, 10, 12, 15, 1e9], ["≤5", "5–8", "8–10", "10–12", "12–15", ">15"]),
)
fw_rates = df.groupby("Firmware_Version").Failure_Flag.agg(["size", "sum"])
firmware = [dict(label=k, n=int(r["size"]), failed=int(r["sum"]), rate=round(float(r["sum"] / r["size"]), 4)) for k, r in fw_rates.iterrows()]

corr = df[COLS].corr().round(2)
OUT.write_text(json.dumps(dict(
    n_drives=len(df), n_failures=len(fails), base_rate=round(len(fails) / len(df), 4),
    max_temp=float(df.Temperature_C.max()), patterns=patterns, missing=missing, bins=bins, firmware=firmware,
    corr=dict(labels=[LABEL[c] for c in COLS], matrix=corr.values.tolist())), ensure_ascii=False, separators=(",", ":")))
for p in patterns: print(p["rank"], p["name"], p["count"], [(d["metric"], d["ratio"]) for d in p["distinct"]])

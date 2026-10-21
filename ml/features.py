"""Shared feature engineering. Mirrored 1:1 in web/lib/model.ts."""
import numpy as np
import pandas as pd

VENDORS = ["VendorA", "VendorB", "VendorC", "VendorD"]
MODELS = ["Model-LITE", "Model-PRO", "Model-ULTRA", "Model-X1", "Model-X2"]
FIRMWARES = ["FW1.0", "FW1.1", "FW1.2", "FW2.0", "FW2.1"]

RAW_NUM = [
    "Power_On_Hours", "Total_TBW_TB", "Total_TBR_TB", "Temperature_C",
    "Percent_Life_Used", "Media_Errors", "Unsafe_Shutdowns", "CRC_Errors",
    "Read_Error_Rate", "Write_Error_Rate",
]
ENG = ["tbw_per_khr", "wear_per_khr", "media_err_per_khr", "rw_ratio"]
# SMART_Warning_Flag and Failure_Mode are 1:1 proxies for the label -> never used.
LEAKY = ["SMART_Warning_Flag", "Failure_Mode"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in RAW_NUM:
        X[c] = df[c].astype(float)
    khr = df["Power_On_Hours"] / 1000.0 + 1.0
    X["tbw_per_khr"] = df["Total_TBW_TB"] / khr
    X["wear_per_khr"] = df["Percent_Life_Used"] / khr
    X["media_err_per_khr"] = df["Media_Errors"] / khr
    X["rw_ratio"] = df["Total_TBR_TB"] / (df["Total_TBW_TB"] + 1.0)
    for v in VENDORS:
        X[f"vendor={v}"] = (df["Vendor"] == v).astype(float)
    for m in MODELS:
        X[f"model={m}"] = (df["Model"] == m).astype(float)
    for f in FIRMWARES:
        X[f"fw={f}"] = (df["Firmware_Version"] == f).astype(float)
    return X

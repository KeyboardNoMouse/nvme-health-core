// In-browser inference for the exported Random Forest. Feature engineering mirrors ml/features.py.
export type Tree = { f: number[]; t: number[]; l: number[]; r: number[]; v: number[] };
export type Model = {
  features: string[]; threshold: number; trees: Tree[];
  medians: Record<string, number>; importances: Record<string, number>;
  metrics: { pr_auc: number; roc_auc: number; precision: number; recall: number; tp: number; fp: number; fn: number; tn: number; n_test: number };
};
export type Drive = {
  Drive_ID?: string; Vendor: string; Model: string; Firmware_Version: string;
  Power_On_Hours: number; Total_TBW_TB: number; Total_TBR_TB: number; Temperature_C: number;
  Percent_Life_Used: number; Media_Errors: number; Unsafe_Shutdowns: number; CRC_Errors: number;
  Read_Error_Rate: number; Write_Error_Rate: number;
};
export const VENDORS = ["VendorA", "VendorB", "VendorC", "VendorD"];
export const MODELS = ["Model-LITE", "Model-PRO", "Model-ULTRA", "Model-X1", "Model-X2"];
export const FIRMWARES = ["FW1.0", "FW1.1", "FW1.2", "FW2.0", "FW2.1"];
export const NUMERIC = ["Power_On_Hours", "Total_TBW_TB", "Total_TBR_TB", "Temperature_C", "Percent_Life_Used",
  "Media_Errors", "Unsafe_Shutdowns", "CRC_Errors", "Read_Error_Rate", "Write_Error_Rate"] as const;

export function featurize(d: Drive): Record<string, number> {
  const x: Record<string, number> = {};
  for (const c of NUMERIC) x[c] = Number(d[c]);
  const khr = d.Power_On_Hours / 1000 + 1;
  x.tbw_per_khr = d.Total_TBW_TB / khr;
  x.wear_per_khr = d.Percent_Life_Used / khr;
  x.media_err_per_khr = d.Media_Errors / khr;
  x.rw_ratio = d.Total_TBR_TB / (d.Total_TBW_TB + 1);
  for (const v of VENDORS) x[`vendor=${v}`] = d.Vendor === v ? 1 : 0;
  for (const m of MODELS) x[`model=${m}`] = d.Model === m ? 1 : 0;
  for (const f of FIRMWARES) x[`fw=${f}`] = d.Firmware_Version === f ? 1 : 0;
  return x;
}

export function risk(m: Model, d: Drive): number {
  const x = featurize(d), v = m.features.map((n) => x[n]);
  let s = 0;
  for (const t of m.trees) {
    let i = 0;
    while (t.l[i] !== -1) i = v[t.f[i]] <= t.t[i] ? t.l[i] : t.r[i];
    s += t.v[i];
  }
  return s / m.trees.length;
}

/** What pushes this drive's risk up? Replace each input with the fleet median and measure the drop. */
export function drivers(m: Model, d: Drive) {
  const base = risk(m, d), out: { name: string; delta: number }[] = [];
  for (const c of NUMERIC) {
    const alt = { ...d, [c]: m.medians[c] } as Drive;
    out.push({ name: c.replace(/_/g, " "), delta: base - risk(m, alt) });
  }
  out.push({ name: "Firmware Version", delta: base - risk(m, { ...d, Firmware_Version: "FW1.1" }) });
  return out.filter((o) => o.delta > 0.005).sort((a, b) => b.delta - a.delta).slice(0, 4);
}

export function band(p: number, thr: number) {
  if (p >= 0.5) return { label: "Critical", cls: "crit" };
  if (p >= thr) return { label: "High", cls: "high" };
  if (p >= thr / 2) return { label: "Watch", cls: "watch" };
  return { label: "Healthy", cls: "ok" };
}

export function parseCSV(text: string): Record<string, string>[] {
  const [h, ...rows] = text.trim().split(/\r?\n/);
  const cols = h.split(",").map((c) => c.trim());
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [cols[i], v.trim()])));
}
export const toDrive = (r: Record<string, string>): Drive => ({
  ...(r as unknown as Drive),
  ...Object.fromEntries(NUMERIC.map((c) => [c, Number(r[c])])),
});

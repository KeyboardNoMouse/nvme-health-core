"use client";
import { useEffect, useMemo, useState } from "react";
import Patterns from "./Patterns";
import { Model, Drive, VENDORS, MODELS, FIRMWARES, risk, drivers, band, modeProbs, topMode, parseCSV, toDrive } from "@/lib/model";

const PRESETS: Record<string, Drive> = {
  "Healthy": { Vendor: "VendorA", Model: "Model-PRO", Firmware_Version: "FW1.1", Power_On_Hours: 24000, Total_TBW_TB: 90, Total_TBR_TB: 95, Temperature_C: 40, Percent_Life_Used: 20, Media_Errors: 0, Unsafe_Shutdowns: 2, CRC_Errors: 0, Read_Error_Rate: 6, Write_Error_Rate: 5 },
  "Wear-out": { Vendor: "VendorB", Model: "Model-ULTRA", Firmware_Version: "FW2.0", Power_On_Hours: 35000, Total_TBW_TB: 450, Total_TBR_TB: 470, Temperature_C: 42, Percent_Life_Used: 99, Media_Errors: 1, Unsafe_Shutdowns: 3, CRC_Errors: 0, Read_Error_Rate: 8, Write_Error_Rate: 8 },
  "Firmware fault": { Vendor: "VendorC", Model: "Model-X1", Firmware_Version: "FW1.0", Power_On_Hours: 30000, Total_TBW_TB: 110, Total_TBR_TB: 115, Temperature_C: 41, Percent_Life_Used: 24, Media_Errors: 4, Unsafe_Shutdowns: 3, CRC_Errors: 1, Read_Error_Rate: 16, Write_Error_Rate: 6 },
  "Early-life": { Vendor: "VendorD", Model: "Model-X2", Firmware_Version: "FW2.0", Power_On_Hours: 1500, Total_TBW_TB: 100, Total_TBR_TB: 95, Temperature_C: 41, Percent_Life_Used: 23, Media_Errors: 1, Unsafe_Shutdowns: 3, CRC_Errors: 1, Read_Error_Rate: 15, Write_Error_Rate: 10 },
};
const NUM_FIELDS = ["Power_On_Hours", "Total_TBW_TB", "Total_TBR_TB", "Temperature_C", "Percent_Life_Used", "Media_Errors", "Unsafe_Shutdowns", "CRC_Errors", "Read_Error_Rate", "Write_Error_Rate"] as const;
const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
type Row = { id: string; vendor: string; model: string; fw: string; hours: number; life: number; temp: number; media: number; p: number; failed?: number; mode?: number };

export default function Home() {
  const [m, setM] = useState<Model | null>(null);
  const [tab, setTab] = useState<"predict" | "fleet" | "patterns">("predict");
  const [d, setD] = useState<Drive>(PRESETS["Healthy"]);
  const [fleet, setFleet] = useState<Row[]>([]);
  const [src, setSrc] = useState("Bundled dataset (out-of-fold scores)");
  const [group, setGroup] = useState<"fw" | "vendor" | "model">("fw");
  const [sort, setSort] = useState<keyof Row>("p");

  useEffect(() => {
    fetch(`${BASE}/model/model.json`).then((r) => r.json()).then(setM);
    fetch(`${BASE}/model/fleet_scored.csv`).then((r) => r.text()).then((t) =>
      setFleet(parseCSV(t).map((r) => ({ id: r.Drive_ID, vendor: r.Vendor, model: r.Model, fw: r.Firmware_Version, hours: +r.Power_On_Hours, life: +r.Percent_Life_Used, temp: +r.Temperature_C, media: +r.Media_Errors, p: +r.risk, failed: +r.Failure_Flag, mode: +r.likely_mode }))));
  }, []);

  const pred = useMemo(() => (m ? risk(m, d) : 0), [m, d]);
  const modeP = useMemo(() => (m ? modeProbs(m, d) : []), [m, d]);
  const why = useMemo(() => (m ? drivers(m, d) : []), [m, d]);
  const b = m ? band(pred, m.threshold) : { label: "…", cls: "ok" };

  function upload(f?: File) {
    if (!f || !m) return;
    f.text().then((t) => {
      const rows = parseCSV(t).map((r, i) => { const x = toDrive(r); return { id: r.Drive_ID || `row-${i + 1}`, vendor: x.Vendor, model: x.Model, fw: x.Firmware_Version, hours: x.Power_On_Hours, life: x.Percent_Life_Used, temp: x.Temperature_C, media: x.Media_Errors, p: risk(m, x), mode: topMode(m, x) }; });
      setFleet(rows); setSrc(`Uploaded: ${f.name} (scored in your browser)`);
    });
  }

  const thr = m?.threshold ?? 0.27;
  const counts = { Critical: 0, High: 0, Watch: 0, Healthy: 0 } as Record<string, number>;
  fleet.forEach((r) => counts[band(r.p, thr).label]++);
  const gOpts = group === "fw" ? FIRMWARES : group === "vendor" ? VENDORS : MODELS;
  const gKey = (r: Row) => (group === "fw" ? r.fw : group === "vendor" ? r.vendor : r.model);
  const byGroup = gOpts.map((g) => { const rs = fleet.filter((r) => gKey(r) === g); return { g, n: rs.length, rate: rs.length ? rs.filter((r) => r.p >= thr).length / rs.length : 0 }; });
  const maxG = Math.max(0.001, ...byGroup.map((x) => x.rate));
  const flagged = fleet.filter((r) => r.p >= thr);
  const modeName = (id?: number) => (m && id !== undefined && m.mode_ids.includes(id) ? m.mode_names[m.mode_ids.indexOf(id)] : "–");
  const byMode = (m?.mode_ids ?? []).map((id) => ({ name: modeName(id), n: flagged.filter((r) => r.mode === id).length }));
  const maxMode = Math.max(1, ...byMode.map((x) => x.n));
  const top = [...fleet].sort((a, b) => (sort === "p" || typeof a[sort] === "number" ? (b[sort] as number) - (a[sort] as number) : String(a[sort]).localeCompare(String(b[sort])))).slice(0, 25);

  return (
    <main>
      <h1>NVMe Health Core</h1>
      <p className="sub">Random Forest failure predictor · runs fully in the browser{m && ` · holdout recall ${(m.metrics.recall * 100).toFixed(0)}%, precision ${(m.metrics.precision * 100).toFixed(0)}%, PR-AUC ${m.metrics.pr_auc}`}</p>
      <div className="tabs">
        <button className={tab === "predict" ? "on" : ""} onClick={() => setTab("predict")}>Predictor</button>
        <button className={tab === "fleet" ? "on" : ""} onClick={() => setTab("fleet")}>Fleet dashboard</button>
        <button className={tab === "patterns" ? "on" : ""} onClick={() => setTab("patterns")}>Failure patterns</button>
      </div>

      {tab === "predict" && (
        <div className="grid g2">
          <div className="card">
            <h2>Drive telemetry</h2>
            <div className="row">{Object.keys(PRESETS).map((k) => <button key={k} onClick={() => setD(PRESETS[k])}>{k}</button>)}</div>
            <div className="form">
              {([["Vendor", VENDORS], ["Model", MODELS], ["Firmware_Version", FIRMWARES]] as const).map(([k, opts]) => (
                <label key={k}>{k.replace("_", " ")}<select value={d[k]} onChange={(e) => setD({ ...d, [k]: e.target.value })}>{opts.map((o) => <option key={o}>{o}</option>)}</select></label>
              ))}
              {NUM_FIELDS.map((k) => (
                <label key={k}>{k.replace(/_/g, " ")}<input type="number" step="any" value={d[k]} onChange={(e) => setD({ ...d, [k]: Number(e.target.value) })} /></label>
              ))}
            </div>
          </div>
          <div className="card">
            <h2>Failure risk</h2>
            <div className="big">{(pred * 100).toFixed(1)}%</div>
            <p><span className={`pill ${b.cls}`}>{b.label}</span> <span className="mut">alert threshold {(thr * 100).toFixed(0)}%</span></p>
            {m && b.cls !== "ok" && (<>
              <h2 style={{ marginTop: 20 }}>If it fails: likely pattern</h2>
              {m.mode_names.map((n, i) => ({ n, p: modeP[i] ?? 0 })).sort((a, b) => b.p - a.p).map((x) => (<div key={x.n}>{x.n} <span className="mut">{(x.p * 100).toFixed(0)}%</span><div className="bar"><i style={{ width: `${x.p * 100}%` }} /></div></div>))}
              <p className="mut" style={{ margin: "0 0 8px" }}>Pattern model: {(m.mode_metrics.accuracy * 100).toFixed(0)}% accurate in cross-validation on {m.mode_metrics.n} failing drives.</p>
            </>)}
            <h2 style={{ marginTop: 20 }}>Top risk drivers</h2>
            {why.length === 0 && <p className="mut">No input stands out versus the fleet median.</p>}
            {why.map((w) => (<div key={w.name}><div>{w.name} <span className="mut">+{(w.delta * 100).toFixed(1)} pts</span></div><div className="bar"><i style={{ width: `${Math.min(100, (w.delta / (why[0].delta || 1)) * 100)}%` }} /></div></div>))}
          </div>
        </div>
      )}

      {tab === "patterns" && <Patterns />}

      {tab === "fleet" && (
        <>
          <div className="row">
            <span className="mut">{src}</span>
            <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}><span className="pill high" style={{ cursor: "pointer", padding: "6px 12px" }}>Upload CSV</span>
              <input type="file" accept=".csv" hidden onChange={(e) => upload(e.target.files?.[0])} /></label>
            <a className="mut" href={`${BASE}/model/sample_upload.csv`} download>sample format</a>
          </div>
          <div className="grid g4" style={{ marginBottom: 14 }}>
            {(["Critical", "High", "Watch", "Healthy"] as const).map((k) => (
              <div className="card" key={k}><span className={`pill ${k === "Healthy" ? "ok" : k.toLowerCase()}`}>{k}</span><div className="big" style={{ marginTop: 8 }}>{counts[k].toLocaleString()}</div></div>
            ))}
          </div>
          <div className="grid g2">
            <div className="card scroll">
              <h2>Highest-risk drives</h2>
              <table><thead><tr>{([["id", "Drive"], ["fw", "FW"], ["hours", "Hours"], ["life", "Life %"], ["media", "Media err"], ["p", "Risk"], ["mode", "Pattern"]] as [keyof Row, string][]).map(([k, l]) => <th key={k} onClick={() => setSort(k)}>{l}</th>)}</tr></thead>
                <tbody>{top.map((r) => (<tr key={r.id}><td>{r.id}</td><td>{r.fw}</td><td>{r.hours.toLocaleString()}</td><td>{r.life}</td><td>{r.media}</td><td><span className={`pill ${band(r.p, thr).cls}`}>{(r.p * 100).toFixed(0)}%</span></td><td>{r.p >= thr ? modeName(r.mode) : "–"}</td></tr>))}</tbody></table>
            </div>
            <div className="grid">
              <div className="card">
                <div className="row" style={{ marginBottom: 8 }}><h2 style={{ margin: 0 }}>Flagged share by</h2>
                  <select value={group} onChange={(e) => setGroup(e.target.value as typeof group)}><option value="fw">firmware</option><option value="vendor">vendor</option><option value="model">model</option></select></div>
                {byGroup.map((x) => (<div key={x.g}><div>{x.g} <span className="mut">{(x.rate * 100).toFixed(1)}% of {x.n.toLocaleString()}</span></div><div className="bar"><i style={{ width: `${(x.rate / maxG) * 100}%` }} /></div></div>))}
              </div>
              <div className="card">
                <h2>Flagged drives by likely pattern</h2>
                {byMode.map((x) => (<div key={x.name}><div>{x.name} <span className="mut">{x.n}</span></div><div className="bar"><i style={{ width: `${(x.n / maxMode) * 100}%` }} /></div></div>))}
              </div>
            </div>
          </div>
        </>
      )}
    </main>
  );
}

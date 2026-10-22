"use client";
import { useEffect, useState } from "react";

type Bin = { label: string; n: number; failed: number; rate: number };
type Pattern = {
  rank: number; mode: number; name: string; desc: string; count: number; share: number;
  distinct: { metric: string; value: number; healthy: number; ratio: number }[];
  firmware: Record<string, number>;
};
type Data = {
  n_drives: number; n_failures: number; base_rate: number; max_temp: number;
  patterns: Pattern[]; missing: { mode: number; name: string; desc: string }[];
  bins: Record<"temperature" | "hours" | "life" | "read_err", Bin[]>; firmware: Bin[];
  corr: { labels: string[]; matrix: number[][] };
};
const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;

function Bars({ title, note, items, base }: { title: string; note?: string; items: Bin[]; base: number }) {
  const max = Math.max(...items.map((i) => i.rate), base, 0.001);
  return (
    <div className="card">
      <h2>{title}</h2>
      {items.map((i) => (
        <div key={i.label}>
          <div>{i.label} <span className="mut">{pct(i.rate)} · {i.failed}/{i.n.toLocaleString()}</span></div>
          <div className="bar"><i style={{ width: `${(i.rate / max) * 100}%`, background: i.rate > base * 2 ? "var(--high)" : "var(--acc)" }} /></div>
        </div>
      ))}
      {note && <p className="mut" style={{ margin: "8px 0 0" }}>{note}</p>}
    </div>
  );
}

export default function Patterns() {
  const [d, setD] = useState<Data | null>(null);
  useEffect(() => { fetch(`${BASE}/model/patterns.json`).then((r) => r.json()).then(setD); }, []);
  if (!d) return <p className="mut">Loading…</p>;

  const L = d.corr.labels, M = d.corr.matrix;
  const ti = L.indexOf("Temperature °C");
  const tempMax = Math.max(...M[ti].filter((_, j) => j !== ti).map(Math.abs));
  const tbwLife = M[L.indexOf("TB written")][L.indexOf("Life used %")];
  const maxFw = Math.max(...d.firmware.map((f) => f.rate));
  const worstFw = d.firmware.filter((f) => f.rate > d.base_rate * 1.5).map((f) => f.label).join(" and ");

  return (
    <>
      <p className="sub" style={{ marginTop: 0 }}>
        {d.n_failures} failures among {d.n_drives.toLocaleString()} drives ({pct(d.base_rate)}). Only {d.patterns.length} of the 5 documented failure modes appear in this dataset.
      </p>

      <div className="grid g3" style={{ marginBottom: 14 }}>
        {d.patterns.map((p) => (
          <div className="card" key={p.mode}>
            <h2>#{p.rank} · {p.name}</h2>
            <div className="big">{p.count}</div>
            <div className="bar"><i style={{ width: pct(p.share, 0) }} /></div>
            <p className="mut" style={{ margin: "0 0 10px" }}>{pct(p.share, 0)} of failures. {p.desc}</p>
            <h2>Signature vs healthy</h2>
            {p.distinct.map((x) => (
              <div key={x.metric}>{x.metric}: <b>{x.value}</b> <span className="mut">vs {x.healthy} ({x.ratio >= 1 ? `${x.ratio.toFixed(1)}×` : `${(1 / x.ratio).toFixed(0)}× lower`})</span></div>
            ))}
            <h2 style={{ marginTop: 12 }}>Firmware mix</h2>
            {Object.entries(p.firmware).filter(([, v]) => v > 0).map(([k, v]) => (
              <div key={k}>{k} <span className="mut">{pct(v, 0)}</span></div>
            ))}
          </div>
        ))}
      </div>

      <div className="card" style={{ marginBottom: 14 }}>
        <h2>Not observed in this dataset</h2>
        {d.missing.map((m) => <div key={m.mode}><b>{m.name}</b> <span className="mut">(mode {m.mode}): 0 drives. {m.desc}</span></div>)}
        <p className="mut" style={{ margin: "8px 0 0" }}>The hottest drive in the data runs at {d.max_temp}°C, below the 70°C thermal threshold.</p>
      </div>

      <h2>Failure rate by condition</h2>
      <div className="grid g2" style={{ marginBottom: 14, gridTemplateColumns: "1fr 1fr" }}>
        <Bars title="Power-on hours" items={d.bins.hours} base={d.base_rate} note="Early-life drives (under 3,000 hours) fail at a far higher rate than any older group." />
        <Bars title="Life used %" items={d.bins.life} base={d.base_rate} note="Risk is flat until about 80% life used, then jumps (small counts at the top end)." />
        <Bars title="Read error rate" items={d.bins.read_err} base={d.base_rate} note="Failure rate climbs steadily once read error rate passes 10." />
        <Bars title="Temperature °C" items={d.bins.temperature} base={d.base_rate} note="No upward trend with temperature in this data." />
      </div>

      <div className="grid g2" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Bars title="Firmware version" items={d.firmware} base={d.base_rate} note={`${worstFw} fail at roughly ${(maxFw / d.base_rate).toFixed(1)}× the fleet average at most.`} />
        <div className="card scroll">
          <h2>Correlation between telemetry fields</h2>
          <table style={{ fontSize: 11 }}>
            <thead><tr><th /> {L.map((l) => <th key={l} title={l}>{l.split(" ")[0].slice(0, 5)}</th>)}</tr></thead>
            <tbody>{M.map((row, i) => (
              <tr key={L[i]}><th>{L[i]}</th>{row.map((v, j) => (
                <td key={j} style={{ background: `rgba(88,166,255,${Math.abs(v) * 0.8})`, textAlign: "center", color: Math.abs(v) > 0.5 ? "#04111f" : undefined }}>{v.toFixed(2)}</td>
              ))}</tr>
            ))}</tbody>
          </table>
          <p className="mut" style={{ margin: "8px 0 0" }}>
            TB written and life used move together (r = {tbwLife.toFixed(2)}). Temperature has no meaningful correlation with any field (|r| ≤ {tempMax.toFixed(2)}), and error counts do not accumulate with age or usage. The failure patterns are encoded in a few separate signals rather than a shared degradation path, which is typical of synthetic data.
          </p>
        </div>
      </div>
    </>
  );
}

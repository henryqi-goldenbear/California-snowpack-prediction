import { useMemo, useState } from "react";
import { CloudRain, Info, MountainSnow, RotateCcw, Sparkles, ThermometerSun, Waves } from "lucide-react";

type Inputs = { year: number; enso: number; pdo: number; ao: number; pna: number };
type Region = { name: string; lat: number; elevation: number; weight: number };
const months = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"];
const monthFactor = [.65, 1.15, 1.3, 1.1, .9, .45];
const regions: Region[] = [
  { name: "North Coast", lat: 40.2, elevation: 900, weight: .16 },
  { name: "Shasta & Cascades", lat: 41, elevation: 1700, weight: .12 },
  { name: "Northern Sierra", lat: 39.6, elevation: 2100, weight: .14 },
  { name: "Central Sierra", lat: 38.3, elevation: 2300, weight: .15 },
  { name: "Southern Sierra", lat: 36.8, elevation: 2400, weight: .13 },
  { name: "Central Coast & Valleys", lat: 36.7, elevation: 250, weight: .18 },
  { name: "Southern California", lat: 34.2, elevation: 750, weight: .12 },
];
const defaults: Inputs = { year: 2027, enso: .8, pdo: .35, ao: 0, pna: .2 };

function calculate(i: Inputs) {
  const predictedTemp = .28 + .35 * i.enso + .12 * i.pdo - .08 * i.ao + .1 * i.pna + .025 * (i.year - 2025);
  const details = regions.map(region => {
    const south = (40.5 - region.lat) / 7;
    const base = 85 + 90 * (1 - south) + 20 * region.elevation / 2400;
    const tele = Math.max(.55, 1 + .11 * i.enso * (2 * south - .55) + .1 * i.pdo + .04 * i.pna - .025 * i.ao);
    const normalTele = 1;
    const snowFraction = Math.max(.03, Math.min(.96, .1 + region.elevation / 2800 - .09 * predictedTemp - .035 * south));
    const monthly = monthFactor.map((factor, m) => ({ month: months[m], precipitation: base * factor * tele, normal: base * factor * normalTele, snow: base * factor * tele * snowFraction * 1.05 }));
    return { ...region, monthly, precip: monthly.reduce((a, b) => a + b.precipitation, 0), normal: monthly.reduce((a, b) => a + b.normal, 0), snow: monthly.reduce((a, b) => a + b.snow, 0) };
  });
  const trajectory = months.map((month, m) => ({ month, precip: details.reduce((a, r) => a + r.monthly[m].precipitation * r.weight, 0), snow: details.reduce((a, r) => a + r.monthly[m].snow * r.weight, 0) }));
  const precip = details.reduce((a, r) => a + r.precip * r.weight, 0), normal = details.reduce((a, r) => a + r.normal * r.weight, 0), snow = details.reduce((a, r) => a + r.snow * r.weight, 0);
  const pct = precip / normal * 100;
  return { details, trajectory, precip, snow, pct, predictedTemp, category: pct >= 110 ? "Wetter than normal" : pct <= 90 ? "Drier than normal" : "Near normal" };
}

function Slider({ label, hint, value, min, max, step, onChange, format = String }: { label: string; hint: string; value: number; min: number; max: number; step: number; onChange: (n: number) => void; format?: (n: number) => string }) {
  const fill = `${((value - min) / (max - min)) * 100}%`;
  return <label className="control"><span><b>{label}</b><output>{format(value)}</output></span><small>{hint}</small><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} style={{ "--fill": fill } as React.CSSProperties} /></label>;
}

export default function App() {
  const [input, setInput] = useState(defaults);
  const [mode, setMode] = useState<"precip" | "snow">("precip");
  const result = useMemo(() => calculate(input), [input]);
  const peak = result.trajectory.reduce((a, b) => b.precip > a.precip ? b : a);
  const set = (key: keyof Inputs, value: number) => setInput(old => ({ ...old, [key]: value }));
  const maxChart = Math.max(...result.trajectory.map(x => mode === "precip" ? x.precip : x.snow));
  return <main>
    <header><div className="brand"><span className="brandmark"><MountainSnow size={23} /></span><span>Sierra Signal</span></div><div className="status"><span></span> Scenario engine ready</div></header>
    <section className="hero"><div><p className="eyebrow">California winter outlook lab</p><h1>See the winter<br/><em>before it arrives.</em></h1><p className="lede">Turn Pacific climate signals into a human-readable precipitation and snowfall outlook for every region of California.</p></div><div className="season"><small>Forecast season</small><select aria-label="Forecast water year" value={input.year} onChange={e => set("year", Number(e.target.value))}>{[2026,2027,2028,2029].map(y => <option key={y} value={y}>{y - 1}–{String(y).slice(2)}</option>)}</select><span>November through April</span></div></section>
    <section className="workspace">
      <aside className="panel controls"><div className="panel-title"><div><span>01</span><h2>Climate signals</h2></div><button onClick={() => setInput(defaults)} title="Reset inputs"><RotateCcw size={15}/> Reset</button></div>
        <p className="helper"><Info size={14}/> Adjust the ocean and atmosphere indices. Temperature is predicted automatically.</p>
        <Slider label="ENSO / ONI" hint="El Niño ← Neutral → La Niña" min={-2.5} max={2.5} step={.1} value={input.enso} onChange={v => set("enso", v)} format={n => `${n > 0 ? "+" : ""}${n.toFixed(1)}`} />
        <Slider label="PDO" hint="Cool phase ← → Warm phase" min={-2} max={2} step={.1} value={input.pdo} onChange={v => set("pdo", v)} format={n => `${n > 0 ? "+" : ""}${n.toFixed(1)}`} />
        <div className="split"><Slider label="AO" hint="Arctic Oscillation" min={-2} max={2} step={.1} value={input.ao} onChange={v => set("ao", v)} format={n => n.toFixed(1)} /><Slider label="PNA" hint="Pacific–North American" min={-2} max={2} step={.1} value={input.pna} onChange={v => set("pna", v)} format={n => n.toFixed(1)} /></div>
        <div className="temperature-result"><ThermometerSun size={18}/><div><small>Model-predicted temperature</small><b>{result.predictedTemp > 0 ? "+" : ""}{result.predictedTemp.toFixed(1)}°C</b><span>seasonal anomaly</span></div></div>
        <div className="signal-note"><Sparkles size={17}/><p><b>What the model sees</b><br/>{input.enso > .5 ? "El Niño tilts storms toward Southern California" : input.enso < -.5 ? "La Niña favors the northern storm track" : "ENSO is not strongly steering the storm track"}; {input.pdo > .4 ? "a warm PDO adds a wetter influence." : input.pdo < -.4 ? "a cool PDO adds a drier influence." : "PDO influence is limited."} Those signals imply a {result.predictedTemp >= .7 ? "warmer" : result.predictedTemp <= -.2 ? "colder" : "near-normal"} winter, which directly changes the rain–snow split.</p></div>
      </aside>
      <section className="results">
        <div className="panel summary"><div className="panel-title"><div><span>02</span><h2>Your winter, translated</h2></div><span className={`badge ${result.category.startsWith("Drier") ? "dry" : ""}`}>{result.category}</span></div>
          <p className="narrative">California is projected to receive <strong>{Math.round(result.pct)}% of normal precipitation</strong>. ENSO and the other climate signals predict a <strong>{result.predictedTemp > 0 ? "+" : ""}{result.predictedTemp.toFixed(1)}°C temperature anomaly</strong>. That temperature projection determines how much precipitation falls as snow, producing roughly <strong>{Math.round(result.snow)} cm</strong> of weighted seasonal snowfall. {result.predictedTemp > 1 ? "Warmer conditions increase rain-on-snow risk and reduce lower-elevation snow." : "Temperatures remain supportive of Sierra snow accumulation."}</p>
          <div className="metrics"><article><CloudRain/><small>Statewide precipitation</small><b>{Math.round(result.precip)} <i>mm</i></b><span>{result.pct >= 100 ? "+" : ""}{Math.round(result.pct - 100)}% vs normal</span></article><article><MountainSnow/><small>Temperature-driven snow</small><b>{Math.round(result.snow)} <i>cm</i></b><span>Nov–Apr total</span></article><article><ThermometerSun/><small>Predicted temperature</small><b>{result.predictedTemp > 0 ? "+" : ""}{result.predictedTemp.toFixed(1)}<i>°C</i></b><span>From ENSO · PDO · AO · PNA</span></article></div>
        </div>
        <div className="panel chart-card"><div className="chart-head"><div><span className="section-kicker">03 · Winter trajectory</span><h2>How the season unfolds</h2></div><div className="toggle"><button className={mode === "precip" ? "active" : ""} onClick={() => setMode("precip")}>Precipitation</button><button className={mode === "snow" ? "active" : ""} onClick={() => setMode("snow")}>Snowfall</button></div></div><div className="chart" aria-label={`${mode} monthly trajectory`}>{result.trajectory.map((x, idx) => { const val = mode === "precip" ? x.precip : x.snow; return <div className="bar-wrap" key={x.month}><span>{Math.round(val)}</span><div className="bar" style={{ height: `${Math.max(8, val / maxChart * 100)}%`, animationDelay: `${idx * 45}ms` }}></div><b>{x.month}</b></div>})}</div><div className="chart-note"><Waves size={16}/> {peak.month} is the modeled precipitation peak; March and April determine how much of the Sierra pack carries into spring.</div></div>
      </section>
    </section>
    <section className="regional"><div className="regional-head"><div><span className="section-kicker">04 · Regional detail</span><h2>One state, seven different winters</h2></div><p>Values are area-weighted regional seasonal totals.</p></div><div className="region-grid">{result.details.map(r => { const pct = r.precip / r.normal * 100; return <article key={r.name}><div className="region-top"><h3>{r.name}</h3><span>{Math.round(pct)}%</span></div><div className="mini-track"><i style={{ width: `${Math.min(100, pct * .8)}%` }}></i><u style={{ left: "80%" }}></u></div><div className="region-values"><span><CloudRain/> {Math.round(r.precip)} mm</span><span><MountainSnow/> {Math.round(r.snow)} cm</span></div></article>})}</div></section>
    <footer><div><MountainSnow size={17}/> Sierra Signal</div><p>Scenario guidance, not an operational weather forecast. Validate decisions against official NOAA and California DWR products.</p><span>Model: climate-index regional outlook</span></footer>
  </main>;
}

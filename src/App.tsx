import { useEffect, useState } from "react";
import { BrainCircuit, CloudRain, Info, LoaderCircle, MountainSnow, RotateCcw, Sparkles, ThermometerSun, Waves } from "lucide-react";

type Inputs = { year: number; enso: number; pdo: number; ao: number; pna: number };
type MonthResult = { month: string; precip: number; snow: number };
type RegionResult = { name: string; precip: number; normal: number; snow: number };
type Forecast = {
  summary: string; category: string; predictedTemp: number; precip: number;
  pct: number; snow: number; trajectory: MonthResult[]; details: RegionResult[];
};

const defaults: Inputs = { year: 2027, enso: .8, pdo: .35, ao: 0, pna: .2 };
const emptyForecast: Forecast = {
  summary: "Run Mistral to turn the selected climate signals into a statewide winter outlook.",
  category: "Awaiting forecast", predictedTemp: 0, precip: 0, pct: 100, snow: 0,
  trajectory: ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"].map(month => ({ month, precip: 0, snow: 0 })),
  details: ["North Coast", "Shasta & Cascades", "Northern Sierra", "Central Sierra", "Southern Sierra", "Central Coast & Valleys", "Southern California"].map(name => ({ name, precip: 0, normal: 1, snow: 0 })),
};

function Slider({ label, hint, value, min, max, step, onChange }: { label: string; hint: string; value: number; min: number; max: number; step: number; onChange: (n: number) => void }) {
  const fill = `${((value - min) / (max - min)) * 100}%`;
  return <label className="control"><span><b>{label}</b><output>{value > 0 ? "+" : ""}{value.toFixed(1)}</output></span><small>{hint}</small><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} style={{ "--fill": fill } as React.CSSProperties} /></label>;
}

export default function App() {
  const [input, setInput] = useState(defaults);
  const [result, setResult] = useState<Forecast>(emptyForecast);
  const [mode, setMode] = useState<"precip" | "snow">("precip");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasForecast, setHasForecast] = useState(false);
  const set = (key: keyof Inputs, value: number) => { setInput(old => ({ ...old, [key]: value })); setHasForecast(false); };

  async function runForecast() {
    setLoading(true); setError("");
    try {
      const response = await fetch("/api/forecast", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(input) });
      const body = await response.json() as Forecast & { error?: string };
      if (!response.ok) throw new Error(body.error || "Mistral could not generate the forecast.");
      setResult(body); setHasForecast(true);
    } catch (err) { setError(err instanceof Error ? err.message : "Forecast request failed."); }
    finally { setLoading(false); }
  }

  useEffect(() => { void runForecast(); }, []);
  const peak = result.trajectory.reduce((a, b) => b.precip > a.precip ? b : a, result.trajectory[0]);
  const maxChart = Math.max(1, ...result.trajectory.map(x => mode === "precip" ? x.precip : x.snow));

  return <main>
    <header><div className="brand"><span className="brandmark"><BrainCircuit size={23}/></span><span>Mistral Winter Lab</span></div><div className="status"><span></span> Mistral API connected</div></header>
    <section className="hero"><div><p className="eyebrow">Mistral-powered California outlook</p><h1>Ask the climate.<br/><em>Read the winter.</em></h1><p className="lede">Mistral turns Pacific climate signals into temperature, precipitation, and snowfall guidance for every region of California.</p></div><div className="season"><small>Forecast season</small><select aria-label="Forecast water year" value={input.year} onChange={e => set("year", Number(e.target.value))}>{[2026,2027,2028,2029].map(y => <option key={y} value={y}>{y - 1}–{String(y).slice(2)}</option>)}</select><span>November through April</span></div></section>
    <section className="workspace">
      <aside className="panel controls"><div className="panel-title"><div><span>01</span><h2>Climate signals</h2></div><button onClick={() => { setInput(defaults); setHasForecast(false); }}><RotateCcw size={15}/> Reset</button></div>
        <p className="helper"><Info size={14}/> Mistral uses these indices to predict temperature first, then the rain–snow split.</p>
        <Slider label="ENSO / ONI" hint="La Niña ← Neutral → El Niño" min={-2.5} max={2.5} step={.1} value={input.enso} onChange={v => set("enso", v)}/>
        <Slider label="PDO" hint="Cool phase ← → Warm phase" min={-2} max={2} step={.1} value={input.pdo} onChange={v => set("pdo", v)}/>
        <div className="split"><Slider label="AO" hint="Arctic Oscillation" min={-2} max={2} step={.1} value={input.ao} onChange={v => set("ao", v)}/><Slider label="PNA" hint="Pacific–North American" min={-2} max={2} step={.1} value={input.pna} onChange={v => set("pna", v)}/></div>
        <button className="mistral-button" onClick={runForecast} disabled={loading}>{loading ? <LoaderCircle className="spin"/> : <Sparkles/>}{loading ? "Mistral is forecasting…" : hasForecast ? "Refresh with Mistral" : "Run Mistral forecast"}</button>
        {error && <p className="api-error">{error}</p>}
        <div className="temperature-result"><ThermometerSun size={18}/><div><small>Mistral-predicted temperature</small><b>{hasForecast && result.predictedTemp > 0 ? "+" : ""}{hasForecast ? result.predictedTemp.toFixed(1) : "—"}°C</b><span>seasonal anomaly</span></div></div>
      </aside>
      <section className="results">
        <div className="panel summary"><div className="panel-title"><div><span>02</span><h2>Mistral’s winter read</h2></div><span className="badge">{result.category}</span></div>
          <p className="narrative">{result.summary}</p>
          <div className="metrics"><article><CloudRain/><small>Statewide precipitation</small><b>{hasForecast ? Math.round(result.precip) : "—"} <i>mm</i></b><span>{hasForecast ? `${result.pct >= 100 ? "+" : ""}${Math.round(result.pct - 100)}% vs normal` : "Mistral estimate"}</span></article><article><MountainSnow/><small>Temperature-driven snow</small><b>{hasForecast ? Math.round(result.snow) : "—"} <i>cm</i></b><span>Nov–Apr total</span></article><article><ThermometerSun/><small>Predicted temperature</small><b>{hasForecast && result.predictedTemp > 0 ? "+" : ""}{hasForecast ? result.predictedTemp.toFixed(1) : "—"}<i>°C</i></b><span>From ENSO · PDO · AO · PNA</span></article></div>
        </div>
        <div className="panel chart-card"><div className="chart-head"><div><span className="section-kicker">03 · Winter trajectory</span><h2>How Mistral sees the season</h2></div><div className="toggle"><button className={mode === "precip" ? "active" : ""} onClick={() => setMode("precip")}>Precipitation</button><button className={mode === "snow" ? "active" : ""} onClick={() => setMode("snow")}>Snowfall</button></div></div><div className="chart" aria-label={`${mode} monthly trajectory`}>{result.trajectory.map((x, idx) => { const val = mode === "precip" ? x.precip : x.snow; return <div className="bar-wrap" key={x.month}><span>{hasForecast ? Math.round(val) : ""}</span><div className="bar" style={{ height: `${hasForecast ? Math.max(8, val / maxChart * 100) : 2}%`, animationDelay: `${idx * 45}ms` }}/><b>{x.month}</b></div>})}</div><div className="chart-note"><Waves size={16}/> {hasForecast ? `${peak.month} is the modeled precipitation peak. Temperature determines how much reaches the ground as snow.` : "Run Mistral to generate the November–April trajectory."}</div></div>
      </section>
    </section>
    <section className="regional"><div className="regional-head"><div><span className="section-kicker">04 · Regional detail</span><h2>One state, seven Mistral outlooks</h2></div><p>Seasonal totals generated from the selected climate signals.</p></div><div className="region-grid">{result.details.map(r => { const pct = r.normal ? r.precip / r.normal * 100 : 0; return <article key={r.name}><div className="region-top"><h3>{r.name}</h3><span>{hasForecast ? `${Math.round(pct)}%` : "—"}</span></div><div className="mini-track"><i style={{ width: `${hasForecast ? Math.min(100, pct * .8) : 0}%` }}/><u style={{ left: "80%" }}/></div><div className="region-values"><span><CloudRain/> {hasForecast ? Math.round(r.precip) : "—"} mm</span><span><MountainSnow/> {hasForecast ? Math.round(r.snow) : "—"} cm</span></div></article>})}</div></section>
    <footer><div><BrainCircuit size={17}/> Mistral Winter Lab</div><p>AI-generated scenario guidance, not an official weather forecast. Validate decisions against NOAA and California DWR products.</p><span>Powered by Mistral API</span></footer>
  </main>;
}

const REGIONS = ["North Coast", "Shasta & Cascades", "Northern Sierra", "Central Sierra", "Southern Sierra", "Central Coast & Valleys", "Southern California"];
const MONTHS = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"];
const SEASON_PHASES = ["early", "mid", "late"];
const STATIC_ASSETS = {};

const RISK_LEVELS = ["Low", "Moderate", "High", "Critical"];
const hasImpact = (impact) => impact && RISK_LEVELS.includes(impact.riskLevel) && typeof impact.summary === "string" && impact.summary.trim();
const corsHeaders = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };
const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: corsHeaders });

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/api/forecast") {
      const assetPath = url.pathname.replace(/\?.*$/, "");
      const asset = STATIC_ASSETS[assetPath] || (assetPath.includes(".") ? null : STATIC_ASSETS["/"]);
      if (asset) {
        const bytes = new TextEncoder().encode(asset.body);
        return new Response(bytes, { headers: { "content-type": asset.type, "access-control-allow-origin": "*", "cache-control": url.pathname === "/" ? "no-cache" : "public, max-age=3600, must-revalidate" } });
      }
      if (env?.ASSETS?.fetch) {
        const assetRequest = url.search
          ? new Request(new URL(assetPath + url.search, url.origin), request)
          : request;
        const assetResponse = await env.ASSETS.fetch(assetRequest);
        if (assetResponse.status !== 404) return assetResponse;
      }
      return new Response("Not found", { status: 404 });
    }
    if (request.method !== "POST") return json({ error: "Method not allowed" }, 405);
    if (!env.MISTRAL_API_KEY) return json({ error: "Mistral API key is not configured yet." }, 503);

    try {
      const input = await request.json();
      const values = [input.year, input.enso, input.pdo, input.ao, input.pna];
      if (!values.every(Number.isFinite)) return json({ error: "All climate inputs must be numeric." }, 400);
const RECENT_WINTERS = `Recent California winter precedent to weight heavily (more recent = more predictive):
- WY 2023: record atmospheric-river sequence, major Sierra snowpack recovery, flood and debris-flow impacts
- WY 2024: near-normal to wet north, warm storms limiting low-elevation snow, early melt concerns
- WY 2020-2022: deepening drought, weak snowpack, reservoir stress, then partial recovery
- WY 2017: wettest winter in northern Sierra records, Oroville spillway stress test
- WY 2012-2016: severe drought, snowpack collapse, rain-dominant warm storms at mid elevations
- Long-term backdrop: warming raises rain-snow line; recent winters show stronger AR clustering than 1980s-1990s`;

      const prompt = `Generate a California winter outlook for water year ${input.year}. Climate indices: ENSO/ONI=${input.enso}, PDO=${input.pdo}, AO=${input.ao}, PNA=${input.pna}.

${RECENT_WINTERS}

Recency rule: anchor the outlook on the last 5-10 water years first. When indices resemble a recent winter, cite that analog explicitly and let it drive precipitation, temperature, and snowpack more than distant climatology. De-emphasize pre-2000 patterns unless the index combination has no modern analog.

Use physical reasoning in this strict causal order: (1) indices predict statewide Nov-Apr temperature anomaly in C; (2) indices predict precipitation; (3) predicted temperature determines rain-versus-snow partition, with elevation differences; (4) produce snowfall. Return plausible scenario guidance, not false precision.

Analyze the season in three phases and explain how signals evolve:
- Early season (Nov-Dec): storm onset, atmospheric river potential, initial snowpack setup
- Mid season (Jan-Feb): core wet season, peak storm activity, snowpack accumulation
- Late season (Mar-Apr): transition, melt risk, late-season storms vs drying

For each region in details, include a risks field describing the main winter hazards that region could face (e.g. flooding, debris flows, drought stress, snowpack deficit, water-supply shortfall, agricultural frost, urban runoff, wildfire-season carryover). Tailor risks to that region's geography and the forecast you generated.

Also assess statewide water allocation and wildfire risk for the following water year:
- waterAllocation: implications for reservoir storage, agricultural deliveries, urban supply, environmental flows, and groundwater recharge given predicted snowpack and runoff timing
- wildfireRisk: implications for fine-fuel moisture, grass crop carryover into the next fire season, early spring green-up vs late dry-down, and wind-driven fire weather potential

Return ONLY JSON with exactly this shape:
{"summary":"120-180 word statewide overview referencing how early, mid, and late season differ","category":"Wetter than normal|Near normal|Drier than normal","predictedTemp":number,"precip":number,"pct":number,"snow":number,"waterAllocation":{"riskLevel":"Low|Moderate|High|Critical","summary":"60-90 word statewide water allocation outlook for reservoirs, agriculture, cities, and ecosystems"},"wildfireRisk":{"riskLevel":"Low|Moderate|High|Critical","summary":"60-90 word statewide wildfire-season carryover outlook from this winter's precipitation, temperature, and dry-down trajectory"},"seasonPhases":[{"id":"early","label":"Early season","months":"Nov–Dec","category":"Wetter than normal|Near normal|Drier than normal","predictedTemp":number,"precip":number,"pct":number,"snow":number,"summary":"40-70 word analysis for Nov-Dec only"},{"id":"mid","label":"Mid season","months":"Jan–Feb","category":"Wetter than normal|Near normal|Drier than normal","predictedTemp":number,"precip":number,"pct":number,"snow":number,"summary":"40-70 word analysis for Jan-Feb only"},{"id":"late","label":"Late season","months":"Mar–Apr","category":"Wetter than normal|Near normal|Drier than normal","predictedTemp":number,"precip":number,"pct":number,"snow":number,"summary":"40-70 word analysis for Mar-Apr only"}],"trajectory":[{"month":"Nov","precip":number,"snow":number}],"details":[{"name":"North Coast","precip":number,"normal":number,"snow":number,"risks":"35-55 word description of this region's primary winter risks"}]}

Units: precipitation mm, snowfall cm, temperature anomaly in C. trajectory must contain exactly ${MONTHS.join(", ")} in order. seasonPhases must contain exactly early, mid, late in that order. Phase precip/snow are period totals; phase predictedTemp is the period mean anomaly. details must contain exactly ${REGIONS.join(", ")} in order with a non-empty risks string for each region. waterAllocation and wildfireRisk must both include riskLevel and summary. Statewide values are area-weighted seasonal totals. Regional normal must be a positive climatological precipitation baseline.`;

      const response = await fetch("https://api.mistral.ai/v1/chat/completions", {
        method: "POST",
        headers: { "authorization": `Bearer ${env.MISTRAL_API_KEY}`, "content-type": "application/json" },
        body: JSON.stringify({ model: env.MISTRAL_MODEL || "mistral-small-latest", temperature: 0.2, max_tokens: 4800, response_format: { type: "json_object" }, messages: [{ role: "system", content: "You are a California hydroclimate forecasting assistant. Weight recent water years more heavily than older history when forming outlooks. Return valid JSON only." }, { role: "user", content: prompt }] }),
      });
      if (!response.ok) return json({ error: `Mistral API returned ${response.status}. Check the API key and model access.` }, 502);
      const completion = await response.json();
      const forecast = JSON.parse(completion.choices?.[0]?.message?.content || "{}");
      if (!Array.isArray(forecast.trajectory) || forecast.trajectory.length !== 6 || !Array.isArray(forecast.details) || forecast.details.length !== 7) return json({ error: "Mistral returned an incomplete forecast. Please retry." }, 502);
      if (!Array.isArray(forecast.seasonPhases) || forecast.seasonPhases.length !== 3 || !forecast.seasonPhases.every((phase, index) => phase?.id === SEASON_PHASES[index])) return json({ error: "Mistral returned incomplete seasonal analysis. Please retry." }, 502);
      if (!forecast.details.every((region, index) => region?.name === REGIONS[index] && typeof region?.risks === "string" && region.risks.trim())) return json({ error: "Mistral returned incomplete regional risk analysis. Please retry." }, 502);
      if (!hasImpact(forecast.waterAllocation) || !hasImpact(forecast.wildfireRisk)) return json({ error: "Mistral returned incomplete water allocation or wildfire analysis. Please retry." }, 502);
      return json(forecast);
    } catch (error) {
      return json({ error: error instanceof Error ? error.message : "Mistral forecast failed." }, 500);
    }
  }
};

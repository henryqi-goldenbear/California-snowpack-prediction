const REGIONS = ["North Coast", "Shasta & Cascades", "Northern Sierra", "Central Sierra", "Southern Sierra", "Central Coast & Valleys", "Southern California"];
const MONTHS = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"];
const STATIC_ASSETS = {};

const corsHeaders = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };
const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: corsHeaders });

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/api/forecast") {
      if (env?.ASSETS?.fetch) return env.ASSETS.fetch(request);
      const asset = STATIC_ASSETS[url.pathname] || (url.pathname.includes(".") ? null : STATIC_ASSETS["/"]);
      if (!asset) return new Response("Not found", { status: 404 });
      return new Response(asset.body, { headers: { "content-type": asset.type, "cache-control": url.pathname === "/" ? "no-cache" : "public, max-age=31536000, immutable" } });
    }
    if (request.method !== "POST") return json({ error: "Method not allowed" }, 405);
    if (!env.MISTRAL_API_KEY) return json({ error: "Mistral API key is not configured yet." }, 503);

    try {
      const input = await request.json();
      const values = [input.year, input.enso, input.pdo, input.ao, input.pna];
      if (!values.every(Number.isFinite)) return json({ error: "All climate inputs must be numeric." }, 400);
      const prompt = `Generate a California winter outlook for water year ${input.year}. Climate indices: ENSO/ONI=${input.enso}, PDO=${input.pdo}, AO=${input.ao}, PNA=${input.pna}.

Use physical reasoning in this strict causal order: (1) indices predict statewide Nov-Apr temperature anomaly in C; (2) indices predict precipitation; (3) predicted temperature determines rain-versus-snow partition, with elevation differences; (4) produce snowfall. Return plausible scenario guidance, not false precision.

Return ONLY JSON with exactly this shape:
{"summary":"120-180 word plain-language explanation","category":"Wetter than normal|Near normal|Drier than normal","predictedTemp":number,"precip":number,"pct":number,"snow":number,"trajectory":[{"month":"Nov","precip":number,"snow":number}],"details":[{"name":"North Coast","precip":number,"normal":number,"snow":number}]}

Units: precipitation mm, snowfall cm. trajectory must contain exactly ${MONTHS.join(", ")} in order. details must contain exactly ${REGIONS.join(", ")} in order. Statewide values are area-weighted seasonal totals. Regional normal must be a positive climatological precipitation baseline.`;

      const response = await fetch("https://api.mistral.ai/v1/chat/completions", {
        method: "POST",
        headers: { "authorization": `Bearer ${env.MISTRAL_API_KEY}`, "content-type": "application/json" },
        body: JSON.stringify({ model: env.MISTRAL_MODEL || "mistral-small-latest", temperature: 0.2, max_tokens: 2200, response_format: { type: "json_object" }, messages: [{ role: "system", content: "You are a California hydroclimate forecasting assistant. Return valid JSON only." }, { role: "user", content: prompt }] }),
      });
      if (!response.ok) return json({ error: `Mistral API returned ${response.status}. Check the API key and model access.` }, 502);
      const completion = await response.json();
      const forecast = JSON.parse(completion.choices?.[0]?.message?.content || "{}");
      if (!Array.isArray(forecast.trajectory) || forecast.trajectory.length !== 6 || !Array.isArray(forecast.details) || forecast.details.length !== 7) return json({ error: "Mistral returned an incomplete forecast. Please retry." }, 502);
      return json(forecast);
    } catch (error) {
      return json({ error: error instanceof Error ? error.message : "Mistral forecast failed." }, 500);
    }
  }
};

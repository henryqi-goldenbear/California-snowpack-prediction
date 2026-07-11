import { createServer } from "node:http";
import { createReadStream, existsSync, readFileSync } from "node:fs";
import { extname, join } from "node:path";
import worker from "../worker/index.js";

for (const file of [".env", ".env.example"]) {
  if (!existsSync(file)) continue;
  for (const line of readFileSync(file, "utf8").split(/\r?\n/)) {
    const match = line.match(/^([A-Z_]+)=(.*)$/);
    if (match && !process.env[match[1]]) process.env[match[1]] = match[2];
  }
  break;
}

const env = {
  MISTRAL_API_KEY: process.env.MISTRAL_API_KEY,
  MISTRAL_MODEL: process.env.MISTRAL_MODEL || "mistral-small-latest",
};

const host = process.env.HOST || "0.0.0.0";
const port = Number(process.env.PORT || 8080);
const dist = join(process.cwd(), "dist");
const mime = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".json": "application/json; charset=utf-8",
};

createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

  if (url.pathname.startsWith("/api/")) {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = chunks.length ? Buffer.concat(chunks) : undefined;
    const response = await worker.fetch(
      new Request(`http://127.0.0.1:${port}${url.pathname}${url.search}`, {
        method: req.method,
        headers: req.headers,
        body,
      }),
      env,
    );
    res.writeHead(response.status, Object.fromEntries(response.headers.entries()));
    res.end(Buffer.from(await response.arrayBuffer()));
    return;
  }

  const assetPath = join(dist, url.pathname === "/" ? "index.html" : url.pathname.replace(/^\//, ""));
  if (existsSync(assetPath) && !assetPath.includes("..")) {
    res.writeHead(200, { "content-type": mime[extname(assetPath)] || "application/octet-stream" });
    createReadStream(assetPath).pipe(res);
    return;
  }

  const index = join(dist, "index.html");
  if (existsSync(index)) {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(readFileSync(index));
    return;
  }

  res.writeHead(503, { "content-type": "text/plain; charset=utf-8" });
  res.end("Build the app first: npm run build");
}).listen(port, host, () => {
  console.log(`Mistral Winter Lab serving on http://${host === "0.0.0.0" ? "localhost" : host}:${port}`);
});

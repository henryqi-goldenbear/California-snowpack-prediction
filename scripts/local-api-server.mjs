import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
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

const port = Number(process.env.API_PORT || 8787);

createServer(async (req, res) => {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = chunks.length ? Buffer.concat(chunks) : undefined;
  const response = await worker.fetch(
    new Request(`http://127.0.0.1:${port}${req.url}`, { method: req.method, headers: req.headers, body }),
    env,
  );
  res.writeHead(response.status, Object.fromEntries(response.headers.entries()));
  res.end(Buffer.from(await response.arrayBuffer()));
}).listen(port, "127.0.0.1", () => {
  console.log(`Mistral API listening on http://127.0.0.1:${port}`);
});

import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";

mkdirSync("dist/server", { recursive: true });
mkdirSync("dist/.openai", { recursive: true });
copyFileSync(".openai/hosting.json", "dist/.openai/hosting.json");

const contentType = path => path.endsWith(".css") ? "text/css; charset=utf-8" : path.endsWith(".js") ? "text/javascript; charset=utf-8" : "text/html; charset=utf-8";
const assets = { "/": { body: readFileSync("dist/index.html", "utf8"), type: contentType(".html") } };
for (const file of readdirSync("dist/assets")) {
  assets[`/assets/${file}`] = { body: readFileSync(`dist/assets/${file}`, "utf8"), type: contentType(file) };
}
const worker = readFileSync("worker/index.js", "utf8").replace("const STATIC_ASSETS = {};", `const STATIC_ASSETS = ${JSON.stringify(assets)};`);
writeFileSync("dist/server/index.js", worker);

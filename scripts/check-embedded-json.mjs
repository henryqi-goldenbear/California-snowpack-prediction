import { readFileSync } from "node:fs";

const code = readFileSync("dist/server/index.js", "utf8");
const match = code.match(/const STATIC_ASSETS = JSON\.parse\(atob\("([^"]+)"\)\);/);
if (!match) throw new Error("STATIC_ASSETS base64 marker not found");

const assets = JSON.parse(Buffer.from(match[1], "base64").toString("utf8"));
const js = assets["/assets/index-xhHPe_co.js"].body;
const local = readFileSync("dist/assets/index-xhHPe_co.js", "utf8");

console.log("embedded len", js.length);
console.log("local len", local.length);
console.log("has double-dollar typeof", js.includes("$$typeof"));
console.log("matches local file", js === local);

import { copyFileSync, mkdirSync } from "node:fs";

mkdirSync("dist/server", { recursive: true });
mkdirSync("dist/.openai", { recursive: true });
copyFileSync(".openai/hosting.json", "dist/.openai/hosting.json");

copyFileSync("worker/index.js", "dist/server/index.js");

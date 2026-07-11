import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [
    react(),
    {
      name: "remove-crossorigin",
      transformIndexHtml(html) {
        return html
          .replace(/ crossorigin/g, "")
          .replace(/src="(\/assets\/[^"]+\.js)"/, 'src="$1?v=20260712"');
      },
    },
  ],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8787", changeOrigin: true },
    },
  },
});

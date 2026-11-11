import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/api": "http://localhost:5174",
      "/assets": "http://localhost:5174",
      "/models": "http://localhost:5174",
      "/ws": { target: "ws://localhost:5174", ws: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});

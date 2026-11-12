import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// the screen talks to two services, both through this dev server so the browser sees one origin:
//   /shop -> the simulated business (only its fake sign-in is used from here)
//   /api  -> the hivemind chat service
const backend = "http://127.0.0.1:8080";
const agents = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/shop": { target: backend, changeOrigin: true, rewrite: (path) => path.replace(/^\/shop/, "") },
      "/api": { target: agents, changeOrigin: true, rewrite: (path) => path.replace(/^\/api/, "") },
    },
  },
});

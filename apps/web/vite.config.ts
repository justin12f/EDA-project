import { defineConfig } from "vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsConfigPaths from "vite-tsconfig-paths";

// The TanStack Start SSR plugin assumes a full Vite dev/build server; it is not
// compatible with the mini Vite server Vitest spins up internally (it crashes in
// @tanstack/server-functions-plugin's configureServer hook because Vitest's server
// object doesn't carry the shape the plugin expects). Skip it under `vitest`.
const isTest = !!process.env.VITEST;

export default defineConfig({
  plugins: [
    tsConfigPaths({ projects: ["./tsconfig.json"] }),
    tailwindcss(),
    ...(isTest ? [] : [tanstackStart({ target: "node-server", customViteReactPlugin: true })]),
    react(),
  ],
  server: { port: 3000 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/unit/**/*.test.{ts,tsx}"],
  },
});

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import tsConfigPaths from "vite-tsconfig-paths";

// Deliberately without the TanStack Start plugin. It assumes a full Vite dev or
// build server and crashes inside @tanstack/server-functions-plugin's
// configureServer hook against the mini server Vitest runs internally. Component
// tests do not need SSR wiring — they need JSX, path aliases and jsdom.
export default defineConfig({
  plugins: [tsConfigPaths({ projects: ["./tsconfig.json"] }), react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/unit/**/*.test.{ts,tsx}"],
  },
});

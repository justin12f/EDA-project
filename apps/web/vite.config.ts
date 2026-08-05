import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import tsConfigPaths from "vite-tsconfig-paths";

// The test configuration lives in vitest.config.ts, not here. Vitest bundles its
// own nested copy of Vite, so its `UserConfig` is a structurally different type
// from this one — putting a `test` block in this file makes the two collide and
// typecheck fails on plugin variance. Two files, no clash.
export default defineConfig({
  plugins: [
    tsConfigPaths({ projects: ["./tsconfig.json"] }),
    tailwindcss(),
    tanstackStart({ target: "node-server", customViteReactPlugin: true }),
    react(),
  ],
  server: { port: 3000 },
});

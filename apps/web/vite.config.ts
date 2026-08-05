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
    tanstackStart({ customViteReactPlugin: true }),
    react(),
  ],
  server: { port: 3000 },
  // The monorepo keeps one .env at the repo root — Python reads it via
  // Settings' env_file, and this points Vite at the same file rather than
  // maintaining a second copy of VITE_SUPABASE_URL etc. that could drift.
  envDir: "../..",
  // A route component reached only through the router's own automatic
  // code-splitting (`?tsr-split=component`) is the FIRST thing to import
  // `@tanstack/react-router` in a cold session. That triggers Vite's
  // mid-session "new dependency found, re-optimizing" path instead of
  // discovering it during the initial scan — and the split chunk that
  // rendered before the reload keeps a reference to the pre-reload module,
  // which is where "React is null" inside that chunk comes from. Listing it
  // (and the other real dependencies) here makes the first scan find
  // everything, so there is no second pass to race against.
  optimizeDeps: {
    include: [
      "@tanstack/react-router",
      "@tanstack/react-query",
      "@supabase/supabase-js",
      "clsx",
      "tailwind-merge",
    ],
  },
});

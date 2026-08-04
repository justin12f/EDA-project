# ADR-0005: Porting the Lovable frontend into the repository

**Status:** Proposed
**Date:** 2026-08-03
**Deciders:** Project owner

## Context

The frontend was designed in Lovable as project **Data Whisperer**, product name **Lumen — Agentic Data Platform**. Its source has been extracted into `frontend/` (see [design-system.md](design-system.md) for the visual language).

What was imported:

```
src/styles.css                          Tailwind v4 theme — the whole design system
src/components/dashboard/types.ts       DataSource model + fixtures
src/components/dashboard/Sidebar.tsx    brand, command bar, source list, user card
src/components/dashboard/ChatPanel.tsx  breadcrumb, agent badge, bubbles, ProposalCard
src/components/dashboard/RightPanel.tsx tabs, KPI tiles, line chart, distribution
src/components/dashboard/DatabaseView.tsx ERD canvas, crow's-foot edges, minimap, SQL view
src/components/dashboard/DataGrid.tsx   cleaned-data preview (built, not yet mounted)
src/components/dashboard/PipelineCanvas.tsx pipeline DAG (built, not yet mounted)
src/routes/__root.tsx, src/routes/index.tsx, src/router.tsx, src/start.ts, src/server.ts
src/lib/utils.ts, src/lib/error-page.ts, src/lib/error-capture.ts
```

Three properties of the imported code shape this decision:

1. **The design is finished and it is good.** A strict dark palette, two accents with distinct jobs (blue = interactive, amber = AI), tabular-numeric mono for every figure, and a coherent motion vocabulary (`ai-pulse`, `amber-fade-in`, `input-glow`).
2. **Every byte of data is a fixture.** `DATA_SOURCES` is a hardcoded array; the ERD is `buildSchema()`; the KPI figures and the 30-point line chart are literals. There is no network call anywhere. `nullsFixed` — a single `useState<boolean>` threaded through three components — is the entire "agent applied a change" simulation.
3. **It depends on Lovable-proprietary packages.** `vite.config.ts` imports `@lovable.dev/vite-tanstack-config`, and `src/routes/mcp.ts` + `src/lib/mcp/*` import `@lovable.dev/mcp-js`. Neither resolves outside Lovable's sandbox. The MCP server it generates is, in its own words, public and unauthenticated.

Note also that `RightPanel` mounts `DatabaseView` but renders placeholders for the `pipeline` and `data` tabs, even though `PipelineCanvas.tsx` and `DataGrid.tsx` are fully built. Two finished components are simply not wired up.

## Decision

Port to a **plain TanStack Start + Vite + Tailwind v4 application** at `apps/web`, preserving the design exactly and replacing everything Lovable-specific.

| Lovable dependency | Replacement |
|--------------------|-------------|
| `@lovable.dev/vite-tanstack-config` | explicit `vite.config.ts` with `@tanstack/react-start/plugin/vite`, `@vitejs/plugin-react`, `@tailwindcss/vite`, `vite-tsconfig-paths` |
| `@lovable.dev/mcp-js` + `src/routes/mcp.ts` | **deleted.** The four read-only tools it exposed (`list_data_sources`, `get_data_quality_report`, `get_cleaning_pipeline`, `get_database_schema`) are reimplemented as authenticated REST endpoints on `services/api`. If an MCP surface is wanted later it is built on the API, behind auth. |
| `componentTagger`, sandbox port detection | dropped |
| Bun (`bun.lock`, `bunfig.toml`) | npm (Node 24 / npm 11 are already installed; no extra package manager to provision) |

**All fixtures move to `src/mocks/`** and are served through a `dataMode` switch (`mock | live`). Until an endpoint exists, `live` falls back to `mock` and logs. This keeps the app runnable and demo-able at every commit while endpoints land one at a time.

**Data access is TanStack Query against `services/api`** through a generated client. The API's OpenAPI schema is the contract; types are generated, never hand-written.

The three simulated behaviours are re-pointed at real ones:

| Prototype | Real |
|-----------|------|
| `nullsFixed: boolean` prop drilling | `Proposal.status` from `GET /v1/proposals/{id}` |
| `onAcceptPipeline()` sets local state | `POST /v1/proposals/{id}/accept` → enqueues a Run |
| static chat transcript | SSE stream from `GET /v1/runs/{id}/events` |

**Routes** grow from the single `/` to: `/` (marketing), `/login`, `/signup`, `/app` (workspace, the imported dashboard), `/app/sources`, `/app/runs/$runId`, `/settings/{profile,members,billing,usage}`.

## Options Considered

### Option A — Keep the project in Lovable, sync via its GitHub integration

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Design fidelity | Perfect |
| Control | Low |

**Pros:** two-way sync; visual editing stays available; zero porting work.
**Cons:** connecting creates a *new* repository (Lovable does not import an existing one), splitting the project across two repos; the proprietary packages remain load-bearing; the unauthenticated MCP server stays in the tree; the frontend cannot be built in CI without Lovable's registry.

### Option B — Port to plain TanStack Start (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Design fidelity | Perfect — the design lives in `styles.css` and Tailwind classes, both portable |
| Control | Total |

**Pros:** one repository; standard toolchain; the app builds anywhere; SSR and file-based routing are retained, so the imported components need no structural change; the Lovable project remains available as a design sandbox.
**Cons:** `vite.config.ts` must be reconstructed by hand; the MCP route is lost (deliberately); future Lovable edits must be re-imported manually.

### Option C — Rewrite in Next.js

| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Design fidelity | Perfect, but every route file is rewritten |
| Control | Total |

**Pros:** the largest ecosystem; the best-known deployment story.
**Cons:** discards working TanStack Router/Query/Start integration for no product gain; `createFileRoute`, `shellComponent`, and the router context all have to be re-expressed. Effort with no user-visible return.

### Option D — Strip SSR, ship a Vite SPA

**Pros:** simplest possible build; static hosting.
**Cons:** loses the server route layer that the BFF pattern in ADR-0001 depends on, and the marketing route loses SSR/SEO. The BFF would have to move somewhere else, so the complexity is relocated, not removed.

## Trade-off Analysis

The design is portable and the plumbing is not — that asymmetry decides it. `styles.css` and the Tailwind class strings carry ~100% of the visual identity and depend on nothing Lovable-specific; `vite.config.ts` and the MCP wiring carry ~0% of it and depend on Lovable entirely. Option B keeps what matters and discards what does not.

Option A's blocker is structural rather than aesthetic: Lovable's Git integration creates a new repository and cannot adopt this one, so choosing it means the Python engine and its frontend live apart permanently. That is the wrong seam for a product where the frontend's whole job is rendering the engine's output.

Deleting the MCP server is not incidental. It is described in the project's own changelog as public and unauthenticated, exposing workspace data sources, quality audits, pipelines, and schema to anyone with the URL. Under ADR-0002 that is disqualifying.

Keeping the fixtures behind a `mock | live` switch rather than deleting them preserves something valuable: a UI that always runs. Deleting them makes the frontend unbootable until the last endpoint exists, which is the worst possible ordering for a solo build.

## Consequences

**Easier**
- One repo, one CI pipeline, one `docker compose up`.
- Building the frontend without network access to a proprietary registry.
- Growing routes for auth, settings and billing — file-based routing already supports it.
- Demoing at any commit, because `mock` mode always renders.

**Harder**
- Vite/TanStack config becomes ours to maintain and upgrade.
- Design changes made in Lovable must be re-imported by hand.
- Two finished components (`DataGrid`, `PipelineCanvas`) need wiring and real data before their tabs can stop showing placeholders.

**To revisit**
- Re-adding an MCP server on `services/api`, authenticated, once the REST surface is stable — the tool definitions in `src/lib/mcp/tools/` are a good specification for it.
- Extracting the design tokens into a shared package if a second surface (docs site, embeddable widget) appears.

## Action Items

1. [ ] Create `apps/web` from the imported `frontend/` tree; write `package.json`, `vite.config.ts`, `tsconfig.json`, `components.json`.
2. [ ] Delete `src/routes/mcp.ts` and `src/lib/mcp/`; record the four tool contracts as API endpoint specs.
3. [ ] Move fixtures to `src/mocks/`; add the `dataMode` switch and an API client seam.
4. [ ] Verify the design against the captured reference: palette, `57px` header rhythm, mono figures, `ai-pulse`/`amber-fade-in` motion.
5. [ ] Add routes for marketing, auth, workspace, settings.
6. [ ] Generate the typed API client from the OpenAPI schema.
7. [ ] Replace `nullsFixed` with real `Proposal` state; wire accept/reject to the API.
8. [ ] Wire `PipelineCanvas` and `DataGrid` into the `pipeline` and `data` tabs.
9. [ ] Add Playwright coverage for the accept-proposal flow and a visual-regression baseline.

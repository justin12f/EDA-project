# Design system — Lumen

Clean · minimal · pastel. Light by default, dark as a first-class variant.

Canonical source: [`apps/web/src/styles.css`](../../frontend/src/styles.css). This document
explains the intent so every new surface stays consistent without re-deriving it.

Lineage: two Lovable prototypes were merged — *Data Whisperer / Lumen* (three-pane layout,
ProposalCard, interactive ERD, KPI tiles) and *Agentic Insights / Nucleus* (active context,
saved pipelines, typed pipeline steps, thinking pills). The **information architecture** comes
from both. The **visual language below is new**: the prototypes' near-black canvas with
saturated blue/amber has been replaced by a pastel, high-air, light-first system.

---

## 1. Colour

Neutrals are cool and near-white — never pure grey, never pure black. Accents are pastel
**as backgrounds** and saturated **as ink**, which is what lets a pastel system stay accessible.

Every accent hue has exactly three stops:

| Stop | Role | May carry text? |
|------|------|-----------------|
| `tint` | pastel wash behind a block | no — it *is* the background |
| `soft` | pastel border, chart fill, progress bar | no |
| `ink`  | text, icon, solid button fill | **yes — this stop only** |

Writing text in `soft` is the single easiest way to break this system. Don't.

### Semantics

| Hue | Token | Means |
|-----|-------|-------|
| **Blue** | `primary` | **Interactive.** The person can act on this: buttons, links, focus rings, selection, the active row. |
| **Violet** | `ai` | **Agent.** The AI did this, is doing this, or proposes it. Agent avatars, thinking pills, proposals awaiting review, freshly-altered schema. |
| **Mint** | `success` | **Applied.** Only ever appears *after* a human accepted something. Never used for "loaded OK". |
| **Amber** | `warning` | Attention, drift, degraded, approaching quota. |
| **Rose** | `destructive` | Anomalies, deletions, failures. |

**The blue/violet split is the identity.** Blue = *you can do this*. Violet = *the AI did this*.
They never substitute for each other, and mint never appears before a human decision.

### Light values

| Token | Value | | Token | Value |
|-------|-------|-|-------|-------|
| `background` | `#FAFAFC` | | `primary` | `#2563EB` |
| `surface` | `#FFFFFF` | | `primary-tint` / `-soft` | `#EEF3FF` / `#C7D9FD` |
| `surface-2` | `#F4F4F7` | | `ai` | `#7C3AED` |
| `foreground` | `#17171C` | | `ai-tint` / `-soft` | `#F4F0FF` / `#DDD0FB` |
| `muted-foreground` | `#6E6E7A` | | `success` | `#0F7A55` |
| `border` | `#E8E8EE` | | `warning` | `#9A5B00` |
| `border-strong` | `#D8D8E0` | | `destructive` | `#C0334B` |
| `sidebar` | `#F7F7FA` | | `radius` | `10px` |

Every `ink` stop clears 4.5:1 on `background` and on `surface`. `muted-foreground` clears 5.6:1
on white — it is safe for real text, not only for labels.

### Dark values

Same semantics, inverted lightness: canvas `#0E0E12`, surface `#16161B`, border `#26262F`.
Accent inks become the *light* end of their hue (`primary #8AB0FB`, `ai #BFA6FB`) and the tints
become deep, desaturated versions of it. `.dark` on `<html>` switches the whole set.

---

## 2. Type

- **Sans:** Inter. `font-feature-settings: "cv02","cv03","cv04","cv11","ss01"`, `letter-spacing: -0.006em`.
- **Mono:** JetBrains Mono — for **every** number, identifier, column name, SQL type, path, duration and status token. If it is a value the system produced, it is mono.
- `.num-tabular` on all figures so digits do not shift while they count up.

| Role | Size / weight | Notes |
|------|---------------|-------|
| Eyebrow / section label | 10.5px · 600 · uppercase · `0.08em` | `.label-eyebrow`, always `muted-foreground` |
| Metadata, units, counts | 11.5px · mono | |
| UI text, buttons, rows | 13px · 500 | |
| Body / chat | 14px · 400 · `leading-relaxed` | |
| Panel title | 15px · 600 · `-0.01em` | |
| KPI figure | 24px · 600 · mono · tabular | the largest thing on any screen |
| Page heading | 20px · 600 | settings and marketing only |

The prototypes ran 9.5–13px. This system runs 10.5–15px with more air. The read should be
**calm instrument**, not dense terminal.

---

## 3. Space, shape, elevation

- Radius `10px` (`--radius`). Cards and inputs `rounded-lg`; pills, chips and badges `rounded-md`; avatars `rounded-lg` (squircle, never circles except user photos).
- **Every top bar is `56px`.** Sidebar header, chat header, and right-panel tabs share one baseline across all three columns. This is the load-bearing measurement of the layout — new panels must match it.
- Layout: `272px` sidebar · fluid centre · `460px` right panel. Focus modes collapse a column to `w-0` over `280ms cubic-bezier(0.16,1,0.3,1)`.
- Row height `34px` for lists, `30px` for ERD columns and table rows.
- Padding: `16px` inside cards, `20px` inside panels, `12px` in dense rows.
- **Structure comes from `1px` borders, not shadow.** Shadows (`--elevation-*`) only lift things that genuinely float: popovers, dropdowns, ERD cards, the minimap, drag previews. They are deliberately near-invisible.

---

## 4. Motion

| Utility | Purpose |
|---------|---------|
| `.ai-pulse` | 2.6s violet halo on the agent-active dot. Ambient presence — calm, never urgent. |
| `.ai-ring` | Static violet ring on a proposal awaiting review. |
| `.ai-flash` | 1600ms one-shot violet wash over an object the agent just changed. |
| `.stream-caret` | Token-streaming caret in agent messages. |
| `.input-focus` | 3px blue focus ring on the composer and search. |
| `.skeleton` | 1.4s shimmer while data loads. Never a spinner for content. |
| KPI count-up | `1 - (1-p)³` over 1200ms via `requestAnimationFrame`. |

Motion says *something changed, and here is where*. It never decorates.
`prefers-reduced-motion` collapses everything to `0.01ms`.

---

## 5. Information architecture (merged)

```
┌─ Sidebar 272px ──────┬─ Agent workspace ──────────┬─ Inspector 460px ──────┐
│ brand · ⌘K search    │ breadcrumb · agent state   │ Pipeline │ Data │       │
│                      │                            │ Analytics │ Schema     │
│ ACTIVE CONTEXT       │ thinking pills             │                        │
│  current source      │  ✓ Inspected schema        │ Pipeline: typed steps  │
│  current table       │  ✓ Sampled 50k rows        │  SOURCE FILTER VALIDATE│
│                      │                            │  TRANSFORM SINK        │
│ DATA SOURCES         │ message bubbles            │  + Run pipeline        │
│  · production_pg  ●  │                            │                        │
│  · analytics_mysql ● │ ┌ ProposalCard ──────────┐ │ Data: grid with        │
│  · users_2024.csv    │ │ awaiting review        │ │  fixed/cast/anomaly    │
│  + Connect source    │ │ + Drop NULLs · country │ │                        │
│                      │ │ ~15,420 rows · 1.2s    │ │ Analytics: KPI tiles   │
│ SAVED PIPELINES      │ │ [Accept]  [Reject]     │ │  + charts              │
│  · Clean signups  6  │ └────────────────────────┘ │                        │
│  · Normalize tx   4  │                            │ Schema: ERD canvas,    │
│                      │ composer                   │  crow's-foot, minimap, │
│ user · plan · usage  │  source · context · attach │  SQL view              │
└──────────────────────┴────────────────────────────┴────────────────────────┘
```

From *Lumen*: three-pane shell, ProposalCard, ERD canvas, KPI tiles, chart widgets.
From *Nucleus*: Active Context, Saved Pipelines, typed pipeline steps with per-step state,
Run pipeline, thinking pills, plan + usage in the user card.

---

## 6. Signature components

- **ProposalCard** — the product's central object. `ai-tint` + `ai-ring` + `awaiting review` → mint border + `applied`, or muted + `rejected`. Always carries: a diff (`+ Drop NULLs · country_code`), an impact estimate (`~15,420 rows · est. 1.2s`), a rationale, and exactly two actions.
- **Thinking pill** — `11.5px`, `ai-tint` background, check glyph, one clause: `✓ Sampled 50k rows`. Shows the agent's work without a wall of text.
- **Pipeline step** — status glyph (done / running / queued), name, typed badge (`SOURCE` `FILTER` `VALIDATE` `TRANSFORM` `SINK`), and a one-line effect (`Removed 18,204 rows missing PK`).
- **KPI tile** — eyebrow label, 24px mono tabular figure, delta with trend arrow. Gains a mint border and a `fixed` chip once a proposal lands.
- **ERD** — `248px` table cards on a `16px` dot grid, `30px` column rows, orthogonal edges with crow's-foot (many) and bar (one) terminators, hover dims unrelated tables, plus a minimap.
- **Chip / kbd / status badge** — `rounded-md`, `1px` border, mono, uppercase, `10.5px`.

---

## 7. Rules for new surfaces

1. Backgrounds are pastel; **text and solid fills use the `ink` stop only.**
2. Numbers are mono and tabular. Always.
3. Blue = the person can act. Violet = the AI acted. Mint = a human approved it.
4. Top bars are `56px`. Panels are separated by `1px` borders, not shadow.
5. Section labels use `.label-eyebrow`. Don't invent a second label style.
6. Any AI-originated change is visibly attributable — a violet badge, an `altered` chip, or a proposal card. **The user must never have to wonder whether they or the agent did something.**
7. Loading is a `.skeleton`, not a spinner. Empty states get one sentence and one action.
8. Support both themes and `prefers-reduced-motion` from the first commit, not as a retrofit.

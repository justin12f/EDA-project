import { useState } from "react";
import { Play, Check, Loader2, Code2, Brain, X, Filter, Sparkles, Database, Plus } from "lucide-react";
import type { DataSource } from "./types";

type NodeStatus = "done" | "running" | "queued";
type NodeKind = "source" | "clean" | "transform" | "ml" | "sink";

type GraphNode = {
  id: string;
  label: string;
  sub: string;
  kind: NodeKind;
  status: NodeStatus;
  x: number; // grid col
  y: number; // grid row
  code: string;
  why: string;
};

const NODES: GraphNode[] = [
  {
    id: "src",
    label: "Raw Source",
    sub: "users · 482k",
    kind: "source",
    status: "done",
    x: 0,
    y: 1,
    code: "SELECT * FROM users WHERE created_at > now() - interval '90 days';",
    why: "Bounded the working set to the last 90 days where I observed schema drift.",
  },
  {
    id: "drop",
    label: "Drop NULLs",
    sub: "country_code",
    kind: "clean",
    status: "done",
    x: 1,
    y: 0,
    code: "df = df.dropna(subset=['country_code'])",
    why: "3.2% of rows had a NULL country_code — well below the 5% loss threshold.",
  },
  {
    id: "cast",
    label: "Cast Dates",
    sub: "ISO 8601 · UTC",
    kind: "transform",
    status: "done",
    x: 1,
    y: 2,
    code: "df['created_at'] = pd.to_datetime(df['created_at'], utc=True)",
    why: "Two formats coexisted (Z suffix vs naïve). Normalizing prevents skewed time-series.",
  },
  {
    id: "dedupe",
    label: "Dedupe",
    sub: "email_hash",
    kind: "clean",
    status: "running",
    x: 2,
    y: 1,
    code: "df = df.drop_duplicates(subset=['email_hash'], keep='last')",
    why: "412 duplicate email_hash values created inflated cohort sizes.",
  },
  {
    id: "anom",
    label: "Anomaly Tag",
    sub: "iforest · k=12",
    kind: "ml",
    status: "queued",
    x: 3,
    y: 0,
    code: "model = IsolationForest(n_estimators=200, contamination=0.02).fit(X)",
    why: "Daily signup variance widened past 3σ on Aug 14 — worth flagging.",
  },
  {
    id: "sink",
    label: "Persist",
    sub: "users_clean_v3",
    kind: "sink",
    status: "queued",
    x: 3,
    y: 2,
    code: "df.to_sql('users_clean_v3', engine, if_exists='replace')",
    why: "Materialize so downstream dashboards can read without re-running the pipeline.",
  },
];

const EDGES: [string, string][] = [
  ["src", "drop"],
  ["src", "cast"],
  ["drop", "dedupe"],
  ["cast", "dedupe"],
  ["dedupe", "anom"],
  ["dedupe", "sink"],
];

const CELL_W = 220;
const CELL_H = 120;
const PAD_X = 32;
const PAD_Y = 32;
const NODE_W = 180;
const NODE_H = 78;

function nodeCenter(n: GraphNode) {
  return {
    cx: PAD_X + n.x * CELL_W + NODE_W / 2,
    cy: PAD_Y + n.y * CELL_H + NODE_H / 2,
  };
}

export function PipelineCanvas({ source }: { source: DataSource }) {
  const [active, setActive] = useState<GraphNode | null>(null);

  const width = PAD_X * 2 + 3 * CELL_W + NODE_W;
  const height = PAD_Y * 2 + 2 * CELL_H + NODE_H;

  return (
    <div className="flex h-full">
      <div className="grid-bg relative flex-1 overflow-auto">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-sidebar/80 px-4 py-2 backdrop-blur">
          <div>
            <div className="text-xs font-semibold">Cleaning pipeline · v3</div>
            <div className="text-[10px] text-muted-foreground">
              scoped to <span className="font-mono text-primary">{source.name}</span> · 6 nodes
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <button className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-[10px] text-muted-foreground hover:text-foreground">
              <Plus className="h-3 w-3" /> Node
            </button>
            <button className="flex items-center gap-1.5 rounded-md bg-primary px-2.5 py-1.5 text-[11px] font-medium text-primary-foreground shadow-[0_0_16px_-4px] shadow-primary/60 hover:brightness-110">
              <Play className="h-3 w-3" /> Run all
            </button>
          </div>
        </div>

        <svg width={width} height={height} className="block">
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="oklch(0.5 0.02 260)" />
            </marker>
            <marker
              id="arrowActive"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="oklch(0.72 0.18 245)" />
            </marker>
          </defs>

          {EDGES.map(([fromId, toId], i) => {
            const from = NODES.find((n) => n.id === fromId)!;
            const to = NODES.find((n) => n.id === toId)!;
            const f = nodeCenter(from);
            const t = nodeCenter(to);
            const x1 = f.cx + NODE_W / 2;
            const y1 = f.cy;
            const x2 = t.cx - NODE_W / 2;
            const y2 = t.cy;
            const mx = (x1 + x2) / 2;
            const isLive = to.status === "running" || from.status === "running";
            return (
              <path
                key={i}
                d={`M ${x1},${y1} C ${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                fill="none"
                stroke={isLive ? "oklch(0.72 0.18 245)" : "oklch(0.4 0.02 260)"}
                strokeWidth={isLive ? 1.5 : 1}
                strokeDasharray={isLive ? "4 4" : undefined}
                markerEnd={isLive ? "url(#arrowActive)" : "url(#arrow)"}
                className={isLive ? "animate-pulse" : ""}
              />
            );
          })}

          {NODES.map((n) => {
            const { cx, cy } = nodeCenter(n);
            const x = cx - NODE_W / 2;
            const y = cy - NODE_H / 2;
            const isActive = active?.id === n.id;
            return (
              <g
                key={n.id}
                transform={`translate(${x},${y})`}
                className="cursor-pointer"
                onClick={() => setActive(n)}
              >
                <rect
                  width={NODE_W}
                  height={NODE_H}
                  rx={10}
                  fill="oklch(0.19 0.009 260)"
                  stroke={
                    isActive
                      ? "oklch(0.72 0.18 245)"
                      : n.status === "running"
                        ? "oklch(0.72 0.18 245 / 0.6)"
                        : "oklch(0.27 0.012 260)"
                  }
                  strokeWidth={isActive ? 1.5 : 1}
                  style={
                    isActive || n.status === "running"
                      ? { filter: "drop-shadow(0 0 12px oklch(0.72 0.18 245 / 0.5))" }
                      : undefined
                  }
                />
                <foreignObject x={0} y={0} width={NODE_W} height={NODE_H}>
                  <div className="flex h-full w-full items-center gap-2.5 px-3">
                    <StatusBadge status={n.status} kind={n.kind} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate text-xs font-semibold text-foreground">
                          {n.label}
                        </span>
                        <span className="rounded border border-border px-1 font-mono text-[9px] uppercase text-muted-foreground">
                          {n.kind}
                        </span>
                      </div>
                      <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
                        {n.sub}
                      </div>
                    </div>
                  </div>
                </foreignObject>
              </g>
            );
          })}
        </svg>
      </div>

      {active && <NodeDetail node={active} onClose={() => setActive(null)} />}
    </div>
  );
}

function StatusBadge({ status, kind }: { status: NodeStatus; kind: NodeKind }) {
  const KindIcon =
    kind === "source" ? Database : kind === "ml" ? Sparkles : kind === "sink" ? Filter : Code2;
  if (status === "done") {
    return (
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-success/15 ring-1 ring-success/40">
        <Check className="h-3.5 w-3.5 text-success" />
      </div>
    );
  }
  if (status === "running") {
    return (
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary/15 ring-1 ring-primary/40">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
      </div>
    );
  }
  return (
    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-accent/60 ring-1 ring-border">
      <KindIcon className="h-3.5 w-3.5 text-muted-foreground" />
    </div>
  );
}

function NodeDetail({ node, onClose }: { node: GraphNode; onClose: () => void }) {
  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-border bg-card">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <StatusBadge status={node.status} kind={node.kind} />
          <div className="leading-tight">
            <div className="text-xs font-semibold">{node.label}</div>
            <div className="text-[10px] text-muted-foreground">{node.sub}</div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-3">
        <section>
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            <Code2 className="h-3 w-3" /> Action
          </div>
          <pre className="overflow-x-auto rounded-md border border-border bg-[oklch(0.13_0.008_260)] p-2.5 font-mono text-[11px] leading-relaxed text-foreground/90">
            <code>{node.code}</code>
          </pre>
        </section>

        <section>
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            <Brain className="h-3 w-3 text-primary" /> AI Justification
          </div>
          <div className="rounded-md border border-primary/30 bg-primary/5 p-2.5 text-[11px] leading-relaxed text-foreground/90">
            {node.why}
          </div>
        </section>

        <section>
          <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Stats
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="rows in" value="482,193" />
            <Stat label="rows out" value="481,247" />
            <Stat label="latency" value="1.8s" />
            <Stat label="memory" value="412MB" />
          </div>
        </section>
      </div>
    </aside>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-background/60 p-2">
      <div className="text-[9px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-0.5 font-mono text-xs">{value}</div>
    </div>
  );
}

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Key,
  Link2,
  Copy,
  Code2,
  Check,
  Database as DatabaseIcon,
  Table2,
  Shield,
  ZoomIn,
  ZoomOut,
  Maximize,
} from "lucide-react";

/* ──────────────────────────────────────────────────────────────────────── */
/* Schema model                                                             */
/* ──────────────────────────────────────────────────────────────────────── */

type ColKind = "pk" | "fk" | "unique" | null;
type Col = {
  name: string;
  type: string;
  kind?: ColKind;
  nullable?: boolean;
  ref?: { table: string; col: string };
};
type Tbl = {
  id: string;
  name: string;
  rows: string;
  cols: Col[];
  x: number; // canvas px
  y: number;
  isNew?: boolean;
  isAltered?: boolean;
};

const CANVAS_W = 980;
const CANVAS_H = 600;
const TBL_W = 240;
const ROW_H = 22;
const HEADER_H = 30;

function buildSchema(nullsFixed: boolean): Tbl[] {
  const base: Tbl[] = [
    {
      id: "users",
      name: "users",
      rows: "12.4M",
      x: 360,
      y: 150,
      isAltered: nullsFixed,
      cols: [
        { name: "id", type: "uuid", kind: "pk", nullable: false },
        { name: "email_hash", type: "varchar(64)", kind: "unique", nullable: false },
        {
          name: "country_code",
          type: "varchar(2)",
          kind: "fk",
          nullable: !nullsFixed,
          ref: { table: "countries", col: "code" },
        },
        {
          name: "plan_type",
          type: "varchar(16)",
          kind: "fk",
          nullable: true,
          ref: { table: "plans", col: "plan_code" },
        },
        { name: "created_at", type: "timestamptz", nullable: false },
      ],
    },
    {
      id: "signups",
      name: "signups",
      rows: "48.2K",
      x: 700,
      y: 40,
      cols: [
        { name: "signup_id", type: "int4", kind: "pk", nullable: false },
        {
          name: "user_id",
          type: "uuid",
          kind: "fk",
          nullable: false,
          ref: { table: "users", col: "id" },
        },
        { name: "source", type: "varchar(32)", nullable: true },
        { name: "occurred_at", type: "timestamptz", nullable: false },
      ],
    },
    {
      id: "orders",
      name: "orders",
      rows: "1.8M",
      x: 700,
      y: 320,
      cols: [
        { name: "id", type: "uuid", kind: "pk", nullable: false },
        {
          name: "user_id",
          type: "uuid",
          kind: "fk",
          nullable: false,
          ref: { table: "users", col: "id" },
        },
        { name: "amount_cents", type: "int8", nullable: false },
        { name: "currency", type: "varchar(3)", nullable: false },
        { name: "status", type: "varchar(16)", nullable: false },
      ],
    },
    {
      id: "plans",
      name: "plans",
      rows: "5",
      x: 40,
      y: 40,
      cols: [
        { name: "plan_code", type: "varchar(16)", kind: "pk", nullable: false },
        { name: "plan_name", type: "varchar(32)", nullable: false },
        { name: "price_cents", type: "int4", nullable: false },
      ],
    },
    {
      id: "countries",
      name: "countries",
      rows: "249",
      x: 40,
      y: 320,
      cols: [
        { name: "code", type: "varchar(2)", kind: "pk", nullable: false },
        { name: "name", type: "varchar(64)", nullable: false },
        { name: "region", type: "varchar(32)", nullable: true },
      ],
    },
  ];
  return base;
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Component                                                                */
/* ──────────────────────────────────────────────────────────────────────── */

export function DatabaseView({ nullsFixed, expanded }: { nullsFixed: boolean; expanded: boolean }) {
  const tables = useMemo(() => buildSchema(nullsFixed), [nullsFixed]);
  const [showSql, setShowSql] = useState(false);
  const [copied, setCopied] = useState(false);
  const [zoom, setZoom] = useState(expanded ? 1 : 0.78);
  const [hovered, setHovered] = useState<string | null>(null);
  const [pulseKey, setPulseKey] = useState(0);

  useEffect(() => {
    setZoom(expanded ? 1 : 0.78);
  }, [expanded]);
  useEffect(() => {
    if (nullsFixed) setPulseKey((k) => k + 1);
  }, [nullsFixed]);

  const copy = () => {
    navigator.clipboard?.writeText("postgres://lumen:****@prod-pg.lumen.io:5432/app");
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header actions */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-card px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="flex h-6 items-center gap-1.5 rounded-md border border-[#3B82F6]/30 bg-[#3B82F6]/[0.08] px-2 text-[10.5px] font-medium text-[#3B82F6]">
            <DatabaseIcon className="h-3 w-3" />
            PostgreSQL
            <span className="font-mono text-muted-foreground">15.4</span>
          </span>
          <span className="font-mono text-[10.5px] text-muted-foreground">
            {tables.length} tables · {tables.reduce((s, t) => s + t.cols.length, 0)} columns ·{" "}
            <span className="text-foreground/80">
              {tables.reduce((s, t) => s + t.cols.filter((c) => c.kind === "fk").length, 0)}
            </span>{" "}
            relations
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowSql((v) => !v)}
            className={`flex h-6 items-center gap-1 rounded-md border px-2 text-[10.5px] transition-colors ${
              showSql
                ? "border-[#3B82F6]/40 bg-[#3B82F6]/10 text-[#3B82F6]"
                : "border-border bg-background text-foreground/80 hover:text-foreground"
            }`}
          >
            <Code2 className="h-3 w-3" />
            SQL Schema
          </button>
          <button
            onClick={copy}
            className="flex h-6 items-center gap-1 rounded-md border border-border bg-background px-2 text-[10.5px] text-foreground/80 transition-colors hover:text-foreground"
          >
            {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
            {copied ? "Copied" : "Connection string"}
          </button>
        </div>
      </div>

      {showSql ? (
        <SqlView tables={tables} />
      ) : (
        <ErdCanvas
          tables={tables}
          zoom={zoom}
          setZoom={setZoom}
          hovered={hovered}
          setHovered={setHovered}
          nullsFixed={nullsFixed}
          pulseKey={pulseKey}
        />
      )}
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────── */
/* ERD canvas                                                               */
/* ──────────────────────────────────────────────────────────────────────── */

function ErdCanvas({
  tables,
  zoom,
  setZoom,
  hovered,
  setHovered,
  nullsFixed,
  pulseKey,
}: {
  tables: Tbl[];
  zoom: number;
  setZoom: (n: number) => void;
  hovered: string | null;
  setHovered: (s: string | null) => void;
  nullsFixed: boolean;
  pulseKey: number;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);

  // Build edges (FK → referenced PK)
  const edges = useMemo(() => {
    const out: { from: Tbl; fromCol: Col; to: Tbl; toCol: Col }[] = [];
    for (const t of tables) {
      for (const c of t.cols) {
        if (c.kind === "fk" && c.ref) {
          const to = tables.find((x) => x.id === c.ref!.table);
          if (!to) continue;
          const toCol = to.cols.find((x) => x.name === c.ref!.col);
          if (!toCol) continue;
          out.push({ from: t, fromCol: c, to, toCol });
        }
      }
    }
    return out;
  }, [tables]);

  return (
    <div
      ref={wrapRef}
      className="relative flex-1 overflow-auto rounded-md border border-border bg-[#0a0a0c]"
    >
      {/* dotted background */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: "radial-gradient(#1b1b20 1px, transparent 1px)",
          backgroundSize: "16px 16px",
        }}
      />

      {/* zoom toolbar */}
      <div className="absolute right-3 top-3 z-20 flex items-center gap-1 rounded-md border border-border bg-card/80 p-1 backdrop-blur">
        <ZoomBtn
          onClick={() => setZoom(Math.max(0.5, +(zoom - 0.1).toFixed(2)))}
          icon={<ZoomOut className="h-3 w-3" />}
        />
        <span className="num-tabular w-9 text-center font-mono text-[10px] text-muted-foreground">
          {Math.round(zoom * 100)}%
        </span>
        <ZoomBtn
          onClick={() => setZoom(Math.min(1.4, +(zoom + 0.1).toFixed(2)))}
          icon={<ZoomIn className="h-3 w-3" />}
        />
        <span className="mx-1 h-3 w-px bg-border" />
        <ZoomBtn onClick={() => setZoom(1)} icon={<Maximize className="h-3 w-3" />} />
      </div>

      {/* canvas */}
      <div
        className="relative"
        style={{
          width: CANVAS_W * zoom,
          height: CANVAS_H * zoom,
        }}
      >
        <div
          className="absolute left-0 top-0 origin-top-left"
          style={{ width: CANVAS_W, height: CANVAS_H, transform: `scale(${zoom})` }}
        >
          {/* SVG edges below the cards */}
          <EdgeLayer edges={edges} hovered={hovered} nullsFixed={nullsFixed} pulseKey={pulseKey} />

          {/* Table cards */}
          {tables.map((t) => (
            <TableCard
              key={`${t.id}-${pulseKey}`}
              table={t}
              hovered={hovered}
              onHover={setHovered}
              edges={edges}
            />
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="pointer-events-none absolute bottom-3 left-3 z-10 flex items-center gap-3 rounded-md border border-border bg-card/80 px-2 py-1 font-mono text-[9.5px] text-muted-foreground backdrop-blur">
        <span className="flex items-center gap-1">
          <Key className="h-2.5 w-2.5 text-[#F5B946]" /> PK
        </span>
        <span className="flex items-center gap-1">
          <Link2 className="h-2.5 w-2.5 text-[#7C8794]" /> FK
        </span>
        <span className="flex items-center gap-1">
          <Shield className="h-2.5 w-2.5 text-[#3B82F6]" /> UNIQUE
        </span>
        <span className="flex items-center gap-1">
          <span className="h-px w-3 bg-[#F59E0B]" /> recently added
        </span>
      </div>

      {/* Minimap */}
      <Minimap tables={tables} hovered={hovered} />
    </div>
  );
}

function ZoomBtn({ icon, onClick }: { icon: React.ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
    >
      {icon}
    </button>
  );
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Table card                                                               */
/* ──────────────────────────────────────────────────────────────────────── */

function TableCard({
  table,
  hovered,
  onHover,
  edges,
}: {
  table: Tbl;
  hovered: string | null;
  onHover: (s: string | null) => void;
  edges: { from: Tbl; to: Tbl }[];
}) {
  const isHovered = hovered === table.id;
  const isRelated = hovered
    ? edges.some(
        (e) =>
          (e.from.id === hovered && e.to.id === table.id) ||
          (e.to.id === hovered && e.from.id === table.id),
      )
    : false;
  const dim = hovered && !isHovered && !isRelated;
  const highlight = table.isNew || table.isAltered;

  return (
    <div
      onMouseEnter={() => onHover(table.id)}
      onMouseLeave={() => onHover(null)}
      className={`absolute overflow-hidden rounded-md border bg-card shadow-[0_8px_30px_-12px_rgba(0,0,0,0.6)] transition-all duration-300 ${
        highlight
          ? "amber-fade-in border-[#F59E0B]/50"
          : isHovered
            ? "border-[#3B82F6]/60"
            : "border-border"
      } ${dim ? "opacity-40" : "opacity-100"}`}
      style={{ left: table.x, top: table.y, width: TBL_W }}
    >
      {/* header */}
      <div className="flex items-center justify-between border-b border-border bg-[#17171b] px-2.5 py-1.5">
        <div className="flex items-center gap-1.5">
          <Table2 className="h-3 w-3 text-muted-foreground" />
          <span className="font-mono text-[11.5px] font-semibold tracking-tight text-foreground">
            {table.name}
          </span>
          {table.isAltered && (
            <span className="rounded-sm border border-[#F59E0B]/30 bg-[#F59E0B]/10 px-1 py-px font-mono text-[8.5px] uppercase tracking-wider text-[#F59E0B]">
              altered
            </span>
          )}
        </div>
        <span className="num-tabular font-mono text-[9.5px] text-muted-foreground">
          {table.rows}
        </span>
      </div>

      {/* columns */}
      <div>
        {table.cols.map((c) => (
          <div
            key={c.name}
            className="flex items-center gap-1.5 border-b border-border/60 px-2.5 font-mono text-[10.5px] last:border-b-0"
            style={{ height: ROW_H }}
          >
            <span className="flex w-3 shrink-0 justify-center">
              {c.kind === "pk" && <Key className="h-2.5 w-2.5 text-[#F5B946]" />}
              {c.kind === "fk" && <Link2 className="h-2.5 w-2.5 text-[#7C8794]" />}
              {c.kind === "unique" && <Shield className="h-2.5 w-2.5 text-[#3B82F6]" />}
            </span>
            <span
              className={`flex-1 truncate ${c.kind === "pk" ? "text-foreground" : "text-foreground/85"}`}
            >
              {c.name}
            </span>
            <span className="text-muted-foreground">{c.type}</span>
            <span
              className={`w-[52px] shrink-0 text-right text-[9px] uppercase tracking-wider ${
                c.nullable ? "text-muted-foreground/70" : "text-foreground/55"
              }`}
            >
              {c.nullable ? "nullable" : "not null"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────── */
/* SVG edges (orthogonal + crow's-foot)                                     */
/* ──────────────────────────────────────────────────────────────────────── */

function colYOffset(table: Tbl, colName: string) {
  const idx = table.cols.findIndex((c) => c.name === colName);
  return HEADER_H + idx * ROW_H + ROW_H / 2;
}

function EdgeLayer({
  edges,
  hovered,
  nullsFixed,
  pulseKey,
}: {
  edges: { from: Tbl; fromCol: Col; to: Tbl; toCol: Col }[];
  hovered: string | null;
  nullsFixed: boolean;
  pulseKey: number;
}) {
  return (
    <svg
      width={CANVAS_W}
      height={CANVAS_H}
      className="pointer-events-none absolute inset-0"
      style={{ overflow: "visible" }}
    >
      <defs>
        <filter id="amber-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {edges.map((e, i) => {
        // FK is the "many" side, PK is the "one" side
        const fromSideRight = e.from.x + TBL_W / 2 < e.to.x + TBL_W / 2;
        const fx = fromSideRight ? e.from.x + TBL_W : e.from.x;
        const tx = fromSideRight ? e.to.x : e.to.x + TBL_W;
        const fy = e.from.y + colYOffset(e.from, e.fromCol.name);
        const ty = e.to.y + colYOffset(e.to, e.toCol.name);

        const ext = 22;
        const fStart = fromSideRight ? fx + ext : fx - ext;
        const tStart = fromSideRight ? tx - ext : tx + ext;
        const midX = (fStart + tStart) / 2;
        // 3-segment stair using midX
        const path = `M ${fx} ${fy} H ${midX} V ${ty} H ${tx}`;

        const isAmber = nullsFixed && e.from.id === "users" && e.fromCol.name === "country_code";
        const stroke = isAmber ? "#F59E0B" : "#2A2A30";
        const strokeHover = isAmber ? "#F59E0B" : "#3B82F6";
        const isHL = hovered && (hovered === e.from.id || hovered === e.to.id);

        return (
          <g key={`${i}-${pulseKey}`} className={isAmber ? "amber-edge" : ""}>
            <path
              d={path}
              fill="none"
              stroke={isHL ? strokeHover : stroke}
              strokeWidth={isAmber ? 1.4 : 1}
              strokeOpacity={hovered && !isHL ? 0.25 : 1}
              filter={isAmber ? "url(#amber-glow)" : undefined}
            />
            {/* one-side: single bar at PK end */}
            <OneMarker
              x={tx}
              y={ty}
              side={fromSideRight ? "left" : "right"}
              color={isHL ? strokeHover : stroke}
            />
            {/* many-side: crow's foot at FK end */}
            <CrowsFoot
              x={fx}
              y={fy}
              side={fromSideRight ? "right" : "left"}
              color={isHL ? strokeHover : stroke}
            />
          </g>
        );
      })}
    </svg>
  );
}

function OneMarker({
  x,
  y,
  side,
  color,
}: {
  x: number;
  y: number;
  side: "left" | "right";
  color: string;
}) {
  const dx = side === "right" ? -10 : 10;
  return (
    <>
      <line x1={x + dx} y1={y - 5} x2={x + dx} y2={y + 5} stroke={color} strokeWidth={1.2} />
      <line x1={x} y1={y} x2={x + dx} y2={y} stroke={color} strokeWidth={1.2} />
    </>
  );
}

function CrowsFoot({
  x,
  y,
  side,
  color,
}: {
  x: number;
  y: number;
  side: "left" | "right";
  color: string;
}) {
  const dx = side === "right" ? 10 : -10;
  return (
    <>
      <line x1={x} y1={y} x2={x + dx} y2={y - 5} stroke={color} strokeWidth={1.2} />
      <line x1={x} y1={y} x2={x + dx} y2={y + 5} stroke={color} strokeWidth={1.2} />
      <line x1={x} y1={y} x2={x + dx} y2={y} stroke={color} strokeWidth={1.2} />
    </>
  );
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Minimap                                                                  */
/* ──────────────────────────────────────────────────────────────────────── */

function Minimap({ tables, hovered }: { tables: Tbl[]; hovered: string | null }) {
  const W = 160,
    H = 100;
  const sx = W / CANVAS_W,
    sy = H / CANVAS_H;
  return (
    <div className="absolute bottom-3 right-3 z-10 rounded-md border border-border bg-card/85 p-1.5 backdrop-blur">
      <div className="mb-1 flex items-center justify-between px-0.5">
        <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
          minimap
        </span>
        <span className="font-mono text-[9px] text-muted-foreground">{tables.length}t</span>
      </div>
      <svg width={W} height={H} className="rounded-sm bg-[#0a0a0c]">
        {tables.map((t) => {
          const colCount = t.cols.length;
          const tH = (HEADER_H + colCount * ROW_H) * sy;
          const tW = TBL_W * sx;
          const active = hovered === t.id;
          return (
            <rect
              key={t.id}
              x={t.x * sx}
              y={t.y * sy}
              width={tW}
              height={tH}
              fill={active ? "#3B82F6" : t.isAltered ? "#F59E0B" : "#2A2A30"}
              fillOpacity={active ? 0.7 : t.isAltered ? 0.55 : 0.65}
              stroke={active ? "#3B82F6" : "#3a3a40"}
              strokeWidth={0.5}
              rx={1}
            />
          );
        })}
      </svg>
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────── */
/* SQL view                                                                 */
/* ──────────────────────────────────────────────────────────────────────── */

function SqlView({ tables }: { tables: Tbl[] }) {
  const sql = tables
    .map((t) => {
      const cols = t.cols
        .map((c) => {
          const parts = [`  ${c.name}`, c.type];
          if (c.kind === "pk") parts.push("PRIMARY KEY");
          if (c.kind === "unique") parts.push("UNIQUE");
          parts.push(c.nullable ? "NULL" : "NOT NULL");
          if (c.kind === "fk" && c.ref) parts.push(`REFERENCES ${c.ref.table}(${c.ref.col})`);
          return parts.join(" ");
        })
        .join(",\n");
      return `CREATE TABLE ${t.name} (\n${cols}\n);`;
    })
    .join("\n\n");
  return (
    <pre className="flex-1 overflow-auto rounded-md border border-border bg-[#0a0a0c] p-4 font-mono text-[11px] leading-5 text-foreground/85">
      {sql}
    </pre>
  );
}

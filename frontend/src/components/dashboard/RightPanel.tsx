import { useEffect, useState } from "react";
import {
  Plus,
  Download,
  MoreHorizontal,
  TrendingUp,
  TrendingDown,
  Maximize2,
  Minimize2,
  PanelLeftOpen,
} from "lucide-react";
import type { DataSource } from "./types";
import type { FocusMode } from "@/routes/index";
import { PanelToggle } from "./ChatPanel";
import { DatabaseView } from "./DatabaseView";

type Tab = "pipeline" | "data" | "analytics" | "database";

export function RightPanel({
  source,
  nullsFixed,
  focus,
  onSetFocus,
  sidebarOpen,
  onToggleSidebar,
}: {
  source: DataSource;
  nullsFixed: boolean;
  focus: FocusMode;
  onSetFocus: (f: FocusMode) => void;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}) {
  const [tab, setTab] = useState<Tab>("database");
  const isFocused = focus === "right";

  return (
    <aside className="flex h-full w-full shrink-0 flex-col border-l border-border bg-background">
      {/* Tabs */}
      <div className="flex h-[57px] shrink-0 items-center justify-between gap-3 border-b border-border px-4">
        <div className="flex items-center gap-5">
          {isFocused && !sidebarOpen && (
            <button
              onClick={onToggleSidebar}
              aria-label="Open sidebar"
              className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
            >
              <PanelLeftOpen className="h-3.5 w-3.5" />
            </button>
          )}
          {(["pipeline", "data", "analytics", "database"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`relative h-[57px] text-[11.5px] font-medium uppercase tracking-[0.12em] transition-colors ${
                tab === t ? "text-foreground" : "text-muted-foreground hover:text-foreground/80"
              }`}
            >
              {t === "database" ? "Schema / ERD" : t}
              {tab === t && <span className="absolute inset-x-0 -bottom-px h-[2px] bg-foreground" />}
            </button>
          ))}
        </div>
        <PanelToggle
          active={isFocused}
          onClick={() => onSetFocus(isFocused ? "none" : "right")}
          label={isFocused ? "Restore layout" : "Focus analytics"}
          icon={isFocused ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
        />
      </div>

      {/* Title */}
      <div className="flex items-center justify-between border-b border-border px-5 py-3">
        <div>
          <div className="text-[13px] font-semibold tracking-tight text-foreground">
            {tab === "database" ? "Relational schema" : "EDA dashboard"}
          </div>
          <div className="mt-0.5 text-[10.5px] text-muted-foreground">
            {tab === "database" ? (
              <>
                AI-architected · scoped to{" "}
                <span className="font-mono text-foreground/80">{source.name}</span>
              </>
            ) : (
              <>
                7 widgets · auto-generated from{" "}
                <span className="font-mono text-foreground/80">{source.table}</span> schema
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <IconBtn>
            <Plus className="h-3 w-3" />
            {tab === "database" ? "Table" : "Widget"}
          </IconBtn>
          <IconBtn>
            <Download className="h-3 w-3" />
            Export
          </IconBtn>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {tab === "analytics" && <AnalyticsView nullsFixed={nullsFixed} />}
        {tab === "pipeline" && <Placeholder label="Pipeline graph" />}
        {tab === "data" && <Placeholder label="Data preview" />}
        {tab === "database" && <DatabaseView nullsFixed={nullsFixed} expanded={isFocused} />}
      </div>
    </aside>
  );
}

function IconBtn({ children }: { children: React.ReactNode }) {
  return (
    <button className="flex items-center gap-1 rounded-md border border-border bg-card px-2 py-1 text-[10.5px] text-foreground/80 transition-colors hover:border-border hover:text-foreground">
      {children}
    </button>
  );
}

function Placeholder({ label }: { label: string }) {
  return (
    <div className="flex h-full items-center justify-center rounded-md border border-dashed border-border text-[11px] text-muted-foreground">
      {label}
    </div>
  );
}

function AnalyticsView({ nullsFixed }: { nullsFixed: boolean }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-3">
        <Kpi label="Signups · 90d" value="48,210" delta="+12.4%" deltaUp />
        <NullRateKpi nullsFixed={nullsFixed} />
        <Kpi label="Anomalies" value="184" delta="+22" deltaUp />
      </div>

      <Widget title="Daily signups" subtitle="created_at · day · last 30">
        <LineChart />
      </Widget>

      <Widget title="Plan distribution" subtitle="plan · categorical">
        <Distribution />
      </Widget>
    </div>
  );
}

function Kpi({
  label,
  value,
  delta,
  deltaUp,
}: {
  label: string;
  value: string;
  delta: string;
  deltaUp?: boolean;
}) {
  return (
    <div className="rounded-md border border-border bg-card p-3">
      <div className="text-[9.5px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
        {label}
      </div>
      <div className="num-tabular mt-1.5 font-mono text-[18px] font-semibold tracking-tight text-foreground">
        {value}
      </div>
      <div
        className={`mt-1 flex items-center gap-1 text-[10px] ${
          deltaUp ? "text-emerald-400" : "text-red-400"
        }`}
      >
        {deltaUp ? <TrendingUp className="h-2.5 w-2.5" /> : <TrendingDown className="h-2.5 w-2.5" />}
        <span className="num-tabular font-mono">{delta}</span>
      </div>
    </div>
  );
}

function NullRateKpi({ nullsFixed }: { nullsFixed: boolean }) {
  const [val, setVal] = useState(3.2);
  const [delta, setDelta] = useState(-0.8);

  useEffect(() => {
    if (!nullsFixed) {
      setVal(3.2);
      setDelta(-0.8);
      return;
    }
    const start = performance.now();
    const from = 3.2;
    const dur = 1200;
    let raf = 0;
    const step = (t: number) => {
      const p = Math.min(1, (t - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(+(from * (1 - eased)).toFixed(1));
      setDelta(+(-(from * eased)).toFixed(1));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [nullsFixed]);

  return (
    <div
      className={`relative overflow-hidden rounded-md border bg-card p-3 transition-all duration-500 ${
        nullsFixed ? "border-emerald-500/40" : "border-border"
      }`}
    >
      {nullsFixed && (
        <span className="absolute right-2 top-2 rounded-sm border border-emerald-500/30 bg-emerald-500/10 px-1 py-px font-mono text-[8.5px] uppercase tracking-wider text-emerald-400">
          fixed
        </span>
      )}
      <div className="text-[9.5px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
        Null rate
      </div>
      <div
        className={`num-tabular mt-1.5 font-mono text-[18px] font-semibold tracking-tight transition-colors ${
          nullsFixed ? "text-emerald-400" : "text-foreground"
        }`}
      >
        {val.toFixed(1)}%
      </div>
      <div className="mt-1 flex items-center gap-1 text-[10px] text-emerald-400">
        <TrendingDown className="h-2.5 w-2.5" />
        <span className="num-tabular font-mono">{delta.toFixed(1)}%</span>
      </div>
    </div>
  );
}

function Widget({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="group rounded-md border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="leading-tight">
          <div className="text-[11.5px] font-semibold text-foreground">{title}</div>
          <div className="font-mono text-[9.5px] text-muted-foreground">{subtitle}</div>
        </div>
        <button className="opacity-0 transition-opacity group-hover:opacity-100">
          <MoreHorizontal className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground" />
        </button>
      </div>
      <div className="p-3">{children}</div>
    </div>
  );
}

function LineChart() {
  const points = [
    22, 26, 24, 30, 28, 32, 30, 35, 38, 36, 40, 42, 39, 44, 46, 50, 48, 53, 56, 54, 60, 58, 63, 66,
    70, 68, 72, 78, 82, 84,
  ];
  const w = 400,
    h = 110,
    pad = 4;
  const max = Math.max(...points);
  const xs = (i: number) => pad + (i / (points.length - 1)) * (w - pad * 2);
  const ys = (v: number) => h - pad - (v / max) * (h - pad * 2);
  const linePts = points.map((v, i) => `${xs(i)},${ys(v)}`).join(" ");
  const areaPts = `${xs(0)},${h} ${linePts} ${xs(points.length - 1)},${h}`;

  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="h-28 w-full">
        <defs>
          <linearGradient id="ls-grad" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
          </linearGradient>
        </defs>
        {/* gridlines */}
        {[0.25, 0.5, 0.75].map((p) => (
          <line key={p} x1={0} x2={w} y1={h * p} y2={h * p} stroke="#222226" strokeDasharray="2 4" />
        ))}
        <polygon points={areaPts} fill="url(#ls-grad)" />
        <polyline
          points={linePts}
          fill="none"
          stroke="#3B82F6"
          strokeWidth="1"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <circle
          cx={xs(points.length - 1)}
          cy={ys(points[points.length - 1])}
          r="2"
          fill="#3B82F6"
        />
      </svg>
      <div className="mt-2 flex items-center justify-between font-mono text-[9.5px] text-muted-foreground">
        <span>Apr 18</span>
        <span>
          peak <span className="text-foreground">8,420</span>
        </span>
        <span>May 17</span>
      </div>
    </div>
  );
}

function Distribution() {
  const data = [
    { label: "free", v: 62 },
    { label: "pro", v: 48 },
    { label: "team", v: 28 },
    { label: "ent", v: 14 },
    { label: "trial", v: 8 },
  ];
  const max = Math.max(...data.map((d) => d.v));
  return (
    <div className="space-y-2">
      {data.map((d) => (
        <div key={d.label} className="flex items-center gap-3 text-[10.5px]">
          <span className="w-10 font-mono text-muted-foreground">{d.label}</span>
          <div className="relative h-[6px] flex-1 overflow-hidden rounded-sm bg-background">
            <div
              className="h-full rounded-sm bg-[#3B82F6]/60"
              style={{ width: `${(d.v / max) * 100}%` }}
            />
          </div>
          <span className="num-tabular w-8 text-right font-mono text-foreground/80">{d.v}k</span>
        </div>
      ))}
    </div>
  );
}

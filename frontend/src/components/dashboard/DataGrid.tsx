import { Filter, Trash2, Wand2, ArrowRightLeft, Sparkles, Download } from "lucide-react";
import type { DataSource } from "./types";

type Cell = { v: string | number; mod?: "fixed" | "cast" | "anomaly" };

const cols = ["id", "email_hash", "country", "plan", "signups", "last_login", "anomaly"];

const rows: Cell[][] = [
  [
    { v: "u_8421" },
    { v: "a3f…b21" },
    { v: "DE" },
    { v: "pro" },
    { v: 12 },
    { v: "2026-05-14T09:21Z", mod: "cast" },
    { v: "—" },
  ],
  [
    { v: "u_8422" },
    { v: "9e1…cc4" },
    { v: "US" },
    { v: "free" },
    { v: 1 },
    { v: "2026-05-13T18:02Z", mod: "cast" },
    { v: "—" },
  ],
  [
    { v: "u_8423" },
    { v: "11d…0a8" },
    { v: "US", mod: "fixed" },
    { v: "pro" },
    { v: 47 },
    { v: "2026-05-14T03:11Z", mod: "cast" },
    { v: "high", mod: "anomaly" },
  ],
  [
    { v: "u_8424" },
    { v: "c92…77f" },
    { v: "FR" },
    { v: "team" },
    { v: 6 },
    { v: "2026-05-12T22:55Z", mod: "cast" },
    { v: "—" },
  ],
  [
    { v: "u_8425" },
    { v: "5b7…91e" },
    { v: "BR" },
    { v: "free" },
    { v: 2 },
    { v: "2026-05-14T11:43Z", mod: "cast" },
    { v: "low", mod: "anomaly" },
  ],
  [
    { v: "u_8426" },
    { v: "fa0…3c2" },
    { v: "JP" },
    { v: "pro" },
    { v: 18 },
    { v: "2026-05-14T06:14Z", mod: "cast" },
    { v: "—" },
  ],
  [
    { v: "u_8427" },
    { v: "2d4…aa1" },
    { v: "BR", mod: "fixed" },
    { v: "free" },
    { v: 88 },
    { v: "2026-05-14T02:09Z", mod: "cast" },
    { v: "high", mod: "anomaly" },
  ],
  [
    { v: "u_8428" },
    { v: "7c6…b09" },
    { v: "GB" },
    { v: "team" },
    { v: 9 },
    { v: "2026-05-13T19:30Z", mod: "cast" },
    { v: "—" },
  ],
  [
    { v: "u_8429" },
    { v: "0ee…fd5" },
    { v: "ES" },
    { v: "pro" },
    { v: 14 },
    { v: "2026-05-14T08:17Z", mod: "cast" },
    { v: "—" },
  ],
  [
    { v: "u_8430" },
    { v: "44a…21b" },
    { v: "CA" },
    { v: "free" },
    { v: 3 },
    { v: "2026-05-14T10:05Z", mod: "cast" },
    { v: "—" },
  ],
];

export function DataGrid({ source }: { source: DataSource }) {
  return (
    <div className="flex h-full flex-col">
      {/* Cleaning summary (sticky) */}
      <div className="sticky top-0 z-10 border-b border-border bg-sidebar/90 px-4 py-3 backdrop-blur">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold">{source.table}_clean_v3</span>
            <span className="rounded border border-success/40 bg-success/10 px-1.5 py-0.5 text-[10px] font-medium text-success">
              preview
            </span>
            <span className="font-mono text-[10px] text-muted-foreground">{source.name}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <button className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-[10px] text-muted-foreground hover:text-foreground">
              <Filter className="h-3 w-3" /> Filter
            </button>
            <button className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-[10px] text-muted-foreground hover:text-foreground">
              <Download className="h-3 w-3" /> Export
            </button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-3 gap-2">
          <SummaryBadge
            tone="destructive"
            icon={<Trash2 className="h-3 w-3" />}
            label="Rows Deleted"
            value="946"
          />
          <SummaryBadge
            tone="warning"
            icon={<Wand2 className="h-3 w-3" />}
            label="Nulls Fixed"
            value="15,420"
          />
          <SummaryBadge
            tone="primary"
            icon={<ArrowRightLeft className="h-3 w-3" />}
            label="Types Transformed"
            value="3"
          />
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px]">
          <thead className="sticky top-0 bg-sidebar">
            <tr className="border-b border-border text-left">
              {cols.map((c) => (
                <th
                  key={c}
                  className="px-3 py-2 font-medium uppercase tracking-wider text-muted-foreground"
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-border/50 transition-colors hover:bg-accent/30">
                {r.map((cell, j) => (
                  <ModCell key={j} cell={cell} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between border-t border-border px-4 py-2 text-[10px] text-muted-foreground">
        <div className="flex items-center gap-3">
          <span>Page 1 of 48,219 · 482,193 rows</span>
          <span className="flex items-center gap-1.5">
            <LegendDot color="success" /> fixed
            <LegendDot color="primary" /> cast
            <LegendDot color="destructive" /> anomaly
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button className="rounded border border-border px-2 py-0.5 hover:text-foreground">
            Prev
          </button>
          <button className="rounded border border-border px-2 py-0.5 hover:text-foreground">
            Next
          </button>
        </div>
      </div>
    </div>
  );
}

function ModCell({ cell }: { cell: Cell }) {
  const v = String(cell.v);
  if (cell.mod === "fixed") {
    return (
      <td className="px-3 py-2">
        <span className="relative inline-flex items-center gap-1 rounded bg-success/10 px-1.5 py-0.5 text-success ring-1 ring-success/30">
          <Sparkles className="h-2.5 w-2.5" />
          {v}
        </span>
      </td>
    );
  }
  if (cell.mod === "cast") {
    return (
      <td className="px-3 py-2">
        <span className="rounded bg-primary/5 px-1.5 py-0.5 text-foreground/90 ring-1 ring-primary/20">
          {v}
        </span>
      </td>
    );
  }
  if (cell.mod === "anomaly") {
    return (
      <td className="px-3 py-2">
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
            v === "high"
              ? "bg-destructive/15 text-destructive ring-1 ring-destructive/30"
              : "bg-warning/15 text-warning ring-1 ring-warning/30"
          }`}
        >
          {v}
        </span>
      </td>
    );
  }
  return (
    <td className={`px-3 py-2 ${v === "—" ? "text-muted-foreground/60" : "text-foreground/90"}`}>
      {v}
    </td>
  );
}

function SummaryBadge({
  tone,
  icon,
  label,
  value,
}: {
  tone: "destructive" | "warning" | "primary";
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  const cls =
    tone === "destructive"
      ? "border-destructive/30 bg-destructive/10 text-destructive"
      : tone === "warning"
        ? "border-warning/30 bg-warning/10 text-warning"
        : "border-primary/30 bg-primary/10 text-primary";
  return (
    <div className={`flex items-center justify-between rounded-md border px-2.5 py-1.5 ${cls}`}>
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider opacity-90">
        {icon} {label}
      </div>
      <span className="font-mono text-sm font-semibold">{value}</span>
    </div>
  );
}

function LegendDot({ color }: { color: "success" | "primary" | "destructive" }) {
  const cls =
    color === "success" ? "bg-success" : color === "primary" ? "bg-primary" : "bg-destructive";
  return <span className={`ml-2 inline-block h-1.5 w-1.5 rounded-full ${cls}`} />;
}

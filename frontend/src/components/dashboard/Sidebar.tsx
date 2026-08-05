import {
  Search,
  Plus,
  Database,
  Server,
  FileSpreadsheet,
  FileJson,
  PanelLeftClose,
} from "lucide-react";
import { DATA_SOURCES, type DataSource } from "./types";

// Kept for backward compat with any leftover imports
export type SectionKey = "pipelines" | "analytics" | "agent";

const ICONS: Record<DataSource["kind"], typeof Database> = {
  postgres: Database,
  mysql: Server,
  csv: FileSpreadsheet,
  json: FileJson,
};

export function Sidebar({
  activeSourceId,
  onSelect,
  onCollapse,
}: {
  activeSourceId: string;
  onSelect: (id: string) => void;
  onCollapse?: () => void;
}) {
  return (
    <aside className="flex h-full w-[260px] shrink-0 flex-col border-r border-border bg-sidebar">
      {/* Brand */}
      <div className="flex h-[57px] items-center gap-2.5 border-b border-border px-3">
        <button
          onClick={onCollapse}
          aria-label="Collapse sidebar"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
        >
          <PanelLeftClose className="h-3.5 w-3.5" />
        </button>
        <LogoMark />
        <div className="leading-tight">
          <div className="text-[13px] font-semibold tracking-tight text-foreground">Lumen</div>
          <div className="text-[9.5px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
            Agentic Data
          </div>
        </div>
      </div>

      {/* Command bar */}
      <div className="border-b border-border px-3 py-2.5">
        <div className="flex items-center gap-2 rounded-md border border-border bg-card px-2.5 py-1.5">
          <Search className="h-3.5 w-3.5 text-muted-foreground" />
          <input
            placeholder="Search…"
            className="w-full bg-transparent text-[12px] text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
          <kbd className="rounded-sm border border-border px-1 py-px font-mono text-[9.5px] text-muted-foreground">
            ⌘K
          </kbd>
        </div>
      </div>

      {/* Sources */}
      <div className="flex-1 overflow-y-auto px-2 py-3">
        <div className="mb-2 flex items-center justify-between px-2">
          <span className="text-[9.5px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            Data Sources
          </span>
          <button className="flex h-5 items-center justify-center rounded-sm text-muted-foreground hover:text-foreground">
            <Plus className="h-3 w-3" />
          </button>
        </div>

        <div className="space-y-0.5">
          {DATA_SOURCES.map((s) => (
            <SourceRow
              key={s.id}
              source={s}
              active={s.id === activeSourceId}
              onClick={() => onSelect(s.id)}
            />
          ))}
        </div>
      </div>

      {/* User card */}
      <div className="border-t border-border p-2.5">
        <div className="flex items-center gap-2.5 rounded-md border border-border bg-card px-2.5 py-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-[#3B82F6]/20 text-[10.5px] font-semibold text-[#3B82F6] ring-1 ring-[#3B82F6]/40">
            AK
          </div>
          <div className="min-w-0 flex-1 leading-tight">
            <div className="truncate text-[12px] font-medium text-foreground">Ana Kovač</div>
            <div className="truncate text-[10px] text-muted-foreground">ana@lumen.dev</div>
          </div>
          <span className="rounded-sm border border-border bg-background px-1.5 py-px font-mono text-[9.5px] uppercase tracking-wider text-muted-foreground">
            Pro
          </span>
        </div>
      </div>
    </aside>
  );
}

function SourceRow({
  source,
  active,
  onClick,
}: {
  source: DataSource;
  active: boolean;
  onClick: () => void;
}) {
  const Icon = ICONS[source.kind];
  return (
    <button
      onClick={onClick}
      className={`group relative flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left transition-colors ${
        active ? "bg-card text-foreground" : "text-foreground/85 hover:bg-card/60"
      }`}
    >
      {active && (
        <span className="absolute left-0 top-1/2 h-4 w-[2px] -translate-y-1/2 rounded-r bg-[#3B82F6]" />
      )}
      <Icon
        className={`h-3.5 w-3.5 shrink-0 ${
          active ? "text-[#3B82F6]" : "text-muted-foreground group-hover:text-foreground"
        }`}
      />
      <span className="min-w-0 flex-1 truncate text-[12px] font-medium">{source.name}</span>
      <span className="num-tabular font-mono text-[10px] text-muted-foreground">{source.rows}</span>
    </button>
  );
}

function LogoMark() {
  return (
    <div className="flex h-6 w-6 items-center justify-center rounded-[5px] border border-border bg-card">
      <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none">
        <path d="M3 3 L8 8 L13 3" stroke="#3B82F6" strokeWidth="1.5" strokeLinecap="square" />
        <path d="M3 13 L8 8 L13 13" stroke="#F59E0B" strokeWidth="1.5" strokeLinecap="square" />
      </svg>
    </div>
  );
}

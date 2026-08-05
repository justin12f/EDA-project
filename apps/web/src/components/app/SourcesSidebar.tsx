import { useRef } from "react";

import { cn } from "../../lib/cn";
import type { Source } from "../../lib/api/types";

const STATUS_DOT: Record<Source["status"], string> = {
  idle: "bg-muted-foreground",
  syncing: "bg-warning",
  live: "bg-success",
  error: "bg-destructive",
};

export function SourcesSidebar({
  sources,
  selectedId,
  onSelect,
  onUpload,
  onNewChat,
  uploading,
  uploadError,
  orgName,
  email,
  onSignOut,
}: {
  sources: Source[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onUpload: (file: File) => void;
  onNewChat: () => void;
  uploading: boolean;
  uploadError: string | null;
  orgName: string;
  email: string;
  onSignOut: () => void;
}) {
  const fileInput = useRef<HTMLInputElement>(null);

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-sidebar-border bg-sidebar">
      <div className="flex h-14 items-center gap-2 border-b border-sidebar-border px-4">
        <div className="flex h-6 w-6 items-center justify-center rounded-md bg-ai text-ai-foreground">
          <span className="text-[11px] font-semibold">L</span>
        </div>
        <span className="truncate text-[13px] font-semibold text-sidebar-foreground">
          {orgName}
        </span>
      </div>

      <div className="px-3 pt-3">
        <button
          onClick={onNewChat}
          className="flex h-9 w-full items-center justify-center gap-1.5 rounded-lg bg-primary text-[13px] font-medium text-primary-foreground transition hover:brightness-110"
        >
          + New chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-4">
        <div className="mb-2 flex items-center justify-between px-1">
          <span className="label-eyebrow">Data sources</span>
        </div>

        <button
          onClick={() => fileInput.current?.click()}
          disabled={uploading}
          className="input-focus mb-2 flex h-9 w-full items-center justify-center gap-2 rounded-lg border border-dashed border-border-strong text-[13px] font-medium text-muted-foreground transition hover:border-primary hover:text-primary disabled:opacity-50"
        >
          {uploading ? "Uploading…" : "+ Upload CSV, JSON or Parquet"}
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".csv,.json,.parquet"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onUpload(file);
            event.target.value = "";
          }}
        />
        {uploadError && <p className="mb-2 px-1 text-[12px] text-destructive">{uploadError}</p>}

        <button
          onClick={() => onSelect(null)}
          className={cn(
            "mb-1 w-full rounded-lg px-2.5 py-2 text-left text-[13px] transition",
            selectedId === null
              ? "bg-primary-tint text-primary"
              : "text-muted-foreground hover:bg-secondary",
          )}
        >
          General chat
        </button>

        {sources.length === 0 && (
          <p className="px-2 py-6 text-center text-[12px] text-muted-foreground">
            No sources yet. Upload a file to get started.
          </p>
        )}

        <ul className="space-y-0.5">
          {sources.map((source) => (
            <li key={source.id}>
              <button
                onClick={() => onSelect(source.id)}
                className={cn(
                  "flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left transition",
                  selectedId === source.id
                    ? "bg-primary-tint text-primary"
                    : "text-sidebar-foreground hover:bg-secondary",
                )}
              >
                <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", STATUS_DOT[source.status])} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px]">{source.name}</span>
                  <span className="block truncate text-[11px] text-muted-foreground">
                    {source.kind} · {source.row_count?.toLocaleString() ?? "—"} rows
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="flex items-center justify-between border-t border-sidebar-border px-4 py-3">
        <span className="truncate text-[12px] text-muted-foreground">{email}</span>
        <button
          onClick={onSignOut}
          className="shrink-0 text-[12px] font-medium text-muted-foreground hover:text-foreground"
        >
          Sign out
        </button>
      </div>
    </aside>
  );
}

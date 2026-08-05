import { useEffect, useState } from "react";
import {
  Send,
  Paperclip,
  Database,
  Check,
  X,
  ChevronRight,
  GitBranch,
  Plus,
  User,
  Sparkles,
  AtSign,
  PanelLeftOpen,
  Maximize2,
  Minimize2,
  Columns2,
} from "lucide-react";
import type { DataSource } from "./types";
import type { FocusMode } from "@/routes/index";

export function ChatPanel({
  source,
  nullsFixed,
  onAcceptPipeline,
  sidebarOpen,
  onToggleSidebar,
  focus,
  onSetFocus,
}: {
  source: DataSource;
  nullsFixed: boolean;
  onAcceptPipeline: () => void;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  focus: FocusMode;
  onSetFocus: (f: FocusMode) => void;
}) {
  const [input, setInput] = useState("");
  const [rejected, setRejected] = useState(false);

  // Reset card when source changes
  useEffect(() => {
    setInput("");
    setRejected(false);
  }, [source.id]);

  const accepted = nullsFixed;
  const isFocused = focus === "chat";

  return (
    <section className="flex h-full min-w-0 flex-1 flex-col bg-background">
      {/* Header — breadcrumb + agent badge */}
      <header className="flex h-[57px] shrink-0 items-center justify-between gap-3 border-b border-border px-4">
        <div className="flex min-w-0 items-center gap-2 text-[12px]">
          {!sidebarOpen && (
            <button
              onClick={onToggleSidebar}
              aria-label="Open sidebar"
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
            >
              <PanelLeftOpen className="h-3.5 w-3.5" />
            </button>
          )}
          <span className="truncate font-medium text-foreground">{source.name}</span>
          <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
          <span className="truncate font-mono text-muted-foreground">{source.table}</span>
          <ChevronRight className="hidden h-3 w-3 shrink-0 text-muted-foreground md:inline" />
          <span className="hidden font-medium text-foreground md:inline">Pipelines</span>

          <span className="ml-2 inline-flex h-6 shrink-0 items-center gap-1.5 rounded-md border border-[#F59E0B]/30 bg-[#F59E0B]/[0.06] px-2">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full rounded-full bg-[#F59E0B] ai-pulse" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-[#F59E0B]" />
            </span>
            <span className="text-[10px] font-medium uppercase leading-none tracking-[0.1em] text-[#F59E0B]">
              agent active
            </span>
          </span>

          <span className="ml-1 hidden h-6 shrink-0 items-center gap-1 rounded-md border border-border bg-card px-2 font-mono text-[10.5px] leading-none text-muted-foreground lg:inline-flex">
            <GitBranch className="h-3 w-3" />
            branch:&nbsp;<span className="text-foreground">clean/v3</span>
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          <PanelToggle
            active={isFocused}
            onClick={() => onSetFocus(isFocused ? "none" : "chat")}
            label={isFocused ? "Restore layout" : "Focus chat"}
            icon={
              isFocused ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />
            }
          />
          <PanelToggle
            active={!sidebarOpen && focus === "none"}
            onClick={onToggleSidebar}
            label="Split: chat + analytics"
            icon={<Columns2 className="h-3 w-3" />}
          />
          <button className="inline-flex h-6 shrink-0 items-center gap-1 whitespace-nowrap rounded-md border border-border bg-card px-2 text-[11px] font-medium leading-none text-foreground/90 transition-colors hover:border-[#3B82F6]/40 hover:text-foreground">
            <Plus className="h-3 w-3 shrink-0" />
            <span className="whitespace-nowrap">New thread</span>
          </button>
        </div>
      </header>

      {/* Chat body */}
      <div className="relative flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-6 px-6 py-7">
          {/* User message */}
          <Bubble role="user">
            Audit users on Production Postgres and propose a cleaning pipeline.
          </Bubble>

          {/* Agent reply */}
          <Bubble role="agent">
            I found <strong className="font-semibold text-foreground">3.2% nulls</strong> in{" "}
            <code className="rounded-sm border border-border bg-card px-1 py-px font-mono text-[11px] text-foreground">
              country_code
            </code>{" "}
            and <strong className="font-semibold text-foreground">412 duplicate</strong>{" "}
            <code className="rounded-sm border border-border bg-card px-1 py-px font-mono text-[11px] text-foreground">
              email_hash
            </code>{" "}
            values. Here's a proposed step:
          </Bubble>

          {/* Pipeline proposal card */}
          <div className="flex gap-3">
            <AgentAvatar />
            <div className="min-w-0 flex-1">
              <ProposalCard
                accepted={accepted}
                rejected={rejected}
                onAccept={onAcceptPipeline}
                onReject={() => setRejected(true)}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-border bg-background px-5 py-3">
        <div className="mx-auto max-w-3xl">
          <div className="input-glow rounded-lg border border-border bg-card transition-all">
            <div className="flex items-center gap-1.5 border-b border-border px-2 py-1.5">
              <ChipButton icon={<Database className="h-3 w-3" />} label="Source" />
              <ChipButton icon={<AtSign className="h-3 w-3" />} label="Context" />
              <ChipButton icon={<Paperclip className="h-3 w-3" />} label="Attach" />
              <span className="ml-auto flex items-center gap-1.5 rounded-sm border border-border bg-background px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                <span className="h-1 w-1 rounded-full bg-[#3B82F6]" />
                {source.name} · {source.table}
              </span>
            </div>
            <div className="flex items-end gap-2 px-3 py-2.5">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                rows={1}
                placeholder="Ask Lumen to clean, transform, or analyze…"
                className="block max-h-40 min-h-[20px] w-full resize-none bg-transparent text-[13px] leading-5 text-foreground placeholder:text-muted-foreground focus:outline-none"
              />
              <button
                disabled={!input.trim()}
                className="flex h-7 w-7 items-center justify-center rounded-md bg-[#3B82F6] text-white transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:bg-card disabled:text-muted-foreground"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
          <div className="mt-1.5 flex items-center justify-between px-1 font-mono text-[10px] text-muted-foreground">
            <span>shift + return for newline</span>
            <span>1,840 tokens · gpt-4o</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function ChipButton({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <button className="flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[11px] text-muted-foreground transition-colors hover:bg-background hover:text-foreground">
      {icon}
      {label}
    </button>
  );
}

export function PanelToggle({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      aria-label={label}
      className={`inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md border transition-colors ${
        active
          ? "border-[#3B82F6]/50 bg-[#3B82F6]/10 text-[#3B82F6]"
          : "border-border bg-card text-muted-foreground hover:border-[#3B82F6]/40 hover:text-foreground"
      }`}
    >
      {icon}
    </button>
  );
}

function AgentAvatar() {
  return (
    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-[#F59E0B]/30 bg-[#F59E0B]/10">
      <Sparkles className="h-3.5 w-3.5 text-[#F59E0B]" />
    </div>
  );
}

function Bubble({ role, children }: { role: "user" | "agent"; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      {role === "agent" ? (
        <AgentAvatar />
      ) : (
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-border bg-card text-muted-foreground">
          <User className="h-3.5 w-3.5" />
        </div>
      )}
      <div className="min-w-0 flex-1 pt-0.5">
        <div className="mb-1 text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
          {role === "agent" ? "Lumen" : "Ana"}
        </div>
        <div className="text-[13px] leading-relaxed text-foreground/90">{children}</div>
      </div>
    </div>
  );
}

function ProposalCard({
  accepted,
  rejected,
  onAccept,
  onReject,
}: {
  accepted: boolean;
  rejected: boolean;
  onAccept: () => void;
  onReject: () => void;
}) {
  const decided = accepted || rejected;

  return (
    <div
      className={`overflow-hidden rounded-lg border bg-card transition-all duration-500 ${
        accepted
          ? "border-emerald-500/40 shadow-[0_0_24px_-12px_rgba(52,211,153,0.6)]"
          : rejected
            ? "border-border opacity-60"
            : "border-[#F59E0B]/40 amber-glow"
      }`}
    >
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <span
            className={`flex h-5 w-5 items-center justify-center rounded ${
              accepted
                ? "bg-emerald-500/15 text-emerald-400"
                : rejected
                  ? "bg-card text-muted-foreground"
                  : "bg-[#F59E0B]/15 text-[#F59E0B]"
            }`}
          >
            {accepted ? <Check className="h-3 w-3" /> : <Sparkles className="h-3 w-3" />}
          </span>
          <div className="text-[12px] font-semibold text-foreground">
            {accepted
              ? "Pipeline update applied"
              : rejected
                ? "Proposal rejected"
                : "Pipeline update proposed"}
          </div>
        </div>
        <span
          className={`rounded-sm px-1.5 py-px font-mono text-[9.5px] uppercase tracking-wider ${
            accepted
              ? "border border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
              : rejected
                ? "border border-border bg-background text-muted-foreground"
                : "border border-[#F59E0B]/30 bg-[#F59E0B]/10 text-[#F59E0B]"
          }`}
        >
          {accepted ? "applied" : rejected ? "rejected" : "awaiting review"}
        </span>
      </div>

      <div className="px-3 py-2.5">
        <div className="space-y-1 rounded-md border border-border bg-background px-3 py-2 font-mono text-[12px] leading-6">
          <DiffLine op="+" label="Drop NULLs" target="country_code" done={accepted} />
          <DiffLine op="+" label="Dedupe" target="email_hash" done={accepted} />
        </div>

        <div className="mt-2 flex items-center justify-between text-[10.5px] text-muted-foreground">
          <span>
            Affects ~<span className="font-mono text-foreground">15,420</span> rows · est.{" "}
            <span className="font-mono text-foreground">1.2s</span>
          </span>
          <button className="text-muted-foreground transition-colors hover:text-foreground">
            View full diff →
          </button>
        </div>
      </div>

      {!decided && (
        <div className="flex items-center gap-2 border-t border-border bg-background/40 px-3 py-2">
          <button
            onClick={onAccept}
            className="flex items-center gap-1.5 rounded-md bg-[#3B82F6] px-3 py-1 text-[11.5px] font-medium text-white shadow-[0_0_18px_-6px_#3B82F6] transition-all hover:brightness-110"
          >
            <Check className="h-3 w-3" /> Accept
          </button>
          <button
            onClick={onReject}
            className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1 text-[11.5px] text-foreground/80 transition-colors hover:border-border hover:text-foreground"
          >
            <X className="h-3 w-3" /> Reject
          </button>
          <span className="ml-auto font-mono text-[10px] text-muted-foreground">⏎ to accept</span>
        </div>
      )}
    </div>
  );
}

function DiffLine({
  op,
  label,
  target,
  done,
}: {
  op: "+" | "-";
  label: string;
  target: string;
  done: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className={`w-3 text-center ${op === "+" ? "text-emerald-400" : "text-red-400"}`}>
        {op}
      </span>
      <span className="text-foreground/90">{label}</span>
      <span className="text-muted-foreground">·</span>
      <span className="text-[#3B82F6]">{target}</span>
      {done && <Check className="ml-auto h-3 w-3 text-emerald-400" />}
    </div>
  );
}

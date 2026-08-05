import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Sidebar } from "@/components/dashboard/Sidebar";
import { ChatPanel } from "@/components/dashboard/ChatPanel";
import { RightPanel } from "@/components/dashboard/RightPanel";
import { DATA_SOURCES } from "@/components/dashboard/types";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Lumen — Agentic Data Platform" },
      {
        name: "description",
        content:
          "Premium agentic AI workspace for cleaning, pipelines, and EDA across your data sources.",
      },
    ],
  }),
  component: Dashboard,
});

export type FocusMode = "none" | "chat" | "right";

function Dashboard() {
  const [activeSourceId, setActiveSourceId] = useState(DATA_SOURCES[0].id);
  const [nullsFixed, setNullsFixed] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [focus, setFocus] = useState<FocusMode>("none");

  const source = DATA_SOURCES.find((s) => s.id === activeSourceId) ?? DATA_SOURCES[0];

  const showSidebar = sidebarOpen && focus === "none";
  const showChat = focus !== "right";
  const showRight = focus !== "chat";

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground">
      <div
        className={`transition-[width,opacity] duration-300 ease-out overflow-hidden ${
          showSidebar ? "w-[260px] opacity-100" : "w-0 opacity-0"
        }`}
      >
        <Sidebar
          activeSourceId={activeSourceId}
          onSelect={setActiveSourceId}
          onCollapse={() => setSidebarOpen(false)}
        />
      </div>

      <div
        className={`transition-[flex-grow,opacity] duration-300 ease-out min-w-0 ${
          showChat
            ? "flex-1 opacity-100"
            : "flex-[0] opacity-0 pointer-events-none w-0 overflow-hidden"
        }`}
      >
        {showChat && (
          <ChatPanel
            source={source}
            nullsFixed={nullsFixed}
            onAcceptPipeline={() => setNullsFixed(true)}
            sidebarOpen={sidebarOpen}
            onToggleSidebar={() => {
              setSidebarOpen((v) => !v);
              setFocus("none");
            }}
            focus={focus}
            onSetFocus={setFocus}
          />
        )}
      </div>

      <div
        className={`transition-[width,opacity] duration-300 ease-out overflow-hidden ${
          showRight
            ? focus === "right"
              ? "w-full opacity-100"
              : "w-[440px] opacity-100"
            : "w-0 opacity-0"
        }`}
      >
        {showRight && (
          <RightPanel
            source={source}
            nullsFixed={nullsFixed}
            focus={focus}
            onSetFocus={setFocus}
            sidebarOpen={sidebarOpen}
            onToggleSidebar={() => {
              setSidebarOpen((v) => !v);
              setFocus("none");
            }}
          />
        )}
      </div>
    </div>
  );
}

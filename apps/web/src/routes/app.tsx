import { useCallback, useEffect, useState } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";

import { ChatPanel } from "../components/app/ChatPanel";
import { SourcesSidebar } from "../components/app/SourcesSidebar";
import { apiGet, apiUpload } from "../lib/api/client";
import type { Proposal, Source, TranscriptItem } from "../lib/api/types";
import { useRequireSession } from "../lib/hooks/useSession";
import { signOut } from "../lib/supabase/auth";
import { replayTranscript } from "../lib/transcript";

interface AppSearch {
  thread?: string;
}

// Not zod: the app has no direct dependency on it (only transitively, via
// the router itself), and a plain validator needs nothing more for one
// optional string field.
function validateSearch(search: Record<string, unknown>): AppSearch {
  return typeof search.thread === "string" && search.thread ? { thread: search.thread } : {};
}

export const Route = createFileRoute("/app")({ validateSearch, component: AppPage });

interface Me {
  email: string;
  display_name: string;
  org_name: string;
}

function AppPage() {
  const session = useRequireSession();
  const { thread } = Route.useSearch();

  if (session.status !== "signed-in") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="skeleton h-8 w-8 rounded-full" />
      </div>
    );
  }

  // Keyed by thread: switching threads (including to a brand new one) is a
  // clean remount rather than every piece of state having to notice the id
  // changed underneath it.
  return (
    <AppShell key={thread ?? "new"} fallbackEmail={session.session.user.email ?? ""} urlThread={thread} />
  );
}

function AppShell({ fallbackEmail, urlThread }: { fallbackEmail: string; urlThread?: string }) {
  const navigate = useNavigate({ from: "/app" });
  const [threadId] = useState(() => urlThread ?? crypto.randomUUID());
  const [me, setMe] = useState<Me | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  // Undefined while a reopened thread's history is still loading, so
  // ChatPanel does not mount (and briefly show its empty state) before the
  // history that belongs there has arrived.
  const [initialItems, setInitialItems] = useState<TranscriptItem[] | undefined>(
    urlThread ? undefined : [],
  );

  const refetchSources = useCallback(async () => {
    const response = await apiGet<{ sources: Source[] }>("/v1/sources");
    setSources(response.sources);
  }, []);

  const refetchProposals = useCallback(async () => {
    const response = await apiGet<{ proposals: Proposal[] }>(
      `/v1/proposals?thread_id=${threadId}`,
    );
    setProposals(response.proposals);
  }, [threadId]);

  useEffect(() => {
    apiGet<Me>("/v1/me").then(setMe).catch(() => {});
    refetchSources().catch(() => {});
  }, [refetchSources]);

  useEffect(() => {
    if (!urlThread) {
      // A brand new thread — put it in the URL now, so a reload from this
      // point on reopens the same conversation instead of starting another.
      navigate({ search: { thread: threadId }, replace: true });
      return;
    }
    let active = true;
    apiGet<{ events: { type: string; payload: Record<string, unknown> }[]; source_id: string | null }>(
      `/v1/threads/${threadId}/events`,
    )
      .then((history) => {
        if (!active) return;
        setInitialItems(replayTranscript(history.events));
        if (history.source_id) setSelectedId(history.source_id);
      })
      .catch(() => {
        if (active) setInitialItems([]);
      });
    refetchProposals().catch(() => {});
    return () => {
      active = false;
    };
    // Intentionally once per mount: this effect exists to load the thread
    // this component was keyed for, not to react to later state changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function startNewChat() {
    navigate({ search: { thread: crypto.randomUUID() } });
  }

  async function handleUpload(file: File) {
    setUploading(true);
    setUploadError(null);
    try {
      const created = await apiUpload<{ id: string }>("/v1/sources", file);
      await refetchSources();
      setSelectedId(created.id);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  const selectedSource = sources.find((source) => source.id === selectedId) ?? null;

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <SourcesSidebar
        sources={sources}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onUpload={handleUpload}
        onNewChat={startNewChat}
        uploading={uploading}
        uploadError={uploadError}
        orgName={me?.org_name ?? "Workspace"}
        email={me?.email ?? fallbackEmail}
        onSignOut={() => void signOut()}
      />
      {initialItems === undefined ? (
        <div className="flex flex-1 items-center justify-center">
          <div className="skeleton h-8 w-8 rounded-full" />
        </div>
      ) : (
        <ChatPanel
          threadId={threadId}
          selectedSource={selectedSource}
          proposals={proposals}
          onProposalsChanged={refetchProposals}
          initialItems={initialItems}
        />
      )}
    </div>
  );
}

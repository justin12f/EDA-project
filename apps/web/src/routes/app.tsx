import { useCallback, useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";

import { ChatPanel } from "../components/app/ChatPanel";
import { SourcesSidebar } from "../components/app/SourcesSidebar";
import { apiGet, apiUpload } from "../lib/api/client";
import type { Proposal, Source } from "../lib/api/types";
import { useRequireSession } from "../lib/hooks/useSession";
import { signOut } from "../lib/supabase/auth";

export const Route = createFileRoute("/app")({ component: AppPage });

interface Me {
  email: string;
  display_name: string;
  org_name: string;
}

function AppPage() {
  const session = useRequireSession();

  if (session.status !== "signed-in") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="skeleton h-8 w-8 rounded-full" />
      </div>
    );
  }

  return <AppShell fallbackEmail={session.session.user.email ?? ""} />;
}

function AppShell({ fallbackEmail }: { fallbackEmail: string }) {
  const [threadId] = useState(() => crypto.randomUUID());
  const [me, setMe] = useState<Me | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

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
        uploading={uploading}
        uploadError={uploadError}
        orgName={me?.org_name ?? "Workspace"}
        email={me?.email ?? fallbackEmail}
        onSignOut={() => void signOut()}
      />
      <ChatPanel
        threadId={threadId}
        selectedSource={selectedSource}
        proposals={proposals}
        onProposalsChanged={refetchProposals}
      />
    </div>
  );
}

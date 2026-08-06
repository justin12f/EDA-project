import { useEffect, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";

import { ErrorBanner, FieldLabel, TextInput } from "../components/auth/AuthShell";
import { apiDelete, apiGet, apiPost, apiPut } from "../lib/api/client";
import type { ApiKey, ApiKeyCreated, TrustPattern } from "../lib/api/types";
import { useRequireSession } from "../lib/hooks/useSession";

export const Route = createFileRoute("/settings")({ component: SettingsPage });

interface Me {
  org_name: string;
  role: string;
}

const SCOPES: Array<{ value: ApiKey["scope"]; label: string }> = [
  { value: "read:glossary", label: "Read glossary" },
  { value: "read:certification", label: "Read certification" },
];

function SettingsPage() {
  const session = useRequireSession();
  if (session.status !== "signed-in") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="skeleton h-8 w-8 rounded-full" />
      </div>
    );
  }
  return <SettingsShell />;
}

function SettingsShell() {
  const [me, setMe] = useState<Me | null>(null);

  useEffect(() => {
    apiGet<Me>("/v1/me").then(setMe).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <header className="flex h-14 items-center gap-3 border-b border-border px-6">
        <Link to="/app" className="text-[13px] font-medium text-muted-foreground hover:text-foreground">
          ← Back
        </Link>
        <h1 className="text-[13px] font-semibold text-foreground">Settings</h1>
        {me && <span className="text-[12px] text-muted-foreground">{me.org_name}</span>}
      </header>

      <div className="mx-auto max-w-xl space-y-10 px-6 py-8">
        {me === null ? null : me.role !== "owner" ? (
          <p className="rounded-lg border border-border bg-card px-4 py-3 text-[13px] text-muted-foreground">
            Only a workspace owner can manage these settings.
          </p>
        ) : (
          <>
            <section>
              <h2 className="label-eyebrow mb-3">API keys</h2>
              <ApiKeyManager />
            </section>
            <section>
              <h2 className="label-eyebrow mb-3">Trust patterns</h2>
              <TrustPatternsManager />
            </section>
          </>
        )}
      </div>
    </div>
  );
}

function ApiKeyManager() {
  const [keys, setKeys] = useState<ApiKey[] | null>(null);
  const [name, setName] = useState("");
  const [scope, setScope] = useState<ApiKey["scope"]>("read:glossary");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [justCreated, setJustCreated] = useState<ApiKeyCreated | null>(null);

  async function refetch() {
    const response = await apiGet<{ keys: ApiKey[] }>("/v1/api-keys");
    setKeys(response.keys);
  }

  useEffect(() => {
    refetch().catch(() => {});
  }, []);

  async function handleCreate(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    setError(null);
    try {
      const created = await apiPost<ApiKeyCreated>("/v1/api-keys", { name: name.trim(), scope });
      setJustCreated(created);
      setName("");
      await refetch();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the key");
    } finally {
      setCreating(false);
    }
  }

  async function handleRevoke(id: string) {
    try {
      await apiDelete(`/v1/api-keys/${id}`);
      await refetch();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not revoke the key");
    }
  }

  return (
    <div className="space-y-6">
      <p className="text-[13px] text-muted-foreground">
        Read-only credentials for external systems — a BI tool or another service asking
        &ldquo;what does this column mean&rdquo; or &ldquo;can I trust this dataset right now&rdquo;
        without opening this app. No key can write anything.
      </p>

      {justCreated && (
        <div className="ai-ring rounded-xl border border-ai-soft bg-card p-4">
          <p className="mb-2 text-[13px] font-medium text-foreground">
            &ldquo;{justCreated.name}&rdquo; created — copy it now, it won&apos;t be shown again.
          </p>
          <code className="block break-all rounded-lg bg-secondary px-3 py-2 text-[12px] text-foreground">
            {justCreated.key}
          </code>
          <button
            onClick={() => {
              navigator.clipboard?.writeText(justCreated.key);
            }}
            className="mt-2 h-7 rounded-lg border border-input bg-background px-3 text-[12px] font-medium text-foreground transition hover:bg-secondary"
          >
            Copy
          </button>
        </div>
      )}

      <form onSubmit={handleCreate} className="rounded-xl border border-border bg-card p-4">
        <div className="mb-3">
          <FieldLabel htmlFor="key-name">Name</FieldLabel>
          <TextInput
            id="key-name"
            placeholder="e.g. Looker dashboard"
            value={name}
            onChange={(event) => setName(event.target.value)}
            required
          />
        </div>
        <div className="mb-3">
          <FieldLabel htmlFor="key-scope">Scope</FieldLabel>
          <select
            id="key-scope"
            value={scope}
            onChange={(event) => setScope(event.target.value as ApiKey["scope"])}
            className="h-9 w-full rounded-lg border border-input bg-background px-3 text-[13px] text-foreground"
          >
            {SCOPES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <ErrorBanner message={error} />
        <button
          type="submit"
          disabled={creating || !name.trim()}
          className="h-9 rounded-lg bg-primary px-4 text-[13px] font-medium text-primary-foreground transition hover:brightness-110 disabled:opacity-50"
        >
          {creating ? "Creating…" : "Create key"}
        </button>
      </form>

      <div>
        <span className="label-eyebrow">Existing keys</span>
        {keys === null ? (
          <p className="mt-2 text-[13px] text-muted-foreground">Loading…</p>
        ) : keys.length === 0 ? (
          <p className="mt-2 text-[13px] text-muted-foreground">No keys yet.</p>
        ) : (
          <ul className="mt-2 space-y-2">
            {keys.map((key) => (
              <li
                key={key.id}
                className="flex items-center justify-between rounded-lg border border-border bg-card px-3 py-2"
              >
                <div className="min-w-0">
                  <p className="truncate text-[13px] text-foreground">{key.name}</p>
                  <p className="text-[11px] text-muted-foreground">
                    <code>{key.key_prefix}…</code> · {key.scope}
                    {key.revoked_at ? " · revoked" : ""}
                  </p>
                </div>
                {!key.revoked_at && (
                  <button
                    onClick={() => handleRevoke(key.id)}
                    className="shrink-0 text-[12px] font-medium text-destructive hover:underline"
                  >
                    Revoke
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function formatPattern(signature: string): string {
  return signature.replace(/[_:]/g, " ").replace(/\+/g, " + ");
}

/**
 * "Trust here is not a black box the org discovers by watching what
 * happens — it is a setting they can inspect and override" (ADR-0011 §5).
 * One row per structural pattern this workspace has decided on; the toggle
 * is a hard override, independent of the score — even a perfect streak
 * can be turned off here.
 */
function TrustPatternsManager() {
  const [patterns, setPatterns] = useState<TrustPattern[] | null>(null);
  const [minStreak, setMinStreak] = useState(20);
  const [error, setError] = useState<string | null>(null);

  async function refetch() {
    const response = await apiGet<{ patterns: TrustPattern[]; min_streak: number }>(
      "/v1/trust-patterns",
    );
    setPatterns(response.patterns);
    setMinStreak(response.min_streak);
  }

  useEffect(() => {
    refetch().catch(() => {});
  }, []);

  async function toggle(pattern: TrustPattern) {
    setError(null);
    try {
      await apiPut(`/v1/trust-patterns/${pattern.id}/auto-apply`, {
        enabled: !pattern.auto_apply_enabled,
      });
      await refetch();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update this pattern");
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-[13px] text-muted-foreground">
        What this workspace has taught the Sentinel to trust. A pattern needs {minStreak}{" "}
        consecutive approvals with no rejection to auto-apply on its own — the toggle overrides
        that regardless of score.
      </p>
      <ErrorBanner message={error} />
      {patterns === null ? (
        <p className="text-[13px] text-muted-foreground">Loading…</p>
      ) : patterns.length === 0 ? (
        <p className="text-[13px] text-muted-foreground">
          No decisions recorded yet — this fills in as proposals get accepted or rejected.
        </p>
      ) : (
        <ul className="space-y-2">
          {patterns.map((pattern) => (
            <li key={pattern.id} className="rounded-lg border border-border bg-card px-3 py-2.5">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="truncate text-[13px] font-medium capitalize text-foreground">
                  {formatPattern(pattern.pattern_signature)}
                </span>
                <label className="flex shrink-0 items-center gap-1.5 text-[11px] text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={pattern.auto_apply_enabled}
                    onChange={() => toggle(pattern)}
                    className="h-3 w-3 accent-ai"
                  />
                  Allow auto-apply
                </label>
              </div>
              <p className="text-[11px] text-muted-foreground">
                {pattern.approvals} approved · {pattern.rejections} rejected ·{" "}
                {pattern.consecutive_approvals} in a row · score {Math.round(pattern.score * 100)}%
                {pattern.eligible ? (
                  <span className="ml-1.5 rounded-full bg-success-tint px-1.5 py-0.5 font-medium text-success">
                    eligible now
                  </span>
                ) : null}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

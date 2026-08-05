import { useEffect, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";

import { currentSession, onAuthChange } from "../lib/supabase/auth";

export const Route = createFileRoute("/auth/callback")({ component: CallbackPage });

function CallbackPage() {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let settled = false;
    const next = new URLSearchParams(window.location.search).get("next") || "/app";

    // A hard navigation, not the SPA router: this page only exists to bridge
    // Supabase's redirect back into a session, and a full load into `/app` is
    // no slower than a client transition here while needing none of its typed
    // route plumbing for a URL we do not know ahead of time.
    const finish = () => {
      if (settled) return;
      settled = true;
      window.location.assign(next);
    };

    // The Supabase client parses `?code=...` from the URL and exchanges it for
    // a session as soon as it initializes (`detectSessionInUrl: true`). That
    // happens asynchronously, so both a direct check and a subscription are
    // here: whichever notices the session first wins.
    currentSession().then((session) => {
      if (session) finish();
    });
    const unsubscribe = onAuthChange((session) => {
      if (session) finish();
    });

    const timeout = window.setTimeout(() => {
      if (!settled) setError("Sign-in didn't complete — the link may have expired.");
    }, 8000);

    return () => {
      unsubscribe();
      window.clearTimeout(timeout);
    };
  }, []);

  return (
    <main className="flex min-h-screen items-center justify-center bg-background px-6">
      <div className="text-center">
        {error ? (
          <>
            <p className="label-eyebrow text-destructive">Sign-in failed</p>
            <p className="mt-2 text-sm text-muted-foreground">{error}</p>
            <Link
              to="/login"
              className="mt-6 inline-flex h-9 items-center rounded-lg bg-primary px-4 text-[13px] font-medium text-primary-foreground transition hover:brightness-110"
            >
              Back to sign in
            </Link>
          </>
        ) : (
          <>
            <div className="skeleton mx-auto h-8 w-8 rounded-full" />
            <p className="mt-4 text-sm text-muted-foreground">Signing you in…</p>
          </>
        )}
      </div>
    </main>
  );
}

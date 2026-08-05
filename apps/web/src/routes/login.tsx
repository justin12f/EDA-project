import { useState } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";

import {
  AuthShell,
  ErrorBanner,
  FieldLabel,
  GoogleButton,
  TextInput,
} from "../components/auth/AuthShell";
import { signInWithEmail, signInWithGoogle } from "../lib/supabase/auth";

export const Route = createFileRoute("/login")({ component: LoginPage });

function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const { error: signInError } = await signInWithEmail(email, password);
    setBusy(false);
    if (signInError) {
      setError(signInError);
      return;
    }
    navigate({ to: "/app" });
  }

  async function handleGoogle() {
    setBusy(true);
    setError(null);
    const { error: oauthError } = await signInWithGoogle();
    if (oauthError) {
      setError(oauthError);
      setBusy(false);
    }
    // On success the browser is already navigating away to Google.
  }

  return (
    <AuthShell
      title="Welcome back"
      subtitle="Sign in to your workspace"
      footer={
        <>
          Don&apos;t have an account?{" "}
          <Link to="/signup" className="font-medium text-primary hover:underline">
            Create one
          </Link>
        </>
      }
    >
      <ErrorBanner message={error} />
      <div className="mb-4">
        <GoogleButton onClick={handleGoogle} disabled={busy} />
      </div>
      <div className="mb-4 flex items-center gap-3">
        <div className="h-px flex-1 bg-border" />
        <span className="label-eyebrow">or</span>
        <div className="h-px flex-1 bg-border" />
      </div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <FieldLabel htmlFor="email">Email</FieldLabel>
          <TextInput
            id="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </div>
        <div>
          <FieldLabel htmlFor="password">Password</FieldLabel>
          <TextInput
            id="password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </div>
        <button
          type="submit"
          disabled={busy}
          className="h-9 w-full rounded-lg bg-primary text-[13px] font-medium text-primary-foreground transition hover:brightness-110 disabled:opacity-50"
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </AuthShell>
  );
}

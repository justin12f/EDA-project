import { useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";

import {
  AuthShell,
  ErrorBanner,
  FieldLabel,
  GoogleButton,
  TextInput,
} from "../components/auth/AuthShell";
import { signInWithGoogle, signUpWithEmail } from "../lib/supabase/auth";

export const Route = createFileRoute("/signup")({ component: SignupPage });

function SignupPage() {
  const [displayName, setDisplayName] = useState("");
  const [orgName, setOrgName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const { error: signUpError } = await signUpWithEmail(
      email,
      password,
      displayName,
      orgName || undefined,
    );
    setBusy(false);
    if (signUpError) {
      setError(signUpError);
      return;
    }
    setSent(true);
  }

  async function handleGoogle() {
    setBusy(true);
    setError(null);
    const { error: oauthError } = await signInWithGoogle();
    if (oauthError) {
      setError(oauthError);
      setBusy(false);
    }
  }

  if (sent) {
    return (
      <AuthShell
        title="Check your inbox"
        subtitle="One more step"
        footer={
          <Link to="/login" className="font-medium text-primary hover:underline">
            Back to sign in
          </Link>
        }
      >
        <p className="text-[13px] text-muted-foreground">
          We sent a confirmation link to <span className="text-foreground">{email}</span>.
          Open it to activate your workspace.
        </p>
      </AuthShell>
    );
  }

  return (
    <AuthShell
      title="Create your workspace"
      subtitle="Start cleaning and forecasting your data"
      footer={
        <>
          Already have an account?{" "}
          <Link to="/login" className="font-medium text-primary hover:underline">
            Sign in
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
          <FieldLabel htmlFor="displayName">Your name</FieldLabel>
          <TextInput
            id="displayName"
            autoComplete="name"
            required
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
          />
        </div>
        <div>
          <FieldLabel htmlFor="orgName">Workspace name</FieldLabel>
          <TextInput
            id="orgName"
            placeholder="Optional — defaults to your name"
            value={orgName}
            onChange={(event) => setOrgName(event.target.value)}
          />
        </div>
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
            autoComplete="new-password"
            required
            minLength={8}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </div>
        <button
          type="submit"
          disabled={busy}
          className="h-9 w-full rounded-lg bg-primary text-[13px] font-medium text-primary-foreground transition hover:brightness-110 disabled:opacity-50"
        >
          {busy ? "Creating…" : "Create account"}
        </button>
      </form>
    </AuthShell>
  );
}

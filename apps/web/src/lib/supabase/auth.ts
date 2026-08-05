/**
 * Authentication actions.
 *
 * Every function returns `{ error }` rather than throwing, because every caller
 * is a form that needs to render the message next to the field. Supabase's own
 * error strings are user-readable, so they are passed through rather than
 * translated into something vaguer.
 */

import type { Provider, Session, User } from "@supabase/supabase-js";

import { supabase } from "./client";

export interface AuthResult {
  error: string | null;
}

/** Where the OAuth provider sends the person back to. */
function redirectTo(next = "/app"): string {
  const origin = typeof window === "undefined" ? "" : window.location.origin;
  return `${origin}/auth/callback?next=${encodeURIComponent(next)}`;
}

export async function signUpWithEmail(
  email: string,
  password: string,
  displayName: string,
  orgName?: string,
): Promise<AuthResult> {
  // `data` lands in raw_user_meta_data, which the handle_new_user trigger reads
  // to name the profile and the personal organization. Sending it at signup is
  // what makes the workspace exist before the first request.
  const { error } = await supabase().auth.signUp({
    email,
    password,
    options: {
      data: { display_name: displayName, ...(orgName ? { org_name: orgName } : {}) },
      emailRedirectTo: redirectTo(),
    },
  });
  return { error: error?.message ?? null };
}

export async function signInWithEmail(email: string, password: string): Promise<AuthResult> {
  const { error } = await supabase().auth.signInWithPassword({ email, password });
  return { error: error?.message ?? null };
}

export async function signInWithOAuth(provider: Provider, next = "/app"): Promise<AuthResult> {
  const { error } = await supabase().auth.signInWithOAuth({
    provider,
    options: {
      redirectTo: redirectTo(next),
      // Ask Google for a refresh token and always show the account chooser, so
      // switching accounts does not silently reuse the last one.
      queryParams:
        provider === "google"
          ? { access_type: "offline", prompt: "select_account" }
          : undefined,
    },
  });
  return { error: error?.message ?? null };
}

export const signInWithGoogle = (next = "/app") => signInWithOAuth("google", next);

export async function signOut(): Promise<AuthResult> {
  const { error } = await supabase().auth.signOut();
  return { error: error?.message ?? null };
}

export async function currentSession(): Promise<Session | null> {
  const { data } = await supabase().auth.getSession();
  return data.session;
}

export async function currentUser(): Promise<User | null> {
  const { data } = await supabase().auth.getUser();
  return data.user;
}

/** Subscribe to sign-in and sign-out. Returns the unsubscribe function. */
export function onAuthChange(handler: (session: Session | null) => void): () => void {
  const { data } = supabase().auth.onAuthStateChange((_event, session) => handler(session));
  return () => data.subscription.unsubscribe();
}

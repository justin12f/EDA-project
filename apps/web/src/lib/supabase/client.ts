/**
 * Supabase browser client.
 *
 * Only the anon key ever reaches this file — it is compiled into the bundle and
 * is public by design. Row-level security, not secrecy, is what protects the
 * data behind it. The service-role key must never appear in anything prefixed
 * `VITE_`.
 *
 * Sessions persist in localStorage and refresh themselves. The access token is
 * what the API verifies, so every request to our own backend carries it as a
 * bearer token — see `lib/api/client.ts`.
 */

import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!url || !anonKey) {
  throw new Error(
    "VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY are required. Copy .env.example to .env " +
      "and fill in the Supabase block.",
  );
}

let browserClient: SupabaseClient | undefined;

export function supabase(): SupabaseClient {
  if (!browserClient) {
    browserClient = createClient(url, anonKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
        flowType: "pkce",
      },
    });
  }
  return browserClient;
}

/** The current access token, or null when signed out. */
export async function accessToken(): Promise<string | null> {
  const { data } = await supabase().auth.getSession();
  return data.session?.access_token ?? null;
}

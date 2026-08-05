/**
 * Client-only session state. Deliberately not read during SSR: the session
 * lives in the browser's localStorage, which the server render has no access
 * to, so a server-side check would always see "signed out" and redirect a
 * signed-in person on every hard load. Reading it in `useEffect` means the
 * check runs exactly once, in the one place it can be correct.
 */

import { useEffect, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import type { Session } from "@supabase/supabase-js";

import { currentSession, onAuthChange } from "../supabase/auth";

export type SessionState =
  | { status: "loading" }
  | { status: "signed-out" }
  | { status: "signed-in"; session: Session };

export function useSession(): SessionState {
  const [state, setState] = useState<SessionState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    const settle = (session: Session | null) => {
      if (active) setState(session ? { status: "signed-in", session } : { status: "signed-out" });
    };
    currentSession().then(settle);
    const unsubscribe = onAuthChange(settle);
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);

  return state;
}

/** Sends a signed-out visitor to /login. Renders nothing useful until settled. */
export function useRequireSession(): SessionState {
  const state = useSession();
  const navigate = useNavigate();

  useEffect(() => {
    if (state.status === "signed-out") navigate({ to: "/login" });
  }, [state.status, navigate]);

  return state;
}

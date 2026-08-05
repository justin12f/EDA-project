import { useEffect } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";

import { useSession } from "../lib/hooks/useSession";

export const Route = createFileRoute("/")({ component: IndexPage });

function IndexPage() {
  const session = useSession();
  const navigate = useNavigate();

  useEffect(() => {
    if (session.status === "signed-in") navigate({ to: "/app" });
    else if (session.status === "signed-out") navigate({ to: "/login" });
  }, [session.status, navigate]);

  return (
    <main className="flex min-h-screen items-center justify-center bg-background">
      <div className="skeleton h-8 w-8 rounded-full" />
    </main>
  );
}

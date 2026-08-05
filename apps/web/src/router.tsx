import { QueryClient } from "@tanstack/react-query";
import { createRouter as createTanStackRouter } from "@tanstack/react-router";
import { routeTree } from "./routeTree.gen";

// The installed @tanstack/react-start-plugin generates its server and client
// entry points with `import { createRouter } from './router'` hard-coded — not
// configurable, and not the `getRouter` name current upstream docs describe.
// This export name is load-bearing for that generated code, not a style choice.
export const createRouter = () => {
  const queryClient = new QueryClient();

  const router = createTanStackRouter({
    routeTree,
    context: { queryClient },
    scrollRestoration: true,
    defaultPreloadStaleTime: 0,
  });

  return router;
};

/**
 * Fetch wrapper for Lumen's API.
 *
 * Every call attaches the current Supabase access token — never stored here,
 * always re-read, because a stale token copied into a variable would keep
 * working right up until it silently didn't.
 *
 * Errors come back as RFC 9457 problem+json (see `services/api/.../errors.py`);
 * `ApiError` carries the one field worth showing a person — `detail` — so every
 * caller has exactly one message to render, not a response shape to parse.
 */

import { accessToken } from "../supabase/client";

const API_URL = import.meta.env.VITE_API_URL;

if (!API_URL) {
  throw new Error("VITE_API_URL is required. Copy .env.example to .env and fill it in.");
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function authHeaders(): Promise<Record<string, string>> {
  const token = await accessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function throwOnError(response: Response): Promise<never> {
  let detail = response.statusText;
  try {
    const body = await response.json();
    detail = body.detail || body.title || detail;
  } catch {
    // Not a problem+json body — the status line is all there is to say.
  }
  throw new ApiError(response.status, detail);
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { headers: await authHeaders() });
  if (!response.ok) return throwOnError(response);
  return response.json();
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { ...(await authHeaders()), "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) return throwOnError(response);
  return response.json();
}

export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "PUT",
    headers: { ...(await authHeaders()), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) return throwOnError(response);
  return response.json();
}

export async function apiDelete<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "DELETE",
    headers: await authHeaders(),
  });
  if (!response.ok) return throwOnError(response);
  return response.json();
}

export async function apiUpload<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: await authHeaders(),
    body: form,
  });
  if (!response.ok) return throwOnError(response);
  return response.json();
}

// ── streaming runs ───────────────────────────────────────────────────────────

export interface AgentEvent {
  type: string;
  payload: Record<string, unknown>;
}

export interface RunRequest {
  prompt: string;
  source_id?: string;
  thread_id?: string;
  backend?: string;
}

/**
 * POST /v1/runs and hand each Server-Sent Event to `onEvent` as it arrives.
 *
 * Not the browser's native `EventSource` — that API can only GET, and starting
 * a run has to carry a JSON body and a bearer token. `fetch` with a streamed
 * body reader and a hand-rolled frame parser does the same job for a POST.
 * The server sends `\n`-separated frames (see `runs.py`'s `sep="\n"`), so a
 * blank line is exactly two consecutive newlines — nothing fancier to parse.
 */
export async function streamRun(
  body: RunRequest,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_URL}/v1/runs`, {
    method: "POST",
    headers: {
      ...(await authHeaders()),
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) return throwOnError(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      parseFrame(frame, onEvent);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function parseFrame(frame: string, onEvent: (event: AgentEvent) => void): void {
  let type = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  // Keepalive pings are a bare `:` comment line — no `data:` field, nothing to
  // hand the caller.
  if (!data) return;
  onEvent({ type, payload: JSON.parse(data) });
}

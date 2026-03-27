import { createHash } from "node:crypto";
import type { spawn } from "node:child_process";

export type FindResultItem = {
  uri: string;
  level?: number;
  abstract?: string;
  overview?: string;
  category?: string;
  score?: number;
  match_reason?: string;
};

export type FindResult = {
  memories?: FindResultItem[];
  resources?: FindResultItem[];
  skills?: FindResultItem[];
  total?: number;
};

export type CaptureMode = "semantic" | "keyword";
export type RuntimeIdentity = {
  accountId: string;
  userId: string;
  agentId: string;
};
export type RequestIdentity = {
  accountId?: string;
  userId?: string;
  agentId?: string;
};
export type LocalClientCacheEntry = {
  client: OpenVikingClient;
  process: ReturnType<typeof spawn> | null;
};

export type PendingClientEntry = {
  promise: Promise<OpenVikingClient>;
  resolve: (c: OpenVikingClient) => void;
  reject: (err: unknown) => void;
};

export const localClientCache = new Map<string, LocalClientCacheEntry>();

// Module-level pending promise map: shared across all plugin registrations so
// that both [gateway] and [plugins] contexts await the same promise and
// don't create duplicate pending promises that never resolve.
export const localClientPendingPromises = new Map<string, PendingClientEntry>();

const MEMORY_URI_PATTERNS = [
  /^viking:\/\/user\/(?:[^/]+\/)?memories(?:\/|$)/,
  /^viking:\/\/agent\/(?:[^/]+\/)?memories(?:\/|$)/,
  /^viking:\/\/resources\/shared-memory(?:\/|$)/,
];
const GLOBAL_MEMORY_ROOT = "viking://resources/shared-memory";
const USER_MEMORY_ROOT = "viking://user/memories";
const AGENT_MEMORY_ROOT = "viking://agent/memories";

function md5Short(input: string): string {
  return createHash("md5").update(input).digest("hex").slice(0, 12);
}

export function sanitizeAgentPathSegment(input: string): string {
  const normalized = input
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "");
  return normalized || "default";
}

export function buildGlobalMemoryRoot(): string {
  return GLOBAL_MEMORY_ROOT;
}

export function buildUserMemoryRoot(): string {
  return USER_MEMORY_ROOT;
}

export function buildConcreteUserMemoryRoot(userId: string | undefined): string {
  const normalized = (userId ?? "").trim();
  if (!normalized) {
    return USER_MEMORY_ROOT;
  }
  return `viking://user/${normalized}/memories`;
}

export function buildAgentMemoryRoot(_agentId: string): string {
  return AGENT_MEMORY_ROOT;
}

export function getResourceMemoryTargets(agentId: string, userId?: string): string[] {
  return [buildGlobalMemoryRoot(), buildConcreteUserMemoryRoot(userId), buildAgentMemoryRoot(agentId)];
}

export function normalizeMemoryTargetUri(targetUri: string, agentId: string, userId?: string): string {
  const trimmed = targetUri.trim().replace(/\/+$/, "");
  if (trimmed === buildGlobalMemoryRoot() || trimmed.startsWith(`${buildGlobalMemoryRoot()}/`)) {
    return trimmed;
  }
  if (trimmed === buildUserMemoryRoot() || trimmed.startsWith(`${buildUserMemoryRoot()}/`)) {
    const userRoot = buildConcreteUserMemoryRoot(userId);
    if (trimmed === buildUserMemoryRoot()) {
      return userRoot;
    }
    return trimmed.replace(buildUserMemoryRoot(), userRoot);
  }
  const agentRoot = buildAgentMemoryRoot(agentId);
  if (trimmed === agentRoot || trimmed.startsWith(`${agentRoot}/`)) {
    return trimmed;
  }
  return trimmed;
}

export function isMemoryUri(uri: string): boolean {
  return MEMORY_URI_PATTERNS.some((pattern) => pattern.test(uri));
}

function resolveRequestIdentity(
  identity: RequestIdentity | string | undefined,
  defaultAgentId: string,
): RuntimeIdentity {
  if (typeof identity === "string") {
    return {
      accountId: "default",
      userId: "default",
      agentId: identity || defaultAgentId || "default",
    };
  }
  return {
    accountId: identity?.accountId?.trim() || "default",
    userId: identity?.userId?.trim() || "default",
    agentId: identity?.agentId?.trim() || defaultAgentId || "default",
  };
}

export class OpenVikingClient {
  constructor(
    private readonly baseUrl: string,
    private readonly apiKey: string,
    private readonly defaultAgentId: string,
    private readonly timeoutMs: number,
  ) {}

  getDefaultAgentId(): string {
    return this.defaultAgentId;
  }

  getDefaultSearchTargets(identity?: RequestIdentity | string): string[] {
    const resolvedIdentity = resolveRequestIdentity(identity, this.defaultAgentId);
    return getResourceMemoryTargets(resolvedIdentity.agentId, resolvedIdentity.userId);
  }

  private async request<T>(path: string, init: RequestInit = {}, identity?: RequestIdentity | string): Promise<T> {
    const resolvedIdentity = resolveRequestIdentity(identity, this.defaultAgentId);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const headers = new Headers(init.headers ?? {});
      if (this.apiKey) {
        headers.set("X-API-Key", this.apiKey);
      }
      if (resolvedIdentity.agentId) {
        headers.set("X-OpenViking-Agent", resolvedIdentity.agentId);
      }
      headers.set("X-OpenViking-Account", resolvedIdentity.accountId);
      headers.set("X-OpenViking-User", resolvedIdentity.userId);
      if (typeof init.body === "string" && !headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
      }

      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers,
        signal: controller.signal,
      });

      const payload = (await response.json().catch(() => ({}))) as {
        status?: string;
        result?: T;
        error?: { code?: string; message?: string };
      };

      if (!response.ok || payload.status === "error") {
        const code = payload.error?.code ? ` [${payload.error.code}]` : "";
        const message = payload.error?.message ?? `HTTP ${response.status}`;
        throw new Error(`OpenViking request failed${code}: ${message}`);
      }

      return (payload.result ?? payload) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  async healthCheck(): Promise<void> {
    await this.request<{ status: string }>("/health");
  }

  private async ls(uri: string, identity?: RequestIdentity | string): Promise<Array<Record<string, unknown>>> {
    return this.request<Array<Record<string, unknown>>>(
      `/api/v1/fs/ls?uri=${encodeURIComponent(uri)}&output=original`,
      {},
      identity,
    );
  }

  private async normalizeTargetUri(targetUri: string, identity?: RequestIdentity | string): Promise<string> {
    const trimmed = targetUri.trim().replace(/\/+$/, "");
    const resolvedIdentity = resolveRequestIdentity(identity, this.defaultAgentId);
    return normalizeMemoryTargetUri(trimmed, resolvedIdentity.agentId, resolvedIdentity.userId);
  }

  async find(
    query: string,
    options: {
      targetUri: string;
      limit: number;
      scoreThreshold?: number;
    },
    identity?: RequestIdentity | string,
  ): Promise<FindResult> {
    const normalizedTargetUri = await this.normalizeTargetUri(options.targetUri, identity);
    const body = {
      query,
      target_uri: normalizedTargetUri,
      limit: options.limit,
      score_threshold: options.scoreThreshold,
    };
    return this.request<FindResult>("/api/v1/search/find", {
      method: "POST",
      body: JSON.stringify(body),
    }, identity);
  }

  async read(uri: string, identity?: RequestIdentity | string): Promise<string> {
    return this.request<string>(
      `/api/v1/content/read?uri=${encodeURIComponent(uri)}`,
      {},
      identity,
    );
  }

  async addSessionMessage(
    sessionId: string,
    role: string,
    content: string,
    identity?: RequestIdentity | string,
  ): Promise<void> {
    await this.request<{ session_id: string }>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`,
      {
        method: "POST",
        body: JSON.stringify({ role, content }),
      },
      identity,
    );
  }

  /** GET session and auto-create if absent. */
  async getSession(sessionId: string, identity?: RequestIdentity | string): Promise<{ message_count?: number }> {
    return this.request<{ message_count?: number }>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}?auto_create=true`,
      { method: "GET" },
      identity,
    );
  }

  /**
   * Commit a session: archive (Phase 1) and extract memories (Phase 2).
   * wait=false (default): Phase 2 runs in background, returns task_id for polling.
   * wait=true: blocks until Phase 2 completes, returns memories_extracted count.
   */
  async commitSession(
    sessionId: string,
    options?: { wait?: boolean; identity?: RequestIdentity | string },
  ): Promise<{
    session_id: string;
    status: string;
    task_id?: string;
    archive_uri?: string;
    archived?: boolean;
    memories_extracted?: number;
  }> {
    const wait = options?.wait ?? false;
    return this.request<{
      session_id: string;
      status: string;
      task_id?: string;
      archive_uri?: string;
      archived?: boolean;
      memories_extracted?: number;
    }>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/commit?wait=${wait}`, {
      method: "POST",
      body: JSON.stringify({}),
    }, options?.identity);
  }

  async deleteSession(sessionId: string, identity?: RequestIdentity | string): Promise<void> {
    await this.request(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" }, identity);
  }

  async deleteUri(uri: string, identity?: RequestIdentity | string): Promise<void> {
    await this.request(`/api/v1/fs?uri=${encodeURIComponent(uri)}&recursive=false`, {
      method: "DELETE",
    }, identity);
  }

  async storeTextResource(
    text: string,
    options?: {
      accountId?: string;
      userId?: string;
      agentId?: string;
      scope?: "global" | "agent" | "both";
      title?: string;
      wait?: boolean;
      reason?: string;
    },
  ): Promise<string[]> {
    const resolvedIdentity = resolveRequestIdentity(
      {
        accountId: options?.accountId,
        userId: options?.userId,
        agentId: options?.agentId,
      },
      this.defaultAgentId,
    );
    const scope = options?.scope ?? "agent";
    const roots = this.getDefaultSearchTargets(resolvedIdentity);
    const sharedRoot = roots[0]!;
    const agentRoot = roots[2]!;
    const targets = scope === "global" ? [sharedRoot] : scope === "agent" ? [agentRoot] : [sharedRoot, agentRoot];
    const titleBase = (options?.title ?? "memory").trim() || "memory";
    const safeTitle = sanitizeAgentPathSegment(titleBase).replace(/\./g, "-");
    const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
    const suffix = md5Short(`${resolvedIdentity.accountId}:${resolvedIdentity.userId}:${resolvedIdentity.agentId}:${text}:${stamp}`);
    const fileName = `${stamp}-${safeTitle}-${suffix}.md`;
    const tempPath = await this.uploadTempText(text, resolvedIdentity);
    const storedUris: string[] = [];
    for (const root of targets) {
      const uri = `${root}/${fileName}`;
      await this.request<{ root_uri?: string; uri?: string }>(
        "/api/v1/resources",
        {
          method: "POST",
          body: JSON.stringify({
            temp_path: tempPath,
            to: uri,
            reason: options?.reason ?? "openclaw openviking memory",
            wait: options?.wait ?? false,
          }),
        },
        resolvedIdentity,
      );
      storedUris.push(uri);
    }
    return storedUris;
  }

  private async uploadTempText(text: string, identity?: RequestIdentity | string): Promise<string> {
    const resolvedIdentity = resolveRequestIdentity(identity, this.defaultAgentId);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const form = new FormData();
      form.append("file", new Blob([text], { type: "text/markdown" }), "memory.md");
      form.append("telemetry", "false");
      const headers = new Headers();
      if (this.apiKey) {
        headers.set("X-API-Key", this.apiKey);
      }
      if (resolvedIdentity.agentId) {
        headers.set("X-OpenViking-Agent", resolvedIdentity.agentId);
      }
      headers.set("X-OpenViking-Account", resolvedIdentity.accountId);
      headers.set("X-OpenViking-User", resolvedIdentity.userId);
      const response = await fetch(`${this.baseUrl}/api/v1/resources/temp_upload`, {
        method: "POST",
        body: form,
        headers,
        signal: controller.signal,
      });
      const payload = (await response.json().catch(() => ({}))) as {
        temp_path?: string;
        result?: { temp_path?: string };
        status?: string;
        error?: { code?: string; message?: string };
      };
      if (!response.ok || payload.status === "error") {
        const code = payload.error?.code ? ` [${payload.error.code}]` : "";
        const message = payload.error?.message ?? `HTTP ${response.status}`;
        throw new Error(`OpenViking temp upload failed${code}: ${message}`);
      }
      const tempPath = payload.result?.temp_path ?? payload.temp_path;
      if (!tempPath) {
        throw new Error("OpenViking temp upload failed: missing temp_path");
      }
      return tempPath;
    } finally {
      clearTimeout(timer);
    }
  }
}

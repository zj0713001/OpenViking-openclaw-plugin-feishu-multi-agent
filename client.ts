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
export type ScopeName = "user" | "agent";
export type RuntimeIdentity = {
  userId: string;
  agentId: string;
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
  /^viking:\/\/resources\/(?:global|agents\/[^/]+)\/memories(?:\/|$)/,
];
const GLOBAL_MEMORY_ROOT = "viking://resources/global/memories";
const AGENT_MEMORY_BASE = "viking://resources/agents";

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

export function buildAgentMemoryRoot(agentId: string): string {
  return `${AGENT_MEMORY_BASE}/${sanitizeAgentPathSegment(agentId)}/memories`;
}

export function getResourceMemoryTargets(agentId: string): string[] {
  return [buildGlobalMemoryRoot(), buildAgentMemoryRoot(agentId)];
}

export function normalizeLegacyMemoryTargetUri(targetUri: string, agentId: string): string {
  const trimmed = targetUri.trim().replace(/\/+$/, "");
  const match = trimmed.match(/^viking:\/\/(user|agent)(?:\/([^/]+))?\/memories(?:\/(.*))?$/);
  if (!match) {
    return trimmed;
  }
  const scope = match[1] as ScopeName;
  const rest = (match[3] ?? "").trim();
  const root = scope === "user" ? buildGlobalMemoryRoot() : buildAgentMemoryRoot(agentId);
  return rest ? `${root}/${rest}` : root;
}

export function isMemoryUri(uri: string): boolean {
  return MEMORY_URI_PATTERNS.some((pattern) => pattern.test(uri));
}

export class OpenVikingClient {
  private identityCache = new Map<string, RuntimeIdentity>();
  private ensuredMemoryRoots = new Set<string>();

  constructor(
    private readonly baseUrl: string,
    private readonly apiKey: string,
    private readonly defaultAgentId: string,
    private readonly timeoutMs: number,
  ) {}

  getDefaultAgentId(): string {
    return this.defaultAgentId;
  }

  getDefaultSearchTargets(agentId?: string): string[] {
    return getResourceMemoryTargets(agentId ?? this.defaultAgentId);
  }

  private async request<T>(path: string, init: RequestInit = {}, agentId?: string): Promise<T> {
    const effectiveAgentId = agentId ?? this.defaultAgentId;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const headers = new Headers(init.headers ?? {});
      if (this.apiKey) {
        headers.set("X-API-Key", this.apiKey);
      }
      if (effectiveAgentId) {
        headers.set("X-OpenViking-Agent", effectiveAgentId);
      }
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

  private async ls(uri: string, agentId?: string): Promise<Array<Record<string, unknown>>> {
    return this.request<Array<Record<string, unknown>>>(
      `/api/v1/fs/ls?uri=${encodeURIComponent(uri)}&output=original`,
      {},
      agentId,
    );
  }

  private async getRuntimeIdentity(agentId?: string): Promise<RuntimeIdentity> {
    const effectiveAgentId = agentId ?? this.defaultAgentId;
    const cached = this.identityCache.get(effectiveAgentId);
    if (cached) {
      return cached;
    }
    const fallback: RuntimeIdentity = { userId: "default", agentId: effectiveAgentId || "default" };
    try {
      const status = await this.request<{ user?: unknown }>("/api/v1/system/status", {}, agentId);
      const userId =
        typeof status.user === "string" && status.user.trim() ? status.user.trim() : "default";
      const identity: RuntimeIdentity = { userId, agentId: effectiveAgentId || "default" };
      this.identityCache.set(effectiveAgentId, identity);
      return identity;
    } catch {
      this.identityCache.set(effectiveAgentId, fallback);
      return fallback;
    }
  }

  private async mkdir(uri: string, agentId?: string): Promise<void> {
    await this.request<{ uri: string }>(
      "/api/v1/fs/mkdir",
      {
        method: "POST",
        body: JSON.stringify({ uri }),
      },
      agentId,
    );
  }

  async ensureMemoryRoots(agentId?: string): Promise<string[]> {
    const effectiveAgentId = agentId ?? this.defaultAgentId;
    const roots = this.getDefaultSearchTargets(effectiveAgentId);
    const cacheKey = `${effectiveAgentId}:${roots.join("|")}`;
    if (this.ensuredMemoryRoots.has(cacheKey)) {
      return roots;
    }
    for (const uri of roots) {
      await this.mkdir(uri, effectiveAgentId).catch(() => {});
    }
    this.ensuredMemoryRoots.add(cacheKey);
    return roots;
  }

  private async normalizeTargetUri(targetUri: string, agentId?: string): Promise<string> {
    const trimmed = targetUri.trim().replace(/\/+$/, "");
    const effectiveAgentId = agentId ?? this.defaultAgentId;
    return normalizeLegacyMemoryTargetUri(trimmed, effectiveAgentId);
  }

  async find(
    query: string,
    options: {
      targetUri: string;
      limit: number;
      scoreThreshold?: number;
    },
    agentId?: string,
  ): Promise<FindResult> {
    await this.ensureMemoryRoots(agentId).catch(() => {});
    const normalizedTargetUri = await this.normalizeTargetUri(options.targetUri, agentId);
    const body = {
      query,
      target_uri: normalizedTargetUri,
      limit: options.limit,
      score_threshold: options.scoreThreshold,
    };
    return this.request<FindResult>("/api/v1/search/find", {
      method: "POST",
      body: JSON.stringify(body),
    }, agentId);
  }

  async read(uri: string, agentId?: string): Promise<string> {
    return this.request<string>(
      `/api/v1/content/read?uri=${encodeURIComponent(uri)}`,
      {},
      agentId,
    );
  }

  async addSessionMessage(sessionId: string, role: string, content: string, agentId?: string): Promise<void> {
    await this.request<{ session_id: string }>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`,
      {
        method: "POST",
        body: JSON.stringify({ role, content }),
      },
      agentId,
    );
  }

  /** GET session — server auto-creates if absent; also loads messages from storage before extract. */
  async getSession(sessionId: string, agentId?: string): Promise<{ message_count?: number }> {
    return this.request<{ message_count?: number }>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}`,
      { method: "GET" },
      agentId,
    );
  }

  /**
   * Commit a session: archive (Phase 1) and extract memories (Phase 2).
   * wait=false (default): Phase 2 runs in background, returns task_id for polling.
   * wait=true: blocks until Phase 2 completes, returns memories_extracted count.
   */
  async commitSession(
    sessionId: string,
    options?: { wait?: boolean; agentId?: string },
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
    }, options?.agentId);
  }

  async deleteSession(sessionId: string, agentId?: string): Promise<void> {
    await this.request(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" }, agentId);
  }

  async deleteUri(uri: string, agentId?: string): Promise<void> {
    await this.request(`/api/v1/fs?uri=${encodeURIComponent(uri)}&recursive=false`, {
      method: "DELETE",
    }, agentId);
  }

  async storeTextResource(
    text: string,
    options?: {
      agentId?: string;
      scope?: "global" | "agent" | "both";
      title?: string;
      wait?: boolean;
      reason?: string;
    },
  ): Promise<string[]> {
    const effectiveAgentId = options?.agentId ?? this.defaultAgentId;
    const scope = options?.scope ?? "agent";
    const roots = await this.ensureMemoryRoots(effectiveAgentId);
    const targets = scope === "global" ? [roots[0]!] : scope === "agent" ? [roots[1]!] : roots;
    const titleBase = (options?.title ?? "memory").trim() || "memory";
    const safeTitle = sanitizeAgentPathSegment(titleBase).replace(/\./g, "-");
    const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
    const suffix = md5Short(`${effectiveAgentId}:${text}:${stamp}`);
    const fileName = `${stamp}-${safeTitle}-${suffix}.md`;
    const tempPath = await this.uploadTempText(text, effectiveAgentId);
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
        effectiveAgentId,
      );
      storedUris.push(uri);
    }
    return storedUris;
  }

  private async uploadTempText(text: string, agentId?: string): Promise<string> {
    const effectiveAgentId = agentId ?? this.defaultAgentId;
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
      if (effectiveAgentId) {
        headers.set("X-OpenViking-Agent", effectiveAgentId);
      }
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

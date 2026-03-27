import { spawn } from "node:child_process";
import { tmpdir } from "node:os";

import { Type } from "@sinclair/typebox";
import { memoryOpenVikingConfigSchema } from "./config.js";

import { OpenVikingClient, localClientCache, localClientPendingPromises, isMemoryUri } from "./client.js";
import type { FindResultItem, PendingClientEntry, RequestIdentity } from "./client.js";
import {
  isTranscriptLikeIngest,
  extractLatestUserText,
} from "./text-utils.js";
import { classifySharedMemories } from "./shared-memory-promoter.js";
import {
  clampScore,
  postProcessMemories,
  formatMemoryLines,
  toJsonLog,
  summarizeInjectionMemories,
  pickMemoriesForInjection,
} from "./memory-ranking.js";
import {
  IS_WIN,
  waitForHealth,
  quickRecallPrecheck,
  withTimeout,
  resolvePythonCommand,
  prepareLocalPort,
} from "./process-manager.js";
import { createMemoryOpenVikingContextEngine } from "./context-engine.js";
import type { ContextEngineWithSessionMapping } from "./context-engine.js";

type PluginLogger = {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
};

type HookAgentContext = {
  agentId?: string;
  sessionId?: string;
  sessionKey?: string;
  accountId?: string;
  userId?: string;
  senderId?: string;
  senderOpenId?: string;
  runtimeContext?: Record<string, unknown>;
};

type OpenClawPluginApi = {
  pluginConfig?: unknown;
  logger: PluginLogger;
  registerTool: (
    tool: {
      name: string;
      label: string;
      description: string;
      parameters: unknown;
      execute: (_toolCallId: string, params: Record<string, unknown>) => Promise<unknown>;
    },
    opts?: { name?: string; names?: string[] },
  ) => void;
  registerService: (service: {
    id: string;
    start: (ctx?: unknown) => void | Promise<void>;
    stop?: (ctx?: unknown) => void | Promise<void>;
  }) => void;
  registerContextEngine?: (id: string, factory: () => unknown) => void;
  on: (
    hookName: string,
    handler: (event: unknown, ctx?: HookAgentContext) => unknown,
    opts?: { priority?: number },
  ) => void;
};

const MAX_OPENVIKING_STDERR_LINES = 200;
const MAX_OPENVIKING_STDERR_CHARS = 256_000;
const AUTO_RECALL_TIMEOUT_MS = 30_000;
const FEISHU_OPEN_ID_RE = /\bou_[a-z0-9]{8,}\b/i;
const sharedSessionRequestIdentities = new Map<string, { accountId: string; userId: string; agentId: string }>();
const sharedLatestIdentityByAgent = new Map<string, { accountId: string; userId: string; agentId: string }>();
let didLogContextEngineRegistration = false;

const contextEnginePlugin = {
  id: "openviking",
  name: "Context Engine (OpenViking)",
  description: "OpenViking-backed context-engine memory with auto-recall/capture",
  kind: "context-engine" as const,
  configSchema: memoryOpenVikingConfigSchema,

  register(api: OpenClawPluginApi) {
    const cfg = memoryOpenVikingConfigSchema.parse(api.pluginConfig);
    const localCacheKey = `${cfg.mode}:${cfg.baseUrl}:${cfg.configPath}:${cfg.apiKey}`;
    const sessionPromotionCandidates = new Map<string, string[]>();
    const sessionRequestIdentities = sharedSessionRequestIdentities;

    const resolveContextUserId = (ctx?: HookAgentContext): string | undefined => {
      const direct = ctx?.userId ?? ctx?.senderOpenId ?? ctx?.senderId;
      if (typeof direct === "string" && direct.trim()) {
        return direct.trim();
      }
      const runtime = ctx?.runtimeContext ?? {};
      const candidate = runtime.senderOpenId ?? runtime.senderId ?? runtime.authProfileId;
      return typeof candidate === "string" && candidate.trim() ? candidate.trim() : undefined;
    };
    const extractUserIdFromEvent = (event: unknown): string | undefined => {
      if (!event || typeof event !== "object") {
        return undefined;
      }
      const eventObj = event as { messages?: unknown[]; prompt?: string } & Record<string, unknown>;
      const directCandidates = [
        eventObj.senderOpenId,
        eventObj.senderId,
        eventObj.userId,
        eventObj.authProfileId,
      ];
      for (const candidate of directCandidates) {
        if (typeof candidate === "string" && candidate.trim()) {
          return candidate.trim();
        }
      }
      const texts: string[] = [];
      if (typeof eventObj.prompt === "string") {
        texts.push(eventObj.prompt);
      }
      if (Array.isArray(eventObj.messages)) {
        for (const msg of eventObj.messages) {
          if (!msg || typeof msg !== "object") {
            continue;
          }
          const msgObj = msg as Record<string, unknown>;
          const msgDirectCandidates = [msgObj.senderOpenId, msgObj.senderId, msgObj.userId, msgObj.authProfileId];
          for (const candidate of msgDirectCandidates) {
            if (typeof candidate === "string" && candidate.trim()) {
              return candidate.trim();
            }
          }
          const content = msgObj.content;
          if (typeof content === "string") {
            texts.push(content);
          } else if (Array.isArray(content)) {
            for (const block of content) {
              if (!block || typeof block !== "object") {
                continue;
              }
              const blockObj = block as Record<string, unknown>;
              if (typeof blockObj.text === "string") {
                texts.push(blockObj.text);
              }
            }
          }
        }
      }
      for (const text of texts) {
        const match = text.match(FEISHU_OPEN_ID_RE);
        if (match?.[0]) {
          return match[0];
        }
      }
      return undefined;
    };
    const resolveContextAccountId = (ctx?: HookAgentContext): string => {
      const direct = ctx?.accountId;
      if (typeof direct === "string" && direct.trim()) {
        return direct.trim();
      }
      const runtime = ctx?.runtimeContext ?? {};
      const candidate = runtime.agentAccountId;
      return typeof candidate === "string" && candidate.trim() ? candidate.trim() : "default";
    };
    const resolveStoredRequestIdentity = (
      sessionId?: string,
      sessionKey?: string,
      fallbackAgentId?: string,
    ): RequestIdentity | undefined => {
      const stored = (sessionId && sessionRequestIdentities.get(sessionId)) ||
        (sessionKey && sessionRequestIdentities.get(sessionKey));
      if (stored) {
        return stored;
      }
      if (!fallbackAgentId) {
        return undefined;
      }
      return {
        accountId: "default",
        userId: "default",
        agentId: fallbackAgentId,
      };
    };
    const rememberSessionIdentity = (ctx?: HookAgentContext) => {
      const agentId = ctx?.agentId?.trim();
      const userId = resolveContextUserId(ctx);
      if (!agentId || !userId) {
        return;
      }
      const identity = {
        accountId: resolveContextAccountId(ctx),
        userId,
        agentId,
      };
      if (ctx.sessionId) {
        sessionRequestIdentities.set(ctx.sessionId, identity);
      }
      if (ctx.sessionKey) {
        sessionRequestIdentities.set(ctx.sessionKey, identity);
      }
      sharedLatestIdentityByAgent.set(identity.agentId, identity);
    };

    let clientPromise: Promise<OpenVikingClient>;
    let localProcess: ReturnType<typeof spawn> | null = null;
    let resolveLocalClient: ((c: OpenVikingClient) => void) | null = null;
    let rejectLocalClient: ((err: unknown) => void) | null = null;
    let localUnavailableReason: string | null = null;
    const markLocalUnavailable = (reason: string, err?: unknown) => {
      if (!localUnavailableReason) {
        localUnavailableReason = reason;
        api.logger.warn(
          `openviking: local mode marked unavailable (${reason})${err ? `: ${String(err)}` : ""}`,
        );
      }
      if (rejectLocalClient) {
        rejectLocalClient(
          err instanceof Error ? err : new Error(`openviking unavailable: ${reason}`),
        );
        rejectLocalClient = null;
      }
      resolveLocalClient = null;
    };

    if (cfg.mode === "local") {
      const cached = localClientCache.get(localCacheKey);
      if (cached) {
        localProcess = cached.process;
        clientPromise = Promise.resolve(cached.client);
      } else {
        const existingPending = localClientPendingPromises.get(localCacheKey);
        if (existingPending) {
          clientPromise = existingPending.promise;
        } else {
          const entry = {} as PendingClientEntry;
          entry.promise = new Promise<OpenVikingClient>((resolve, reject) => {
            entry.resolve = resolve;
            entry.reject = reject;
          });
          clientPromise = entry.promise;
          localClientPendingPromises.set(localCacheKey, entry);
        }
      }
    } else {
      clientPromise = Promise.resolve(new OpenVikingClient(cfg.baseUrl, cfg.apiKey, cfg.agentId, cfg.timeoutMs));
    }

    const getClient = (): Promise<OpenVikingClient> => clientPromise;
    const recordSessionCandidate = (sessionKey: string, text: string): void => {
      const normalized = text.trim();
      if (!sessionKey || !normalized) {
        return;
      }
      const existing = sessionPromotionCandidates.get(sessionKey) ?? [];
      existing.push(normalized);
      sessionPromotionCandidates.set(sessionKey, existing);
    };
    const promoteSessionCandidatesToGlobal = async (sessionKey: string, agentId: string): Promise<void> => {
      const texts = sessionPromotionCandidates.get(sessionKey) ?? [];
      if (texts.length === 0) {
        return;
      }
      try {
        const decision = await classifySharedMemories(
          {
            enabled: cfg.sharedMemoryPromotionEnabled,
            provider: cfg.sharedMemoryPromotionProvider,
            baseUrl: cfg.sharedMemoryPromotionBaseUrl,
            apiKey: cfg.sharedMemoryPromotionApiKey,
            model: cfg.sharedMemoryPromotionModel,
            maxCandidates: cfg.sharedMemoryPromotionMaxCandidates,
          },
          agentId,
          texts,
        );
        if (decision.promote.length === 0) {
          api.logger.info(
            `openviking: shared-memory promotion skipped (sessionKey=${sessionKey}, reason=${decision.reason ?? "no_candidates_selected"})`,
          );
          return;
        }
        void (async () => {
          const client = await getClient();
          const storedUris: string[] = [];
          for (const text of decision.promote) {
            const uris = await client.storeTextResource(text, {
              agentId,
              scope: "global",
              title: sessionKey,
              wait: false,
              reason: "openclaw shared memory promotion",
            });
            storedUris.push(...uris);
          }
          api.logger.info(
            `openviking: promoted ${decision.promote.length} shared memories to global for sessionKey=${sessionKey} ` +
              `${toJsonLog({ reason: decision.reason ?? "", uris: storedUris, promoted: decision.promote })}`,
          );
        })().catch((err) => {
          api.logger.warn(`openviking: async shared-memory promotion failed for sessionKey=${sessionKey}: ${String(err)}`);
        });
      } finally {
        sessionPromotionCandidates.delete(sessionKey);
      }
    };

    async function searchMemoryTargets(
      client: OpenVikingClient,
      query: string,
      requestLimit: number,
      identity: RequestIdentity,
      explicitTargetUri?: string,
    ) {
      const targets = explicitTargetUri
        ? [explicitTargetUri]
        : client.getDefaultSearchTargets(identity);
      const settled = await Promise.allSettled(
        targets.map((targetUri) =>
          client.find(
            query,
            {
              targetUri,
              limit: requestLimit,
              scoreThreshold: 0,
            },
            identity,
          ),
        ),
      );
      const allMemories = settled.flatMap((entry) =>
        entry.status === "fulfilled" ? (entry.value.memories ?? []) : [],
      );
      const uniqueMemories = allMemories.filter(
        (memory, index, self) => index === self.findIndex((m) => m.uri === memory.uri),
      );
      return {
        targets,
        memories: uniqueMemories,
        total: uniqueMemories.length,
        settled,
      };
    }

    api.registerTool(
      {
        name: "memory_recall",
        label: "Memory Recall (OpenViking)",
        description:
          "Search long-term memories from OpenViking. Use when you need past user preferences, facts, or decisions.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          limit: Type.Optional(
            Type.Number({ description: "Max results (default: plugin config)" }),
          ),
          scoreThreshold: Type.Optional(
            Type.Number({ description: "Minimum score (0-1, default: plugin config)" }),
          ),
          targetUri: Type.Optional(
            Type.String({ description: "Search scope URI (default: plugin config)" }),
          ),
        }),
        async execute(_toolCallId: string, params: Record<string, unknown>) {
          const { query } = params as { query: string };
          const limit =
            typeof (params as { limit?: number }).limit === "number"
              ? Math.max(1, Math.floor((params as { limit: number }).limit))
              : cfg.recallLimit;
          const scoreThreshold =
            typeof (params as { scoreThreshold?: number }).scoreThreshold === "number"
              ? Math.max(0, Math.min(1, (params as { scoreThreshold: number }).scoreThreshold))
              : cfg.recallScoreThreshold;
          const targetUri =
            typeof (params as { targetUri?: string }).targetUri === "string"
              ? (params as { targetUri: string }).targetUri
              : undefined;
          const requestLimit = Math.max(limit * 4, 20);

          const client = await getClient();
          const recallIdentity: RequestIdentity = {
            accountId: "default",
            userId: "default",
            agentId: client.getDefaultAgentId(),
          };
          const result = await searchMemoryTargets(
            client,
            query,
            requestLimit,
            recallIdentity,
            targetUri,
          );

          const memories = postProcessMemories(result.memories ?? [], {
            limit,
            scoreThreshold,
          });
          if (memories.length === 0) {
            return {
              content: [{ type: "text", text: "No relevant OpenViking memories found." }],
              details: { count: 0, total: result.total ?? 0, scoreThreshold },
            };
          }
          return {
            content: [
              {
                type: "text",
                text: `Found ${memories.length} memories:\n\n${formatMemoryLines(memories)}`,
              },
            ],
            details: {
              count: memories.length,
              memories,
              total: result.total ?? memories.length,
              scoreThreshold,
              requestLimit,
              searchedTargets: result.targets,
            },
          };
        },
      },
      { name: "memory_recall" },
    );

    api.registerTool(
      {
        name: "memory_store",
        label: "Memory Store (OpenViking)",
        description:
          "Store text in OpenViking memory pipeline by writing to a session and running memory extraction.",
        parameters: Type.Object({
          text: Type.String({ description: "Information to store as memory source text" }),
          role: Type.Optional(Type.String({ description: "Session role, default user" })),
          sessionId: Type.Optional(Type.String({ description: "Existing OpenViking session ID" })),
          sessionKey: Type.Optional(Type.String({ description: "OpenClaw sessionKey — uses the persistent mapped OV session" })),
        }),
        async execute(_toolCallId: string, params: Record<string, unknown>) {
          const { text } = params as { text: string };
          const role =
            typeof (params as { role?: string }).role === "string"
              ? (params as { role: string }).role
              : "user";
          const sessionIdIn = (params as { sessionId?: string }).sessionId;
          const sessionKeyIn = (params as { sessionKey?: string }).sessionKey;
          const inferredAgentId = (sessionKeyIn ? resolveAgentId(sessionKeyIn) : undefined) ?? cfg.agentId;
          const storedIdentity = (sessionIdIn && sessionRequestIdentities.get(sessionIdIn)) ||
            (sessionKeyIn && sessionRequestIdentities.get(sessionKeyIn)) ||
            sharedLatestIdentityByAgent.get(inferredAgentId) ||
            undefined;

          api.logger.info?.(
            `openviking: memory_store invoked (textLength=${text?.length ?? 0}, sessionId=${sessionIdIn ?? "auto"}, sessionKey=${sessionKeyIn ?? "none"})`,
          );

          const storeAgentId = sessionKeyIn ? resolveAgentId(sessionKeyIn) : undefined;
          try {
            const c = await getClient();
            if (!storedIdentity?.userId) {
              throw new Error(
                "memory_store requires a resolved session user identity; call it from an active conversation session or provide sessionKey/sessionId bound to a real user",
              );
            }
            const identity: RequestIdentity = {
              accountId: storedIdentity?.accountId ?? "default",
              userId: storedIdentity.userId,
              agentId: storedIdentity?.agentId ?? storeAgentId ?? cfg.agentId,
            };
            const ovSessionId = sessionKeyIn
              ? contextEngineRef?.getOVSessionForKey(sessionKeyIn) ?? sessionKeyIn
              : sessionIdIn ?? `memory_store_${Date.now()}`;
            await c.getSession(ovSessionId, identity);
            await c.addSessionMessage(ovSessionId, role, text, identity);
            const commitResult = await c.commitSession(ovSessionId, { wait: false, identity });
            api.logger.info?.(
              `openviking: memory_store committed session ovSessionId=${ovSessionId} status=${commitResult.status} task_id=${commitResult.task_id ?? "none"}`,
            );
            return {
              content: [
                {
                  type: "text",
                  text: `Stored memory via session commit: ${ovSessionId}`,
                },
              ],
              details: {
                action: "stored",
                mode: "session_commit",
                sessionId: ovSessionId,
                status: commitResult.status,
                taskId: commitResult.task_id,
                archiveUri: commitResult.archive_uri,
              },
            };
          } catch (err) {
            api.logger.warn(`openviking: memory_store failed: ${String(err)}`);
            throw err;
          }
        },
      },
      { name: "memory_store" },
    );

    api.registerTool(
      {
        name: "memory_forget",
        label: "Memory Forget (OpenViking)",
        description:
          "Forget memory by URI, or search then delete when a strong single match is found.",
        parameters: Type.Object({
          uri: Type.Optional(Type.String({ description: "Exact memory URI to delete" })),
          query: Type.Optional(Type.String({ description: "Search query to find memory URI" })),
          targetUri: Type.Optional(
            Type.String({ description: "Search scope URI (default: plugin config)" }),
          ),
          limit: Type.Optional(Type.Number({ description: "Search limit (default: 5)" })),
          scoreThreshold: Type.Optional(
            Type.Number({ description: "Minimum score (0-1, default: plugin config)" }),
          ),
        }),
        async execute(_toolCallId: string, params: Record<string, unknown>) {
          const uri = (params as { uri?: string }).uri;
          const toolIdentity: RequestIdentity = {
            accountId: "default",
            userId: "default",
            agentId: cfg.agentId,
          };
          if (uri) {
            if (!isMemoryUri(uri)) {
              return {
                content: [{ type: "text", text: `Refusing to delete non-memory URI: ${uri}` }],
                details: { action: "rejected", uri },
              };
            }
            await (await getClient()).deleteUri(uri, toolIdentity);
            return {
              content: [{ type: "text", text: `Forgotten: ${uri}` }],
              details: { action: "deleted", uri },
            };
          }

          const query = (params as { query?: string }).query;
          if (!query) {
            return {
              content: [{ type: "text", text: "Provide uri or query." }],
              details: { error: "missing_param" },
            };
          }

          const limit =
            typeof (params as { limit?: number }).limit === "number"
              ? Math.max(1, Math.floor((params as { limit: number }).limit))
              : 5;
          const scoreThreshold =
            typeof (params as { scoreThreshold?: number }).scoreThreshold === "number"
              ? Math.max(0, Math.min(1, (params as { scoreThreshold: number }).scoreThreshold))
              : cfg.recallScoreThreshold;
          const targetUri =
            typeof (params as { targetUri?: string }).targetUri === "string"
              ? (params as { targetUri: string }).targetUri
              : undefined;
          const requestLimit = Math.max(limit * 4, 20);

          const client = await getClient();
          const result = await searchMemoryTargets(
            client,
            query,
            requestLimit,
            toolIdentity,
            targetUri,
          );
          const candidates = postProcessMemories(result.memories ?? [], {
            limit: requestLimit,
            scoreThreshold,
            leafOnly: true,
          }).filter((item) => isMemoryUri(item.uri));
          if (candidates.length === 0) {
            return {
              content: [
                {
                  type: "text",
                  text: "No matching leaf memory candidates found. Try a more specific query.",
                },
              ],
              details: { action: "none", scoreThreshold },
            };
          }
          const top = candidates[0];
          if (candidates.length === 1 && clampScore(top.score) >= 0.85) {
            await (await getClient()).deleteUri(top.uri, toolIdentity);
            return {
              content: [{ type: "text", text: `Forgotten: ${top.uri}` }],
              details: { action: "deleted", uri: top.uri, score: top.score ?? 0 },
            };
          }

          const list = candidates
            .map((item) => `- ${item.uri} (${(clampScore(item.score) * 100).toFixed(0)}%)`)
            .join("\n");

          return {
            content: [
              {
                type: "text",
                text: `Found ${candidates.length} candidates. Specify uri:\n${list}`,
              },
            ],
            details: { action: "candidates", candidates, scoreThreshold, requestLimit },
          };
        },
      },
      { name: "memory_forget" },
    );
    let contextEngineRef: ContextEngineWithSessionMapping | null = null;

    const sessionAgentIds = new Map<string, string>();
    const rememberSessionAgentId = (ctx: {
      agentId?: string;
      sessionId?: string;
      sessionKey?: string;
    }) => {
      if (!ctx?.agentId) {
        return;
      }
      if (ctx.sessionId) {
        sessionAgentIds.set(ctx.sessionId, ctx.agentId);
      }
      if (ctx.sessionKey) {
        sessionAgentIds.set(ctx.sessionKey, ctx.agentId);
      }
    };
    const resolveAgentId = (sessionId?: string, fallbackConfigAgentId?: string): string | undefined => {
      if (!sessionId) {
        return fallbackConfigAgentId;
      }
      const resolved = sessionAgentIds.get(sessionId);
      if (typeof resolved === "string" && resolved.trim()) {
        return resolved;
      }
      return fallbackConfigAgentId;
    };

    api.on("session_start", async (_event: unknown, ctx?: HookAgentContext) => {
      rememberSessionIdentity(ctx);
      const sessionId = ctx?.sessionId ?? "none";
      const sessionKey = ctx?.sessionKey ?? "none";
      const agentId = ctx?.agentId ?? "none";
      if (!ctx?.agentId) {
        api.logger.warn?.(`openviking: session_start ctx missing agentId sessionId=${sessionId} sessionKey=${sessionKey}`);
      }
      rememberSessionAgentId(ctx ?? {});
    });
    api.on("session_end", async (_event: unknown, ctx?: HookAgentContext) => {
      rememberSessionIdentity(ctx);
      rememberSessionAgentId(ctx ?? {});
    });
    api.on("before_prompt_build", async (event: unknown, ctx?: HookAgentContext) => {
      rememberSessionIdentity(ctx);
      const hookSessionId = ctx?.sessionId ?? ctx?.sessionKey;
      const agentId = ctx?.agentId ?? resolveAgentId(hookSessionId, cfg.agentId);
      let requestIdentity = hookSessionId ? sessionRequestIdentities.get(hookSessionId) : undefined;
      if (!requestIdentity?.userId || requestIdentity.userId === "default") {
        const eventUserId = extractUserIdFromEvent(event);
        if (hookSessionId && eventUserId && agentId) {
          requestIdentity = {
            accountId: "default",
            userId: eventUserId,
            agentId,
          };
          sessionRequestIdentities.set(hookSessionId, requestIdentity);
          sharedLatestIdentityByAgent.set(agentId, requestIdentity);
        }
      }
      if (hookSessionId && agentId) {
        sessionAgentIds.set(hookSessionId, agentId);
      }
      let client: OpenVikingClient;
      try {
        client = await withTimeout(
          getClient(),
          5000,
          "openviking: client initialization timeout (OpenViking service not ready yet)"
        );
      } catch (err) {
        api.logger.warn?.(`openviking: failed to get client: ${String(err)}`);
        return;
      }

      const eventObj = (event ?? {}) as { messages?: unknown[]; prompt?: string };
      const queryText =
        extractLatestUserText(eventObj.messages) ||
        (typeof eventObj.prompt === "string" ? eventObj.prompt.trim() : "");
      if (!queryText) {
        return;
      }

      const prependContextParts: string[] = [];
      if (cfg.autoRecall && queryText.length >= 5) {
        const precheck = await quickRecallPrecheck(cfg.mode, cfg.baseUrl, cfg.port, localProcess);
        if (!precheck.ok) {
          api.logger.info(
            `openviking: skipping auto-recall because precheck failed (${precheck.reason})`,
          );
        } else {
          try {
            await withTimeout(
              (async () => {
                const candidateLimit = Math.max(cfg.recallLimit * 4, 20);
                const searchResult = await searchMemoryTargets(
                  client,
                  queryText,
                  candidateLimit,
                  requestIdentity ?? { accountId: "default", userId: "default", agentId },
                );
                for (const settled of searchResult.settled) {
                  if (settled.status === "rejected") {
                    api.logger.warn(`openviking: memory search failed: ${String(settled.reason)}`);
                  }
                }
                const processed = postProcessMemories(searchResult.memories ?? [], {
                  limit: candidateLimit,
                  scoreThreshold: cfg.recallScoreThreshold,
                });
                api.logger.info(
                  `openviking: recall-search-detail ${toJsonLog({
                    query: queryText,
                    targets: searchResult.targets,
                    rawCount: searchResult.memories?.length ?? 0,
                    rawMemories: summarizeInjectionMemories(searchResult.memories ?? []),
                    processedCount: processed.length,
                    processedMemories: summarizeInjectionMemories(processed),
                  })}`,
                );
                const memories = pickMemoriesForInjection(processed, cfg.recallLimit, queryText);
                api.logger.info(
                  `openviking: recall-picked-detail ${toJsonLog({
                    query: queryText,
                    pickedCount: memories.length,
                    pickedMemories: summarizeInjectionMemories(memories),
                  })}`,
                );

                if (memories.length > 0) {
                  const { lines: memoryLines, estimatedTokens } = await buildMemoryLinesWithBudget(
                    memories,
                      (uri) => client.read(uri, requestIdentity ?? { accountId: "default", userId: "default", agentId }),
                    {
                      recallPreferAbstract: cfg.recallPreferAbstract,
                      recallMaxContentChars: cfg.recallMaxContentChars,
                      recallTokenBudget: cfg.recallTokenBudget,
                    },
                  );
                  const memoryContext = memoryLines.join("\n");
                  api.logger.info(
                    `openviking: injecting ${memoryLines.length} memories (~${estimatedTokens} tokens, budget=${cfg.recallTokenBudget})`,
                  );
                  api.logger.info(
                    `openviking: inject-detail ${toJsonLog({ count: memories.length, memories: summarizeInjectionMemories(memories) })}`,
                  );
                  prependContextParts.push(
                    "<relevant-memories>\n" +
                      "The following OpenViking memories may be relevant.\n" +
                      "When the user asks about their preferences, profile, saved facts, or prior decisions, treat user-scoped memories as durable memory unless the user is explicitly correcting them.\n" +
                      "If multiple user preference memories are present, merge them into one consolidated answer instead of picking only one item.\n" +
                      "Do not ignore a higher-confidence user preference memory just because another lower-value memory is also present.\n" +
                      "If memories appear inconsistent, mention the conflict briefly and ask for confirmation only when necessary.\n" +
                      `${memoryContext}\n` +
                    "</relevant-memories>",
                  );
                }

              })(),
              AUTO_RECALL_TIMEOUT_MS,
              "openviking: auto-recall search timeout",
            );
          } catch (err) {
            api.logger.warn(`openviking: auto-recall failed: ${String(err)}`);
          }
        }
      }

      if (cfg.ingestReplyAssist) {
        const decision = isTranscriptLikeIngest(queryText, {
          minSpeakerTurns: cfg.ingestReplyAssistMinSpeakerTurns,
          minChars: cfg.ingestReplyAssistMinChars,
        });
        if (decision.shouldAssist) {
          api.logger.info(
            `openviking: ingest-reply-assist applied (reason=${decision.reason}, speakerTurns=${decision.speakerTurns}, chars=${decision.chars})`,
          );
          prependContextParts.push(
            "<ingest-reply-assist>\n" +
              "The latest user input looks like a multi-speaker transcript used for memory ingestion.\n" +
              "Reply with 1-2 concise sentences to acknowledge or summarize key points.\n" +
              "Do not output NO_REPLY or an empty reply.\n" +
              "Do not fabricate facts beyond the provided transcript and recalled memories.\n" +
              "</ingest-reply-assist>",
          );
        }
      }

      if (prependContextParts.length > 0) {
        return {
          prependContext: prependContextParts.join("\n\n"),
        };
      }
    });
    api.on("agent_end", async (_event: unknown, ctx?: HookAgentContext) => {
      rememberSessionAgentId(ctx ?? {});
    });
    api.on("before_reset", async (_event: unknown, ctx?: HookAgentContext) => {
      const sessionKey = ctx?.sessionKey;
      if (sessionKey && contextEngineRef) {
        try {
          await contextEngineRef.commitOVSession(sessionKey);
          api.logger.info(`openviking: committed OV session on reset for sessionKey=${sessionKey}`);
        } catch (err) {
          api.logger.warn(`openviking: failed to commit OV session on reset: ${String(err)}`);
        }
      }
    });
    api.on("after_compaction", async (_event: unknown, _ctx?: HookAgentContext) => {
      // Reserved hook registration for future post-compaction memory integration.
    });

    if (typeof api.registerContextEngine === "function") {
      api.registerContextEngine(contextEnginePlugin.id, () => {
        contextEngineRef = createMemoryOpenVikingContextEngine({
          id: contextEnginePlugin.id,
          name: contextEnginePlugin.name,
          version: "0.1.0",
          cfg,
          logger: api.logger,
          getClient,
          resolveAgentId,
          resolveRequestIdentity: resolveStoredRequestIdentity,
          recordSessionCandidate,
          promoteSessionCandidatesToGlobal,
        });
        return contextEngineRef;
      });
      if (!didLogContextEngineRegistration) {
        didLogContextEngineRegistration = true;
        api.logger.info(
          "openviking: registered context-engine (before_prompt_build=auto-recall, afterTurn=auto-capture, sessionKey=stable mapped session)",
        );
      }
    } else {
      api.logger.warn(
        "openviking: registerContextEngine is unavailable; context-engine behavior will not run",
      );
    }

    api.registerService({
      id: "openviking",
      start: async () => {
        // Claim the pending entry — only the first start() call to claim it spawns the process.
        // Subsequent start() calls (from other registrations sharing the same promise) fall through.
        const pendingEntry = localClientPendingPromises.get(localCacheKey);
        const isSpawner = cfg.mode === "local" && !!pendingEntry;
        if (isSpawner) {
          localClientPendingPromises.delete(localCacheKey);
          resolveLocalClient = pendingEntry!.resolve;
          rejectLocalClient = pendingEntry!.reject;
        }
        if (isSpawner) {
          const timeoutMs = 60_000;
          const intervalMs = 500;

          // Prepare port: kill stale OpenViking, or auto-find free port if occupied by others
          const actualPort = await prepareLocalPort(cfg.port, api.logger);
          const baseUrl = `http://127.0.0.1:${actualPort}`;

          const pythonCmd = resolvePythonCommand(api.logger);

          // Inherit system environment; optionally override Go/Python paths via env vars
          const pathSep = IS_WIN ? ";" : ":";
	  const { ALL_PROXY, all_proxy, HTTP_PROXY, http_proxy, HTTPS_PROXY, https_proxy, ...filteredEnv } = process.env;
          const env = {
            ...filteredEnv,
            PYTHONUNBUFFERED: "1",
            PYTHONWARNINGS: "ignore::RuntimeWarning",
            OPENVIKING_CONFIG_FILE: cfg.configPath,
            OPENVIKING_START_CONFIG: cfg.configPath,
            OPENVIKING_START_HOST: "127.0.0.1",
            OPENVIKING_START_PORT: String(actualPort),
            ...(process.env.OPENVIKING_GO_PATH && { PATH: `${process.env.OPENVIKING_GO_PATH}${pathSep}${process.env.PATH || ""}` }),
            ...(process.env.OPENVIKING_GOPATH && { GOPATH: process.env.OPENVIKING_GOPATH }),
            ...(process.env.OPENVIKING_GOPROXY && { GOPROXY: process.env.OPENVIKING_GOPROXY }),
          };
          // Run OpenViking server: use run_path on the module file to avoid RuntimeWarning from
          // "parent package import loads submodule before execution" (exit 3). Fallback to run_module with warning suppressed.
          const runpyCode = `import sys,os,warnings; warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*sys.modules.*'); sys.argv=['openviking.server.bootstrap','--config',os.environ['OPENVIKING_START_CONFIG'],'--host',os.environ.get('OPENVIKING_START_HOST','127.0.0.1'),'--port',os.environ['OPENVIKING_START_PORT']]; import runpy, importlib.util; spec=importlib.util.find_spec('openviking.server.bootstrap'); (runpy.run_path(spec.origin, run_name='__main__') if spec and getattr(spec,'origin',None) else runpy.run_module('openviking.server.bootstrap', run_name='__main__', alter_sys=True))`;
          const child = spawn(
            pythonCmd,
            ["-c", runpyCode],
            { env, cwd: IS_WIN ? tmpdir() : "/tmp", stdio: ["ignore", "pipe", "pipe"] },
          );
          localProcess = child;
          const stderrChunks: string[] = [];
          let stderrCharCount = 0;
          let stderrDroppedChunks = 0;
          const pushStderrChunk = (chunk: string) => {
            if (!chunk) return;
            stderrChunks.push(chunk);
            stderrCharCount += chunk.length;
            while (
              stderrChunks.length > MAX_OPENVIKING_STDERR_LINES ||
              stderrCharCount > MAX_OPENVIKING_STDERR_CHARS
            ) {
              const dropped = stderrChunks.shift();
              if (!dropped) break;
              stderrCharCount -= dropped.length;
              stderrDroppedChunks += 1;
            }
          };
          const formatStderrOutput = () => {
            if (!stderrChunks.length && !stderrDroppedChunks) return "";
            const truncated =
              stderrDroppedChunks > 0
                ? `[truncated ${stderrDroppedChunks} earlier stderr chunk(s)]\n`
                : "";
            return `\n[openviking stderr]\n${truncated}${stderrChunks.join("\n")}`;
          };
          child.on("error", (err: Error) => api.logger.warn(`openviking: local server error: ${String(err)}`));
          child.stderr?.on("data", (chunk: Buffer) => {
            const s = String(chunk).trim();
            pushStderrChunk(s);
            api.logger.debug?.(`[openviking] ${s}`);
          });
          child.on("exit", (code: number | null, signal: string | null) => {
            if (localProcess === child) {
              localProcess = null;
              localClientCache.delete(localCacheKey);
            }
            if (code != null && code !== 0 || signal) {
              const out = formatStderrOutput();
              api.logger.warn(`openviking: subprocess exited (code=${code}, signal=${signal})${out}`);
            }
          });
          try {
            await waitForHealth(baseUrl, timeoutMs, intervalMs);
            const client = new OpenVikingClient(baseUrl, cfg.apiKey, cfg.agentId, cfg.timeoutMs);
            localClientCache.set(localCacheKey, { client, process: child });
            resolveLocalClient!(client);
            rejectLocalClient = null;
            api.logger.info(
              `openviking: local server started (${baseUrl}, config: ${cfg.configPath})`,
            );
          } catch (err) {
            localProcess = null;
            child.kill("SIGTERM");
            markLocalUnavailable("startup failed", err);
            if (stderrChunks.length) {
              api.logger.warn(
                `openviking: startup failed (health check timeout or error).${formatStderrOutput()}`,
              );
            }
            throw err;
          }
        } else {
          await (await getClient()).healthCheck().catch(() => {});
          api.logger.info(
            `openviking: initialized (url: ${cfg.baseUrl}, targetUri: ${cfg.targetUri}, search: hybrid endpoint)`,
          );
        }
      },
      stop: () => {
        if (localProcess) {
          localProcess.kill("SIGTERM");
          localClientCache.delete(localCacheKey);
          localClientPendingPromises.delete(localCacheKey);
          localProcess = null;
          api.logger.info("openviking: local server stopped");
        } else {
          api.logger.info("openviking: stopped");
        }
      },
    });
  },
};

/** Estimate token count using chars/4 heuristic (adequate for budget enforcement). */
export function estimateTokenCount(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

export type BuildMemoryLinesOptions = {
  recallPreferAbstract: boolean;
  recallMaxContentChars: number;
};

async function resolveMemoryContent(
  item: FindResultItem,
  readFn: (uri: string) => Promise<string>,
  options: BuildMemoryLinesOptions,
): Promise<string> {
  let content: string;

  if (options.recallPreferAbstract && item.abstract?.trim()) {
    content = item.abstract.trim();
  } else if (item.level === 2) {
    try {
      const fullContent = await readFn(item.uri);
      content =
        fullContent && typeof fullContent === "string" && fullContent.trim()
          ? fullContent.trim()
          : (item.abstract?.trim() || item.uri);
    } catch {
      content = item.abstract?.trim() || item.uri;
    }
  } else {
    content = item.abstract?.trim() || item.uri;
  }

  if (content.length > options.recallMaxContentChars) {
    content = content.slice(0, options.recallMaxContentChars) + "...";
  }

  return content;
}

export async function buildMemoryLines(
  memories: FindResultItem[],
  readFn: (uri: string) => Promise<string>,
  options: BuildMemoryLinesOptions,
): Promise<string[]> {
  const lines: string[] = [];
  for (const item of memories) {
    const content = await resolveMemoryContent(item, readFn, options);
    lines.push(`- [${item.category ?? "memory"}] ${content}`);
  }
  return lines;
}

export type BuildMemoryLinesWithBudgetOptions = BuildMemoryLinesOptions & {
  recallTokenBudget: number;
};

/**
 * Build memory lines with a token budget constraint.
 *
 * The first memory is always included even if its token count exceeds the
 * remaining budget. This is intentional (spec Section 6.2): with
 * `recallMaxContentChars=500`, a single line is at most ~128 tokens — well
 * within the 2000-token default budget — so overshoot is bounded and
 * guarantees at least one memory is surfaced.
 */
export async function buildMemoryLinesWithBudget(
  memories: FindResultItem[],
  readFn: (uri: string) => Promise<string>,
  options: BuildMemoryLinesWithBudgetOptions,
): Promise<{ lines: string[]; estimatedTokens: number }> {
  let budgetRemaining = options.recallTokenBudget;
  const lines: string[] = [];
  let totalTokens = 0;

  for (const item of memories) {
    if (budgetRemaining <= 0) {
      break;
    }

    const content = await resolveMemoryContent(item, readFn, options);
    const line = `- [${item.category ?? "memory"}] ${content}`;
    const lineTokens = estimateTokenCount(line);

    // First line is always included even if it exceeds the budget (spec §6.2).
    if (lineTokens > budgetRemaining && lines.length > 0) {
      break;
    }

    lines.push(line);
    totalTokens += lineTokens;
    budgetRemaining -= lineTokens;
  }

  return { lines, estimatedTokens: totalTokens };
}

export default contextEnginePlugin;

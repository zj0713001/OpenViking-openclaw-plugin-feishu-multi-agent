import { createHash } from "node:crypto";

import type { OpenVikingClient, RequestIdentity } from "./client.js";
import type { MemoryOpenVikingConfig } from "./config.js";
import {
  getCaptureDecision,
  extractAutoCaptureTexts,
} from "./text-utils.js";
import {
  trimForLog,
  toJsonLog,
} from "./memory-ranking.js";

type AgentMessage = {
  role?: string;
  content?: unknown;
};

type ContextEngineInfo = {
  id: string;
  name: string;
  version?: string;
};

type AssembleResult = {
  messages: AgentMessage[];
  estimatedTokens: number;
  systemPromptAddition?: string;
};

type IngestResult = {
  ingested: boolean;
};

type IngestBatchResult = {
  ingestedCount: number;
};

type CompactResult = {
  ok: boolean;
  compacted: boolean;
  reason?: string;
  result?: unknown;
};

type CompactDelegate = (arg: {
  sessionId: string;
  sessionFile: string;
  tokenBudget?: number;
  force?: boolean;
  currentTokenCount?: number;
  compactionTarget?: "budget" | "threshold";
  customInstructions?: string;
  runtimeContext?: Record<string, unknown>;
}) => Promise<CompactResult>;

type ContextEngine = {
  info: ContextEngineInfo;
  ingest: (params: { sessionId: string; message: AgentMessage; isHeartbeat?: boolean }) => Promise<IngestResult>;
  ingestBatch?: (params: {
    sessionId: string;
    messages: AgentMessage[];
    isHeartbeat?: boolean;
  }) => Promise<IngestBatchResult>;
  afterTurn?: (params: {
    sessionId: string;
    sessionFile: string;
    messages: AgentMessage[];
    prePromptMessageCount: number;
    autoCompactionSummary?: string;
    isHeartbeat?: boolean;
    tokenBudget?: number;
    runtimeContext?: Record<string, unknown>;
  }) => Promise<void>;
  assemble: (params: { sessionId: string; messages: AgentMessage[]; tokenBudget?: number }) => Promise<AssembleResult>;
  compact: (params: {
    sessionId: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
    currentTokenCount?: number;
    compactionTarget?: "budget" | "threshold";
    customInstructions?: string;
    runtimeContext?: Record<string, unknown>;
  }) => Promise<CompactResult>;
};

export type ContextEngineWithSessionMapping = ContextEngine & {
  /** Return the OV session ID for an OpenClaw sessionKey using a stable cross-platform-safe mapping. */
  getOVSessionForKey: (sessionKey: string) => string;
  /** Ensure an OV session exists on the server for the given OpenClaw sessionKey (auto-created by getSession if absent). */
  resolveOVSession: (sessionKey: string) => Promise<string>;
  /** Commit (extract + archive) then delete the OV session, so a fresh one is created on next use. */
  commitOVSession: (sessionKey: string) => Promise<void>;
};

type Logger = {
  info: (msg: string) => void;
  warn?: (msg: string) => void;
  error: (msg: string) => void;
};

function estimateTokens(messages: AgentMessage[]): number {
  return Math.max(1, messages.length * 80);
}

function parseAgentIdFromSessionKey(sessionKey: string | undefined): string | undefined {
  if (!sessionKey) {
    return undefined;
  }
  const trimmed = sessionKey.trim();
  if (!trimmed) {
    return undefined;
  }
  const match = trimmed.match(/^agent:([^:]+):/);
  if (!match) {
    return undefined;
  }
  const agentId = match[1]?.trim();
  return agentId || undefined;
}

function describeError(err: unknown): string {
  if (err instanceof Error) {
    return err.message || err.name || String(err);
  }
  return String(err);
}

function isModuleResolutionError(err: unknown): boolean {
  if (!(err instanceof Error)) {
    return false;
  }
  const message = `${err.name}: ${err.message}`.toLowerCase();
  return (
    message.includes("cannot find module") ||
    message.includes("module not found") ||
    message.includes("package path not exported") ||
    message.includes("is not defined by 'exports'") ||
    message.includes("unsupported dir import") ||
    message.includes("failed to resolve module specifier")
  );
}

async function tryDelegatedCompact(
  params: {
    sessionId: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
    currentTokenCount?: number;
    compactionTarget?: "budget" | "threshold";
    customInstructions?: string;
    runtimeContext?: Record<string, unknown>;
  },
  logger: Logger,
): Promise<CompactResult | null> {
  let delegateCompactionToRuntime: CompactDelegate | null;
  try {
    delegateCompactionToRuntime = await loadCompactDelegate(logger);
  } catch (err) {
    return {
      ok: false,
      compacted: false,
      reason: `delegate_compact_import_failed:${describeError(err)}`,
    };
  }

  if (!delegateCompactionToRuntime) {
    if (cachedCompactUnavailableReason) {
      warnOrInfo(
        logger,
        `openviking: delegated compaction unavailable (${cachedCompactUnavailableReason})`,
      );
    }
    return null;
  }

  try {
    return await delegateCompactionToRuntime(params);
  } catch (err) {
    logger.error(`openviking: delegated compaction failed: ${describeError(err)}`);
    return {
      ok: false,
      compacted: false,
      reason: `delegate_compact_failed:${describeError(err)}`,
    };
  }
}

let cachedCompactDelegate: CompactDelegate | null | undefined;
let cachedCompactUnavailableReason: string | undefined;

async function loadCompactDelegate(logger: Logger): Promise<CompactDelegate | null> {
  if (cachedCompactDelegate !== undefined) {
    return cachedCompactDelegate;
  }
  const candidates = [
    "openclaw/plugin-sdk/core",
    "openclaw/plugin-sdk",
  ];
  const importErrors: string[] = [];

  for (const path of candidates) {
    try {
      const mod = (await import(path)) as {
        delegateCompactionToRuntime?: CompactDelegate;
      };
      if (!mod?.delegateCompactionToRuntime) {
        importErrors.push(`${path}: delegateCompactionToRuntime export missing`);
        continue;
      }
      cachedCompactDelegate = mod.delegateCompactionToRuntime;
      cachedCompactUnavailableReason = undefined;
      return cachedCompactDelegate;
    } catch (err) {
      const detail = `${path}: ${describeError(err)}`;
      importErrors.push(detail);
      if (!isModuleResolutionError(err)) {
        logger.error(`openviking: delegated compaction import failed: ${detail}`);
        throw err;
      }
    }
  }

  cachedCompactUnavailableReason =
    `failed to load compact delegate from candidates: ${importErrors.join(" | ")}`;
  cachedCompactDelegate = null;
  return null;
}

function warnOrInfo(logger: Logger, message: string): void {
  if (typeof logger.warn === "function") {
    logger.warn(message);
    return;
  }
  logger.info(message);
}

function md5Short(input: string): string {
  return createHash("md5").update(input).digest("hex").slice(0, 12);
}

const SAFE_SESSION_KEY_RE = /^[A-Za-z0-9_-]+$/;

export function mapSessionKeyToOVSessionId(sessionKey: string): string {
  const normalized = sessionKey.trim();
  if (!normalized) {
    return "openclaw_session";
  }
  if (SAFE_SESSION_KEY_RE.test(normalized)) {
    return normalized;
  }

  const readable = normalized
    .replace(/[^A-Za-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48);
  const digest = md5Short(normalized);
  return readable ? `openclaw_${readable}_${digest}` : `openclaw_session_${digest}`;
}

export function createMemoryOpenVikingContextEngine(params: {
  id: string;
  name: string;
  version?: string;
  cfg: Required<MemoryOpenVikingConfig>;
  logger: Logger;
  getClient: () => Promise<OpenVikingClient>;
  resolveAgentId: (sessionId?: string) => string | undefined;
  resolveRequestIdentity: (sessionId?: string, sessionKey?: string, agentId?: string) => RequestIdentity | undefined;
  recordSessionCandidate: (sessionKey: string, text: string) => void;
  promoteSessionCandidatesToGlobal: (sessionKey: string, agentId: string) => Promise<void>;
}): ContextEngineWithSessionMapping {
  const {
    id,
    name,
    version,
    cfg,
    logger,
    getClient,
    resolveAgentId,
    resolveRequestIdentity,
    recordSessionCandidate,
    promoteSessionCandidatesToGlobal,
  } = params;

  async function doCommitOVSession(sessionKey: string): Promise<void> {
    const agentId = resolveAgentId(sessionKey);
    const requestIdentity = resolveRequestIdentity(sessionKey, sessionKey, agentId);
    if (!agentId) {
      warnOrInfo(
        logger,
        `openviking: commit skipped for sessionKey=${sessionKey} because agentId could not be resolved. (fallback disabled or empty)`,
      );
      return;
    }
    try {
      const client = await getClient();
      const ovSessionId = mapSessionKeyToOVSessionId(sessionKey);
      const commitResult = await client.commitSession(ovSessionId, { wait: true, identity: requestIdentity ?? agentId });
      logger.info(
        `openviking: committed OV session for sessionKey=${sessionKey}, ovSessionId=${ovSessionId}, agentId=${agentId}, archived=${commitResult.archived ?? false}, memories=${commitResult.memories_extracted ?? 0}, task_id=${commitResult.task_id ?? "none"}`,
      );
      await promoteSessionCandidatesToGlobal(sessionKey, agentId);
      await client.deleteSession(ovSessionId, requestIdentity ?? agentId).catch(() => {});
    } catch (err) {
      warnOrInfo(logger, `openviking: commit failed for sessionKey=${sessionKey}: ${String(err)}`);
    }
  }

  function extractSessionKey(runtimeContext: Record<string, unknown> | undefined): string | undefined {
    if (!runtimeContext) {
      return undefined;
    }
    const key = runtimeContext.sessionKey;
    return typeof key === "string" && key.trim() ? key.trim() : undefined;
  }

  return {
    info: {
      id,
      name,
      version,
    },

    // --- session-mapping extensions ---

    getOVSessionForKey: (sessionKey: string) => mapSessionKeyToOVSessionId(sessionKey),

    async resolveOVSession(sessionKey: string): Promise<string> {
      return mapSessionKeyToOVSessionId(sessionKey);
    },

    commitOVSession: doCommitOVSession,

    // --- standard ContextEngine methods ---

    async ingest(): Promise<IngestResult> {
      return { ingested: false };
    },

    async ingestBatch(): Promise<IngestBatchResult> {
      return { ingestedCount: 0 };
    },

    async assemble(assembleParams): Promise<AssembleResult> {
      return {
        messages: assembleParams.messages,
        estimatedTokens: estimateTokens(assembleParams.messages),
      };
    },

    async afterTurn(afterTurnParams): Promise<void> {
      if (!cfg.autoCapture) {
        return;
      }

      try {
        const sessionKey = extractSessionKey(afterTurnParams.runtimeContext);
        const stableSessionId = afterTurnParams.sessionId;
        const mappedAgentId = resolveAgentId(stableSessionId);
        const parsedAgentId = parseAgentIdFromSessionKey(sessionKey);
        const agentId = mappedAgentId ?? parsedAgentId;
        if (!agentId) {
          logger.info(`openviking: auto-capture skipped (agentId unresolved for sessionId=${stableSessionId ?? "none"}, sessionKey=${sessionKey ?? "none"})`);
          return;
        }

        const messages = afterTurnParams.messages ?? [];
        if (messages.length === 0) {
          logger.info("openviking: auto-capture skipped (messages=0)");
          return;
        }

        const start =
          typeof afterTurnParams.prePromptMessageCount === "number" &&
          afterTurnParams.prePromptMessageCount >= 0
            ? afterTurnParams.prePromptMessageCount
            : 0;

        const { texts: newTexts, newCount, usedAssistantContext } = extractAutoCaptureTexts(messages, start);

        if (newTexts.length === 0) {
          logger.info("openviking: auto-capture skipped (no new user messages)");
          return;
        }

        const turnText = newTexts.join("\n");
        const decision = getCaptureDecision(turnText, cfg.captureMode, cfg.captureMaxLength);
        const requestIdentity = resolveRequestIdentity(stableSessionId, sessionKey, agentId) ?? {
          accountId: "default",
          userId: "default",
          agentId,
        };
        const ovSessionId = mapSessionKeyToOVSessionId(sessionKey ?? stableSessionId ?? `openclaw_${Date.now()}`);
        if (!decision.shouldCapture) {
          logger.info("openviking: auto-capture skipped (capture decision rejected)");
          return;
        }

        const client = await getClient();
        await client.getSession(ovSessionId, requestIdentity);
        await client.addSessionMessage(ovSessionId, "user", decision.normalizedText, requestIdentity);
        const commitResult = await client.commitSession(ovSessionId, { wait: false, identity: requestIdentity });
        if (sessionKey) {
          recordSessionCandidate(sessionKey, decision.normalizedText);
        }
        logger.info(
          `openviking: captured ${newCount} messages via session commit, ` +
            `ovSessionId=${ovSessionId} status=${commitResult.status} task_id=${commitResult.task_id ?? "none"} assistantContext=${usedAssistantContext}`,
        );
      } catch (err) {
        warnOrInfo(logger, `openviking: auto-capture failed: ${String(err)}`);
      }
    },

    async compact(compactParams): Promise<CompactResult> {
      const delegated = await tryDelegatedCompact(compactParams, logger);
      if (delegated) {
        return delegated;
      }

      warnOrInfo(
        logger,
        "openviking: delegated compaction unavailable; skipping compact",
      );

      return {
        ok: true,
        compacted: false,
        reason: "delegate_compact_unavailable",
      };
    },
  };
}

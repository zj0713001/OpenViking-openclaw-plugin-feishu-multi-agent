export type SharedMemoryPromotionProvider = "openai" | "ollama";

export type SharedMemoryPromotionConfig = {
  enabled: boolean;
  provider: SharedMemoryPromotionProvider;
  baseUrl: string;
  apiKey: string;
  model: string;
  maxCandidates: number;
};

export type SharedMemoryPromotionDecision = {
  promote: string[];
  skipped: string[];
  reason?: string;
};

function normalizeText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

export function dedupePromotionCandidates(texts: string[], limit: number): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of texts) {
    const text = normalizeText(raw);
    if (!text) {
      continue;
    }
    const key = text.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(text);
    if (out.length >= limit) {
      break;
    }
  }
  return out;
}

export function buildSharedMemoryPromotionMessages(agentId: string, texts: string[]): Array<{ role: "system" | "user"; content: string }> {
  const numbered = texts.map((text, index) => `${index + 1}. ${text}`).join("\n\n");
  return [
    {
      role: "system",
      content:
        "You classify candidate memories for promotion from a single agent memory space to a global shared memory space. " +
        "Promote only memories useful across multiple agents, such as shared rules, reusable conventions, common facts, team-wide preferences, collaboration protocols, or durable global knowledge. " +
        "Do not promote private preferences, one-off task context, temporary plans, sensitive personal data, or agent-local working notes. " +
        'Return strict JSON only in the shape {"promote":[number],"reason":"string"}.',
    },
    {
      role: "user",
      content:
        `Current agentId: ${agentId}\n` +
        "Candidate memories captured during this session:\n\n" +
        `${numbered}\n\n` +
        "Select only the candidate indexes that should also be written to global shared memory.",
    },
  ];
}

export function parseSharedMemoryPromotionResponse(content: string, texts: string[]): SharedMemoryPromotionDecision {
  const normalized = content.trim();
  const fenced = normalized.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const payloadText = (fenced?.[1] ?? normalized).trim();
  const parsed = JSON.parse(payloadText) as { promote?: unknown; reason?: unknown };
  const indexes = Array.isArray(parsed.promote) ? parsed.promote : [];
  const selectedIndexes = indexes
    .map((value) => (typeof value === "number" ? Math.floor(value) : Number(value)))
    .filter((value) => Number.isFinite(value) && value >= 1 && value <= texts.length);
  const uniqueIndexes = Array.from(new Set(selectedIndexes));
  const promote = uniqueIndexes.map((index) => texts[index - 1]!).filter(Boolean);
  const promoteSet = new Set(promote.map((item) => item.toLowerCase()));
  const skipped = texts.filter((item) => !promoteSet.has(item.toLowerCase()));
  return {
    promote,
    skipped,
    reason: typeof parsed.reason === "string" ? parsed.reason.trim() : undefined,
  };
}

function extractOpenAICompatibleText(payload: {
  choices?: Array<{ message?: { content?: string | Array<{ type?: string; text?: string }> } }>;
}): string {
  const content = payload.choices?.[0]?.message?.content;
  return Array.isArray(content)
    ? content
        .map((item) => (typeof item?.text === "string" ? item.text : ""))
        .join("\n")
        .trim()
    : typeof content === "string"
      ? content.trim()
      : "";
}

function extractOllamaText(payload: {
  message?: { content?: string };
  response?: string;
}): string {
  if (typeof payload.message?.content === "string" && payload.message.content.trim()) {
    return payload.message.content.trim();
  }
  return typeof payload.response === "string" ? payload.response.trim() : "";
}

async function classifyWithOpenAICompatible(
  cfg: SharedMemoryPromotionConfig,
  agentId: string,
  candidates: string[],
): Promise<SharedMemoryPromotionDecision> {
  if (!cfg.baseUrl || !cfg.apiKey || !cfg.model) {
    return { promote: [], skipped: candidates, reason: "promotion_disabled_or_empty" };
  }
  const response = await fetch(`${cfg.baseUrl.replace(/\/+$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify({
      model: cfg.model,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: buildSharedMemoryPromotionMessages(agentId, candidates),
    }),
  });
  const payload = (await response.json().catch(() => ({}))) as {
    choices?: Array<{ message?: { content?: string | Array<{ type?: string; text?: string }> } }>;
    error?: { message?: string };
  };
  if (!response.ok) {
    throw new Error(payload.error?.message ?? `share promotion LLM request failed: HTTP ${response.status}`);
  }
  const textContent = extractOpenAICompatibleText(payload);
  if (!textContent) {
    throw new Error("share promotion LLM request failed: empty response content");
  }
  return parseSharedMemoryPromotionResponse(textContent, candidates);
}

async function classifyWithOllama(
  cfg: SharedMemoryPromotionConfig,
  agentId: string,
  candidates: string[],
): Promise<SharedMemoryPromotionDecision> {
  if (!cfg.baseUrl || !cfg.model) {
    return { promote: [], skipped: candidates, reason: "promotion_disabled_or_empty" };
  }
  const response = await fetch(`${cfg.baseUrl.replace(/\/+$/, "")}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: cfg.model,
      stream: false,
      format: "json",
      options: { temperature: 0 },
      messages: buildSharedMemoryPromotionMessages(agentId, candidates),
    }),
  });
  const payload = (await response.json().catch(() => ({}))) as {
    message?: { content?: string };
    response?: string;
    error?: string;
  };
  if (!response.ok) {
    throw new Error(payload.error ?? `ollama promotion request failed: HTTP ${response.status}`);
  }
  const textContent = extractOllamaText(payload);
  if (!textContent) {
    throw new Error("ollama promotion request failed: empty response content");
  }
  return parseSharedMemoryPromotionResponse(textContent, candidates);
}

export async function classifySharedMemories(
  cfg: SharedMemoryPromotionConfig,
  agentId: string,
  texts: string[],
): Promise<SharedMemoryPromotionDecision> {
  const candidates = dedupePromotionCandidates(texts, cfg.maxCandidates);
  if (!cfg.enabled || candidates.length === 0) {
    return { promote: [], skipped: candidates, reason: "promotion_disabled_or_empty" };
  }
  if (cfg.provider === "ollama") {
    return classifyWithOllama(cfg, agentId, candidates);
  }
  return classifyWithOpenAICompatible(cfg, agentId, candidates);
}

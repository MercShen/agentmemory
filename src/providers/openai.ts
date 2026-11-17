import type { MemoryProvider } from "../types.js";
import { getEnvVar } from "../config.js";
import { fetchWithTimeout } from "./_fetch.js";
import {
  DEFAULT_AZURE_API_VERSION,
  buildAuthHeaders,
  buildChatUrl,
  detectAzure,
  normalizeBaseUrl,
} from "./_openai-shared.js";

const DEFAULT_MODEL = "gpt-5.6-luna";
const DEFAULT_TIMEOUT_MS = 60_000;

/**
 * Models served by OpenAI's own chat surface reject `max_tokens` and require
 * `max_completion_tokens`; OpenAI-compatible endpoints (DeepSeek, Qwen, vLLM,
 * Ollama, …) only accept `max_tokens`. A 400 from the wrong field blocks every
 * memory write, so the field is chosen per model and overridable.
 */
const COMPLETION_TOKEN_MODEL = /^(chatgpt-|o[0-9]|gpt-[5-9])/i;

export type TokenLimitField = "max_tokens" | "max_completion_tokens";

/** @returns the output-token field this provider should send for `model`. */
export function tokenLimitField(model: string, override?: string | null): TokenLimitField {
  const explicit = override?.trim().toLowerCase();
  if (explicit === "max_tokens" || explicit === "max_completion_tokens") return explicit;
  return COMPLETION_TOKEN_MODEL.test(model) ? "max_completion_tokens" : "max_tokens";
}

/**
 * OpenAI-compatible LLM provider.
 *
 * Uses raw fetch (no SDK) to support any OpenAI-compatible endpoint:
 *   - OpenAI official
 *   - Azure OpenAI (auto-detected from .openai.azure.com host)
 *   - DeepSeek
 *   - 硅基流动 (SiliconFlow)
 *   - vLLM / LM Studio / Ollama (with OpenAI compatibility layer)
 *   - Any other proxy implementing /v1/chat/completions
 *
 * Required env vars:
 *   OPENAI_API_KEY  — API key
 *
 * Optional:
 *   OPENAI_BASE_URL          — base URL without path (default: https://api.openai.com).
 *                              Azure: https://<resource>.openai.azure.com/openai/deployments/<deployment>
 *   OPENAI_MODEL             — model name (default: gpt-5.6-luna)
 *   OPENAI_API_VERSION       — Azure api-version query param (default: 2024-08-01-preview)
 *   OPENAI_TIMEOUT_MS        — outbound fetch timeout in ms (OpenAI-scoped alias,
 *                              takes precedence over AGENTMEMORY_LLM_TIMEOUT_MS
 *                              for back-compat with the v0.9.17 shipping name).
 *   AGENTMEMORY_LLM_TIMEOUT_MS — outbound fetch timeout in ms shared across all
 *                              raw-fetch LLM + embedding providers. Used when
 *                              OPENAI_TIMEOUT_MS is not set. Default: 60000.
 *   MAX_TOKENS               — max output tokens (default: from config or 4096)
 *   OPENAI_TOKEN_LIMIT_FIELD — "max_tokens" | "max_completion_tokens". Forces the
 *                              output-token field for endpoints whose model name
 *                              does not reveal the expected one.
 *   OPENAI_REASONING_EFFORT  — "low" | "medium" | "high" | "none"
 *                              Passthrough for reasoning models (e.g. Ollama Cloud
 *                              thinking models). Set to "none" to ensure
 *                              message.content is populated instead of only
 *                              message.reasoning.
 */
export class OpenAIProvider implements MemoryProvider {
  name = "openai";
  private apiKey: string;
  private model: string;
  private maxTokens: number;
  private baseUrl: string;
  private reasoningEffort?: string;
  private timeoutMs: number;
  private isAzure: boolean;
  private azureApiVersion: string;
  private tokenField: TokenLimitField;

  constructor(apiKey: string, model: string, maxTokens: number, baseURL?: string) {
    this.apiKey = apiKey;
    this.model = model;
    this.maxTokens = maxTokens;
    this.baseUrl = normalizeBaseUrl(baseURL || getEnvVar("OPENAI_BASE_URL"));
    this.reasoningEffort = getEnvVar("OPENAI_REASONING_EFFORT") || undefined;
    this.timeoutMs = resolveTimeout();
    this.azureApiVersion =
      getEnvVar("OPENAI_API_VERSION") || DEFAULT_AZURE_API_VERSION;
    this.isAzure = detectAzure(this.baseUrl);
    this.tokenField = tokenLimitField(model, getEnvVar("OPENAI_TOKEN_LIMIT_FIELD"));
  }

  async compress(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(systemPrompt, userPrompt);
  }

  async summarize(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(systemPrompt, userPrompt);
  }

  private async call(systemPrompt: string, userPrompt: string): Promise<string> {
    const url = buildChatUrl(this.baseUrl, this.isAzure, this.azureApiVersion);
    const body: Record<string, unknown> = {
      model: this.model,
      [this.tokenField]: this.maxTokens,
      stream: false,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    };
    if (this.reasoningEffort) {
      body.reasoning_effort = this.reasoningEffort;
    }

    try {
      const response = await fetchWithTimeout(
        url,
        {
          method: "POST",
          headers: buildAuthHeaders(this.apiKey, this.isAzure),
          body: JSON.stringify(body),
        },
        this.timeoutMs,
      );

      if (!response.ok) {
        const text = await response.text();
        const hint =
          response.status === 400 && /max_(completion_)?tokens/i.test(text)
            ? ` The endpoint rejected "${this.tokenField}"; set OPENAI_TOKEN_LIMIT_FIELD to "${
                this.tokenField === "max_tokens" ? "max_completion_tokens" : "max_tokens"
              }" for this model.`
            : "";
        throw new Error(`OpenAI API error (${response.status}): ${text}${hint}`);
      }

      const data = (await response.json()) as {
        choices?: Array<{
          finish_reason?: string;
          message?: { content?: string | null; reasoning?: string; reasoning_content?: string };
        }>;
      };
      const choice = data.choices?.[0];
      const message = choice?.message;
      const content = message?.content;
      if (content?.trim()) {
        return content;
      }
      if (choice?.finish_reason === "length") {
        throw new Error(
          `OpenAI response hit ${this.tokenField} (${this.maxTokens}) before any content. Raise MAX_TOKENS or lower OPENAI_REASONING_EFFORT.`,
        );
      }
      const reasoning = message?.reasoning ?? message?.reasoning_content;
      if (reasoning) {
        return reasoning;
      }
      throw new Error(
        `OpenAI returned unexpected response: ${JSON.stringify(data).slice(0, 200)}`,
      );
    } catch (err) {
      const aborted =
        err instanceof Error &&
        (err.name === "AbortError" || err.name === "TimeoutError");
      if (aborted) {
        throw new Error(
          `OpenAI API request timed out after ${this.timeoutMs}ms — set OPENAI_TIMEOUT_MS (or AGENTMEMORY_LLM_TIMEOUT_MS) to raise the bound or check the provider status.`,
        );
      }
      throw err;
    }
  }
}

function resolveTimeout(): number {
  const openaiRaw = getEnvVar("OPENAI_TIMEOUT_MS");
  const openai = parsePositiveInt(openaiRaw);
  if (openai !== undefined) return openai;

  const globalRaw = getEnvVar("AGENTMEMORY_LLM_TIMEOUT_MS");
  const globalMs = parsePositiveInt(globalRaw);
  if (globalMs !== undefined) return globalMs;

  return DEFAULT_TIMEOUT_MS;
}

function parsePositiveInt(raw: string | null | undefined): number | undefined {
  if (!raw) return undefined;
  const trimmed = raw.trim();
  // Reject malformed values like "30ms" or "1_000" — parseInt would
  // silently return 30 / 1, swallowing user typos as valid timeouts.
  // The regex enforces pure digits (no sign, no trailing units, no
  // separators) before we hand off to Number.
  if (!/^\d+$/.test(trimmed)) return undefined;
  const n = Number(trimmed);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

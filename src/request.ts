// Shaping that makes an OpenCode request to Anthropic look like one from Claude
// Code. Shared by the OpenCode 1 fetch wrapper and the OpenCode 2 HTTP hooks.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { hostname, userInfo } from "node:os";
import { join } from "node:path";
import {
  BETA_FLAGS,
  BILLING_CCH,
  CLI_BUILD_ID,
  CLI_VERSION,
  ENTRYPOINT,
  OAUTH_BETA_FLAG,
  PROFILE_URL,
  USER_AGENT,
} from "./constants.js";

export type OAuthProfile = {
  account?: { uuid?: string };
};

export type ClaudeSystemPromptCache = {
  system?: Array<{ type?: string; text?: string; cache_control?: unknown }>;
  billing?: {
    ccVersion?: string;
    ccVersionSuffix?: string;
    ccEntrypoint?: string;
  };
};

export const CLAUDE_PREFIX =
  "You are a Claude agent, built on Anthropic's Claude Agent SDK.";

/**
 * Derive a human-readable display name from a Claude model ID.
 *
 * Matches `claude-{family}-{major}-{minor}[-{date}]` (the convention used
 * for every Claude model family to date) and renders e.g.
 *   claude-opus-4-7            -> "Opus 4.7"
 *   claude-haiku-4-5-20251001  -> "Haiku 4.5"
 *   claude-sonnet-4-6          -> "Sonnet 4.6"
 *
 * Falls back to the raw model ID if the convention doesn't match, so future
 * naming changes are surfaced truthfully instead of silently wrong.
 */
export function deriveModelDisplayName(modelId: string): string {
  const m = modelId.match(/^claude-([a-z]+)-(\d+)-(\d+)(?:-\d+)?$/i);
  if (!m) return modelId;
  const [, family, major, minor] = m;
  const capitalized = family.charAt(0).toUpperCase() + family.slice(1).toLowerCase();
  return `${capitalized} ${major}.${minor}`;
}

/**
 * Rewrite model-identity lines inside the cached Claude Code system prompt
 * so they reflect the model actually being requested. Without this, every
 * request (regardless of selected model) gets a system prompt claiming
 * "You are powered by the model named Sonnet 4.6", which can bias behavior.
 */
export function rewriteSystemBlocksForModel(
  blocks: Array<{ type?: string; text?: string }>,
  modelId: string | undefined,
): Array<{ type?: string; text?: string }> {
  if (!modelId) return blocks;
  const display = deriveModelDisplayName(modelId);

  return blocks.map((block) => {
    if (block?.type !== "text" || typeof block.text !== "string") return block;
    let text = block.text;

    text = text.replace(
      /You are powered by the model named [^\n]+? The exact model ID is [a-z0-9.-]+\./g,
      `You are powered by the model named ${display}. The exact model ID is ${modelId}.`,
    );

    return { ...block, text };
  });
}

export function stripSystemCacheControl(
  system: Array<{ type?: string; text?: string; cache_control?: unknown }>,
): Array<{ type?: string; text?: string }> {
  return system.map((block) => {
    if (!("cache_control" in block)) return block;
    const { cache_control, ...rest } = block;
    return rest;
  });
}

const oauthProfileCache = new Map<string, Promise<OAuthProfile | null>>();
const SYSTEM_PROMPT_CACHE_PATH = process.env.ANTHROPIC_SYSTEM_PROMPT_PATH
  || join(process.env.HOME || "", ".cache", "opencode-claude-bridge", "claude-system-prompt.json");

/** Merge HeadersInit (Headers | string[][] | Record) onto a Headers object. */
export function mergeHeaders(target: Headers, source: HeadersInit) {
  if (source instanceof Headers) {
    source.forEach((v, k) => target.set(k, v));
  } else if (Array.isArray(source)) {
    for (const [k, v] of source) {
      if (v !== undefined) target.set(k, String(v));
    }
  } else {
    for (const [k, v] of Object.entries(source)) {
      if (v !== undefined) target.set(k, String(v));
    }
  }
}

/** Build deduplicated beta flags string. */
export function buildBetaFlags(existing: string): string {
  // Preserve SDK betas that enable fields already present in the request body.
  const flags = [
    "claude-code-20250219",
    OAUTH_BETA_FLAG,
    ...BETA_FLAGS.split(","),
    ...existing.split(","),
  ];
  return [...new Set(flags.map((flag) => flag.trim()).filter(Boolean))].join(",");
}

/** Deduplicate repeated Claude Code prefix in a text block. */
function deduplicatePrefix(text: string): string {
  const doubled = `${CLAUDE_PREFIX}\n\n${CLAUDE_PREFIX}`;
  while (text.includes(doubled)) {
    text = text.replace(doubled, CLAUDE_PREFIX);
  }
  return text;
}

export function maybeUnquoteText(text: string): string {
  const trimmed = text.trim();
  if (!trimmed.startsWith("\"")) return text;
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed === "string") return parsed;
  } catch {
    // Expected for non-JSON text — don't log.
  }
  return text;
}

export function stripAssistantPrefillForClaude(
  messages: Array<{ role?: string; content?: unknown }>,
): Array<{ role?: string; content?: unknown }> {
  if (messages.length === 0) return messages;
  const last = messages[messages.length - 1];
  if (last?.role !== "assistant") return messages;
  if (typeof last.content !== "string") return messages;
  if (last.content.trim() !== "Continue with your tasks.") return messages;
  return messages.slice(0, -1);
}

export function normalizeSystemBlocks(
  system: Array<{ type?: string; text?: string }>,
): Array<{ type?: string; text?: string }> {
  const normalized = system.map((item) => {
    if (item.type === "text" && item.text) {
      return {
        ...item,
        text: deduplicatePrefix(
          item.text
            .replace(/OpenCode/g, "Claude Code")
            .replace(/opencode/gi, "Claude"),
        ),
      };
    }
    return item;
  });

  const billing = normalized[0];
  const prompt = normalized[1];
  if (
    billing?.type === "text"
    && billing.text?.startsWith("x-anthropic-billing-header:")
    && prompt?.type === "text"
    && prompt.text?.startsWith(`${CLAUDE_PREFIX}\n\n`)
  ) {
    const rest = prompt.text.slice(`${CLAUDE_PREFIX}\n\n`.length);
    return [
      billing,
      { ...prompt, text: CLAUDE_PREFIX },
      { ...prompt, text: rest },
      ...normalized.slice(2),
    ];
  }

  return normalized;
}

export function readCachedClaudePromptCache(): ClaudeSystemPromptCache | null {
  try {
    const raw = readFileSync(SYSTEM_PROMPT_CACHE_PATH, "utf8");
    const parsed = JSON.parse(raw) as ClaudeSystemPromptCache;
    if (!Array.isArray(parsed.system) || parsed.system.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function getStableDeviceId(): string {
  let who = "unknown";
  try {
    who = `${userInfo().username}@${hostname()}`;
  } catch {
    try {
      who = `${process.env.USER || process.env.USERNAME || "unknown"}@${hostname()}`;
    } catch {}
  }
  return createHash("sha256").update(who).digest("hex");
}

export function buildBillingHeader(
  sessionId: string,
  cachedClaudePromptCache: ClaudeSystemPromptCache | null,
  dynamicFingerprint?: string,
): string {
  const cch = BILLING_CCH
    || createHash("sha256").update(sessionId).digest("hex").slice(0, 5);
  // Prefer dynamic fingerprint (computed per-request from first user message)
  // over static cached value, matching Claude Code's actual behavior.
  const suffix = dynamicFingerprint
    || cachedClaudePromptCache?.billing?.ccVersionSuffix
    || CLI_BUILD_ID;
  const ccVersion = cachedClaudePromptCache?.billing?.ccVersion
    || `${CLI_VERSION}.${suffix}`;
  const ccEntrypoint = cachedClaudePromptCache?.billing?.ccEntrypoint || ENTRYPOINT;
  return `x-anthropic-billing-header: cc_version=${ccVersion}; cc_entrypoint=${ccEntrypoint}; cch=${cch};`;
}

export async function fetchOAuthProfile(accessToken: string): Promise<OAuthProfile | null> {
  const cached = oauthProfileCache.get(accessToken);
  if (cached) return cached;

  const pending = (async () => {
    try {
      const response = await globalThis.fetch(PROFILE_URL, {
        headers: {
          authorization: `Bearer ${accessToken}`,
          "content-type": "application/json",
          "user-agent": USER_AGENT,
          "anthropic-beta": OAUTH_BETA_FLAG,
        },
      });
      if (!response.ok) return null;
      return await response.json() as OAuthProfile;
    } catch {
      return null;
    }
  })();

  oauthProfileCache.set(accessToken, pending);
  return pending;
}

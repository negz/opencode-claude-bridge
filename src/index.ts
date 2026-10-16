import { randomUUID } from "node:crypto";
import {
  createAuthorizationRequest,
  exchangeCodeForTokens,
  parseAuthCode,
  refreshTokens,
  startOAuthCallbackServer,
} from "./oauth.js";
import { getClaudeTokens, readClaudeCredentials } from "./keychain.js";
import {
  SESSION_ID,
  PLUGIN_ID,
} from "./constants.js";
import {
  CLAUDE_PREFIX,
  buildClaudeHeaders,
  buildClaudeUrl,
  getStableDeviceId,
  mergeHeaders,
  readCachedClaudePromptCache,
  shapeClaudeBody,
} from "./request.js";

export {
  buildBetaFlags,
  deriveModelDisplayName,
  rewriteSystemBlocksForModel,
  stripAssistantPrefillForClaude,
  stripSystemCacheControl,
} from "./request.js";
import {
  getClaudeTools,
  translateArgsOpencodeToClaude,
  translateToolArgsJsonString,
  shouldUseClaudeToolSchemas,
} from "./claude-tools.js";
import type { ToolDefinition } from "./claude-tools.js";
import { createSseProcessor } from "./stream.js";
import { setup } from "./v2.js";

// ── Types ──────────────────────────────────────────────────────────

type AuthType = {
  type: string;
  access?: string;
  refresh?: string;
  expires?: number;
};

type ProviderModel = {
  cost?: unknown;
  name?: string;
  attachment?: boolean;
  reasoning?: boolean;
  tool_call?: boolean;
  temperature?: boolean;
  limit?: { context: number; output: number };
  modalities?: { input: string[]; output: string[] };
};

const OUTBOUND_TOOL_NAME_MAP: Record<string, string> = {
  bash: "Bash",
  read: "Read",
  glob: "Glob",
  grep: "Grep",
  edit: "Edit",
  write: "Write",
  task: "Agent",
  webfetch: "WebFetch",
  todowrite: "TodoWrite",
  skill: "Skill",
  question: "AskUserQuestion",
  plan_enter: "EnterPlanMode",
  plan_exit: "ExitPlanMode",
};

const ACTIVE_TOOL_SCHEMA_NAME_MAP: Record<string, string> = {
  bash: "Bash",
  read: "Read",
  glob: "Glob",
  grep: "Grep",
  edit: "Edit",
  write: "Write",
  task: "Agent",
  webfetch: "WebFetch",
  todowrite: "TodoWrite",
  skill: "Skill",
  question: "AskUserQuestion",
  plan_enter: "EnterPlanMode",
  plan_exit: "ExitPlanMode",
};

const INBOUND_TOOL_NAME_MAP: Record<string, string> = {
  Bash: "bash",
  Read: "read",
  Glob: "glob",
  Grep: "grep",
  Edit: "edit",
  Write: "write",
  Agent: "task",
  WebFetch: "webfetch",
  TodoWrite: "todowrite",
  Skill: "skill",
  AskUserQuestion: "question",
  EnterPlanMode: "plan_enter",
  ExitPlanMode: "plan_exit",
};

const ANTHROPIC_MODELS: Record<string, ProviderModel> = {
  "claude-opus-4-6": {
    name: "Claude Opus 4.6",
    attachment: true, reasoning: true, tool_call: true, temperature: false,
    limit: { context: 200000, output: 32000 },
    modalities: { input: ["text", "image"], output: ["text"] },
    cost: { input: 15, output: 75, cache_read: 1.5, cache_write: 18.75 },
  },
  "claude-opus-4-5": {
    name: "Claude Opus 4.5",
    attachment: true, reasoning: true, tool_call: true, temperature: false,
    limit: { context: 200000, output: 32000 },
    modalities: { input: ["text", "image"], output: ["text"] },
    cost: { input: 15, output: 75, cache_read: 1.5, cache_write: 18.75 },
  },
  "claude-sonnet-4-6": {
    name: "Claude Sonnet 4.6",
    attachment: true, reasoning: true, tool_call: true, temperature: false,
    limit: { context: 200000, output: 64000 },
    modalities: { input: ["text", "image"], output: ["text"] },
    cost: { input: 3, output: 15, cache_read: 0.3, cache_write: 3.75 },
  },
  "claude-sonnet-4-5": {
    name: "Claude Sonnet 4.5",
    attachment: true, reasoning: true, tool_call: true, temperature: false,
    limit: { context: 200000, output: 8192 },
    modalities: { input: ["text", "image"], output: ["text"] },
    cost: { input: 3, output: 15, cache_read: 0.3, cache_write: 3.75 },
  },
  "claude-haiku-4-5-20251001": {
    name: "Claude Haiku 4.5",
    attachment: true, reasoning: false, tool_call: true, temperature: true,
    limit: { context: 200000, output: 8192 },
    modalities: { input: ["text", "image"], output: ["text"] },
    cost: { input: 0.8, output: 4, cache_read: 0.08, cache_write: 1 },
  },
};

type PluginClient = {
  auth: {
    set: (args: {
      path: { id: string };
      body: Record<string, unknown>;
    }) => Promise<void>;
  };
};

// ── Helpers ────────────────────────────────────────────────────────

export function shouldInjectClaudeTools(input: {
  model?: string;
  requestUrl?: string;
  tools?: unknown;
}): boolean {
  if (!shouldUseClaudeToolSchemas({ model: input.model, requestUrl: input.requestUrl })) {
    return false;
  }
  return Array.isArray(input.tools) && input.tools.length > 0;
}

function getToolName(tool: unknown): string | undefined {
  if (!tool || typeof tool !== "object") return undefined;
  const name = (tool as { name?: unknown }).name;
  return typeof name === "string" ? name : undefined;
}

function getToolInputSchema(tool: unknown): Record<string, unknown> | undefined {
  if (!tool || typeof tool !== "object") return undefined;
  const inputSchema = (tool as { input_schema?: unknown }).input_schema;
  if (!inputSchema || typeof inputSchema !== "object" || Array.isArray(inputSchema)) return undefined;
  return inputSchema as Record<string, unknown>;
}

export function getClaudeToolsForActiveOpenCodeTools(
  tools: unknown,
): ToolDefinition[] {
  if (!Array.isArray(tools)) return [];
  const claudeToolsByName = new Map(getClaudeTools().map((tool) => [tool.name, tool]));
  const selectedToolsByName = new Map<string, ToolDefinition>();

  for (const tool of tools) {
    const name = getToolName(tool);
    if (!name) continue;

    const claudeName = ACTIVE_TOOL_SCHEMA_NAME_MAP[name];
    if (claudeName) {
      const claudeTool = claudeToolsByName.get(claudeName);
      if (!claudeTool) continue;
      selectedToolsByName.set(claudeName, claudeTool);
      continue;
    }

    const inputSchema = getToolInputSchema(tool);
    if (!inputSchema) {
      console.error(
        `[opencode-claude-bridge] Dropping active tool "${name}" — missing or invalid input_schema`,
      );
      continue;
    }

    const description = (tool as { description?: unknown }).description;
    selectedToolsByName.set(name, {
      name,
      description: typeof description === "string" ? description : "",
      input_schema: inputSchema,
    });
  }

  return Array.from(selectedToolsByName.values()).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
}

export function getInboundToolNameMapForActiveOpenCodeTools(
  tools: unknown,
): Record<string, string> {
  if (!Array.isArray(tools)) return {};
  const inboundToolNameMap: Record<string, string> = {};

  for (const tool of tools) {
    const name = getToolName(tool);
    if (!name) continue;
    const claudeName = ACTIVE_TOOL_SCHEMA_NAME_MAP[name];
    if (claudeName) inboundToolNameMap[claudeName] = name;
  }

  return inboundToolNameMap;
}

function buildAuthFailureResponse(detail: string): Response {
  const message = [
    "Claude OAuth credentials are no longer valid.",
    detail,
    "Reconnect 'Claude Pro / Max (OAuth)' in OpenCode or run `claude login` again.",
  ].filter(Boolean).join(" ");

  return new Response(JSON.stringify({
    type: "error",
    error: {
      type: "authentication_error",
      message,
    },
  }), {
    status: 401,
    headers: {
      "content-type": "application/json",
    },
  });
}

/** Persist fresh tokens to OpenCode's auth store. */
async function storeAuth(
  client: PluginClient,
  tokens: { access: string; refresh: string; expires: number },
) {
  await client.auth.set({
    path: { id: "anthropic" },
    body: {
      type: "oauth",
      refresh: tokens.refresh,
      access: tokens.access,
      expires: tokens.expires,
    },
  });
}

/** Layered token refresh: valid keychain access → stored refresh → CLI refresh token. */
async function refreshAuth(
  auth: AuthType,
  client: PluginClient,
): Promise<string | null> {
  type Tokens = { access: string; refresh: string; expires: number };
  let fresh: Tokens | null = null;
  const errors: string[] = [];
  const attemptedRefreshTokens = new Set<string>();

  const tryRefresh = (refreshToken: string | undefined): Tokens | null => {
    if (!refreshToken || attemptedRefreshTokens.has(refreshToken)) return null;
    attemptedRefreshTokens.add(refreshToken);
    return refreshTokens(refreshToken);
  };

  // Layer 1: Claude CLI keychain, but only if the access token is already
  // valid. Expired keychain refresh tokens are often stale after re-login, so
  // do not try them before OpenCode's own stored OAuth refresh token.
  try {
    const kt = getClaudeTokens({ refreshExpired: false });
    if (kt && kt.expires > Date.now() + 60_000) fresh = kt;
  } catch (err) {
    errors.push(String(err));
  }

  // Layer 2: Stored refresh token
  if (!fresh && auth.refresh) {
    try { fresh = tryRefresh(auth.refresh); }
    catch (err) { errors.push(String(err)); }
  }

  // Layer 3: CLI refresh token, only if it differs from tokens already tried.
  if (!fresh) {
    try {
      const creds = readClaudeCredentials();
      if (creds?.claudeAiOauth?.refreshToken)
        fresh = tryRefresh(creds.claudeAiOauth.refreshToken);
    } catch (err) {
      errors.push(String(err));
    }
  }

  if (fresh) {
    await storeAuth(client, fresh);
    auth.access = fresh.access;
    auth.refresh = fresh.refresh;
    auth.expires = fresh.expires;
    return fresh.access;
  }

  if (errors.length > 0) {
    throw new Error(errors[errors.length - 1]);
  }

  return null;
}

/** Best-effort browser open — tries multiple methods on each platform. */
function openBrowser(url: string): void {
  (async () => {
    const { execFileSync } = await import("node:child_process");

    if (process.platform === "darwin") {
      const uid = String(process.getuid?.() ?? "");
      const attempts: Array<[string, () => void]> = [
        ["launchctl asuser", () => execFileSync("/bin/launchctl", ["asuser", uid, "/usr/bin/open", url], { timeout: 5000 })],
        ["open", () => execFileSync("/usr/bin/open", [url], { timeout: 3000 })],
        ["osascript", () => execFileSync("/usr/bin/osascript", ["-e", `open location "${url}"`], { timeout: 3000 })],
      ];
      for (const [name, attempt] of attempts) {
        try { attempt(); break; }
        catch {}
      }
    } else if (process.platform === "win32") {
      try { execFileSync("cmd", ["/c", "start", "", url], { timeout: 3000 }); }
      catch {}
    } else {
      const attempts: Array<[string, () => void]> = [
        ["xdg-open", () => execFileSync("/usr/bin/xdg-open", [url], { timeout: 3000 })],
        ["open", () => execFileSync("/usr/bin/open", [url], { timeout: 3000 })],
      ];
      for (const [name, attempt] of attempts) {
        try { attempt(); break; }
        catch {}
      }
    }
  })().catch(() => {});
}

export function mapOutboundToolName(name: string | undefined): string | undefined {
  if (!name) return name;
  return OUTBOUND_TOOL_NAME_MAP[name] || name;
}

// ── Plugin ─────────────────────────────────────────────────────────

const OpenCodeClaudeBridge = async ({ client }: { client: PluginClient }) => {
  // Save and clear ANTHROPIC_API_KEY so SDK doesn't bypass our fetch.
  // Restored if OAuth isn't active (API key providers still work).
  const savedApiKey = process.env.ANTHROPIC_API_KEY;
  const sessionId = SESSION_ID || randomUUID();
  const deviceId = getStableDeviceId();
  const cachedClaudePromptCache = readCachedClaudePromptCache();

  // Bootstrap auth from Claude CLI keychain at plugin init time — before
  // OpenCode builds its provider state. This ensures the loader runs on
  // startup so models appear immediately without requiring a restart.
  try {
    const tokens = getClaudeTokens({ refreshExpired: false });
    if (tokens) await storeAuth(client, tokens);
  } catch {}

  return {
    "experimental.chat.system.transform": (
      input: { model?: { providerID: string } },
      output: { system: string[] },
    ) => {
      if (input.model?.providerID !== "anthropic") return;
      if (output.system.some((s) => s.includes(CLAUDE_PREFIX))) return;
      if (output.system.length > 0) {
        output.system[0] = `${CLAUDE_PREFIX}\n\n${output.system[0]}`;
      } else {
        output.system.push(CLAUDE_PREFIX);
      }
    },

    auth: {
      provider: "anthropic",

      async loader(
        getAuth: () => Promise<AuthType>,
        provider: { models: Record<string, ProviderModel> },
      ) {
        // Always inject Claude models — runs before any auth check so models
        // appear in the selector even on a fresh install with no auth configured
        if (provider?.models !== undefined) {
          for (const [id, def] of Object.entries(ANTHROPIC_MODELS)) {
            if (!provider.models[id]) provider.models[id] = { ...def };
          }
        }

        let auth = await getAuth();

        // Auto-bootstrap from Claude CLI keychain if no OAuth tokens stored
        if (auth.type !== "oauth") {
          try {
            const tokens = getClaudeTokens({ refreshExpired: false });
            if (tokens) {
              await storeAuth(client, tokens);
              auth = { type: "oauth", ...tokens };
            }
          } catch {}
        }

        // API key mode — set the key in env and let the SDK handle everything
        if (auth.type === "apikey" && auth.access) {
          process.env.ANTHROPIC_API_KEY = auth.access;
          return {};
        }

        if (auth.type !== "oauth") {
          // Not configured — restore env API key if present so SDK can use it
          if (savedApiKey) process.env.ANTHROPIC_API_KEY = savedApiKey;
          return {};
        }

        // OAuth active — set a placeholder so the SDK doesn't error on missing key.
        // Our fetch wrapper sets the real Authorization: Bearer header and removes x-api-key.
        process.env.ANTHROPIC_API_KEY = "oauth-placeholder";

        return {
          // OpenCode 1.4.0 now rejects Anthropic provider config when both an
          // apiKey and authToken are present. Keep the placeholder apiKey for
          // provider init, and let the fetch wrapper supply the real bearer token.
          async fetch(input: string | URL | Request, init?: RequestInit) {
            const auth = await getAuth();
            if (auth.type !== "oauth") return fetch(input, init);

            let accessToken = auth.access || null;
            if (!accessToken || !auth.expires || auth.expires < Date.now()) {
              try {
                accessToken = await refreshAuth(auth, client);
              } catch (err) {
                const detail = err instanceof Error ? err.message : String(err);
                console.error(`[opencode-claude-bridge] OAuth refresh failed: ${detail}`);
                return buildAuthFailureResponse(detail);
              }
            }

            if (!accessToken) {
              return buildAuthFailureResponse("No usable OAuth access token is available.");
            }

            const url = typeof input === "string" ? input
              : input instanceof URL ? input.toString()
              : input.url;

            // ── Headers ──
            const sourceHeaders = new Headers();
            if (input instanceof Request) mergeHeaders(sourceHeaders, input.headers);
            if (init?.headers) mergeHeaders(sourceHeaders, init.headers);
            const headers = buildClaudeHeaders(sourceHeaders, accessToken, sessionId);

            // ── Body ──
            let body = init?.body;
            let inboundToolNameMap = INBOUND_TOOL_NAME_MAP;
            if (body && typeof body === "string") {
              try {
                const parsed = JSON.parse(body);
                let requestUrl = "";
                try {
                  requestUrl = String(new URL(url));
                } catch {}

                await shapeClaudeBody(parsed, requestUrl, {
                  sessionId,
                  deviceId,
                  accessToken: auth.access,
                  cachedClaudePromptCache,
                  // Keep OpenCode's default tool schemas for non-Claude targets.
                  // Claude wire schemas are only useful for Anthropic-native
                  // requests or Claude-family models on Anthropic-compatible
                  // routers such as OpenRouter.
                  replaceTools: (parsed, requestUrl) => {
                    delete parsed.tool_choice;
                    if (!shouldInjectClaudeTools({ model: parsed.model, requestUrl, tools: parsed.tools })) return;
                    inboundToolNameMap = getInboundToolNameMapForActiveOpenCodeTools(parsed.tools);
                    parsed.tools = getClaudeToolsForActiveOpenCodeTools(parsed.tools);
                  },
                  translateToolUse: (block) => {
                    block.name = mapOutboundToolName(block.name);
                    // Consolidated OpenCode → Claude argument translation:
                    // key renames, Agent/AskUserQuestion/Skill/WebFetch/TodoWrite
                    // field bridging. Mirrors translateToolArgsJsonString
                    // (inbound) so round-trips preserve meaning. See
                    // translateArgsOpencodeToClaude in claude-tools.ts.
                    if (block.input && typeof block.input === "object") {
                      block.input = translateArgsOpencodeToClaude(
                        block.name,
                        block.input as Record<string, unknown>,
                      );
                    }
                  },
                });

                body = JSON.stringify(parsed);
              } catch {}
            }

            // ── URL: add ?beta=true ──
            const finalUrl = buildClaudeUrl(url);

            // ── Request (with 429 auto-refresh retry) ──
            const outHeaders: Record<string, string> = {};
            headers.forEach((v, k) => { outHeaders[k] = v; });

            const doFetch = () => globalThis.fetch(finalUrl, {
              method: init?.method || "POST",
              body,
              headers: outHeaders,
              signal: init?.signal,
            });

            let response = await doFetch();

            // 429 auto-refresh: rate limits are per-access-token, so refreshing
            // the token gives us a fresh rate limit bucket. Try up to 2 retries.
            if (response.status === 429) {
              for (let retry = 0; retry < 2; retry++) {
                console.error(`[opencode-claude-bridge] 429 rate limited (attempt ${retry + 1}/2), refreshing token...`);
                let freshToken: string | null = null;
                try {
                  freshToken = await refreshAuth(auth, client);
                } catch (err) {
                  const detail = err instanceof Error ? err.message : String(err);
                  console.error(`[opencode-claude-bridge] OAuth refresh failed after 429: ${detail}`);
                  return buildAuthFailureResponse(detail);
                }
                if (!freshToken) {
                  console.error("[opencode-claude-bridge] Token refresh failed, returning 429");
                  break;
                }
                outHeaders["authorization"] = `Bearer ${freshToken}`;
                // Back off briefly: 1s first retry, 2s second
                await new Promise((r) => setTimeout(r, 1000 * (retry + 1)));
                response = await doFetch();
                if (response.status !== 429) break;
              }
            }

            // ── Map Claude-style tool names and arguments back to OpenCode ──
            if (!response.body) return response;

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const encoder = new TextEncoder();

            // Parse SSE events on \n\n boundaries, buffer tool_use
            // input_json_delta fragments per content-block index, and
            // translate the assembled JSON once at content_block_stop.
            // This avoids chunk-boundary corruption (e.g. "file_" + "path"
            // split across TCP chunks) that regex-on-raw-bytes can't
            // handle. See src/stream.ts for the processor implementation.
            const processor = createSseProcessor({
              inboundToolNameMap,
              translateToolArgs: translateToolArgsJsonString,
            });

            return new Response(
              new ReadableStream({
                start(controller) {
                  // Use start() with an async loop rather than pull():
                  // Bun's ReadableStream doesn't reliably re-invoke pull()
                  // when the handler resolves without enqueuing (which
                  // happens while we're buffering input_json_delta
                  // fragments), so pull() can stall the stream.
                  (async () => {
                    try {
                      while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                          const tail = processor.flush();
                          if (tail) controller.enqueue(encoder.encode(tail));
                          controller.close();
                          return;
                        }
                        const out = processor.feedChunk(decoder.decode(value, { stream: true }));
                        if (out) controller.enqueue(encoder.encode(out));
                      }
                    } catch (e) {
                      controller.error(e);
                    }
                  })();
                },
              }),
              { status: response.status, statusText: response.statusText, headers: response.headers },
            );
          },
        };
      },

      methods: [
        {
          label: "Claude Pro / Max (OAuth)",
          type: "oauth" as const,
          authorize: async () => {
            // First try: auto-bootstrap from Claude CLI keychain
            const tokens = getClaudeTokens({ refreshExpired: false });
            if (tokens) {
              await storeAuth(client, tokens);
              return {
                instructions: "Connected via Claude CLI — press Esc, then restart OpenCode if Anthropic models aren't visible",
                method: "code" as const,
                callback: async () => ({ type: "success" as const, ...tokens }),
              };
            }

            // Start a local HTTP server to capture the OAuth redirect.
            // This gives us an ephemeral port so the redirect_uri is
            // http://localhost:{port}/callback — matching Claude Code's flow.
            let callbackServer: Awaited<ReturnType<typeof startOAuthCallbackServer>> | null = null;
            try {
              callbackServer = await startOAuthCallbackServer();
            } catch (err) {
              console.error(`[opencode-oauth] Failed to start callback server: ${err}`);
            }

            if (callbackServer) {
              const { url, manualUrl, verifier, port } = callbackServer;
              const redirectUri = `http://localhost:${port}/callback`;

              // Best-effort browser open
              openBrowser(url);

              // Race: localhost callback vs manual code paste.
              // If the browser redirects to localhost, the callback server
              // captures the code automatically. Otherwise, the user can
              // paste the code from the manual-flow URL.
              //
              // We set up the automatic flow listener in the background.
              // If the user pastes a code first, that wins.
              let autoCode: string | null = null;
              callbackServer.waitForCode().then((code) => {
                autoCode = code;
              }).catch(() => {});

              return {
                url,
                instructions: `Opening browser for OAuth...\n\nIf the browser redirected back successfully, just press Enter.\nOtherwise, open this URL and paste the code:\n${manualUrl}`,
                method: "code" as const,
                callback: async (manualInput: string) => {
                  try {
                    // Brief pause to let the auto-flow resolve if it just arrived
                    if (!autoCode && !manualInput?.trim()) {
                      await new Promise((r) => setTimeout(r, 500));
                    }

                    if (autoCode) {
                      // Auto-flow captured the code via localhost redirect
                      const tokens = exchangeCodeForTokens(autoCode, verifier, redirectUri);
                      await storeAuth(client, tokens);
                      callbackServer?.close();
                      return { type: "success" as const, ...tokens };
                    }

                    const code = parseAuthCode(manualInput);
                    if (!code) {
                      console.error("[opencode-oauth] No authorization code received");
                      callbackServer?.close();
                      return { type: "failed" as const };
                    }

                    // Manual flow — redirect_uri defaults to MANUAL_REDIRECT_URL
                    const tokens = exchangeCodeForTokens(code, verifier);
                    await storeAuth(client, tokens);
                    callbackServer?.close();
                    return { type: "success" as const, ...tokens };
                  } catch (err) {
                    console.error(`[opencode-oauth] Token exchange failed: ${err}`);
                    callbackServer?.close();
                    return { type: "failed" as const };
                  }
                },
              };
            }

            // Fallback: no callback server — use manual redirect URL flow only.
            // This is the path for environments where we can't bind localhost.
            const { url, verifier } = createAuthorizationRequest();

            openBrowser(url);

            return {
              url,
              instructions: `Opening browser for OAuth...\n\nAfter authorizing, you'll see a code — paste it below:\n${url}`,
              method: "code" as const,
              callback: async (code: string) => {
                try {
                  // redirect_uri defaults to MANUAL_REDIRECT_URL in exchangeCodeForTokens
                  const tokens = exchangeCodeForTokens(parseAuthCode(code), verifier);
                  await storeAuth(client, tokens);
                  return { type: "success" as const, ...tokens };
                } catch (err) {
                  console.error(`[opencode-oauth] Token exchange failed: ${err}`);
                  return { type: "failed" as const };
                }
              },
            };
          },
        },
        {
          label: "Claude API Key",
          type: "oauth" as const,
          authorize: async () => {
            // Auto-detect from environment
            const envKey = process.env.ANTHROPIC_API_KEY || savedApiKey;
            if (envKey) {
              await client.auth.set({
                path: { id: "anthropic" },
                body: { type: "apikey", access: envKey, expires: Date.now() + 365 * 24 * 60 * 60 * 1000 },
              });
              return {
                instructions: "✓ API key found in environment — press Esc, then restart OpenCode if Anthropic models aren't visible",
                method: "code" as const,
                callback: async () => ({ type: "success" as const, access: envKey, refresh: "", expires: Date.now() + 365 * 24 * 60 * 60 * 1000 }),
              };
            }
            // Prompt user to paste key
            return {
              instructions: "Paste your Anthropic API key (sk-ant-...):",
              method: "code" as const,
              callback: async (key: string) => {
                const k = key.trim();
                if (!k) return { type: "failed" as const };
                await client.auth.set({
                  path: { id: "anthropic" },
                  body: { type: "apikey", access: k, expires: Date.now() + 365 * 24 * 60 * 60 * 1000 },
                });
                return { type: "success" as const, access: k, refresh: "", expires: Date.now() + 365 * 24 * 60 * 60 * 1000 };
              },
            };
          },
        },
      ],
    },
  };
};

/**
 * opencode plugin module.
 *
 * The `{ id, server }` shape is what OpenCode 1's plugin loader looks for. A
 * bare function default export makes it fall back to its legacy path, which
 * treats every named export of this module as a plugin factory and invokes it
 * with `(input, options)` — crashing on the helpers exported here for tests.
 *
 * OpenCode 2 calls `setup` instead, so one package serves both.
 */
export default {
  id: PLUGIN_ID,
  server: OpenCodeClaudeBridge,
  setup,
};

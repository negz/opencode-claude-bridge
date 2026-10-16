// OpenCode 2 entrypoint. OpenCode 2 stores the Claude OAuth credential, asks
// this plugin to refresh it, and sends it as a bearer token, so all the plugin
// does per request is make it look like one from Claude Code.
//
// OpenCode 2 sends its own tool definitions. Unlike the OpenCode 1 entrypoint,
// this one doesn't swap them for Claude Code's, so it keeps tool_choice too,
// which OpenCode 2 uses to stop tool calls at an agent's step limit. Anthropic
// accepts requests with OpenCode 2's tools and bills them to the Claude plan.
//
// It also leaves retries to OpenCode, rather than refreshing the token on a
// 429, because each refresh spends a single-use refresh token.
import { randomUUID } from "node:crypto";
import { SESSION_ID } from "./constants.js";
import { getClaudeTokens, readClaudeCredentials } from "./keychain.js";
import {
  createAuthorizationRequest,
  exchangeCodeForTokens,
  parseAuthCode,
  refreshTokens,
} from "./oauth.js";
import type { OAuthTokens } from "./oauth.js";
import {
  CLAUDE_PREFIX,
  buildClaudeHeaders,
  buildClaudeUrl,
  getStableDeviceId,
  readCachedClaudePromptCache,
  shapeClaudeBody,
} from "./request.js";
import type { ClaudeSystemPromptCache } from "./request.js";

// The parts of OpenCode 2's plugin API this plugin uses, from @opencode/plugin
// 2.0.30's promise API (packages/plugin/src/promise in anomalyco/opencode).
// Declared here because that package ships TypeScript sources whose
// dependencies don't fit this package's build.
export type OAuthCredential = {
  type: "oauth";
  methodID: string;
  access: string;
  refresh: string;
  expires: number;
};

export type OAuthAuthorization = {
  url: string;
  instructions: string;
} & (
  | { mode: "auto"; callback: Promise<OAuthCredential> }
  | { mode: "code"; callback: (code: string) => Promise<OAuthCredential> }
);

type IntegrationEditor = {
  method: {
    update(input: {
      integrationID: string;
      method: { id: string; type: "oauth"; label: string };
      authorize: () => Promise<OAuthAuthorization>;
      refresh: (credential: OAuthCredential) => Promise<OAuthCredential>;
    }): void;
  };
};

type HttpRequestEvent = { request: Request };

export type PluginContext = {
  integration: {
    transform(callback: (editor: IntegrationEditor) => void): Promise<unknown>;
  };
  session: {
    hook(
      name: "http.request",
      callback: (event: HttpRequestEvent) => Promise<void>,
      options: { providerID: string },
    ): Promise<unknown>;
  };
};

// OpenCode 2 files the Claude credential it imports from OpenCode 1's auth.json
// under this method ID, so registering it lets this plugin refresh that
// credential as well as ones it creates.
const METHOD_ID = "oauth";

// OpenCode 2 refreshes a credential five minutes before it expires. Only treat
// a token as fresh when it outlives that comfortably, or OpenCode would ask to
// refresh it again on the next request.
const FRESH_MARGIN_MS = 10 * 60 * 1000;

function toCredential(tokens: OAuthTokens): OAuthCredential {
  return { type: "oauth", methodID: METHOD_ID, ...tokens };
}

export async function authorize(): Promise<OAuthAuthorization> {
  const tokens = getClaudeTokens({ refreshExpired: false });
  if (tokens) {
    return {
      url: "https://claude.ai",
      instructions: "Connected with the Claude CLI's credentials.",
      mode: "auto",
      callback: Promise.resolve(toCredential(tokens)),
    };
  }

  const { url, verifier } = createAuthorizationRequest();
  return {
    url,
    instructions: "Sign in to Claude, then paste the code it shows you.",
    mode: "code",
    callback: async (code) => toCredential(exchangeCodeForTokens(parseAuthCode(code), verifier)),
  };
}

export type RefreshDeps = {
  /** The Claude CLI's tokens, if its access token is still valid. */
  cliTokens: () => OAuthTokens | null;
  /** The Claude CLI's refresh token, whether or not its access token is valid. */
  cliRefreshToken: () => string | undefined;
  refreshTokens: (refreshToken: string) => OAuthTokens;
  now: () => number;
};

const defaultRefreshDeps: RefreshDeps = {
  cliTokens: () => getClaudeTokens({ refreshExpired: false }),
  cliRefreshToken: () => readClaudeCredentials()?.claudeAiOauth?.refreshToken,
  refreshTokens,
  now: () => Date.now(),
};

/** Layered token refresh: valid Claude CLI token → stored refresh token → CLI refresh token. */
async function refreshOnce(
  credential: OAuthCredential,
  deps: RefreshDeps,
): Promise<OAuthCredential> {
  const cli = deps.cliTokens();
  if (cli && cli.expires > deps.now() + FRESH_MARGIN_MS) return toCredential(cli);

  // The CLI's refresh token is often stale once its access token has expired,
  // so try the stored one first. The CLI's lets the plugin recover after a
  // fresh `claude login`.
  const tried = new Set<string>();
  let failure: unknown = new Error("No Claude refresh token is available. Run `claude login`.");
  for (const candidate of [() => credential.refresh, deps.cliRefreshToken]) {
    const token = candidate();
    if (!token || tried.has(token)) continue;
    tried.add(token);
    try {
      return toCredential(deps.refreshTokens(token));
    } catch (err) {
      failure = err;
    }
  }
  throw failure;
}

/**
 * Build OpenCode 2's refresh callback.
 *
 * OpenCode 2 doesn't serialize refreshes, so concurrent requests, such as a
 * session's title and its first turn, each ask to refresh the same expired
 * credential. Anthropic's refresh tokens are single use, so all but the first
 * would fail. The callback hands later callers the result for a refresh token
 * it has already spent, for as long as that result stays fresh.
 */
export function createRefresh(deps: RefreshDeps = defaultRefreshDeps) {
  let last:
    | { refresh: string; result: Promise<OAuthCredential>; value?: OAuthCredential }
    | undefined;

  return (credential: OAuthCredential): Promise<OAuthCredential> => {
    if (
      last?.refresh === credential.refresh &&
      (!last.value || last.value.expires > deps.now() + FRESH_MARGIN_MS)
    ) {
      return last.result;
    }

    const entry: NonNullable<typeof last> = {
      refresh: credential.refresh,
      result: refreshOnce(credential, deps),
    };
    last = entry;
    entry.result.then(
      (value) => { entry.value = value; },
      () => { if (last === entry) last = undefined; },
    );
    return entry.result;
  };
}

/** Prefix the system prompt the way the OpenCode 1 entrypoint's system hook does. */
export function prependClaudePrefix(parsed: { system?: unknown }): void {
  if (!Array.isArray(parsed.system)) parsed.system = [];
  const system = parsed.system as Array<{ type?: string; text?: string }>;
  if (system.some((block) => block?.text?.includes(CLAUDE_PREFIX))) return;
  const first = system[0];
  if (first?.type === "text" && typeof first.text === "string") {
    system[0] = { ...first, text: `${CLAUDE_PREFIX}\n\n${first.text}` };
    return;
  }
  system.unshift({ type: "text", text: CLAUDE_PREFIX });
}

export type RequestState = {
  sessionId: string;
  deviceId: string;
  cachedClaudePromptCache: ClaudeSystemPromptCache | null;
};

/** Rewrite an OpenCode 2 request to Anthropic so it looks like Claude Code's. */
export async function shapeRequest(request: Request, state: RequestState): Promise<Request> {
  // Only OAuth requests need to pass as Claude Code. API key requests don't.
  const authorization = request.headers.get("authorization");
  if (!authorization?.startsWith("Bearer ")) return request;
  const accessToken = authorization.slice("Bearer ".length);

  const headers = buildClaudeHeaders(request.headers, accessToken, state.sessionId);
  // Session headers OpenCode 2 adds and Claude Code doesn't send. OpenCode adds
  // its tracing headers (b3, traceparent) after this hook runs, so they still
  // reach Anthropic.
  const opencodeHeaders: string[] = [];
  headers.forEach((_value, name) => {
    if (name.startsWith("x-opencode-")) opencodeHeaders.push(name);
  });
  for (const name of opencodeHeaders) headers.delete(name);
  headers.delete("x-parent-session-id");
  // The body is about to change length.
  headers.delete("content-length");

  let body = await request.text();
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    parsed = undefined;
  }
  if (parsed && typeof parsed === "object") {
    prependClaudePrefix(parsed as { system?: unknown });
    await shapeClaudeBody(parsed, request.url, {
      sessionId: state.sessionId,
      deviceId: state.deviceId,
      accessToken,
      cachedClaudePromptCache: state.cachedClaudePromptCache,
    });
    body = JSON.stringify(parsed);
  }

  return new Request(buildClaudeUrl(request.url), {
    method: request.method,
    headers,
    body,
    signal: request.signal,
  });
}

export async function setup(ctx: PluginContext): Promise<void> {
  const state: RequestState = {
    sessionId: SESSION_ID || randomUUID(),
    deviceId: getStableDeviceId(),
    cachedClaudePromptCache: readCachedClaudePromptCache(),
  };

  await ctx.integration.transform((editor) => {
    editor.method.update({
      integrationID: "anthropic",
      method: { id: METHOD_ID, type: "oauth", label: "Claude Pro / Max (OAuth)" },
      authorize,
      refresh: createRefresh(),
    });
  });

  await ctx.session.hook(
    "http.request",
    async (event) => {
      event.request = await shapeRequest(event.request, state);
    },
    { providerID: "anthropic" },
  );
}

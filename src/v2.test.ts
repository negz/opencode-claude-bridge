import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  ANTHROPIC_VERSION,
  BETA_FLAGS,
  CLI_VERSION,
  ENTRYPOINT,
  STAINLESS_HEADERS,
  USER_AGENT,
} from "./constants.js";
import { computeFingerprint } from "./claude-tools.js";
import { CLAUDE_PREFIX } from "./request.js";
import { createRefresh, prependClaudePrefix, shapeRequest } from "./v2.js";
import type { OAuthCredential, RefreshDeps } from "./v2.js";

const NOW = 1_800_000_000_000;
const MINUTE = 60_000;
const HOUR = 60 * MINUTE;

function credential(name: string, expires: number): OAuthCredential {
  return { type: "oauth", methodID: "oauth", access: `${name}-access`, refresh: `${name}-refresh`, expires };
}

function tokens(name: string, expires: number) {
  return { access: `${name}-access`, refresh: `${name}-refresh`, expires };
}

// A fake token endpoint. It records every refresh token it's asked to spend.
function fakeDeps(input: {
  cli?: ReturnType<typeof tokens> | null;
  cliRefreshToken?: string;
  responses: Record<string, ReturnType<typeof tokens> | Error>;
  now?: () => number;
}): RefreshDeps & { spent: string[] } {
  const spent: string[] = [];
  return {
    spent,
    cliTokens: () => input.cli ?? null,
    cliRefreshToken: () => input.cliRefreshToken,
    refreshTokens: (token) => {
      spent.push(token);
      const response = input.responses[token];
      if (!response) throw new Error(`unexpected refresh token ${token}`);
      if (response instanceof Error) throw response;
      return response;
    },
    now: input.now ?? (() => NOW),
  };
}

describe("refresh", () => {
  const cases: Array<{
    name: string;
    deps: Parameters<typeof fakeDeps>[0];
    want: OAuthCredential | Error;
    wantSpent: string[];
  }> = [
    {
      name: "uses the Claude CLI's token while it stays fresh",
      deps: { cli: tokens("cli", NOW + HOUR), responses: {} },
      want: credential("cli", NOW + HOUR),
      wantSpent: [],
    },
    {
      name: "refreshes the stored token when the CLI's is about to expire",
      deps: {
        cli: tokens("cli", NOW + 5 * MINUTE),
        cliRefreshToken: "cli-refresh",
        responses: { "stored-refresh": tokens("new", NOW + 8 * HOUR) },
      },
      want: credential("new", NOW + 8 * HOUR),
      wantSpent: ["stored-refresh"],
    },
    {
      name: "falls back to the CLI's refresh token when the stored one fails",
      deps: {
        cli: null,
        cliRefreshToken: "cli-refresh",
        responses: {
          "stored-refresh": new Error("invalid_grant"),
          "cli-refresh": tokens("new", NOW + 8 * HOUR),
        },
      },
      want: credential("new", NOW + 8 * HOUR),
      wantSpent: ["stored-refresh", "cli-refresh"],
    },
    {
      name: "doesn't spend a refresh token twice when the CLI and OpenCode share it",
      deps: {
        cli: null,
        cliRefreshToken: "stored-refresh",
        responses: { "stored-refresh": new Error("invalid_grant") },
      },
      want: new Error("invalid_grant"),
      wantSpent: ["stored-refresh"],
    },
  ];

  for (const tc of cases) {
    it(tc.name, async () => {
      const deps = fakeDeps(tc.deps);
      const refresh = createRefresh(deps);
      const got = refresh(credential("stored", NOW));
      if (tc.want instanceof Error) await assert.rejects(got, tc.want);
      else assert.deepEqual(await got, tc.want);
      assert.deepEqual(deps.spent, tc.wantSpent);
    });
  }

  it("spends a refresh token once for concurrent callers", async () => {
    const deps = fakeDeps({ responses: { "stored-refresh": tokens("new", NOW + 8 * HOUR) } });
    const refresh = createRefresh(deps);
    const got = await Promise.all([
      refresh(credential("stored", NOW)),
      refresh(credential("stored", NOW)),
    ]);
    assert.deepEqual(got, [credential("new", NOW + 8 * HOUR), credential("new", NOW + 8 * HOUR)]);
    assert.deepEqual(deps.spent, ["stored-refresh"]);
  });

  it("hands a caller holding a spent refresh token the result of that refresh", async () => {
    const deps = fakeDeps({ responses: { "stored-refresh": tokens("new", NOW + 8 * HOUR) } });
    const refresh = createRefresh(deps);
    await refresh(credential("stored", NOW));
    assert.deepEqual(await refresh(credential("stored", NOW)), credential("new", NOW + 8 * HOUR));
    assert.deepEqual(deps.spent, ["stored-refresh"]);
  });

  it("doesn't hand out an earlier result once it's about to expire", async () => {
    // A credential adopted from the Claude CLI keeps the CLI's refresh token,
    // so OpenCode later asks to refresh with the same token it passed before.
    let now = NOW;
    let cli = tokens("cli", NOW + HOUR);
    const deps = fakeDeps({ responses: {} });
    deps.cliTokens = () => cli;
    deps.now = () => now;
    const refresh = createRefresh(deps);
    const first = await refresh(credential("cli", NOW));
    assert.deepEqual(first, credential("cli", NOW + HOUR));
    now = NOW + HOUR - 5 * MINUTE;
    cli = tokens("cli", NOW + 9 * HOUR);
    assert.deepEqual(await refresh(first), credential("cli", NOW + 9 * HOUR));
    assert.deepEqual(deps.spent, []);
  });

  it("tries again after a failed refresh", async () => {
    const responses: Record<string, ReturnType<typeof tokens> | Error> = {
      "stored-refresh": new Error("network down"),
    };
    const deps = fakeDeps({ responses });
    const refresh = createRefresh(deps);
    await assert.rejects(refresh(credential("stored", NOW)), new Error("network down"));
    responses["stored-refresh"] = tokens("new", NOW + 8 * HOUR);
    assert.deepEqual(await refresh(credential("stored", NOW)), credential("new", NOW + 8 * HOUR));
    assert.deepEqual(deps.spent, ["stored-refresh", "stored-refresh"]);
  });
});

describe("prependClaudePrefix", () => {
  const cases: Array<{ name: string; system: unknown; want: unknown }> = [
    {
      name: "adds the prefix as a block when there's no system prompt",
      system: undefined,
      want: [{ type: "text", text: CLAUDE_PREFIX }],
    },
    {
      name: "prefixes the first text block",
      system: [
        { type: "text", text: "You are a title generator.", cache_control: { type: "ephemeral" } },
        { type: "text", text: "Second" },
      ],
      want: [
        { type: "text", text: `${CLAUDE_PREFIX}\n\nYou are a title generator.`, cache_control: { type: "ephemeral" } },
        { type: "text", text: "Second" },
      ],
    },
    {
      name: "leaves a system prompt that already has the prefix alone",
      system: [{ type: "text", text: `${CLAUDE_PREFIX}\n\nYou are a title generator.` }],
      want: [{ type: "text", text: `${CLAUDE_PREFIX}\n\nYou are a title generator.` }],
    },
  ];

  for (const tc of cases) {
    it(tc.name, () => {
      const parsed = { system: tc.system };
      prependClaudePrefix(parsed);
      assert.deepEqual(parsed.system, tc.want);
    });
  }
});

describe("shapeRequest", () => {
  // Trimmed from a request OpenCode 2.0.30 sent to Anthropic, as it reaches the
  // http.request hook: no Accept or tracing headers yet, and the bearer token
  // in place. It's a subagent's final step, so it carries the parent session
  // header and the tool_choice OpenCode 2 sends at an agent's step limit.
  const url = "https://api.anthropic.com/v1/messages?beta=true";
  const body = {
    model: "claude-opus-5-5",
    max_tokens: 128000,
    thinking: { type: "adaptive", block_binding: { prefix_mismatch_behavior: "drop_block" } },
    stream: true,
    system: [
      {
        type: "text",
        text: "You are an AI agent running in OpenCode, a coding agent harness.",
        cache_control: { type: "ephemeral" },
      },
      {
        type: "text",
        text: "# Your Model\n- Name: Claude Opus 5.5\n- Provider ID: anthropic",
        cache_control: null,
      },
    ],
    messages: [
      { role: "user", content: [{ type: "text", text: "\"Say hello\"", cache_control: { type: "ephemeral" } }] },
    ],
    tools: [
      {
        name: "read",
        description: "Read the contents of a file or directory.",
        input_schema: {
          type: "object",
          properties: { path: { type: "string", description: "File or directory to read" } },
          required: ["path"],
          additionalProperties: false,
        },
      },
    ],
    tool_choice: { type: "none" },
  };
  const headers = {
    "Content-Type": "application/json",
    "User-Agent": "opencode/latest/2.0.30/cli",
    "anthropic-beta": "interleaved-thinking-2025-05-14,thinking-binding-controls-2026-08-01",
    "anthropic-version": "2023-06-01",
    "x-opencode-client": "cli",
    "x-opencode-project": "8ef1749d7ba7f80f67330d104607f598c1e8fe7b",
    "x-opencode-session": "ses_ebceb3da8ffeHnLLqFxDuAf9ZY",
    "x-opencode-session-id": "ses_ebceb3da8ffeHnLLqFxDuAf9ZY",
    "x-opencode-parent-session-id": "ses_ebcf00000000ParentSession0",
    "x-parent-session-id": "ses_ebcf00000000ParentSession0",
    "x-session-affinity": "ses_ebceb3da8ffeHnLLqFxDuAf9ZY",
    "x-session-id": "ses_ebceb3da8ffeHnLLqFxDuAf9ZY",
  };
  // The cch (db805) comes from a request the OpenCode 1 entrypoint sent for
  // this session ID. The fingerprint folds in the Claude CLI version, which
  // varies by machine, so it's computed. For 2.1.280 that request had b4a.
  const sessionId = "00000000-0000-4000-8000-000000000000";
  const state = { sessionId, deviceId: "device-id", cachedClaudePromptCache: null };

  it("passes requests that don't use OAuth through untouched", async () => {
    const request = new Request(url, {
      method: "POST",
      headers: { ...headers, "x-api-key": "sk-ant-api-key" },
      body: JSON.stringify(body),
    });
    assert.equal(await shapeRequest(request, state), request);
  });

  it("makes an OAuth request look like Claude Code's", async (t) => {
    t.mock.method(globalThis, "fetch", async () =>
      new Response(JSON.stringify({ account: { uuid: "account-uuid" } })),
    );
    const request = new Request(url, {
      method: "POST",
      headers: { ...headers, authorization: "Bearer shape-test-token" },
      body: JSON.stringify(body),
    });

    const got = await shapeRequest(request, state);

    assert.equal(got.url, url);
    assert.equal(got.method, "POST");
    const gotHeaders: Record<string, string> = {};
    got.headers.forEach((value, name) => { gotHeaders[name] = value; });
    assert.deepEqual(gotHeaders, {
      accept: "application/json",
      "anthropic-beta": [
        "claude-code-20250219",
        "oauth-2025-04-20",
        // The configured flags already include claude-code-20250219.
        ...BETA_FLAGS.split(",").filter((flag) => flag !== "claude-code-20250219"),
        // OpenCode's own flag, which thinking.block_binding needs.
        "thinking-binding-controls-2026-08-01",
      ].join(","),
      "anthropic-dangerous-direct-browser-access": "true",
      "anthropic-version": ANTHROPIC_VERSION,
      authorization: "Bearer shape-test-token",
      "content-type": "application/json",
      "user-agent": USER_AGENT,
      "x-app": "cli",
      "x-claude-code-session-id": sessionId,
      ...STAINLESS_HEADERS,
      "x-stainless-retry-count": "0",
      "x-stainless-timeout": "600",
    });
    assert.deepEqual(JSON.parse(await got.text()), {
      model: "claude-opus-5-5",
      max_tokens: 128000,
      thinking: { type: "adaptive", block_binding: { prefix_mismatch_behavior: "drop_block" } },
      stream: true,
      system: [
        {
          type: "text",
          text: `x-anthropic-billing-header: cc_version=${CLI_VERSION}.${computeFingerprint("\"Say hello\"")}; cc_entrypoint=${ENTRYPOINT}; cch=db805;`,
        },
        { type: "text", text: CLAUDE_PREFIX },
        { type: "text", text: "You are an AI agent running in Claude Code, a coding agent harness." },
        { type: "text", text: "# Your Model\n- Name: Claude Opus 5.5\n- Provider ID: anthropic" },
      ],
      messages: [{ role: "user", content: [{ type: "text", text: "Say hello" }] }],
      tools: body.tools,
      tool_choice: { type: "none" },
      metadata: {
        user_id: JSON.stringify({ device_id: "device-id", account_uuid: "account-uuid", session_id: sessionId }),
      },
    });
  });
});

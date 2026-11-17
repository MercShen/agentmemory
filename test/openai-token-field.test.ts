import { createServer, type RequestListener, type Server } from "node:http";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OpenAIProvider, tokenLimitField } from "../src/providers/openai.js";

const servers: Server[] = [];

async function serve(handler: RequestListener): Promise<string> {
  const server = createServer(handler);
  servers.push(server);
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Missing server port");
  return `http://127.0.0.1:${address.port}`;
}

/** Records the parsed JSON body of every request and answers 200. */
async function captureBodies(): Promise<{ url: string; bodies: Array<Record<string, unknown>> }> {
  const bodies: Array<Record<string, unknown>> = [];
  const url = await serve((req, res) => {
    let raw = "";
    req.on("data", chunk => {
      raw += String(chunk);
    });
    req.on("end", () => {
      bodies.push(JSON.parse(raw) as Record<string, unknown>);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ choices: [{ message: { content: "ok" } }] }));
    });
  });
  return { url, bodies };
}

afterEach(async () => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  await Promise.all(servers.splice(0).map(server => new Promise<void>(resolve => {
    server.closeAllConnections();
    server.close(() => resolve());
  })));
});

describe("tokenLimitField", () => {
  it.each([
    ["gpt-5.6-luna", "max_completion_tokens"],
    ["gpt-6-luna", "max_completion_tokens"],
    ["gpt-5.4-mini", "max_completion_tokens"],
    ["o3-mini", "max_completion_tokens"],
    ["chatgpt-4o-latest", "max_completion_tokens"],
    ["gpt-4.1-mini", "max_tokens"],
    ["gpt-oss-20b", "max_tokens"],
    ["deepseek-chat", "max_tokens"],
    ["qwen3.5-plus", "max_tokens"],
  ] as const)("selects the accepted field for %s", (model, expected) => {
    expect(tokenLimitField(model)).toBe(expected);
  });

  it("honours an explicit override over the model heuristic", () => {
    expect(tokenLimitField("gpt-6-luna", "max_tokens")).toBe("max_tokens");
    expect(tokenLimitField("deepseek-chat", "max_completion_tokens")).toBe("max_completion_tokens");
    expect(tokenLimitField("deepseek-chat", "nonsense")).toBe("max_tokens");
  });
});

describe("OpenAIProvider output-token field", () => {
  it("sends max_completion_tokens for a gpt-5+ model and omits max_tokens", async () => {
    const { url, bodies } = await captureBodies();
    const provider = new OpenAIProvider("fixture", "gpt-6-luna", 128, url);
    await expect(provider.compress("system", "user")).resolves.toBe("ok");
    expect(bodies[0]?.max_completion_tokens).toBe(128);
    expect(bodies[0]).not.toHaveProperty("max_tokens");
  });

  it("sends max_tokens for a compatible endpoint model", async () => {
    const { url, bodies } = await captureBodies();
    const provider = new OpenAIProvider("fixture", "deepseek-chat", 128, url);
    await expect(provider.compress("system", "user")).resolves.toBe("ok");
    expect(bodies[0]?.max_tokens).toBe(128);
    expect(bodies[0]).not.toHaveProperty("max_completion_tokens");
  });

  it("lets OPENAI_TOKEN_LIMIT_FIELD override the heuristic", async () => {
    vi.stubEnv("OPENAI_TOKEN_LIMIT_FIELD", "max_tokens");
    const { url, bodies } = await captureBodies();
    const provider = new OpenAIProvider("fixture", "gpt-6-luna", 64, url);
    await expect(provider.compress("system", "user")).resolves.toBe("ok");
    expect(bodies[0]?.max_tokens).toBe(64);
    expect(bodies[0]).not.toHaveProperty("max_completion_tokens");
  });

  it("names the rejected field when the endpoint refuses it", async () => {
    const url = await serve((_req, res) => {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({
        error: {
          message: "Unsupported parameter: 'max_completion_tokens' is not supported with this model.",
          type: "invalid_request_error",
        },
      }));
    });
    const provider = new OpenAIProvider("fixture", "gpt-6-luna", 64, url);
    await expect(provider.compress("system", "user")).rejects.toThrow(
      /set OPENAI_TOKEN_LIMIT_FIELD to "max_tokens"/,
    );
  });
});

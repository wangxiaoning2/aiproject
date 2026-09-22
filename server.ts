import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

/* ------------------------------------------------------------------ *
 * 配置：全部走环境变量，代码里不出现任何密钥
 * ------------------------------------------------------------------ */
const PUBLIC_DIR = fileURLToPath(new URL("./public", import.meta.url));
const PORT = Number(process.env.PORT || 5173);

const BASE_URL = (process.env.LLM_BASE_URL || "https://api.deepseek.com/v1").replace(/\/+$/, "");
const API_KEY = process.env.LLM_API_KEY || "";
const MODEL = process.env.LLM_MODEL || "deepseek-chat";

const MAX_CONTEXT_TOKENS = Number(process.env.MAX_CONTEXT_TOKENS || 8000);
const PRICE_IN = Number(process.env.PRICE_IN || 2);   // 元 / 百万 token（输入）
const PRICE_OUT = Number(process.env.PRICE_OUT || 8);  // 元 / 百万 token（输出）

// 没有 API Key 就自动进 mock 模式 —— 保证你 clone 下来就能跑
const MOCK = process.env.LLM_MOCK === "1" || !API_KEY;

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/* ------------------------------------------------------------------ *
 * 静态文件服务
 * ------------------------------------------------------------------ */
const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

async function serveStatic(pathname: string, res: ServerResponse): Promise<void> {
  const rel = pathname === "/" ? "/index.html" : pathname;
  // 关键：挡住 ../../etc/passwd 这类路径穿越
  const safe = normalize(rel).replace(/^(\.\.[/\\])+/, "");
  const abs = join(PUBLIC_DIR, safe);

  if (!abs.startsWith(PUBLIC_DIR)) {
    res.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("forbidden");
    return;
  }

  try {
    const buf = await readFile(abs);
    res.writeHead(200, {
      "Content-Type": MIME[extname(abs)] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(buf);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("404 not found");
  }
}

/* ------------------------------------------------------------------ *
 * token 估算与上下文裁剪 —— 面试高频题，先在这里亲手实现一遍
 * 经验值：一个中文字 ≈ 1 token，四个英文字符 ≈ 1 token
 * ------------------------------------------------------------------ */
function estimateTokens(text: string): number {
  let cjk = 0;
  for (const ch of text) {
    if (/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]/.test(ch)) cjk++;
  }
  const rest = text.length - cjk;
  return Math.ceil(cjk + rest / 4);
}

function countTokens(messages: ChatMessage[]): number {
  // 每条消息有约 4 token 的固定开销（role、分隔符等）
  return messages.reduce((sum, m) => sum + estimateTokens(m.content) + 4, 0);
}

function trimContext(messages: ChatMessage[], budget: number): { kept: ChatMessage[]; trimmed: number } {
  const system = messages.filter((m) => m.role === "system");
  const rest = messages.filter((m) => m.role !== "system");
  const used = countTokens(system);
  const kept: ChatMessage[] = [];
  let total = used;

  // 从最新的消息往前保留，超出预算的老消息直接丢掉（最朴素的一种策略）
  for (let i = rest.length - 1; i >= 0; i--) {
    const cost = countTokens([rest[i]]);
    if (total + cost > budget) break;
    kept.unshift(rest[i]);
    total += cost;
  }

  return { kept: [...system, ...kept], trimmed: rest.length - kept.length };
}

/* ------------------------------------------------------------------ *
 * SSE 写帧：一个事件 = "event: xxx\ndata: {...}\n\n"
 * ------------------------------------------------------------------ */
function sse(res: ServerResponse, event: string, data: unknown): void {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const parts: Buffer[] = [];
    req.on("data", (c: Buffer) => parts.push(c));
    req.on("end", () => resolve(Buffer.concat(parts).toString("utf8")));
    req.on("error", reject);
  });
}

/* ------------------------------------------------------------------ *
 * 真正调用模型：手写解析上游的 SSE
 * 这一段的每一行都是"框架帮你藏起来的东西"
 * ------------------------------------------------------------------ */
async function streamUpstream(
  res: ServerResponse,
  signal: AbortSignal,
  kept: ChatMessage[],
  temperature: number,
  started: number,
  trimmed: number,
): Promise<void> {
  const payload: Record<string, unknown> = {
    model: MODEL,
    messages: kept,
    temperature,
    stream: true,
  };
  // 多数 OpenAI 兼容端点支持用 include_usage 拿最终用量；个别不支持，所以做成开关
  if (process.env.LLM_INCLUDE_USAGE === "1") {
    payload.stream_options = { include_usage: true };
  }

  const upstream = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!upstream.ok) {
    const detail = await upstream.text().catch(() => "");
    throw new Error(`上游返回 ${upstream.status}：${detail.slice(0, 200)}`);
  }
  if (!upstream.body) throw new Error("上游没有返回流式 body");

  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let firstTokenAt = 0;
  let usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } | null = null;

  for await (const chunk of upstream.body as unknown as AsyncIterable<Uint8Array>) {
    buffer += decoder.decode(chunk, { stream: true });

    // SSE 的一条消息用空行结尾，所以要按空行切，而不是按 chunk 切
    let cut = -1;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, cut).replace(/\r/g, "");
      buffer = buffer.slice(cut + 2);

      for (const line of block.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const raw = line.slice(5).trim();
        if (raw === "[DONE]") continue;

        let json: any;
        try {
          json = JSON.parse(raw);
        } catch {
          continue; // 半截的 JSON 直接跳过，流式解析里很常见
        }

        const delta: string | undefined = json.choices?.[0]?.delta?.content;
        if (delta) {
          if (!firstTokenAt) firstTokenAt = Date.now();
          text += delta;
          sse(res, "token", { t: delta });
        }
        if (json.usage) usage = json.usage;
      }
    }
  }

  const totalMs = Date.now() - started;
  const promptTokens = usage?.prompt_tokens ?? countTokens(kept);
  const completionTokens = usage?.completion_tokens ?? estimateTokens(text);

  sse(res, "done", {
    finish: "stop",
    model: MODEL,
    usedRealUsage: Boolean(usage),
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
    cost: (promptTokens / 1e6) * PRICE_IN + (completionTokens / 1e6) * PRICE_OUT,
    ttftMs: firstTokenAt ? firstTokenAt - started : totalMs,
    totalMs,
    context: { sent: kept.length, trimmed },
  });
}

/* ------------------------------------------------------------------ *
 * Mock 模式：不花钱，照样把流式、中断、报错全练一遍
 * ------------------------------------------------------------------ */
const MOCK_REPLY = `现在是 mock 模式，你看到的是本地逐字"假装"出来的流。

但它走的链路和真实模型完全一样：HTTP 长连接 → SSE 数据帧 → 前端流式渲染。

建议你按顺序做三件事：

第一，打开右边的"原始 SSE 报文"，看清楚每一个 data: 帧长什么样——你会发现所谓"AI 打字"，本质就是一堆小 JSON 拼在一起。

第二，点"停止生成"。注意看服务端日志：客户端一断开，上游请求也会被一起 abort，不然你就得为一个没人看的回答继续付钱。

第三，输入"测试报错"，感受一下中途失败是什么样，再体会重试该怎么做。

跑通这三步，你就不再会觉得大模型是个黑盒了——它只是一个慢一点、按流返回、可能中途挂掉的接口。`;

async function streamMock(
  res: ServerResponse,
  signal: AbortSignal,
  started: number,
  kept: ChatMessage[],
  trimmed: number,
): Promise<void> {
  const lastUser = [...kept].reverse().find((m) => m.role === "user")?.content ?? "";
  const shouldFail = lastUser.includes("测试报错");

  await sleep(260, signal); // 模拟首 token 之前的等待，让 TTFT 有意义

  let text = "";
  let firstTokenAt = 0;

  for (let i = 0; i < MOCK_REPLY.length; i += 2) {
    if (signal.aborted) return; // 客户端断了就停，别做无用功
    if (shouldFail && i > 30) {
      throw new Error("模拟的上游故障：第 30 个字符之后连接被重置");
    }
    const piece = MOCK_REPLY.slice(i, i + 2);
    if (!firstTokenAt) firstTokenAt = Date.now();
    text += piece;
    sse(res, "token", { t: piece });
    await sleep(26, signal);
  }

  const totalMs = Date.now() - started;
  const promptTokens = countTokens(kept);
  const completionTokens = estimateTokens(text);

  sse(res, "done", {
    finish: "stop",
    model: "mock",
    usedRealUsage: false,
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
    cost: (promptTokens / 1e6) * PRICE_IN + (completionTokens / 1e6) * PRICE_OUT,
    ttftMs: firstTokenAt ? firstTokenAt - started : totalMs,
    totalMs,
    context: { sent: kept.length, trimmed },
  });
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(t);
      resolve();
    }, { once: true });
  });
}

/* ------------------------------------------------------------------ *
 * POST /api/chat —— 和模型对话的唯一入口
 * ------------------------------------------------------------------ */
async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let payload: any;
  try {
    payload = JSON.parse(await readBody(req));
  } catch {
    res.writeHead(400, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ error: "请求体不是合法 JSON" }));
    return;
  }

  const systemPrompt = String(payload.systemPrompt ?? "").trim();
  const temperature = Number.isFinite(payload.temperature)
    ? Math.min(2, Math.max(0, Number(payload.temperature)))
    : 0.7;

  const incoming: ChatMessage[] = (Array.isArray(payload.messages) ? payload.messages : [])
    .filter((m: any) => m && typeof m.content === "string" && ["user", "assistant", "system"].includes(m.role))
    .map((m: any) => ({ role: m.role, content: m.content }));

  const withSystem: ChatMessage[] = systemPrompt
    ? [{ role: "system", content: systemPrompt }, ...incoming]
    : incoming;

  const { kept, trimmed } = trimContext(withSystem, MAX_CONTEXT_TOKENS);

  // 流式响应必须带这几个头，nginx 那类反代还要额外关掉缓冲
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });
  res.flushHeaders();

  const started = Date.now();
  const controller = new AbortController();

  // 用户点"停止"或关掉页面 → 立刻掐掉上游请求
  res.on("close", () => {
    if (!controller.signal.aborted) {
      controller.abort();
      console.log(`[abort] 客户端断开，已取消上游请求`);
    }
  });

  console.log(
    `[chat] model=${MOCK ? "mock" : MODEL} temp=${temperature} ` +
    `messages=${kept.length}${trimmed ? ` (裁剪掉 ${trimmed} 条)` : ""} ` +
    `≈${countTokens(kept)} tokens`,
  );

  try {
    if (MOCK) {
      await streamMock(res, controller.signal, started, kept, trimmed);
    } else {
      await streamUpstream(res, controller.signal, kept, temperature, started, trimmed);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (!controller.signal.aborted && !res.writableEnded) {
      sse(res, "error", { message });
      console.error(`[error] ${message}`);
    }
  } finally {
    if (!res.writableEnded) res.end();
  }
}

/* ------------------------------------------------------------------ *
 * 启动
 * ------------------------------------------------------------------ */
const server = createServer((req, res) => {
  const url = new URL(req.url || "/", "http://localhost");

  if (url.pathname === "/api/chat" && req.method === "POST") {
    void handleChat(req, res);
    return;
  }

  if (url.pathname === "/api/config" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({
      mock: MOCK,
      model: MOCK ? "mock" : MODEL,
      baseUrl: MOCK ? "" : BASE_URL,
      maxContextTokens: MAX_CONTEXT_TOKENS,
      priceIn: PRICE_IN,
      priceOut: PRICE_OUT,
    }));
    return;
  }

  void serveStatic(url.pathname, res);
});

server.listen(PORT, () => {
  console.log("");
  console.log(`  流式对话 demo 已启动 →  http://localhost:${PORT}`);
  console.log(`  模式：${MOCK ? "mock（未检测到 LLM_API_KEY，本地假装流式）" : `${MODEL} @ ${BASE_URL}`}`);
  console.log(`  上下文预算：${MAX_CONTEXT_TOKENS} tokens`);
  console.log("");
  console.log("  试试：在输入框里发「测试报错」，看中途失败长什么样。");
  console.log("");
});

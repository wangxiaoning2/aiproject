import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { DEFAULT_INDEX_FILE, buildIndex } from "./rag/ingest.ts";
import { queryEmbedderFor, type Embedder } from "./rag/embed.ts";
import { indexStats, loadIndex, searchIndex } from "./rag/store.ts";
import type { Hit, IndexFile } from "./rag/types.ts";

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

// 检索参数：召回几条。这个数字是 RAG 里最常被调的一个旋钮
const RAG_TOP_K = Number(process.env.RAG_TOP_K || 4);
const RAG_INDEX_FILE = process.env.RAG_INDEX || DEFAULT_INDEX_FILE;

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
 * RAG：在线检索
 *
 * 离线部分（读文件 → 切块 → 算向量）在 rag/ingest.ts，只跑一次。
 * 这里只负责提问时的那一半：算问题向量 → 取最像的几段 → 拼进提示词。
 * ------------------------------------------------------------------ */
let ragIndex: IndexFile | null = null;
let ragEmbedder: Embedder | null = null;

async function initRag(): Promise<void> {
  let index = await loadIndex(RAG_INDEX_FILE);

  // 没索引就顺手建一份。只在"本地 mock 嵌入"时自动做，
  // 否则一启动就悄悄花掉你真实嵌入接口的钱。
  if (!index && !process.env.EMBED_MODEL) {
    console.log("[rag] 没找到索引，用本地伪嵌入自动构建一份…");
    try {
      await buildIndex({ quiet: true });
      index = await loadIndex(RAG_INDEX_FILE);
    } catch (err) {
      console.warn(`[rag] 自动建索引失败：${err instanceof Error ? err.message : String(err)}`);
    }
  }

  if (!index) {
    console.log("[rag] 知识库未就绪。放几篇笔记进 data/，再跑 node rag/ingest.ts");
    return;
  }

  ragIndex = index;
  ragEmbedder = queryEmbedderFor(index);
  console.log(
    `[rag] 知识库就绪：${index.chunks.length} 个块 · ${index.provider} · ${index.model} · ${index.dim} 维`,
  );
}

async function retrieve(query: string, topK: number): Promise<{ hits: Hit[]; tookMs: number }> {
  if (!ragIndex || !ragEmbedder) throw new Error("知识库未就绪：先跑 node rag/ingest.ts");

  const started = Date.now();
  const [vector] = await ragEmbedder.embed([query]);
  const hits = searchIndex(ragIndex, vector, topK);
  return { hits, tookMs: Date.now() - started };
}

/**
 * 把召回的几段拼成"小抄"，塞进 system prompt。
 * 面试里这段的每一句都可以被追问：
 *   为什么要写"资料里没有就说不知道"？→ 不加这句，模型一定会用自己脑子里的知识补
 *   为什么要标来源？→ 让用户能自己核对，这是 RAG 相对纯聊天最大的产品价值
 */
function buildRagSystemPrompt(hits: Hit[]): string {
  const blocks = hits
    .map((h, i) => `【资料 ${i + 1}】来源：${h.chunk.source}（小节：${h.chunk.heading}）\n${h.chunk.text}`)
    .join("\n\n");

  return [
    "下面是从我的知识库里检索到的资料，请只依据这些资料回答问题。",
    "要求：",
    "1. 只用资料里出现过的信息，不要用自己的知识补充；",
    "2. 资料里没有的内容，直接说「笔记里没有相关记录」，不要猜；",
    "3. 用【资料 N】标注依据，方便我核对。",
    "",
    blocks,
  ].join("\n");
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

/**
 * mock 模式下"基于资料回答"长什么样。
 * 注意：这几段话不是我编的，是刚刚真的从 data/ 里检索出来的原文——
 * 模型在真实链路里干的事，就是把它改写成通顺的人话。
 */
function mockRagReply(hits: Hit[], question: string): string {
  const used = hits.slice(0, Math.min(3, hits.length));
  const lines = used.map((h) => {
    const firstSentence = h.chunk.text.replace(/\n/g, " ").split(/(?<=[。！？])/)[0] || h.chunk.text;
    return `依据【资料 ${h.rank}】（${h.chunk.source} · ${h.chunk.heading}）：${firstSentence}`;
  });

  return `（mock 模式，这段流是本地伪造的，没有花你一分钱）

关于「${question}」，检索到了 ${hits.length} 段笔记，答案大致是这样：

${lines.join("\n\n")}

上面每一条都能追溯到原始笔记，这就是 RAG 相对"直接问模型"最大的差别：答案有出处，编造的空间被压掉了。`;
}

async function streamMock(
  res: ServerResponse,
  signal: AbortSignal,
  started: number,
  kept: ChatMessage[],
  trimmed: number,
  ragHits: Hit[] | null = null,
): Promise<void> {
  const lastUser = [...kept].reverse().find((m) => m.role === "user")?.content ?? "";
  const shouldFail = lastUser.includes("测试报错");
  const reply = ragHits && ragHits.length > 0 ? mockRagReply(ragHits, lastUser) : MOCK_REPLY;

  await sleep(260, signal); // 模拟首 token 之前的等待，让 TTFT 有意义

  let text = "";
  let firstTokenAt = 0;

  for (let i = 0; i < reply.length; i += 2) {
    if (signal.aborted) return; // 客户端断了就停，别做无用功
    if (shouldFail && i > 30) {
      throw new Error("模拟的上游故障：第 30 个字符之后连接被重置");
    }
    const piece = reply.slice(i, i + 2);
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

  // 知识库开关：关掉就是普通聊天，打开就是 RAG
  const useRag = payload.useRag === true;
  const topK = Number.isFinite(payload.topK)
    ? Math.min(10, Math.max(1, Number(payload.topK)))
    : RAG_TOP_K;

  const incoming: ChatMessage[] = (Array.isArray(payload.messages) ? payload.messages : [])
    .filter((m: any) => m && typeof m.content === "string" && ["user", "assistant", "system"].includes(m.role))
    .map((m: any) => ({ role: m.role, content: m.content }));

  // 检索必须在写响应头之前完成——查不到资料，就没法给出一个正常的流
  let ragHits: Hit[] | null = null;
  let ragTookMs = 0;

  if (useRag) {
    const question = [...incoming].reverse().find((m) => m.role === "user")?.content ?? "";
    try {
      const result = await retrieve(question, topK);
      ragHits = result.hits;
      ragTookMs = result.tookMs;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      res.writeHead(500, { "Content-Type": "application/json; charset=utf-8" });
      res.end(JSON.stringify({ error: message }));
      return;
    }
  }

  const systemParts = [systemPrompt];
  if (ragHits && ragHits.length > 0) systemParts.push(buildRagSystemPrompt(ragHits));
  const mergedSystem = systemParts.filter(Boolean).join("\n\n");

  const withSystem: ChatMessage[] = mergedSystem
    ? [{ role: "system", content: mergedSystem }, ...incoming]
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

  // 先把"这次检索到了什么"发给前端。
  // 这一步是 RAG 调试的关键：回答不对时，九成问题在召回，不在这里。
  if (useRag && ragHits) {
    sse(res, "rag", {
      topK,
      tookMs: ragTookMs,
      provider: ragIndex?.provider ?? "",
      model: ragIndex?.model ?? "",
      hits: publicHits(ragHits),
    });
  }

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
    `≈${countTokens(kept)} tokens` +
    (useRag ? ` · RAG 召回 ${ragHits?.length ?? 0} 段 / ${ragTookMs}ms` : ""),
  );

  try {
    if (MOCK) {
      await streamMock(res, controller.signal, started, kept, trimmed, ragHits);
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

/** 给前端的检索结果：只保留需要展示的字段，别把向量也发过去 */
function publicHits(hits: Hit[]) {
  return hits.map((h) => ({
    rank: h.rank,
    score: Number(h.score.toFixed(4)),
    source: h.chunk.source,
    heading: h.chunk.heading,
    chars: h.chunk.text.length,
    text: h.chunk.text,
  }));
}

/* ------------------------------------------------------------------ *
 * POST /api/rag/search —— 只检索、不调模型
 * 调 RAG 时用得最多的一个接口：答案不对，先看这里召回了什么
 * ------------------------------------------------------------------ */
async function handleRagSearch(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let payload: any;
  try {
    payload = JSON.parse(await readBody(req));
  } catch {
    res.writeHead(400, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ error: "请求体不是合法 JSON" }));
    return;
  }

  const query = String(payload.query ?? "").trim();
  if (!query) {
    res.writeHead(400, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ error: "query 不能为空" }));
    return;
  }

  const topK = Number.isFinite(payload.topK)
    ? Math.min(10, Math.max(1, Number(payload.topK)))
    : RAG_TOP_K;

  try {
    const { hits, tookMs } = await retrieve(query, topK);
    res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({
      query,
      topK,
      tookMs,
      provider: ragIndex?.provider ?? "",
      model: ragIndex?.model ?? "",
      dim: ragIndex?.dim ?? 0,
      hits: publicHits(hits),
    }));
  } catch (err) {
    res.writeHead(500, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
  }
}

/** POST /api/rag/rebuild —— 改了 data/ 或切块参数之后重建索引 */
async function handleRagRebuild(_req: IncomingMessage, res: ServerResponse): Promise<void> {
  try {
    const { index, files } = await buildIndex({ quiet: true });
    ragIndex = index;
    ragEmbedder = queryEmbedderFor(index);
    res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ ok: true, files, ...indexStats(index) }));
  } catch (err) {
    res.writeHead(500, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
  }
}

/* ------------------------------------------------------------------ *
 * 启动
 * ------------------------------------------------------------------ */
await initRag();

const server = createServer((req, res) => {
  const url = new URL(req.url || "/", "http://localhost");

  if (url.pathname === "/api/chat" && req.method === "POST") {
    void handleChat(req, res);
    return;
  }

  if (url.pathname === "/api/rag/search" && req.method === "POST") {
    void handleRagSearch(req, res);
    return;
  }

  if (url.pathname === "/api/rag/rebuild" && req.method === "POST") {
    void handleRagRebuild(req, res);
    return;
  }

  if (url.pathname === "/api/rag/stats" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify(
      ragIndex
        ? indexStats(ragIndex)
        : { ready: false, hint: "没有索引。放几篇笔记到 data/，然后跑 node rag/ingest.ts" },
    ));
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
      rag: ragIndex ? indexStats(ragIndex) : { ready: false },
      ragTopK: RAG_TOP_K,
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
  console.log(
    `  知识库：${ragIndex
      ? `${ragIndex.chunks.length} 个块 · ${ragIndex.provider} · 召回 ${RAG_TOP_K} 条`
      : "未就绪（放几篇笔记到 data/，跑 node rag/ingest.ts）"}`,
  );
  console.log("");
  console.log("  试试：在输入框里发「测试报错」，看中途失败长什么样。");
  console.log("  打开右侧「知识库」开关，同一个问题会变成「基于你的笔记」回答。");
  console.log("");
});

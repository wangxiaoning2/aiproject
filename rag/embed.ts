/* ------------------------------------------------------------------ *
 * 嵌入（Embedding）
 *
 * 一句话解释：把一段文字压成一串固定长度的数字。
 * 语义相近的文字，在这串数字的空间里"方向"也接近——检索就是靠这个。
 *
 * 这里提供两个实现：
 *   mock  —— 本地 IDF 加权的词面向量，零成本、零依赖，用来把整条链路跑通并看清结构
 *   api   —— 真实嵌入服务（任何 OpenAI 兼容的 /embeddings 接口）
 *
 * ⚠ mock 向量只有"字面相似"的能力，没有真正的语义能力。
 *   它的用途是让你在没配 Key 时也能把 RAG 全链路跑起来、看清每一步；
 *   真实效果必须换成 api。这个差别本身就是第 3 周要亲手验证的练习。
 * ------------------------------------------------------------------ */

import type { IndexFile } from "./types.ts";

export interface Embedder {
  provider: string;
  model: string;
  /** 向量维度。换模型会变，索引必须重建 */
  dim: number;
  embed(texts: string[]): Promise<number[][]>;
}

const MOCK_DIM = 2048;

/**
 * 单个汉字的权重远低于双字词。
 * 因为"的、了、一、是"这类单字到处都是，它们只制造噪音；
 * 真正的信号在"切块""召回""延迟"这种双字词里。
 */
const SINGLE_CHAR_WEIGHT = 0.35;

/* ------------------------------------------------------------------ *
 * L2 归一化：把向量长度变成 1
 * 归一化之后，"余弦相似度"就退化成"点积"，计算更快也更好比较
 * ------------------------------------------------------------------ */
export function l2normalize(v: number[]): number[] {
  let sum = 0;
  for (const x of v) sum += x * x;
  const norm = Math.sqrt(sum);
  if (norm === 0) return v;
  return v.map((x) => x / norm);
}

/* ------------------------------------------------------------------ *
 * 本地伪嵌入：字符 n-gram 哈希到固定维度
 * 只是为了让"向量检索"这件事在没有 Key 的时候也能真实发生
 * ------------------------------------------------------------------ */
function fnv1a(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * 把文本拆成参与打分的单元：英文按词，中文按单字 + 双字（近似 bigram）。
 * 每种单元带前缀标记，避免"单字"和"双字"哈希到同一块空间里互相干扰。
 */
export function termsOf(text: string): string[] {
  const lower = text.toLowerCase();
  const out: string[] = [];

  for (const m of lower.matchAll(/[a-z0-9_]{2,}/g)) out.push(`w:${m[0]}`);

  const cjkRuns = lower.replace(/[^\u4e00-\u9fff]+/g, " ").split(" ").filter(Boolean);
  for (const run of cjkRuns) {
    for (let i = 0; i < run.length; i++) {
      out.push(`c:${run[i]}`);
      if (i + 1 < run.length) out.push(`b:${run.slice(i, i + 2)}`);
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * IDF：让"稀有词"比"的、了、是"更重要
 *
 * 不加这个，检索结果会被"怎么""一个""不会"这类到处都有的词带偏——
 * 这是词面检索的头号噪音来源，也是 BM25 里那一项 IDF 的由来。
 * 真实嵌入模型不需要这一步，因为它是在大规模语料上训出来的；
 * 我们自己造向量，就得自己补上。
 * ------------------------------------------------------------------ */
export type IdfTable = Map<string, number>;

export function buildIdf(texts: string[]): IdfTable {
  const df = new Map<string, number>();
  for (const text of texts) {
    for (const term of new Set(termsOf(text))) df.set(term, (df.get(term) ?? 0) + 1);
  }

  const n = texts.length;
  const idf: IdfTable = new Map();
  for (const [term, count] of df) {
    // 平滑版 IDF：只在极少数块里出现的词权重最高
    idf.set(term, Math.log(1 + (n - count + 0.5) / (count + 0.5)));
  }
  return idf;
}

export function maxIdfOf(idf: IdfTable): number {
  let max = 1;
  for (const v of idf.values()) if (v > max) max = v;
  return max;
}

export function mockEmbed(text: string, idf?: IdfTable, maxIdf = 1): number[] {
  const v = new Array<number>(MOCK_DIM).fill(0);

  for (const term of termsOf(text)) {
    const h = fnv1a(term);
    const idx = h % MOCK_DIM;
    // 用哈希高位当符号位：不同 term 落到同一维时不至于永远互相加分
    const sign = (h >>> 31) % 2 === 0 ? 1 : -1;
    // 越稀有（IDF 越高）的词越重要；没有 IDF 表时退化成纯词频
    const rarity = idf ? (idf.get(term) ?? maxIdf) : 1 + Math.log(1 + term.length);
    const weight = rarity * (term.startsWith("c:") ? SINGLE_CHAR_WEIGHT : 1);
    v[idx] += sign * weight;
  }

  return l2normalize(v);
}

function mockEmbedder(idf?: IdfTable): Embedder {
  const maxIdf = idf ? maxIdfOf(idf) : 1;
  return {
    provider: "mock",
    model: `mock-idf-${MOCK_DIM}d`,
    dim: MOCK_DIM,
    async embed(texts) {
      return texts.map((t) => mockEmbed(t, idf, maxIdf));
    },
  };
}

/* ------------------------------------------------------------------ *
 * 真实嵌入：调 OpenAI 兼容的 /embeddings
 * ------------------------------------------------------------------ */
interface ApiOptions {
  baseUrl: string;
  apiKey: string;
  model: string;
}

async function apiEmbed(texts: string[], opt: ApiOptions): Promise<number[][]> {
  const BATCH = 32; // 一次别发太多，多数服务商有单次条数上限
  const out: number[][] = [];

  for (let i = 0; i < texts.length; i += BATCH) {
    const input = texts.slice(i, i + BATCH);
    const res = await fetch(`${opt.baseUrl}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${opt.apiKey}`,
      },
      body: JSON.stringify({ model: opt.model, input, encoding_format: "float" }),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`嵌入接口返回 ${res.status}：${detail.slice(0, 200)}`);
    }

    const json: any = await res.json();
    const rows = (json.data ?? []).sort((a: any, b: any) => (a.index ?? 0) - (b.index ?? 0));
    for (const row of rows) out.push(l2normalize(row.embedding as number[]));
  }

  return out;
}

/* ------------------------------------------------------------------ *
 * 根据环境变量挑一个实现
 * 规则：没配 EMBED_MODEL 就自动用 mock —— 保证你 clone 下来就能跑
 *
 * DeepSeek 目前没有嵌入接口，真实嵌入请用通义 / 智谱 / OpenAI，
 * 或本地 Ollama（bge-m3、nomic-embed-text）。
 * ------------------------------------------------------------------ */
export function createEmbedder(
  env: NodeJS.ProcessEnv = process.env,
  idf?: IdfTable,
): Embedder {
  const model = (env.EMBED_MODEL || "").trim();
  const forceMock = env.EMBED_MOCK === "1";
  const baseUrl = (env.EMBED_BASE_URL || env.LLM_BASE_URL || "").replace(/\/+$/, "");
  const apiKey = env.EMBED_API_KEY || env.LLM_API_KEY || "";

  if (forceMock || !model || !baseUrl || !apiKey) return mockEmbedder(idf);

  return {
    provider: "openai-compatible",
    model,
    dim: Number(env.EMBED_DIM || 1024),
    async embed(texts) {
      const vectors = await apiEmbed(texts, { baseUrl, apiKey, model });
      if (vectors[0]) this.dim = vectors[0].length;
      return vectors;
    },
  };
}

/* ------------------------------------------------------------------ *
 * 查询用的嵌入器：必须和建索引时用的是同一个模型
 * 否则两个向量根本不在同一个空间里，相似度算出来是垃圾
 * ——这是 RAG 里最隐蔽、也最常见的一类错误
 * ------------------------------------------------------------------ */
export function queryEmbedderFor(index: IndexFile, env: NodeJS.ProcessEnv = process.env): Embedder {
  if (index.provider === "mock") {
    const idf: IdfTable = new Map(Object.entries(index.idf ?? {}));
    return mockEmbedder(idf);
  }

  return {
    provider: index.provider,
    model: index.model,
    dim: index.dim,
    async embed(texts) {
      const baseUrl = (env.EMBED_BASE_URL || env.LLM_BASE_URL || "").replace(/\/+$/, "");
      const apiKey = env.EMBED_API_KEY || env.LLM_API_KEY || "";
      if (!baseUrl || !apiKey) {
        throw new Error("索引是用真实嵌入模型建的，但当前环境没有 EMBED_API_KEY / LLM_API_KEY，无法检索");
      }
      return apiEmbed(texts, { baseUrl, apiKey, model: index.model });
    },
  };
}

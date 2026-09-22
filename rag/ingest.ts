import { readdir, readFile } from "node:fs/promises";
import { extname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { DEFAULT_CHUNK_OPTIONS, chunkDoc, embedTextOf, type ChunkOptions } from "./chunk.ts";
import { buildIdf, createEmbedder } from "./embed.ts";
import { INDEX_VERSION, saveIndex } from "./store.ts";
import type { IndexFile, IndexedChunk, RawDoc } from "./types.ts";

/* ------------------------------------------------------------------ *
 * 索引构建（离线阶段）—— 只跑一次，跟提问频率无关
 *
 *   读 data/ 下的笔记  →  切块  →  算向量  →  存成 rag/index.json
 *
 * 跑法：
 *   node rag/ingest.ts
 *   node rag/ingest.ts --data=./data --out=./rag/index.json
 * ------------------------------------------------------------------ */

const PROJECT_ROOT = fileURLToPath(new URL("..", import.meta.url));
export const DEFAULT_DATA_DIR = join(PROJECT_ROOT, "data");
export const DEFAULT_INDEX_FILE = join(PROJECT_ROOT, "rag", "index.json");

const TEXT_EXT = new Set([".md", ".markdown", ".txt", ".text"]);

/** 递归找出所有文本文件。真实项目这里还要处理 pdf/docx/html，是个体力活 */
async function walk(dir: string, acc: string[] = []): Promise<string[]> {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return acc; // 目录不存在就当空
  }

  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) await walk(full, acc);
    else if (TEXT_EXT.has(extname(entry.name).toLowerCase())) acc.push(full);
  }
  return acc;
}

export interface BuildResult {
  index: IndexFile;
  files: number;
  ms: number;
}

export async function buildIndex(options: {
  dataDir?: string;
  outFile?: string;
  chunkOptions?: Partial<ChunkOptions>;
  env?: NodeJS.ProcessEnv;
  quiet?: boolean;
} = {}): Promise<BuildResult> {
  const dataDir = options.dataDir ?? DEFAULT_DATA_DIR;
  const outFile = options.outFile ?? DEFAULT_INDEX_FILE;
  const chunkOptions = { ...DEFAULT_CHUNK_OPTIONS, ...(options.chunkOptions ?? {}) };
  const log = options.quiet ? () => {} : (...a: unknown[]) => console.log(...a);

  const started = Date.now();
  const embedderCfg = createEmbedder(options.env ?? process.env);

  const files = await walk(dataDir);
  const docs: RawDoc[] = [];
  for (const file of files) {
    const text = await readFile(file, "utf8");
    // 用相对路径当来源标识：界面上显示"这段话出自哪"，同时避免暴露本机绝对路径
    docs.push({ source: relative(PROJECT_ROOT, file), text });
  }

  // 第一步：切块（纯字符串处理，和模型完全无关）
  const chunks = docs.flatMap((doc) => chunkDoc(doc.source, doc.text, chunkOptions));

  if (chunks.length === 0) {
    throw new Error(`在 ${dataDir} 里没找到可索引的文本文件（.md / .txt）。放几篇笔记进去再试。`);
  }

  const texts = chunks.map(embedTextOf);

  // 第二步：如果需要本地伪嵌入，先用全量块统计一次 IDF
  // （真实嵌入模型自带语义能力，不需要这一步）
  const useMock = embedderCfg.provider === "mock";
  const idf = useMock ? buildIdf(texts) : undefined;
  const embedder = useMock ? createEmbedder(options.env ?? process.env, idf) : embedderCfg;

  // 第三步：算向量（这一步才开始花钱、花时间）
  log(`\n  正在嵌入 ${chunks.length} 个块（${embedder.provider} / ${embedder.model}）…`);
  const embedStarted = Date.now();
  const vectors = await embedder.embed(texts);
  const embedMs = Date.now() - embedStarted;

  // 落盘前把浮点截断一下：全精度的 JSON 光是小数位就能让索引文件膨胀好几倍，
  // 而相似度排序几乎不受影响。真实系统这里存的是 float32 二进制。
  const indexed: IndexedChunk[] = chunks.map((chunk, i) => ({
    ...chunk,
    vector: vectors[i].map((x) => Number(x.toFixed(5))),
  }));

  const index: IndexFile = {
    version: INDEX_VERSION,
    provider: embedder.provider,
    model: embedder.model,
    dim: vectors[0]?.length ?? 0,
    builtAt: new Date().toISOString(),
    chunkOptions,
    idf: idf ? Object.fromEntries(idf) : undefined,
    chunks: indexed,
  };

  await saveIndex(outFile, index);

  const avgChars = Math.round(chunks.reduce((s, c) => s + c.text.length, 0) / chunks.length);
  log(`\n  索引构建完成`);
  log(`  文档 ${docs.length} 篇 → 切出 ${chunks.length} 个块（平均 ${avgChars} 字/块）`);
  log(`  向量维度 ${index.dim}，嵌入耗时 ${embedMs}ms，总计 ${Date.now() - started}ms`);
  log(`  已写入 ${relative(PROJECT_ROOT, outFile)}\n`);
  log(`  下一步：node server.ts，然后在页面上打开「知识库」开关。\n`);

  return { index, files: docs.length, ms: Date.now() - started };
}

/* 直接运行本文件时才走 CLI 分支，被 server.ts import 时不会执行 */
const isCli = process.argv[1]?.endsWith("ingest.ts") ?? false;

if (isCli) {
  const args = Object.fromEntries(
    process.argv.slice(2).map((a) => {
      const [k, v = ""] = a.replace(/^--/, "").split("=");
      return [k, v];
    }),
  );

  buildIndex({
    dataDir: args.data ? join(process.cwd(), args.data) : DEFAULT_DATA_DIR,
    outFile: args.out ? join(process.cwd(), args.out) : DEFAULT_INDEX_FILE,
  }).catch((err) => {
    console.error(`\n  索引构建失败：${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
}

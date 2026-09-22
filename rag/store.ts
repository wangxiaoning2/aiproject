import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { Hit, IndexFile, IndexedChunk } from "./types.ts";

/* ------------------------------------------------------------------ *
 * 向量存储与检索
 *
 * 真实项目这里会换成向量数据库（Chroma / Qdrant / pgvector……），
 * 但对几万条以内的数据，一个 JSON 文件 + 暴力算相似度就够了，
 * 而且你能看清每一步。面试时"为什么先不上向量库"是个好答案。
 * ------------------------------------------------------------------ */

export const INDEX_VERSION = 2;

export async function saveIndex(path: string, index: IndexFile): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(index), "utf8");
}

export async function loadIndex(path: string): Promise<IndexFile | null> {
  try {
    const index = JSON.parse(await readFile(path, "utf8")) as IndexFile;
    if (index.version !== INDEX_VERSION) {
      console.warn(`[rag] 索引版本不匹配（文件 ${index.version}，期望 ${INDEX_VERSION}），请重建`);
      return null;
    }
    return index;
  } catch {
    return null;
  }
}

/** 余弦相似度。向量已经归一化过，所以点积就等于余弦 */
export function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`向量维度不一致：索引 ${a.length} 维，查询 ${b.length} 维，请重建索引`);
  }
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/** 最朴素的检索：全部算一遍，取前 topK。数据量大了再说优化的事 */
export function searchIndex(index: IndexFile, queryVector: number[], topK: number): Hit[] {
  const scored = index.chunks.map((c: IndexedChunk) => ({
    chunk: { id: c.id, source: c.source, heading: c.heading, text: c.text, seq: c.seq },
    score: cosine(c.vector, queryVector),
    rank: 0,
  }));

  scored.sort((x, y) => y.score - x.score);
  return scored.slice(0, Math.max(1, topK)).map((h, i) => ({ ...h, rank: i + 1 }));
}

export function indexStats(index: IndexFile) {
  const bySource = new Map<string, number>();
  for (const c of index.chunks) bySource.set(c.source, (bySource.get(c.source) ?? 0) + 1);

  return {
    ready: true,
    provider: index.provider,
    model: index.model,
    dim: index.dim,
    builtAt: index.builtAt,
    chunkOptions: index.chunkOptions,
    totalChunks: index.chunks.length,
    sources: [...bySource.entries()].map(([source, chunks]) => ({ source, chunks })),
  };
}

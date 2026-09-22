import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { queryEmbedderFor } from "./embed.ts";
import { DEFAULT_INDEX_FILE } from "./ingest.ts";
import { loadIndex, searchIndex } from "./store.ts";

/* ------------------------------------------------------------------ *
 * 命令行检索：不花一分钱、不调模型，只看"召回了什么"
 *
 *   node rag/query.ts "回答打到一半就断了怎么办"
 *   node rag/query.ts "怎么省钱" 5
 *
 * 这一步是调 RAG 时最该反复用的工具：
 * 回答不对的时候，先来这里看召回的几段对不对——八成问题就在这。
 * ------------------------------------------------------------------ */

const query = process.argv[2];
const topK = Number(process.argv[3] || 4);

if (!query) {
  console.error("\n  用法：node rag/query.ts \"你的问题\" [topK]\n");
  process.exit(1);
}

const indexPath = join(fileURLToPath(new URL("..", import.meta.url)), "rag", "index.json");
const index = await loadIndex(process.env.RAG_INDEX || indexPath);

if (!index) {
  console.error("\n  还没建索引。先跑：node rag/ingest.ts\n");
  process.exit(1);
}

const embedder = queryEmbedderFor(index);
const [queryVector] = await embedder.embed([query]);
const hits = searchIndex(index, queryVector, topK);

console.log(`\n  问题：${query}`);
console.log(`  检索：${index.chunks.length} 个块中取前 ${topK}（${index.provider} / ${index.model}）\n`);

for (const hit of hits) {
  const bar = "█".repeat(Math.max(1, Math.round(hit.score * 30)));
  console.log(`  #${hit.rank}  相似度 ${hit.score.toFixed(4)}  ${bar}`);
  console.log(`      ${hit.chunk.source}  ·  ${hit.chunk.heading}`);
  console.log(`      ${hit.chunk.text.replace(/\n/g, " ").slice(0, 90)}…\n`);
}

if (embedder.provider === "mock") {
  console.log("  ⚠ 当前用的是 mock 向量，只有字面相似能力。换成真实嵌入模型后语义召回才会生效。\n");
}

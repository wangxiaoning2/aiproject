import type { Chunk } from "./types.ts";

/* ------------------------------------------------------------------ *
 * 切块（Chunking）
 *
 * 这是 RAG 里最不起眼、但最影响效果的一步：
 *   切太小 → 一个完整的方法被劈成两半，语义断了，模型拿到半句话
 *   切太大 → 一个块里塞了三件事，噪音淹没信号，检索分数被稀释
 *
 * 这里实现的是一个"够用且可解释"的策略：
 *   按 Markdown 标题分节 → 节内按段落聚合到目标长度 → 相邻块留一点重叠
 * ------------------------------------------------------------------ */

export interface ChunkOptions {
  /** 目标块大小（字符）。中文里 1 字 ≈ 1 token，所以 320 字 ≈ 320 token */
  targetSize: number;
  /** 相邻块的重叠字符数：防止一句话正好被切断、两边都不完整 */
  overlap: number;
  /** 单块硬上限：万一有个超长段落，先按句子拆开 */
  maxSize: number;
}

export const DEFAULT_CHUNK_OPTIONS: ChunkOptions = {
  targetSize: 320,
  overlap: 60,
  maxSize: 700,
};

/** 超长段落先按句子切开，避免一个块大到把信号淹没 */
function splitLong(paragraph: string, maxSize: number): string[] {
  if (paragraph.length <= maxSize) return [paragraph];

  // 中英文句子边界都覆盖：句号/问号/叹号/换行
  const sentences = paragraph.split(/(?<=[。！？!?；;\n])/).filter((s) => s.trim());
  const out: string[] = [];
  let buf = "";

  for (const s of sentences) {
    if (buf && buf.length + s.length > maxSize) {
      out.push(buf);
      buf = "";
    }
    buf += s;
  }
  if (buf.trim()) out.push(buf);
  return out;
}

export function chunkDoc(
  source: string,
  raw: string,
  options: Partial<ChunkOptions> = {},
): Chunk[] {
  const opts = { ...DEFAULT_CHUNK_OPTIONS, ...options };
  const chunks: Chunk[] = [];

  let heading = "";
  let buf = "";
  let seq = 0;

  const emit = (): void => {
    const body = buf.trim();
    buf = "";
    if (!body) return;

    seq += 1;
    chunks.push({ id: `${source}#${seq}`, source, heading, text: body, seq });

    // 关键：留下尾部一段作为下一块的开头（重叠）
    buf = opts.overlap > 0 && body.length > opts.overlap ? body.slice(-opts.overlap) : "";
  };

  for (const rawLine of raw.split("\n")) {
    const line = rawLine.replace(/\s+$/, "");

    // 标题行：换节了。先把上一节的内容结账，绝不让两节的内容混进同一个块
    const m = /^(#{1,6})\s+(.*)$/.exec(line);
    if (m) {
      emit();
      heading = m[2].trim();
      continue;
    }

    // 空行 = 段落边界：这里选择"继续累积"，让一个块里能容纳几个相关段落
    if (!line.trim()) continue;

    for (const piece of splitLong(line, opts.maxSize)) {
      // 已经攒够目标长度了，先结账再接着装
      if (buf && buf.length >= opts.targetSize) emit();
      buf = buf ? `${buf}\n${piece}` : piece;
    }
  }

  emit();
  return chunks;
}

/* ------------------------------------------------------------------ *
 * 参与嵌入的文本 = 小节标题 + 正文
 * 标题往往就包含关键词（"成本失控""上下文裁剪"），把它带进去，
 * 召回率能明显变好——这是最便宜的一次效果提升，几乎零成本。
 * ------------------------------------------------------------------ */
export function embedTextOf(chunk: Chunk): string {
  return chunk.heading ? `${chunk.heading}\n${chunk.text}` : chunk.text;
}

/* ------------------------------------------------------------------ *
 * RAG 里流动的数据长什么样
 * 把类型先定清楚，后面每一步在做什么就不会糊
 * ------------------------------------------------------------------ */

/** 一篇原始文档（例如 data/ 下的一个 .md 文件） */
export interface RawDoc {
  /** 来源标识，通常就是相对路径，用来在界面上显示"这句话出自哪" */
  source: string;
  text: string;
}

/** 切块之后的最小检索单元 */
export interface Chunk {
  /** 唯一 id：来源 + 序号 */
  id: string;
  source: string;
  /** 所在小节标题，作为上下文前缀一起参与嵌入——这是提升召回最便宜的一招 */
  heading: string;
  /** 块的正文（不含标题），用来展示给人看 */
  text: string;
  /** 在原文中的序号，方便调试"为什么这块被切开了" */
  seq: number;
}

/** 嵌入之后的块：多了一个向量 */
export interface IndexedChunk extends Chunk {
  vector: number[];
}

/** 落盘的索引文件（rag/index.json） */
export interface IndexFile {
  version: number;
  /** 嵌入服务商标识，例如 mock / openai-compatible */
  provider: string;
  model: string;
  /** 向量维度。换嵌入模型后维度会变，必须重建索引 */
  dim: number;
  builtAt: string;
  /** 切块参数，方便对比"改了参数之后召回变了多少" */
  chunkOptions: { targetSize: number; overlap: number; maxSize: number };
  /** mock 向量用的 IDF 权重表（真实嵌入模型不需要这个字段） */
  idf?: Record<string, number>;
  chunks: IndexedChunk[];
}

/** 一次检索命中的结果 */
export interface Hit {
  chunk: Chunk;
  /** 余弦相似度，越大越像（理论上限 1） */
  score: number;
  rank: number;
}

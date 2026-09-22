/* ==================================================================
 * 这个文件里没有任何框架。
 * 目的就是让你看清：一次"AI 打字机"到底发生了什么。
 * ================================================================== */

const els = {
  stream: document.getElementById('stream'),
  input: document.getElementById('input'),
  send: document.getElementById('sendBtn'),
  stop: document.getElementById('stopBtn'),
  clear: document.getElementById('clearBtn'),
  systemPrompt: document.getElementById('systemPrompt'),
  temperature: document.getElementById('temperature'),
  tempVal: document.getElementById('tempVal'),
  raw: document.getElementById('raw'),
  modeBadge: document.getElementById('modeBadge'),
  ttft: document.getElementById('mTtft'),
  total: document.getElementById('mTotal'),
  tokIn: document.getElementById('mIn'),
  tokOut: document.getElementById('mOut'),
  cost: document.getElementById('mCost'),
  ctx: document.getElementById('mCtx'),
  usageHint: document.getElementById('usageHint'),
  // —— 知识库（RAG）相关 ——
  useRag: document.getElementById('useRag'),
  topK: document.getElementById('topK'),
  topKVal: document.getElementById('topKVal'),
  ragStats: document.getElementById('ragStats'),
  ragSources: document.getElementById('ragSources'),
  ragMeta: document.getElementById('ragMeta'),
  ragHits: document.getElementById('ragHits'),
  searchOnly: document.getElementById('searchOnlyBtn'),
  rebuild: document.getElementById('rebuildBtn'),
};

/* 对话历史存在内存里，每次请求整包发过去 —— 模型本身没有记忆 */
const state = {
  messages: [],
  controller: null,
  busy: false,
};

/* ------------------------------------------------------------------
 * 1. 渲染：用 requestAnimationFrame 攒批
 *    每个 token 都改一次 DOM 会让长回答掉帧，攒到下一帧一次性写入才顺滑
 * ------------------------------------------------------------------ */
const bubbles = new WeakMap();

function makeBubble(role, text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;

  const who = document.createElement('p');
  who.className = 'who';
  who.textContent = role === 'user' ? '你' : '模型';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (role === 'user') {
    bubble.textContent = text;
  } else {
    const span = document.createElement('span');
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    bubble.append(span, cursor);
    bubbles.set(wrap, { span, cursor, pending: '', raf: 0 });
  }

  wrap.append(who, bubble);
  els.stream.append(wrap);
  els.stream.scrollTop = els.stream.scrollHeight;
  return wrap;
}

function pushToken(wrap, text) {
  const rec = bubbles.get(wrap);
  if (!rec) return;
  rec.pending += text;
  if (rec.raf) return;                       // 这一帧已经排好队了
  rec.raf = requestAnimationFrame(() => {
    rec.span.textContent += rec.pending;
    rec.pending = '';
    rec.raf = 0;
    els.stream.scrollTop = els.stream.scrollHeight;
  });
}

function finishBubble(wrap) {
  const rec = bubbles.get(wrap);
  if (!rec) return;
  if (rec.pending) {
    rec.span.textContent += rec.pending;
    rec.pending = '';
  }
  rec.cursor.remove();
}

function setBubbleError(wrap, message) {
  const rec = bubbles.get(wrap);
  if (rec) {
    rec.cursor.remove();
    rec.span.textContent = '';
  }
  const bubble = wrap.querySelector('.bubble');
  bubble.classList.add('err');
  const p = document.createElement('div');
  p.textContent = '⚠ ' + message;
  bubble.append(p);
}

/* ------------------------------------------------------------------
 * 2. 核心：手写 SSE 解析
 *    为什么不用 EventSource？它只支持 GET，没法带 JSON body 和自定义请求头。
 *    真实项目里基本都是 fetch + ReadableStream 自己解析。
 * ------------------------------------------------------------------ */
async function sendMessage(userText) {
  if (state.busy) return;

  state.messages.push({ role: 'user', content: userText });
  makeBubble('user', userText);
  els.input.value = '';
  autoGrow();

  await runStream(0);
}

async function runStream(attempt) {
  state.busy = true;
  els.send.disabled = true;
  els.stop.disabled = false;

  const wrap = makeBubble('assistant', '');
  const clientStart = performance.now();

  state.controller = new AbortController();
  let firstTokenAt = 0;
  let received = '';
  let shouldRetry = false;

  els.raw.textContent = '';
  logRaw('—— 新请求 ——\n');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: state.messages,
        systemPrompt: els.systemPrompt.value,
        temperature: Number(els.temperature.value),
        // 打开开关就走 RAG：服务端会先检索，再把资料拼进 system prompt
        useRag: els.useRag.checked,
        topK: Number(els.topK.value),
      }),
      signal: state.controller.signal,
    });

    if (!res.ok) {
      throw new Error(`服务端返回 ${res.status}：${(await res.text()).slice(0, 160)}`);
    }
    if (!res.body) {
      throw new Error('这个浏览器不支持 response.body 流式读取');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let upstreamError = null;

    // 解析一条 SSE 帧：一个帧里可能有 event: 行和一到多行 data:
    function handleFrame(frame) {
      if (!frame.trim()) return;
      logRaw(frame + '\n\n');

      let event = 'message';
      const dataLines = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
      }

      let payload = null;
      try {
        payload = JSON.parse(dataLines.join('\n'));
      } catch {
        return; // 半截 JSON，忽略
      }

      if (event === 'token') {
        if (!firstTokenAt) {
          firstTokenAt = performance.now();
          els.ttft.innerHTML = Math.round(firstTokenAt - clientStart) + '<small>ms</small>';
        }
        received += payload.t;
        pushToken(wrap, payload.t);
      } else if (event === 'rag') {
        // 服务端在生成之前先把"这次检索到了什么"推过来
        renderRagHits(payload);
      } else if (event === 'done') {
        applyDone(payload);
      } else if (event === 'error') {
        upstreamError = payload.message || '上游出错';
      }
    }

    try {
      // 循环读，直到服务端关闭连接
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;

        // ⚠ 关键点：一个 chunk 未必是一条完整消息，可能把一条 SSE 消息切成两半。
        //   所以必须自己缓存 buffer，按空行切分，不能假设 chunk 等于一帧。
        buffer += decoder.decode(value, { stream: true }).replace(/\r/g, '');

        let cut;
        while ((cut = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, cut);
          buffer = buffer.slice(cut + 2);
          handleFrame(frame);
        }

        if (upstreamError) break;
      }
    } finally {
      // 出错或被中断时，把流关掉，别让它继续占着连接
      try {
        await reader.cancel();
      } catch {
        /* 已经关了 */
      }
    }

    if (upstreamError) throw new Error(upstreamError);

    finishBubble(wrap);
    if (received) state.messages.push({ role: 'assistant', content: received });

  } catch (err) {
    finishBubble(wrap);

    if (err && err.name === 'AbortError') {
      setBubbleError(wrap, '已中断（服务端的上游请求也一起被取消了）');
      if (received) state.messages.push({ role: 'assistant', content: received });
    } else if (attempt < 1) {
      // 失败重试一次。真实项目里一定要做，而且要有次数上限和退避策略。
      shouldRetry = true;
      logRaw(`\n—— 请求失败，1 秒后自动重试一次 ——\n${err.message}\n`);
      wrap.remove();
    } else {
      // 失败的回答不进上下文，否则模型会以为自己说过这话
      setBubbleError(wrap, err.message || String(err));
    }
  } finally {
    state.busy = false;
    els.send.disabled = false;
    els.stop.disabled = true;
    state.controller = null;
  }

  if (shouldRetry) {
    await new Promise((r) => setTimeout(r, 1000));
    return runStream(1);
  }

  els.send.focus();
}

/* ------------------------------------------------------------------
 * 3. 召回面板：把"为什么是这几段"摊开给你看
 *    RAG 调不好的时候，九成问题在这里，不在模型那里。
 * ------------------------------------------------------------------ */
function renderRagHits(payload) {
  const hits = payload.hits || [];
  els.ragHits.innerHTML = '';

  if (!hits.length) {
    els.ragMeta.textContent = '这一轮没有任何召回结果——知识库可能是空的，或者索引还没建。';
    return;
  }

  const max = Math.max(...hits.map((h) => h.score), 0.0001);
  const chars = hits.reduce((s, h) => s + (h.chars || 0), 0);
  els.ragMeta.textContent =
    `检索 ${payload.tookMs}ms · 最高相似度 ${hits[0].score.toFixed(4)} · ` +
    `注入提示词 ${hits.length} 段共 ${chars} 字（≈${chars} token，都是要付费的输入）`;

  for (const h of hits) {
    const box = document.createElement('div');
    box.className = 'hit';

    const top = document.createElement('div');
    top.className = 'top';
    const rank = document.createElement('span');
    rank.className = 'rank';
    rank.textContent = '#' + h.rank;
    const score = document.createElement('span');
    score.className = 'score';
    score.textContent = '相似度 ' + h.score.toFixed(4);
    top.append(rank, score);

    // 条形长度按本次最高分归一化：绝对值很难有直觉，相对高低一眼就懂
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.width = Math.max(3, Math.round((h.score / max) * 100)) + '%';

    const src = document.createElement('p');
    src.className = 'src';
    src.textContent = `${h.source} · ${h.heading}`;

    const txt = document.createElement('div');
    txt.className = 'txt';
    txt.textContent = h.text;

    const more = document.createElement('button');
    more.className = 'more';
    more.textContent = '展开全文';
    more.addEventListener('click', () => {
      box.classList.toggle('open');
      more.textContent = box.classList.contains('open') ? '收起' : '展开全文';
    });

    box.append(top, bar, src, txt, more);
    els.ragHits.append(box);
  }
}

function applyRagStats(stats) {
  if (!stats || !stats.ready) {
    els.ragStats.textContent =
      (stats && stats.hint) ||
      '知识库未就绪：放几篇 .md 笔记到 data/ 文件夹，再点下面的重建按钮。';
    els.useRag.checked = false;
    return;
  }

  els.ragStats.textContent =
    `${stats.totalChunks} 个块 · ${stats.sources.length} 篇笔记 · ${stats.model} · ${stats.dim} 维`;

  els.ragSources.innerHTML = '';
  for (const s of stats.sources) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = `${s.source} (${s.chunks})`;
    els.ragSources.append(chip);
  }
}

/* ------------------------------------------------------------------
 * 4. 把服务端给的真实数据摊到面板上
 * ------------------------------------------------------------------ */
function applyDone(payload) {
  const u = payload.usage || {};

  els.total.innerHTML = Math.round(payload.totalMs) + '<small>ms</small>';
  els.tokIn.textContent = u.prompt_tokens ?? '—';
  els.tokOut.textContent = u.completion_tokens ?? '—';
  els.cost.innerHTML = '¥' + Number(payload.cost || 0).toFixed(4);
  els.ctx.textContent =
    (payload.context?.sent ?? '—') +
    (payload.context?.trimmed ? `（−${payload.context.trimmed}）` : '');

  els.usageHint.textContent = payload.usedRealUsage
    ? 'usage 来自服务端真实返回，成本按价格配置估算。注意首 token 延迟是浏览器实测的，那才是用户真正感知到的那一段。'
    : '这次用的是估算 token（中文按字、英文按 4 字符 ≈ 1 token），不是服务端真实 usage。首 token 延迟为浏览器实测。';
}

function logRaw(text) {
  const atBottom = els.raw.scrollHeight - els.raw.scrollTop - els.raw.clientHeight < 40;
  const span = document.createElement('span');
  if (/^event:/.test(text)) span.className = 'ev';
  else if (/^data:/.test(text)) span.className = 'dm';
  span.textContent = text;
  els.raw.append(span);
  if (atBottom) els.raw.scrollTop = els.raw.scrollHeight;
}

/* ------------------------------------------------------------------
 * 5. 交互细节
 * ------------------------------------------------------------------ */
function autoGrow() {
  els.input.style.height = 'auto';
  els.input.style.height = Math.min(els.input.scrollHeight, 140) + 'px';
}

els.input.addEventListener('input', autoGrow);

els.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    const v = els.input.value.trim();
    if (v) sendMessage(v);
  }
});

els.send.addEventListener('click', () => {
  const v = els.input.value.trim();
  if (v) sendMessage(v);
});

els.stop.addEventListener('click', () => {
  // 一按就断：前端停止读流 + 服务端收到 close 事件后取消上游请求
  if (state.controller) state.controller.abort();
});

els.clear.addEventListener('click', () => {
  state.messages = [];
  els.stream.innerHTML = '';
  els.raw.textContent = '等待请求…';
  els.ragHits.innerHTML = '';
  els.ragMeta.textContent = '还没有检索记录。问一句，或者点「只检索」。';
  ['ttft', 'total', 'tokIn', 'tokOut', 'cost', 'ctx'].forEach((k) => {
    els[k].textContent = '—';
  });
});

els.temperature.addEventListener('input', () => {
  els.tempVal.textContent = Number(els.temperature.value).toFixed(1);
});

els.topK.addEventListener('input', () => {
  els.topKVal.textContent = els.topK.value;
});

/* 「只检索」：不调模型，所以一分钱不花。
   调 RAG 时最常用的一步——先确认召回对不对，再去看回答。 */
els.searchOnly.addEventListener('click', async () => {
  const query = els.input.value.trim();
  if (!query) {
    els.ragMeta.textContent = '先在下面的输入框里写一句问题，再点「只检索」。';
    els.input.focus();
    return;
  }

  els.searchOnly.disabled = true;
  els.ragMeta.textContent = '正在检索…';

  try {
    const res = await fetch('/api/rag/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, topK: Number(els.topK.value) }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `服务端返回 ${res.status}`);
    renderRagHits(data);
  } catch (err) {
    els.ragMeta.textContent = `检索失败：${err.message || err}`;
    els.ragHits.innerHTML = '';
  } finally {
    els.searchOnly.disabled = false;
  }
});

els.rebuild.addEventListener('click', async () => {
  els.rebuild.disabled = true;
  els.ragStats.textContent = '正在重建索引…';

  try {
    const res = await fetch('/api/rag/rebuild', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `服务端返回 ${res.status}`);
    applyRagStats(data);
    els.ragHits.innerHTML = '';
    els.ragMeta.textContent = `索引已重建：${data.files} 篇笔记 → ${data.totalChunks} 个块。现在可以重新检索了。`;
  } catch (err) {
    els.ragStats.textContent = `重建失败：${err.message || err}`;
  } finally {
    els.rebuild.disabled = false;
  }
});

/* ------------------------------------------------------------------
 * 6. 启动时问一下服务端当前是什么模式
 * ------------------------------------------------------------------ */
(async function boot() {
  try {
    const cfg = await (await fetch('/api/config')).json();
    els.modeBadge.textContent = cfg.mock ? 'mock 模式（本地伪造，不花钱）' : `已接入 ${cfg.model}`;
    els.modeBadge.className = cfg.mock ? 'badge' : 'badge live';
    els.usageHint.textContent = cfg.mock
      ? '当前是 mock 模式：流量是本地伪造的，不花钱。配好 LLM_API_KEY 重启就能切到真实模型。'
      : `成本按 ¥${cfg.priceIn}/百万(输入)、¥${cfg.priceOut}/百万(输出) 估算。上下文预算 ${cfg.maxContextTokens} tokens。`;

    if (cfg.ragTopK) {
      els.topK.value = String(cfg.ragTopK);
      els.topKVal.textContent = String(cfg.ragTopK);
    }
    applyRagStats(cfg.rag);
  } catch {
    els.modeBadge.textContent = '服务端未响应';
    els.modeBadge.className = 'badge';
  }
  els.input.focus();
})();

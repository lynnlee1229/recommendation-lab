/* All experiment state is server-backed; no fabricated metrics or browser-only training. */
const $ = (id) => document.getElementById(id);
const esc = (x) =>
  String(x ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const number = (x) => Number(x).toLocaleString("en-US");
const fmt = (x) => Number(x).toFixed(4);
const date = (x) =>
  new Date(x).toISOString().slice(0, 19).replace("T", " ") + " UTC";
let state, traceResult;
async function api(path, options) {
  const r = await fetch(path, options);
  const body = await r.json();
  if (!r.ok)
    throw Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  return body;
}
function notice(s = "") {
  $("notice").textContent = s;
}
function table(head, rows, className = "") {
  return `<div class="table-wrap ${className}"><table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
const labels = {
  research: ["实验对照", "同一系统的推荐、搜索与广告；各领域独立评估。"],
  overview: ["实验概览", "先建立可信的比较口径，再观察模型学到了什么。"],
  trace: ["推荐过程", "追踪召回、过滤、融合与 MMR。每个候选都保留来源和分数。"],
  features: ["特征与样本", "检查时间边界、Embedding 输入和真实训练样本。"],
  experiments: ["训练与实验", "一次改变一个变量，让假设接受真实数据检验。"],
  course: [
    "课程地图",
    "45 份讲义的实验路线。区分机制复现、待实现内容与数据缺口。",
  ],
  code: ["代码与 Notebook", "读真实 Python 源码，在独立内核里逐格运行实验。"],
};
document.querySelectorAll("nav button").forEach((b) =>
  b.addEventListener("click", () => {
    document
      .querySelectorAll("nav button,.tab")
      .forEach((e) => e.classList.remove("active"));
    b.classList.add("active");
    $(b.dataset.tab).classList.add("active");
    $("page-title").textContent = labels[b.dataset.tab][0];
    $("page-desc").textContent = labels[b.dataset.tab][1];
    notice();
    window.scrollTo({ top: 0, behavior: "auto" });
    if (b.dataset.tab === "code") {
      loadCode();
      loadNotebookCatalog();
    }
  }),
);
async function loadNotebookCatalog() {
  try {
    const result = await api("/api/notebooks");
    $("notebook-list-summary").textContent =
      `共 ${result.notebooks.length} 个 Notebook · 课程与竞赛复现`;
    $("notebook-list").innerHTML = result.notebooks
      .map((item) => {
        const content = `<strong>${esc(item.title)}</strong><span>${esc(item.mode)}${item.available ? "" : " · 文件暂不可用"}</span>`;
        return item.available
          ? `<a class="notebook-entry" href="/notebooks?target=${encodeURIComponent(item.target)}" target="_blank" rel="noopener">${content}</a>`
          : `<div class="notebook-entry">${content}</div>`;
      })
      .join("");
  } catch (error) {
    $("notebook-list-summary").textContent = "";
    $("notebook-list").innerHTML =
      `<p role="alert">课程目录加载失败：${esc(error.message)}。请刷新重试。</p>`;
  }
}
let codeCatalogReady = false;
async function loadCode() {
  if (!codeCatalogReady) {
    try {
      const catalog = await api('/api/code');
      const selector = $('source-module');
      for (const item of catalog.files) {
        if (!Array.from(selector.options).some(option => option.value === item.id)) {
          const option = document.createElement('option');
          option.value = item.id; option.textContent = `${item.path} · ${item.description}`;
          selector.appendChild(option);
        }
      }
      codeCatalogReady = true;
    } catch (error) { notice(error.message); }
  }
  const module = $("source-module").value;
  try {
    const result = await api(`/api/code/${module}`);
    $("source-caption").textContent =
      `${result.path} · ${result.lines} 行 · ${result.description} · 本页只读`;
    $("source-content").innerHTML = `<ol>${result.source
      .split("\n")
      .map((line) => `<li><code>${esc(line) || " "}</code></li>`)
      .join("")}</ol>`;
    const editorTarget = {
      search: "search_code",
      search_lessons: "search_lessons_code",
      ads: "ads_code",
      models: "models",
      data: "sampling",
      retrieval: "retrieval",
      train: "training",
      service: "lab",
      xmrec_data: "xmrec_code_data",
      xmrec_memory: "xmrec_code_memory",
      xmrec_graph: "xmrec_code_graph",
      xmrec_ranking: "xmrec_code_ranking",
      xmrec_experiment: "xmrec_code_experiment",
      xmrec_metrics: "xmrec_code_metrics",

    }[module];
    $("edit-source").hidden = !editorTarget;
    if (editorTarget) $("edit-source").href = `/notebooks?target=${editorTarget}`;
  } catch (error) {
    notice(error.message);
  }
}
$("source-module").addEventListener("change", loadCode);
if (location.hash === "#code")
  document.querySelector('[data-tab="code"]').click();
function chart(curve, field, title) {
  if (!curve.length) return "";
  const w = 660,
    h = 210,
    p = 42;
  const vs = curve.map((r) => r[field]);
  let lo = Math.min(...vs),
    hi = Math.max(...vs);
  if (hi - lo < 1e-8) {
    lo -= 0.01;
    hi += 0.01;
  }
  const x = (i) => p + (i / Math.max(curve.length - 1, 1)) * (w - p - 18),
    y = (v) => h - p - ((v - lo) / (hi - lo)) * (h - p - 22);
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><text x="${p}" y="16">${esc(title)}</text><line x1="${p}" y1="${h - p}" x2="${w - 18}" y2="${h - p}" stroke="#a6bab0"/><text x="3" y="38">${hi.toFixed(3)}</text><text x="3" y="${h - p}">${lo.toFixed(3)}</text><polyline fill="none" stroke="#006e60" stroke-width="2" points="${vs.map((v, i) => `${x(i)},${y(v)}`).join(" ")}"/>${vs.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="3" fill="#006e60"><title>epoch ${i + 1}: ${v}</title></circle><text x="${x(i)}" y="${h - 18}" text-anchor="middle">${i + 1}</text>`).join("")}<text x="${w - 50}" y="${h - 2}">epoch</text></svg>`;
}
function metricsTable() {
  const rows = Object.entries(state.baselines.models).map(([name, r]) => [
    esc(name),
    fmt(r.metrics["recall@20"]),
    fmt(r.metrics["ndcg@20"]),
    fmt(r.metrics["recall@100"]),
    fmt(r.metrics["catalog_coverage@20"]),
    number(r.audit.evaluated_users),
  ]);
  for (const r of state.runs) {
    const m = r.validation.metrics;
    rows.push([
      `${esc(r.config.name)}<br><span class="mono">${esc(r.run_id)}</span>`,
      fmt(m["recall@20"]),
      fmt(m["ndcg@20"]),
      fmt(m["recall@100"]),
      fmt(m["catalog_coverage@20"]),
      number(r.validation.audit.evaluated_users),
    ]);
  }
  return table(
    [
      "模型 / 实验",
      "Recall@20",
      "NDCG@20",
      "Recall@100",
      "Catalog coverage@20",
      "评估用户",
    ],
    rows,
    "metrics",
  );
}
function drawOverview() {
  const m = state.manifest;
  const audit = state.baselines.models.popular.audit;
  $("version").textContent = m.dataset_version.slice(0, 12);
  $("overview-content").innerHTML = `<div class="stats">${[
    [m.raw_events, "原始真实行为"],
    [m.cohort_users, "训练期选定用户"],
    [m.catalog_items, "训练商品词表"],
    [state.runs.length, "已完成双塔实验"],
  ]
    .map(
      ([v, l]) =>
        `<div class="stat"><strong>${number(v)}</strong><span>${l}</span></div>`,
    )
    .join(
      "",
    )}</div><div class="section-head"><h2>同一口径下的验证集结果</h2><span>全词表检索 · 已看过滤 · 无采样候选</span></div>${metricsTable()}<div class="columns"><div><h3>时间边界</h3>${table(
    ["区间", "边界"],
    [
      ["训练结束", date(m.train_end_ms)],
      ["验证结束", date(m.valid_end_ms)],
      ["数据结束", date(m.end_ms)],
    ],
  )}<p class="muted">两次评估都使用训练截止时刻的冻结历史；不把验证行为带入测试上下文。验证集选模型，最终测试集留作一次验收。</p></div><div><h3>哪些目标没有进入这个分数？</h3>${table(
    ["验证期统计", "数量"],
    [
      ["可评估用户", number(audit.eligible_users)],
      ["实际评估用户", number(audit.evaluated_users)],
      ["词表外的新物品目标", number(audit.cold_oov_targets_excluded)],
      ["训练期已看过的目标", number(audit.previously_seen_targets_excluded)],
    ],
  )}<p class="muted">这些是用户-物品对数量；指标只描述所选活跃用户群的新物品、暖商品推荐，不能外推全站或冷启动。</p></div></div><div class="note"><strong>第一轮动手：历史真的有用吗？</strong><p>进入“推荐过程”，固定用户和模型，先正常推荐，再清空序列特征。预测哪些商品会变化，然后查看实际结果。注意：用户 ID Embedding 仍然保留。</p></div><h3>当前链路的真实边界</h3><p>已实现：热门 / ItemCF / 双塔召回、已看过滤、RRF 融合、MMR 重排、特征和样本检查。粗排、精排、多目标模型和实时特征尚未接入。</p><p class="muted">数据来源：<a href="${esc(m.source_card)}" target="_blank" rel="noopener">Retailrocket 发布页 ↗</a> · SHA-256 <span class="mono">${esc(m.archive_sha256)}</span></p>`;
}
function populate() {
  for (const id of ["trace-user", "feature-user"]) {
    const previous = $(id).value;
    $(id).innerHTML = state.users
      .map(
        (u) =>
          `<option value="${u.user_idx}">用户 ${u.user_id} · ${u.distinct_items} 个商品</option>`,
      )
      .join("");
    if (previous) $(id).value = previous;
  }
  for (const id of ["trace-run", "feature-run"]) {
    const previous = $(id).value;
    $(id).innerHTML = state.runs
      .map(
        (r) =>
          `<option value="${esc(r.run_id)}">${esc(r.config.name)} · dim ${r.config.dim}</option>`,
      )
      .join("");
    if (previous && state.runs.some((r) => r.run_id === previous))
      $(id).value = previous;
    else if (state.runs.length) $(id).value = state.runs.at(-1).run_id;
  }
}
async function refresh() {
  state = await api("/api/summary");
  populate();
  drawOverview();
  drawExperiments();
}
$("recommend").addEventListener("click", async () => {
  const button = $("recommend");
  button.disabled = true;
  notice();
  try {
    traceResult = await api("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_idx: Number($("trace-user").value),
        run_id: $("trace-run").value,
        channel: $("channel").value,
        candidate_budget: Number($("budget").value),
        mmr_lambda: Number($("lambda").value),
        history_override: $("clear-history").checked ? [] : null,
      }),
    });
    drawTrace();
  } catch (e) {
    notice(e.message);
  } finally {
    button.disabled = false;
  }
});
function drawTrace() {
  const r = traceResult,
    t = r.trace;
  $("trace-content").className = "";
  $("trace-content").innerHTML = `<div class="flow">${[
    [t.catalog, "训练商品池"],
    [t.available_after_filter, "已看过滤后"],
    [t.merged_pool, "召回候选"],
    ["—", "粗排 / 待实现"],
    ["—", "精排 / 待实现"],
    [t.returned, "融合 / MMR 输出"],
  ]
    .map(
      ([v, l]) =>
        `<div class="${v === "—" ? "pending" : ""}"><strong>${v}</strong><span>${l}</span></div>`,
    )
    .join(
      "",
    )}</div><p class="muted">request <span class="mono">${t.request_id}</span> · ${t.latency_ms} ms · 特征截止 ${date(t.snapshot_ms)}${t.counterfactual ? " · 反事实输入" : ""}</p>${table(
    [
      "位置",
      "商品 / 类目",
      "召回来源与名次",
      "双塔 cosine",
      "ItemCF 分数",
      "融合分 / MMR",
      "检查",
    ],
    r.items.map((i) => [
      i.position,
      `${i.item_id}<br><span class="muted">category ${esc(i.category)}</span>`,
      Object.entries(i.channels)
        .map(([c, n]) => `<span class="tag">${esc(c)} #${n}</span>`)
        .join(""),
      fmt(i.scores.two_tower),
      fmt(i.scores.itemcf),
      `${fmt(i.fusion_score)} / ${fmt(i.rerank.mmr_score)}`,
      `<button class="detail-button" data-item="${i.item_idx}">展开</button>`,
    ]),
  )}<div id="item-detail"></div><details><summary>各通道的原始候选（点击后查看）</summary>${Object.entries(
    r.channel_candidates,
  )
    .map(
      ([name, items]) =>
        `<h3>${esc(name)}</h3>${table(
          ["rank", "item ID", "item index", "raw score"],
          items.map((i) => [i.rank, i.item_id, i.item_idx, fmt(i.score)]),
        )}`,
    )
    .join(
      "",
    )}</details><p class="muted">不同通道原始分数不直接相加。RRF 使用 1/(60+rank) 融合；MMR 以该候选池内归一化相关性和双塔相似度逐项选取。未训练点击率排序器。</p>`;
  document
    .querySelectorAll(".detail-button")
    .forEach((b) =>
      b.addEventListener("click", () => showDetail(Number(b.dataset.item))),
    );
}
function showDetail(id) {
  const i = traceResult.items.find((x) => x.item_idx === id);
  $("item-detail").innerHTML =
    `<div class="detail"><h3>商品 ${i.item_id} · index ${i.item_idx}</h3><p>双塔得分是归一化用户向量与商品向量的点积。它是相似度，不是点击概率。</p><p>ItemCF 的主要历史贡献：</p>${
      i.itemcf_evidence.length
        ? table(
            ["历史商品", "加权相似度贡献"],
            i.itemcf_evidence.map((e) => [
              e.history_item_id,
              fmt(e.contribution),
            ]),
          )
        : '<p class="muted">没有正的 ItemCF 历史贡献；该物品可能通过其他通道或零分并列候选进入。</p>'
    }<p>MMR：归一化相关性 ${fmt(i.rerank.relevance_normalized)}；与已选商品的最大相似度 ${fmt(i.rerank.max_similarity_to_selected)}。</p><h3>实际商品向量 (${i.vector.length} 维)</h3><div class="vector">${i.vector.map((v, n) => `<span title="维度 ${n}">${v.toFixed(3)}</span>`).join("")}</div><p class="muted">类目来自训练截止前的商品属性快照，只用于查看，未送入第一版双塔。</p></div>`;
  $("item-detail").scrollIntoView({ block: "nearest" });
}
$("inspect").addEventListener("click", async () => {
  notice();
  try {
    const f = await api(
      `/api/users/${$("feature-user").value}/features?run_id=${encodeURIComponent($("feature-run").value)}`,
    );
    $("feature-content").className = "";
    $("feature-content").innerHTML =
      `<div class="note"><strong>实际模型输入</strong><p>用户 index = ${f.user_idx}；序列长度 ${f.model_inputs.history_item_indices.length}；padding = 0；masked mean pooling。</p><p class="mono">${f.model_inputs.history_item_indices.join(" → ")}</p><p class="muted">只含训练期事件。均值池化不识别窗口内的顺序；这正是后续 DIN / 序列模型实验的对照。</p></div><div class="columns"><div><h3>最近的模型输入商品</h3>${table(
        ["item ID", "index", "类目（仅查看）"],
        f.history.map((i) => [i.item_id, i.item_idx, esc(i.category)]),
      )}</div><div><h3>用户塔的输出向量</h3><div class="vector">${f.user_vector.map((v) => `<span>${v.toFixed(3)}</span>`).join("")}</div><h3>训练快照的辅助统计</h3><p>${f.metadata_not_model_inputs.training_event_count} 次行为 / ${f.metadata_not_model_inputs.distinct_items} 个商品</p><p class="muted">这些统计和类目分布未作为当前模型输入。特征版本 ${f.feature_version}；快照 ${date(f.snapshot_ms)}。</p></div></div><details open><summary>最近原始训练行为</summary><div class="event-list">${table(
        ["时间", "行为", "item ID", "index"],
        f.events.map((e) => [
          date(e.timestamp),
          esc(e.event),
          e.item_id,
          e.item_idx,
        ]),
      )}</div></details>`;
  } catch (e) {
    notice(e.message);
  }
});
$("samples").addEventListener("click", async () => {
  try {
    const run = $("feature-run").value;
    const s = await api(`/api/runs/${run}/samples`);
    const d = await api(`/api/runs/${run}/diagnostics`);
    $("sample-content").innerHTML =
      `<p class="muted">${esc(s.negative_policy)}</p>${table(
        ["用户", "样本时间", "此前历史 (item index)", "正目标"],
        s.samples
          .slice(0, 20)
          .map((x) => [
            x.user_id,
            date(x.event_time_ms),
            `<span class="mono">${x.history.join(", ")}</span>`,
            x.positive_item_idx,
          ]),
      )}<h3>8 条真实样本重组的诊断 Batch</h3><p class="muted">这是用训练样本重新组成的教学诊断批次，不是训练时原始 Batch。行=用户，列=目标物品；绿色对角线为正样本，× 为重复目标屏蔽。单元格是 softmax 概率。</p><div class="table-wrap"><table class="matrix"><thead><tr><th>u / item</th>${d.targets.map((t) => `<th>${t}</th>`).join("")}</tr></thead><tbody>${d.probabilities.map((row, i) => `<tr><th>${d.users[i]}</th>${row.map((p, j) => `<td class="${i === j ? "diag" : ""}" title="logit=${d.logits[i][j]}; dL/dlogit=${d.gradients[i][j]}">${p === 0 ? "×" : p.toFixed(3)}</td>`).join("")}</tr>`).join("")}</tbody></table></div><p>交叉熵 loss = <code>${d.loss.toFixed(4)}</code>。鼠标悬停查看 logit 和 dL/dlogit。τ=${d.temperature}，LogQ=${d.logq_correction}。</p><p class="muted">${esc(d.note)}</p>`;
  } catch (e) {
    notice(e.message);
  }
});
function drawExperiments() {
  $("experiment-results").innerHTML =
    `<div class="section-head"><h2>验证集对比</h2><span>模型选择不读取测试集</span></div>${metricsTable()}${state.runs.map((r) => `<details><summary>${esc(r.config.name)} · ${r.training_samples} 样本 · best epoch ${r.best_epoch} · ${r.elapsed_seconds}s</summary><p class="mono">${esc(JSON.stringify(r.config))}</p><div class="columns">${chart(r.curve, "loss", "训练 / sampled-softmax loss")}${chart(r.curve, "validation_ndcg@20", "验证 / NDCG@20")}</div><p class="muted">参数量 ${number(r.parameter_count)}。${esc(r.reproduction_scope)}</p></details>`).join("")}`;
}
$("train-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target),
    body = {};
  for (const [k, v] of form.entries()) body[k] = k === "name" ? v : Number(v);
  body.use_history = form.has("use_history");
  body.logq_correction = form.has("logq_correction");
  try {
    await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("train-button").disabled = true;
    pollJob();
  } catch (err) {
    notice(err.message);
  }
});
let previousJob = "";
async function pollJob() {
  try {
    const j = await api("/api/job");
    if (j.state === "idle") return;
    const p = j.progress || {};
    $("job-status").textContent =
      `${j.state} · run ${j.run_id} · epoch ${p.epoch ?? 0}${p.error ? " · " + p.error : ""}`;
    $("train-button").disabled = j.state === "running";
    if (j.state === "running") setTimeout(pollJob, 3000);
    else if (previousJob !== j.run_id) {
      previousJob = j.run_id;
      if (j.state === "complete") {
        await refresh();
        notice("新实验已完成，模型和验证集结果已保存。");
      } else notice("训练失败，查看 artifacts/logs 中对应日志。");
    }
  } catch (e) {
    notice(e.message);
  }
}
async function drawCourse() {
  const c = await api("/api/curriculum");
  $("course-content").innerHTML =
    `<div class="note"><strong>重拾知识的顺序</strong><p>${esc(c.learning_path)}</p></div><p>讲义共 ${c.total_files} 份、${c.total_pages} 页。已完成全量本地 OCR 与目录映射；公式与经验值需要结合原页阅读。下列状态表示本工作台的实现范围。</p><div class="section-head"><h2>先做这三个实验</h2></div>${c.first_experiments.map((x, i) => `<details ${i === 0 ? "open" : ""}><summary>${i + 1}. ${esc(x.title)}</summary><p><strong>先预测：</strong>${esc(x.predict)}</p><p><strong>动手：</strong>${esc(x.action)}</p><p><strong>检查：</strong>${esc(x.check)}</p></details>`).join("")}<div class="section-head"><h2>讲义中的数值与经验，作为待验证假设</h2></div>${table(
      ["来源", "讲义示例", "本地怎样验证"],
      c.heuristics.map((h) => [
        `${esc(h.file)} p.${h.pages.join(",")}`,
        esc(h.statement),
        esc(h.local_experiment),
      ]),
    )}<p class="muted">以上是特定教学/业务语境的示例，不是推荐系统通用默认参数。</p>${c.groups
      .map(
        (g) =>
          `<div class="course-group"><h2>${esc(g.title)}</h2>${c.lessons
            .filter((l) => l.group === g.id)
            .map(
              (l, i) =>
                `<div class="lesson"><span class="mono">${String(i + 1).padStart(2, "0")}</span><div><strong>${esc(l.title)}</strong>${l.notebook ? `<p><a href="/notebooks?target=${encodeURIComponent(l.notebook.target)}" target="_blank" rel="noopener"><strong>打开配套 Notebook ↗</strong></a> · ${esc(l.notebook.mode)} · ${l.notebook.state === "executed" ? "已运行" : "待验证"}</p>` : ""}<p>${esc(l.practice)}</p><p class="mono">${esc(l.code || "专题练习独立于工作台服务")} · ${l.pages} 页</p><a href="/api/lectures/${encodeURIComponent(l.file)}" target="_blank" rel="noopener">回查讲义 ↗</a> · <a href="${esc(l.public_url)}" target="_blank" rel="noopener">原始仓库 ↗</a></div><span class="status ${l.status}">工作台：${{ ready: "已实现", partial: "部分实现", planned: "未集成", data: "缺数据" }[l.status]}</span></div>`,
            )
            .join("")}</div>`,
      )
      .join(
        "",
      )}<h2>论文与复现边界</h2>${c.papers.map((p) => `<details><summary>${esc(p.title)} · ${esc(p.status)}</summary><p>${esc(p.scope)}</p><a href="${esc(p.url)}" target="_blank" rel="noopener">论文原始来源 ↗</a></details>`).join("")}`;
}
refresh()
  .then(() => {
    drawCourse().catch((e) => notice(e.message));
    pollJob();
  })
  .catch((e) => notice("工作台数据尚未就绪：" + e.message));

async function drawResearch() {
  const target = $("research-results");
  try {
    const {experiments} = await api('/api/experiments');
    target.innerHTML = '';
    for (const run of experiments.filter(x => x.state === 'completed' && !x.annotation?.superseded).reverse()) {
      const result = await api(`/api/experiments/${encodeURIComponent(run.domain)}/${encodeURIComponent(run.run_id)}`);
      const panel = document.createElement('details');
      panel.open = true;
      panel.innerHTML = `<summary>${esc(run.domain)} · ${esc(run.run_id)}</summary><p>${esc(result.scope || '')}</p>`;
      if (run.domain === 'recommendation') {
        panel.innerHTML += table(['编码器','负采样','种子','NDCG@20','Recall@20','打乱历史NDCG','覆盖率','训练秒'], result.rows.map(r => [esc(r.encoder),esc(r.sampler),r.seed,fmt(r['ndcg@20']),fmt(r['recall@20']),fmt(r.shuffled['ndcg@20']),fmt(r['coverage@20']),r.seconds.toFixed(1)]));
        panel.innerHTML += '<p>短训练与有限样本的教学对照；参数量不同，不能声称模型结构普遍优劣。loss不是点击概率，不跨采样策略比较loss绝对值。</p>';
      }
      if (run.domain === 'search' && result.kind !== 'workshop') {
        panel.innerHTML += table(['方法','NDCG@10','MRR@10','Recall@10','Recall@100','P50毫秒','P95毫秒'], result.rows.map(r => [esc(r.method),fmt(r['ndcg@10']),fmt(r['mrr@10']),fmt(r['recall@10']),fmt(r['recall@100']),r.latency_p50_ms.toFixed(1),r.latency_p95_ms.toFixed(1)]));
        panel.innerHTML += `<p>重排候选Recall@20：${fmt(result['rerank_candidate_recall@20'])}。未判断文档按不相关计算，真实相关性可能不完整。延迟包含查询编码与重排，不含离线建索引。<a href="/notebooks?target=search">打开搜索实验 Notebook ↗</a></p>`;
        panel.innerHTML += table(['失败查询ID','BM25 NDCG','TinyBERT NDCG'],result.failures.slice(0,5).map(r => [esc(r.qid),fmt(r.metrics.bm25['ndcg@10']),fmt(r.metrics.tinybert_rerank['ndcg@10'])]));
      }
      if (run.domain === 'search' && result.kind === 'workshop') {
        panel.innerHTML += table(['教学重排特征','select NDCG@10','select MRR@10','select Recall@10','维度','拟合秒'], result.rows.map(r=>[esc(r.method),fmt(r.select['ndcg@10']),fmt(r.select['mrr@10']),fmt(r.select['recall@10']),r.features,r.fit_seconds.toFixed(3)]));
        panel.innerHTML += '<p>训练查询按160/40分组的局部练习；select成绩，不替换正式搜索服务。</p>';
      }
      if (run.domain === 'ads' && result.kind === 'workshop') {
        panel.innerHTML += table(['教学方法','select AUC','select logloss','特征维度','拟合秒'],result.rows.map(r=>[esc(r.method),fmt(r.select.auc),fmt(r.select.logloss),r.features,r.fit_seconds.toFixed(2)]));
        panel.innerHTML += '<p>小前缀教学训练，只报告select；不替换正式200k实验，不混比audit。</p>';
      }
      if (run.domain === 'ads' && result.kind !== 'workshop') {
        panel.innerHTML += table(['方法','审计AUC','logloss','Brier','ECE(10等宽箱)','拟合秒'], result.rows.map(r => [esc(r.method),fmt(r.audit.auc),fmt(r.audit.logloss),fmt(r.audit.brier),fmt(r.audit.ece),r.fit_seconds.toFixed(2)]));
        panel.innerHTML += `<p>树模型家族按select logloss选定：${esc(result.selected_base)}；单独calibration拟合校准器，audit未参与选型。公开数据正负抽样比例未知，这些概率不代表线上CTR。<a href="/notebooks?target=ads">打开广告实验 Notebook ↗</a></p>`;
        panel.innerHTML += table(['审计分段','曝光数','真实点击率','原始logloss','sigmoid logloss'],result.segments.map(r => [esc(r.segment),r.raw.n,fmt(r.raw.click_rate),fmt(r.raw.logloss),fmt(r.sigmoid.logloss)]));
        panel.innerHTML += '<p>下列为独立模拟市场（每广告主预算60），收入、价值和货币均为模拟。不能用来证明线上收益。</p>';
        panel.innerHTML += table(['模拟策略','seed','模拟收入','期望点击价值','期望广告主盈余','预算约束'],result.simulations.map(r => [esc(r.name),r.seed,r.spend.toFixed(2),r.expected_click_value.toFixed(2),r.expected_advertiser_surplus.toFixed(2),r.budget_ok?'通过':'失败']));
      }
      panel.innerHTML += `<p><a href="/api/experiments/${encodeURIComponent(run.domain)}/${encodeURIComponent(run.run_id)}" target="_blank" rel="noopener">完整实验记录 JSON ↗</a></p>`;
      target.appendChild(panel);
    }
    if (!target.children.length) target.textContent = '尚无完成实验。运行对应的项目实验命令后刷新。';
  } catch (e) { target.textContent = `实验读取失败：${e.message}`; }
}
drawResearch();

$('search-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const output = $('search-output');
  output.textContent = '正在检索；首次请求需要加载本机小模型…';
  try {
    const result = await api('/api/search', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:$('search-query').value, method:$('search-method').value, limit:5})});
    output.innerHTML = `<p>${esc(result.method)} · ${result.latency_ms.toFixed(1)} 毫秒 · 分数是排序分数，不能解释为科学声明成立概率。</p>` + result.items.map(x => `<details><summary>${esc(x.document_id)} · ${esc(x.title)}</summary><p>${esc(x.text)}</p><p>score: ${fmt(x.score)}</p></details>`).join('');
  } catch (e) { output.textContent = `检索失败：${e.message}`; }
});


$('ads-example').addEventListener('click', async () => {
  try { $('ads-input').value = JSON.stringify(await api('/api/ads/example'), null, 2); }
  catch (e) { $('ads-output').textContent = e.message; }
});
$('ads-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const input = JSON.parse($('ads-input').value);
    const r = await api('/api/ads/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)});
    $('ads-output').innerHTML = table(['基模型','原始概率','sigmoid概率','isotonic概率'],[[esc(r.base),fmt(r.raw),fmt(r.sigmoid),fmt(r.isotonic)]])+'<p>仅公开样本分布概率；匿名特征含义未知，修改不能解释为因果效果。</p>';
  } catch(e) { $('ads-output').textContent=e.message; }
});
$('auction-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const r=await api('/api/ads/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({strategy:$('auction-strategy').value,budget:Number($('auction-budget').value),multiplier:Number($('auction-multiplier').value),paced:$('auction-paced').checked,seed:42})});
    $('auction-output').innerHTML=table(['模拟收入','期望点击价值','模拟点击','预算约束'],[[r.spend.toFixed(2),r.expected_click_value.toFixed(2),r.clicks,r.budget_ok?'通过':'失败']])+table(['模拟广告主','花费','赢得曝光','点击'],r.advertisers.map((a,i)=>[i+1,a.spend.toFixed(2),a.wins,a.clicks]));
  } catch(e) { $('auction-output').textContent=e.message; }
});

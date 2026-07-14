// Evaluation walkthrough — drives eval.html against the /eval/* API.

const state = {
  queryset: null,
  results: null,
  perQueryById: {},
};

const METRIC_KEYS = ["P@10", "R@10", "nDCG@10", "MRR"];
const SYSTEM_LABEL = {
  Filter: "Filter",
  VectorRAG: "VectorRAG",
  WeightedGraphRAG: "GraphRAG",
};
const COMPONENT_ORDER = ["spatial", "accessibility", "facility", "economic", "disruption", "event"];

// --- helpers ---------------------------------------------------------------
async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) { /* non-JSON body */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(message, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.style.borderColor = isError ? "rgba(154,63,63,0.6)" : "";
  el.classList.add("show");
  window.setTimeout(() => el.classList.remove("show"), 3200);
}

function fmt(x, digits = 3) {
  if (x === null || x === undefined || x === "") return "—";
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(digits) : String(x);
}

function money(x) {
  if (x === null || x === undefined) return "—";
  return "Rs " + Number(x).toLocaleString();
}

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function goldConstraints(gold) {
  const parts = [];
  if ("max_price" in gold) parts.push(`≤ Rs ${gold.max_price.toLocaleString()}`);
  if ("min_price" in gold) parts.push(`≥ Rs ${gold.min_price.toLocaleString()}`);
  if ("min_rating" in gold) parts.push(`rating ≥ ${gold.min_rating}`);
  if ("min_star" in gold) parts.push(`${gold.min_star}★+`);
  if ("max_travel_time" in gold) parts.push(`≤ ${gold.max_travel_time} min`);
  if ("required_amenities" in gold) parts.push(`has ${gold.required_amenities.join(", ")}`);
  return parts.length ? parts.map((p) => `<code>${p}</code>`).join(" ") : "—";
}

// --- stepper ---------------------------------------------------------------
function showStep(step) {
  document.querySelectorAll(".step-panel").forEach((p) => {
    p.classList.toggle("active", p.dataset.step === String(step));
  });
  document.querySelectorAll(".step-chip").forEach((c) => {
    c.classList.toggle("active", c.dataset.step === String(step));
  });
  document.querySelectorAll(".step-nav").forEach((n) => {
    n.classList.toggle("active", n.dataset.step === String(step));
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setupStepper() {
  document.querySelectorAll("[data-step]").forEach((node) => {
    if (node.classList.contains("step-panel")) return;
    node.addEventListener("click", (evt) => {
      evt.preventDefault();
      showStep(node.dataset.step);
    });
  });
}

// --- step 1: query set -----------------------------------------------------
async function loadQueryset() {
  const spec = await request("/eval/queryset");
  state.queryset = spec;

  document.getElementById("pillCity").textContent = `city · ${spec.city}`;
  document.getElementById("pillK").textContent = `K · ${spec.k}`;

  // category counts
  const catCounts = {};
  for (const q of spec.queries) {
    const c = q.category || "general";
    catCounts[c] = (catCounts[c] || 0) + 1;
  }
  const catWrap = document.getElementById("qsCategories");
  catWrap.innerHTML = "";
  Object.entries(catCounts).sort().forEach(([c, n]) => {
    catWrap.appendChild(el("span", "tag", `${c} · ${n}`));
  });

  // stats
  const stats = [
    { key: "Queries", val: spec.queries.length },
    { key: "Categories", val: Object.keys(catCounts).length },
    { key: "K (top-N)", val: spec.k },
    { key: "City", val: spec.city },
  ];
  document.getElementById("qsStats").innerHTML = stats
    .map((s) => `<div class="stat"><p>${s.key}</p><strong>${s.val}</strong></div>`)
    .join("");

  renderQuerysetTable();
  populateInspectSelect();
}

function renderQuerysetTable() {
  const body = document.querySelector("#qsTable tbody");
  body.innerHTML = "";
  for (const q of state.queryset.queries) {
    const pq = state.perQueryById[q.id];
    const nrel = pq ? pq.n_relevant : "—";
    const source = pq ? pq.gold_source : "—";
    const tr = el("tr");
    tr.innerHTML =
      `<td>${q.id}</td>` +
      `<td>${q.question}</td>` +
      `<td>${q.category || "general"}</td>` +
      `<td class="gold-cell">${goldConstraints(q.gold)}</td>` +
      `<td class="num">${nrel}</td>` +
      `<td>${source}</td>`;
    body.appendChild(tr);
  }
}

// --- step 4: results -------------------------------------------------------
function applyResults(data) {
  state.results = data;
  state.perQueryById = {};
  for (const row of data.per_query || []) state.perQueryById[row.id] = row;

  const meta = data.gold_meta || {};
  document.getElementById("pillGold").textContent =
    "gold · " + (meta.used_human ? `human (${meta.human_queries})` : "rule-based");

  renderQuerysetTable(); // now that per-query n_relevant/source is known
  renderOverall(data);
  renderCategory(data);
}

function renderOverall(data) {
  const order = data.system_order || Object.keys(data.overall || {});
  const stats = [
    { key: "Systems", val: order.length },
    { key: "Queries", val: data.n_queries },
    { key: "Pool", val: `${data.pool_size} hotels` },
    { key: "Best nDCG@10", val: `${data.best_system} · ${fmt(data.best_ndcg)}` },
  ];
  document.getElementById("overallStats").innerHTML = stats
    .map((s) => `<div class="stat"><p>${s.key}</p><strong>${s.val}</strong></div>`)
    .join("");

  const body = document.querySelector("#overallTable tbody");
  body.innerHTML = "";
  for (const name of order) {
    const m = data.overall[name] || {};
    const tr = el("tr", name === data.best_system ? "row-best" : "");
    tr.innerHTML =
      `<td>${SYSTEM_LABEL[name] || name}</td>` +
      METRIC_KEYS.map((k) => `<td class="num">${fmt(m[k])}</td>`).join("");
    body.appendChild(tr);
  }
}

function renderCategory(data) {
  const order = data.system_order || Object.keys(data.overall || {});
  const cats = Object.keys(data.by_category || {}).sort();

  const head = document.getElementById("catHeadRow");
  head.innerHTML = `<th>System</th>` + cats.map((c) => `<th class="num">${c}</th>`).join("");

  // per-category best system (for highlight)
  const bestByCat = {};
  for (const c of cats) {
    let best = null, bestVal = -1;
    for (const name of order) {
      const v = (data.by_category[c][name] || {})["nDCG@10"] || 0;
      if (v > bestVal) { bestVal = v; best = name; }
    }
    bestByCat[c] = best;
  }

  const body = document.querySelector("#categoryTable tbody");
  body.innerHTML = "";
  for (const name of order) {
    const tr = el("tr");
    let cells = `<td>${SYSTEM_LABEL[name] || name}</td>`;
    for (const c of cats) {
      const v = (data.by_category[c][name] || {})["nDCG@10"];
      const cls = bestByCat[c] === name ? "num cat-best" : "num";
      cells += `<td class="${cls}">${fmt(v)}</td>`;
    }
    tr.innerHTML = cells;
    body.appendChild(tr);
  }
}

async function loadResults(live = false) {
  const status = document.getElementById("runStatus");
  const btns = [document.getElementById("loadResultsBtn"), document.getElementById("runLiveBtn")];
  btns.forEach((b) => (b.disabled = true));
  status.textContent = live
    ? "Running the harness against Neo4j — this replays 50 queries × 3 systems…"
    : "Loading last saved results…";
  try {
    const data = live ? await request("/eval/run", { method: "POST" }) : await request("/eval/results");
    if (!data.available) {
      status.textContent = "No saved results yet. Press “Run live” to generate them.";
      return;
    }
    applyResults(data);
    status.textContent =
      `Done · ${data.n_queries} queries · pool ${data.pool_size} · ` +
      `gold: ${data.gold_meta?.used_human ? "human+rule" : "rule-based"}` +
      (live ? " · results.json refreshed" : " · from results.json");
    if (live) toast("Evaluation complete");
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    toast(err.message, true);
  } finally {
    btns.forEach((b) => (b.disabled = false));
  }
}

// --- step 5: inspector -----------------------------------------------------
function populateInspectSelect() {
  const sel = document.getElementById("inspectSelect");
  sel.innerHTML = "";
  for (const q of state.queryset.queries) {
    const opt = el("option");
    opt.value = q.id;
    const short = q.question.length > 52 ? q.question.slice(0, 52) + "…" : q.question;
    opt.textContent = `${q.id} · ${short}`;
    sel.appendChild(opt);
  }
}

async function inspect() {
  const sel = document.getElementById("inspectSelect");
  const grid = document.getElementById("inspectGrid");
  const head = document.getElementById("inspectHead");
  const qid = sel.value;
  if (!qid) return;
  head.innerHTML = "";
  grid.innerHTML = `<p class="content-placeholder">Retrieving “${qid}” across all systems…</p>`;
  try {
    const data = await request(`/eval/inspect/${qid}`);
    renderInspect(data);
  } catch (err) {
    grid.innerHTML = `<p class="content-placeholder">Error: ${err.message}</p>`;
    toast(err.message, true);
  }
}

function renderInspect(data) {
  const head = document.getElementById("inspectHead");
  const weights = COMPONENT_ORDER.filter((k) => k in (data.weights || {}))
    .map((k) => `<span class="badge">${k[0].toUpperCase()}${k.slice(1, 4)} <b>${fmt(data.weights[k], 2)}</b></span>`)
    .join("");
  head.innerHTML =
    `<div class="ih-q">${data.question}</div>` +
    `<div class="ih-meta">` +
    `<span class="pill">${data.category}</span>` +
    `<span class="pill">gold · ${data.gold_source}</span>` +
    `<span class="pill">${data.n_relevant} relevant / ${data.pool_size}</span>` +
    `</div>` +
    `<div class="ih-meta"><span class="muted" style="align-self:center">GraphRAG weights:</span>${weights}</div>` +
    `<div class="ih-meta gold-cell"><span class="muted" style="align-self:center">Gold:</span> ${goldConstraints(data.gold)}</div>`;

  // find winner by nDCG@10 for subtle highlight
  let winner = null, best = -1;
  for (const s of data.systems) {
    const v = s.metrics["nDCG@10"] || 0;
    if (v > best) { best = v; winner = s.name; }
  }

  const grid = document.getElementById("inspectGrid");
  grid.innerHTML = "";
  for (const s of data.systems) {
    grid.appendChild(renderSystemColumn(s, s.name === winner));
  }
}

function renderSystemColumn(sys, isWinner) {
  const col = el("div", "sys-col" + (isWinner ? " winner" : ""));
  col.appendChild(el("h3", null, SYSTEM_LABEL[sys.name] || sys.name));

  const badges = el("div", "metric-badges");
  badges.innerHTML = METRIC_KEYS
    .map((k) => `<span class="badge">${k} <b>${fmt(sys.metrics[k], 3)}</b></span>`)
    .join("");
  col.appendChild(badges);

  const list = el("div", "rank-list");
  if (!sys.ranked.length) {
    list.appendChild(el("p", "muted", "No results."));
  }
  for (const item of sys.ranked) {
    list.appendChild(renderRankItem(item, sys.name));
  }
  col.appendChild(list);
  return col;
}

function renderRankItem(item, systemName) {
  const node = el("div", "rank-item" + (item.relevant ? " hit" : ""));

  const row = el("div", "rank-row");
  row.innerHTML =
    `<span class="rank-num">#${item.rank}</span>` +
    `<span class="rank-name">${item.name}</span>` +
    (item.relevant ? `<span class="hit-dot" title="relevant (gold)"></span>` : "") +
    (item.score !== undefined ? `<span class="g-score">${fmt(item.score, 3)}</span>` : "");
  node.appendChild(row);

  const attrs = el("div", "rank-attrs");
  attrs.innerHTML =
    `<span>${money(item.price_lkr)}</span>` +
    `<span>★${fmt(item.rating, 1)}</span>` +
    `<span>${item.star ?? "—"}-star</span>` +
    `<span>${item.travel_time_min != null ? fmt(item.travel_time_min, 0) + " min" : "—"}</span>`;
  node.appendChild(attrs);

  // GraphRAG component bars + reasons
  if (item.components) {
    const bars = el("div", "comp-bars");
    for (const k of COMPONENT_ORDER) {
      if (!(k in item.components)) continue;
      const v = item.components[k];
      const bar = el("div", "comp-bar");
      bar.innerHTML =
        `<span class="comp-label">${k}</span>` +
        `<span class="comp-track"><span class="comp-fill" style="width:${Math.round(v * 100)}%"></span></span>` +
        `<span class="comp-val">${fmt(v, 2)}</span>`;
      bars.appendChild(bar);
    }
    node.appendChild(bars);

    if (item.reasons && item.reasons.length) {
      const reasons = el("div", "rank-reasons");
      reasons.innerHTML = item.reasons.map((r) => `<span class="rank-reason">${r}</span>`).join("");
      node.appendChild(reasons);
    }
  }
  return node;
}

// --- init ------------------------------------------------------------------
async function init() {
  setupStepper();
  document.getElementById("loadResultsBtn").addEventListener("click", () => loadResults(false));
  document.getElementById("runLiveBtn").addEventListener("click", () => loadResults(true));
  document.getElementById("inspectBtn").addEventListener("click", inspect);

  try {
    await loadQueryset();
  } catch (err) {
    toast(`Could not load query set: ${err.message}`, true);
  }
  // Non-blocking: pull cached results if they exist so tables + gold source fill in.
  try {
    const data = await request("/eval/results");
    if (data.available) applyResults(data);
  } catch (_) { /* leave step 4 empty until user runs */ }
}

init();

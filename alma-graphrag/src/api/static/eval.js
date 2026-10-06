// Evaluation walkthrough — drives eval.html against the /eval/* API.
//
// The page mirrors thesis Chapters 4-5: Route 1 (rule-based benchmark) and
// Route 2 (held-out human choices). By default it reads the frozen runs the
// thesis reports, so the numbers on screen match the printed tables; a live
// run is available but drifts slightly because the graph's traffic and prices
// keep changing.

const state = {
  queryset: null,
  runs: {},           // run name -> results payload (thesis | price | latest)
  current: null,      // the run shown in step 4
  mainPerQuery: {},   // per-query rows of the main-set run, for steps 1 and 3
};

const METRIC_KEYS = ["P@10", "R@10", "nDCG@10", "MRR"];
const PROPOSED = "WeightedGraphRAG";

// The eight methods the thesis compares in its overall table, in its order.
const COMPARED = ["Random", "Popularity", "Filter", "Keyword", "SemanticVec",
  "Hybrid", "CrossEncoder", PROPOSED];
// Rows the harness produces that are not part of the comparison.
const DIAGNOSTIC = ["LTR", "LTR[doc-only]", "WeightedGraphRAG[no-diffusion]"];

const LABEL = {
  Random: "Random",
  Popularity: "Popularity",
  Filter: "Filter-and-sort (current practice)",
  Keyword: "Keyword (Postgres full-text)",
  SemanticVec: "Meaning-based",
  Hybrid: "Hybrid (RRF)",
  CrossEncoder: "Cross-encoder",
  LTR: "LTR, trained on the answer key (circular)",
  "LTR[doc-only]": "LTR, hotel features only",
  WeightedGraphRAG: "Weighted GraphRAG (proposed)",
  "WeightedGraphRAG[no-diffusion]": "Proposed without the neighbor estimate",
  "WeightedGraphRAG[human]": "Whole study weights",
  "WeightedGraphRAG[human_informed]": "Study price weight, all queries",
  "WeightedGraphRAG[human-price-aware]": "Study price weight, price-focused queries only",
  "WeightedGraphRAG[handset]": "Scoring model, hand-set weights",
  "WeightedGraphRAG[fitted-on-train]": "Scoring model, fitted on the 67",
  "WeightedGraphRAG[elicited]": "Scoring model, elicited profile",
  "WeightedGraphRAG[blended]": "Scoring model, blended profile",
  "WeightedGraphRAG[balanced]": "Scoring model, balanced profile",
};
const SHORT = {
  Filter: "Filter-and-sort", Keyword: "Keyword", SemanticVec: "Meaning-based",
  Hybrid: "Hybrid", CrossEncoder: "Cross-encoder", WeightedGraphRAG: "Proposed",
};
const label = (name) => LABEL[name] || name;

const CATEGORY_LABEL = {
  accessibility: "Accessibility", amenity: "Amenity", disruption: "Disruption",
  economic: "Economic", multi_dimensional: "Multi-dimensional", quality: "Quality",
  economic_ranked: "Price-ranked",
};

const CRITERIA = ["spatial", "accessibility", "facility", "economic", "disruption"];
const COMPONENT_ORDER = [...CRITERIA, "event"];
const HANDSET = { spatial: 0.25, accessibility: 0.20, facility: 0.25, economic: 0.15, disruption: 0.15 };

const RUN_LABEL = {
  thesis: "thesis · main set",
  price: "thesis · price set",
  latest: "latest live run",
};

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
  const node = document.getElementById("toast");
  node.textContent = message;
  node.style.borderColor = isError ? "rgba(154,63,63,0.6)" : "";
  node.classList.add("show");
  window.setTimeout(() => node.classList.remove("show"), 3200);
}

// Round half away from zero, as the thesis tables do. A plain toFixed turns
// 0.9374999999999999 (float noise for 0.9375) into 0.937, not 0.938.
function round(n, digits) {
  return (Math.abs(n) + 1e-9).toFixed(digits);
}

function fmt(x, digits = 3) {
  if (x === null || x === undefined || x === "") return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return String(x);
  return (n < 0 && Number(round(n, digits)) !== 0 ? "-" : "") + round(n, digits);
}

function signed(x, digits = 3) {
  if (x === null || x === undefined) return "—";
  return (x >= 0 ? "+" : "−") + round(x, digits);
}

function pct(x, digits = 1) {
  return x === null || x === undefined || !Number.isFinite(x)
    ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(digits)}%`;
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

function statCards(id, cards) {
  document.getElementById(id).innerHTML = cards
    .map((s) => `<div class="stat"><p>${s.key}</p><strong>${s.val}</strong></div>`)
    .join("");
}

function fillBody(selector, rows) {
  const body = document.querySelector(`${selector} tbody`);
  body.innerHTML = "";
  rows.forEach((tr) => body.appendChild(tr));
}

function row(cells, className) {
  const tr = el("tr", className || "");
  tr.innerHTML = cells.join("");
  return tr;
}

const td = (v, cls = "") => `<td${cls ? ` class="${cls}"` : ""}>${v}</td>`;

function goldConstraints(gold) {
  const parts = [];
  if ("max_price" in gold) parts.push(`≤ Rs ${gold.max_price.toLocaleString()}`);
  if ("min_price" in gold) parts.push(`≥ Rs ${gold.min_price.toLocaleString()}`);
  if ("min_rating" in gold) parts.push(`rating ≥ ${gold.min_rating}`);
  if ("min_star" in gold) parts.push(`${gold.min_star}★+`);
  if ("max_travel_time" in gold) parts.push(`≤ ${gold.max_travel_time} min`);
  if ("max_disruption" in gold) parts.push(`delay ≤ ${gold.max_disruption} min`);
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
  if (window.location.hash !== `#step-${step}`) {
    history.replaceState(null, "", `#step-${step}`);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function stepFromHash() {
  const m = /^#step-([1-6])$/.exec(window.location.hash);
  return m ? m[1] : null;
}

function setupStepper() {
  document.querySelectorAll("[data-step]").forEach((node) => {
    if (node.classList.contains("step-panel")) return;
    node.addEventListener("click", (evt) => {
      evt.preventDefault();
      showStep(node.dataset.step);
    });
  });
  window.addEventListener("hashchange", () => {
    const step = stepFromHash();
    if (step) showStep(step);
  });
  const initial = stepFromHash();
  if (initial) showStep(initial);
}

// --- step 1: query set -----------------------------------------------------
async function loadQueryset() {
  const spec = await request("/eval/queryset");
  state.queryset = spec;

  document.getElementById("pillCity").textContent = `city · ${spec.city}`;
  document.getElementById("pillK").textContent = `k · ${spec.k}`;

  const catCounts = {};
  for (const q of spec.queries) {
    const c = q.category || "general";
    catCounts[c] = (catCounts[c] || 0) + 1;
  }
  const catWrap = document.getElementById("qsCategories");
  catWrap.innerHTML = "";
  Object.entries(catCounts).sort().forEach(([c, n]) => {
    catWrap.appendChild(el("span", "tag", `${CATEGORY_LABEL[c] || c} · ${n}`));
  });

  statCards("qsStats", [
    { key: "Main queries", val: spec.queries.length },
    { key: "Price queries", val: 20 },
    { key: "Categories", val: Object.keys(catCounts).length },
    { key: "Hotels in pool", val: 77 },
    { key: "k (top-N)", val: spec.k },
    { key: "City", val: spec.city },
  ]);

  renderQuerysetTable();
  populateInspectSelect();
}

function renderQuerysetTable() {
  if (!state.queryset) return;
  fillBody("#qsTable", state.queryset.queries.map((q) => {
    const pq = state.mainPerQuery[q.id];
    return row([
      td(q.id), td(q.question), td(CATEGORY_LABEL[q.category] || q.category || "general"),
      td(goldConstraints(q.gold), "gold-cell"),
      td(pq ? pq.n_relevant : "—", "num"), td(pq ? pq.gold_source : "—"),
    ]);
  }));
}

// --- step 3: worked example ------------------------------------------------
function populateExampleSelect(perQuery) {
  const sel = document.getElementById("exampleSelect");
  // Default to the query the thesis worked example uses (ids differ in case
  // between runs, so match case-insensitively).
  const wanted = (sel.value || "q0005").toLowerCase();
  const keep = (perQuery.find((q) => q.id.toLowerCase() === wanted) || {}).id;
  sel.innerHTML = "";
  for (const q of perQuery) {
    const opt = el("option");
    opt.value = q.id;
    const short = q.question.length > 70 ? q.question.slice(0, 70) + "…" : q.question;
    opt.textContent = `${q.id} · ${short}`;
    sel.appendChild(opt);
  }
  if (keep) sel.value = keep;
  renderExample();
}

function renderExample() {
  const qid = document.getElementById("exampleSelect").value;
  const q = (state.current?.per_query || []).find((r) => r.id === qid);
  if (!q) return;
  const p = q.scores[PROPOSED] || {};
  const f = q.scores.Filter || {};
  const G = q.n_relevant;
  const hits = (m) => Math.round((m["P@10"] || 0) * 10);
  const first = (m) => (m.MRR ? Math.round(1 / m.MRR) : "none");
  document.getElementById("exampleLead").innerHTML =
    `“${q.question}” · ${G} of ${state.current.pool_size} hotels are relevant. ` +
    `The proposed method placed ${hits(p)} of them in its top ten and filter-and-sort ${hits(f)}.`;
  fillBody("#exampleTable", [
    row([td("Relevant hotels |G|"), td("Grade 1 or 2 in the answer key"), td(G, "num"), td(G, "num")]),
    row([td("Relevant hotels in the top ten"), td("|R<sub>10</sub> ∩ G|"), td(hits(p), "num"), td(hits(f), "num")]),
    row([td("P@10"), td("|R<sub>10</sub> ∩ G| / 10"),
      td(`${hits(p)}/10 = ${fmt(p["P@10"])}`, "num"), td(`${hits(f)}/10 = ${fmt(f["P@10"])}`, "num")]),
    row([td("R@10"), td("|R<sub>10</sub> ∩ G| / |G|"),
      td(`${hits(p)}/${G} = ${fmt(p["R@10"])}`, "num"), td(`${hits(f)}/${G} = ${fmt(f["R@10"])}`, "num")]),
    row([td("nDCG@10"), td("DCG@10 / IDCG@10, gains 1 and 2"), td(fmt(p["nDCG@10"]), "num"), td(fmt(f["nDCG@10"]), "num")]),
    row([td("MRR"), td("1 / rank of the first relevant hotel"),
      td(`1/${first(p)} = ${fmt(p.MRR)}`, "num"), td(`1/${first(f)} = ${fmt(f.MRR)}`, "num")]),
  ]);
}

// --- step 4: Route 1 results -----------------------------------------------
function applyResults(data) {
  state.current = data;
  if (data.run !== "price") {
    // Steps 1 and 3 describe the main query set, so they only follow main-set runs.
    state.mainPerQuery = {};
    for (const r of data.per_query || []) state.mainPerQuery[r.id] = r;
    renderQuerysetTable();
  }
  const meta = data.gold_meta || {};
  document.getElementById("pillGold").textContent =
    "answer key · " + (meta.used_human ? `human (${meta.human_queries})` : "rule-based");
  document.getElementById("pillRun").textContent = `run · ${RUN_LABEL[data.run] || data.run}`;

  renderOverall(data);
  renderWinTieLoss(data);
  renderSignificance(data);
  renderCategory(data);
  renderDiagnostics(data);
  renderHumanWeights();
  populateExampleSelect(data.per_query || []);
}

function comparedIn(data) {
  return COMPARED.filter((n) => n in (data.overall || {}));
}

function renderOverall(data) {
  const order = comparedIn(data);
  const nd = (n) => (data.overall[n] || {})["nDCG@10"];
  const prop = nd(PROPOSED);
  const filt = nd("Filter");
  statCards("overallStats", [
    { key: "Methods compared", val: order.length },
    { key: "Queries", val: data.n_queries },
    { key: "Hotel pool", val: `${data.pool_size} hotels` },
    { key: "Proposed nDCG@10", val: fmt(prop) },
    { key: "Filter-and-sort nDCG@10", val: fmt(filt) },
    { key: "Gain over current practice", val: pct((prop - filt) / filt) },
  ]);

  const colBest = {};
  for (const k of METRIC_KEYS) colBest[k] = Math.max(...order.map((n) => (data.overall[n] || {})[k] || 0));

  fillBody("#overallTable", order.map((name) => {
    const m = data.overall[name] || {};
    const gain = name === PROPOSED ? "—" : pct((prop - m["nDCG@10"]) / m["nDCG@10"]);
    return row([
      td(label(name)),
      ...METRIC_KEYS.map((k) => td(fmt(m[k]), m[k] === colBest[k] ? "num col-best" : "num")),
      td(gain, "num"),
    ], name === PROPOSED ? "row-best" : "");
  }));
}

// Wins / ties / losses and the paired effect size d_z, recomputed from the
// per-query scores exactly as the thesis number guard does (population sd).
function renderWinTieLoss(data) {
  const pq = data.per_query || [];
  const others = comparedIn(data).filter((n) => n !== PROPOSED);
  const rows = others.map((b) => {
    const d = pq.map((q) => q.scores[PROPOSED]["nDCG@10"] - q.scores[b]["nDCG@10"]);
    const wins = d.filter((x) => x > 1e-9).length;
    const losses = d.filter((x) => x < -1e-9).length;
    const mean = d.reduce((a, x) => a + x, 0) / d.length;
    const sd = Math.sqrt(d.reduce((a, x) => a + (x - mean) ** 2, 0) / d.length);
    const base = data.overall[b]["nDCG@10"];
    return { b, wins, ties: d.length - wins - losses, losses, dz: sd ? mean / sd : 0,
      gain: (data.overall[PROPOSED]["nDCG@10"] - base) / base };
  }).sort((x, y) => x.dz - y.dz);
  fillBody("#wtlTable", rows.map((r) => row([
    td(label(r.b)), td(`${r.wins} / ${r.ties} / ${r.losses}`, "num"),
    td(fmt(r.dz, 2), "num"), td(pct(r.gain), "num"),
  ])));
}

function renderSignificance(data) {
  const block = document.getElementById("sigBlock");
  const sig = data.significance;
  if (!sig || !sig.vs || !Object.keys(sig.vs).length) {
    block.classList.add("hidden");
    return;
  }
  block.classList.remove("hidden");
  const order = (data.system_order || Object.keys(sig.vs))
    .filter((n) => n in sig.vs && !DIAGNOSTIC.includes(n));
  fillBody("#sigTable", order.map((name) => {
    const s = sig.vs[name];
    const verdict = s.significant
      ? (s.mean_diff > 0 ? `<span class="sig-win">Proposed better</span>` : `<span class="sig-loss">Proposed worse</span>`)
      : `<span class="sig-ns">Not significant</span>`;
    return row([
      td(`vs ${label(name)}`), td(signed(s.mean_diff), "num"),
      td(`[${signed(s.ci_low)}, ${signed(s.ci_high)}]`, "num"),
      td(s.p < 0.0001 ? "&lt; 0.0001" : fmt(s.p, 4), "num"),
      td(s.p_holm < 0.0001 ? "&lt; 0.0001" : fmt(s.p_holm, 4), "num"), td(verdict),
    ]);
  }));
}

function renderCategory(data) {
  const order = comparedIn(data);
  const cats = Object.keys(data.by_category || {}).sort();
  document.getElementById("catHeadRow").innerHTML =
    `<th>Method</th>` + cats.map((c) => `<th class="num">${CATEGORY_LABEL[c] || c}</th>`).join("");

  const bestByCat = {};
  for (const c of cats) {
    bestByCat[c] = Math.max(...order.map((n) => (data.by_category[c][n] || {})["nDCG@10"] || 0));
  }
  fillBody("#categoryTable", order.map((name) => row([
    td(label(name)),
    ...cats.map((c) => {
      const v = (data.by_category[c][name] || {})["nDCG@10"];
      return td(fmt(v), v === bestByCat[c] ? "num cat-best" : "num");
    }),
  ], name === PROPOSED ? "row-best" : "")));
}

function renderDiagnostics(data) {
  const names = DIAGNOSTIC.filter((n) => n in (data.overall || {}));
  document.getElementById("diagBlock").classList.toggle("hidden", !names.length);
  fillBody("#diagTable", names.map((name) => {
    const m = data.overall[name];
    return row([td(label(name)), ...METRIC_KEYS.map((k) => td(fmt(m[k]), "num"))], "row-diag");
  }));
}

// Thesis table "Study weights inside the ranking system": the study weights run inside the ranker, on the Route 1
// main set, price set and accessibility queries. Needs both saved thesis runs.
function renderHumanWeights() {
  const main = state.runs.thesis;
  const price = state.runs.price;
  const block = document.getElementById("humanWeightsBlock");
  if (!main || !price) { block.classList.add("hidden"); return; }
  const variants = [
    [PROPOSED, "Hand-set (0.25 / 0.20 / 0.25 / 0.15 / 0.15)"],
    ["WeightedGraphRAG[human]", "Whole study weights (0.473 / 0.026 / 0.111 / 0.253 / 0.137)"],
    ["WeightedGraphRAG[human_informed]", "Study price weight, all queries"],
    ["WeightedGraphRAG[human-price-aware]", "Study price weight, price-focused queries only"],
  ].filter(([n]) => n in main.overall && n in price.overall);
  block.classList.toggle("hidden", !variants.length);
  const acc = main.by_category.accessibility || {};
  fillBody("#humanWeightsTable", variants.map(([n, text]) => row([
    td(text), td(fmt(main.overall[n]["nDCG@10"]), "num"),
    td(fmt(price.overall[n]["nDCG@10"]), "num"), td(fmt((acc[n] || {})["nDCG@10"]), "num"),
  ], n === PROPOSED ? "row-best" : "")));
}

async function fetchRun(run) {
  if (!state.runs[run]) {
    const data = await request(`/eval/results?run=${run}`);
    if (data.available) state.runs[run] = data;
  }
  return state.runs[run];
}

async function loadResults(live = false) {
  const status = document.getElementById("runStatus");
  const select = document.getElementById("runSelect");
  const btns = [document.getElementById("loadResultsBtn"), document.getElementById("runLiveBtn")];
  btns.forEach((b) => (b.disabled = true));
  status.textContent = live
    ? "Running the harness against Neo4j and pgvector: 60 queries × every method. This takes a few minutes…"
    : "Loading the saved run…";
  try {
    let data;
    if (live) {
      data = await request("/eval/run", { method: "POST" });
      data.run = "latest";
      state.runs.latest = data;
      select.value = "latest";
    } else {
      await Promise.all(["thesis", "price"].map((r) => fetchRun(r).catch(() => null)));
      data = await fetchRun(select.value);
    }
    if (!data) {
      status.textContent = "That run has not been saved yet. Press “Run live” to generate it.";
      return;
    }
    applyResults(data);
    status.textContent =
      `${RUN_LABEL[data.run]} · ${data.n_queries} queries · ${data.pool_size} hotels · ` +
      `answer key: ${data.gold_meta?.used_human ? "human + rules" : "rule-based"} · ` +
      `file: ${data.source_file || "results.json"}` +
      (data.run === "latest" ? " · live data drifts slightly from the thesis run" : "");
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
  grid.innerHTML = `<p class="content-placeholder">Retrieving “${qid}” across all methods…</p>`;
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
    `<span class="pill">${CATEGORY_LABEL[data.category] || data.category}</span>` +
    `<span class="pill">answer key · ${data.gold_source}</span>` +
    `<span class="pill">${data.n_relevant} relevant / ${data.pool_size}</span>` +
    `</div>` +
    `<div class="ih-meta"><span class="muted" style="align-self:center">Weights after request adjustments:</span>${weights}</div>` +
    `<div class="ih-meta gold-cell"><span class="muted" style="align-self:center">Answer-key limits:</span> ${goldConstraints(data.gold)}</div>`;

  let winner = null, best = -1;
  for (const s of data.systems) {
    const v = s.metrics["nDCG@10"] || 0;
    if (v > best) { best = v; winner = s.name; }
  }
  const grid = document.getElementById("inspectGrid");
  grid.innerHTML = "";
  for (const s of data.systems) grid.appendChild(renderSystemColumn(s, s.name === winner));
}

function renderSystemColumn(sys, isWinner) {
  const col = el("div", "sys-col" + (isWinner ? " winner" : ""));
  col.appendChild(el("h3", null, label(sys.name)));
  const badges = el("div", "metric-badges");
  badges.innerHTML = METRIC_KEYS
    .map((k) => `<span class="badge">${k} <b>${fmt(sys.metrics[k], 3)}</b></span>`)
    .join("");
  col.appendChild(badges);
  const list = el("div", "rank-list");
  if (!sys.ranked.length) list.appendChild(el("p", "muted", "No results."));
  for (const item of sys.ranked) list.appendChild(renderRankItem(item));
  col.appendChild(list);
  return col;
}

function renderRankItem(item) {
  const node = el("div", "rank-item" + (item.relevant ? " hit" : ""));
  const top = el("div", "rank-row");
  top.innerHTML =
    `<span class="rank-num">#${item.rank}</span>` +
    `<span class="rank-name">${item.name}</span>` +
    (item.relevant ? `<span class="hit-dot" title="relevant (answer key)"></span>` : "") +
    (item.score !== undefined ? `<span class="g-score">${fmt(item.score, 3)}</span>` : "");
  node.appendChild(top);

  const attrs = el("div", "rank-attrs");
  attrs.innerHTML =
    `<span>${money(item.price_lkr)}</span>` +
    `<span>★${fmt(item.rating, 1)}</span>` +
    `<span>${item.star ?? "—"}-star</span>` +
    `<span>${item.travel_time_min != null ? fmt(item.travel_time_min, 0) + " min" : "—"}</span>`;
  node.appendChild(attrs);

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

// --- step 6: Route 2 choice study ------------------------------------------
//
// Ground truth is the hotel each held-out participant chose. The thesis
// reports three scoring-model rows (hand-set, whole study weights, fitted on
// the 67); the other weight profiles the harness also scores follow them,
// de-emphasised, so nothing in the saved run is hidden.

const THESIS_HUMAN_ROWS = ["Filter", "Keyword", "SemanticVec", "Hybrid",
  "WeightedGraphRAG[handset]", "WeightedGraphRAG[human]", "WeightedGraphRAG[fitted-on-train]"];

function humanOrder(data) {
  const main = THESIS_HUMAN_ROWS.filter((n) => data.system_order.includes(n));
  const extra = data.system_order.filter((n) => !main.includes(n));
  return { main, extra };
}

function humanRow(name, m, keys, best, extra) {
  const cls = [name.startsWith("WeightedGraphRAG") ? "row-graph" : "", extra ? "row-extra" : ""].join(" ");
  const text = name === "WeightedGraphRAG[human]" ? `${label(name)} <span class="muted">(not a clean test)</span>` : label(name);
  return row([
    td(text, "sys-name"),
    ...keys.map((k) => {
      const digits = k === "mean_rank" ? 2 : 3;
      const v = m[k];
      return td(v === undefined ? "—" : fmt(v, digits), best[k] === name ? "num col-best" : "num");
    }),
  ], cls);
}

function bestOf(names, section, key) {
  let best = null;
  for (const n of names) {
    const v = (section[n] || {})[key];
    if (v === undefined) continue;
    if (best === null || (key === "mean_rank" ? v < best[1] : v > best[1])) best = [n, v];
  }
  return best ? best[0] : null;
}

function renderHumanTable(tableId, data, section, keys) {
  const { main, extra } = humanOrder(data);
  const best = {};
  keys.forEach((k) => { best[k] = bestOf(main, data[section], k); });
  const rows = [];
  if (section === "per_choice" && data.random_baseline) {
    const rb = data.random_baseline;
    rows.push(row([td("Random order", "sys-name"),
      ...keys.map((k) => td(rb[k] === undefined ? "—" : fmt(rb[k], k === "mean_rank" ? 2 : 3), "num"))], "row-random"));
  }
  main.forEach((n) => rows.push(humanRow(n, data[section][n] || {}, keys, best, false)));
  extra.forEach((n) => rows.push(humanRow(n, data[section][n] || {}, keys, {}, true)));
  fillBody(`#${tableId}`, rows);
}

function renderHumanSignificance(data) {
  const sig = data.significance || {};
  document.getElementById("humanSigRef").textContent =
    `(reference: ${label(sig.reference || "—")}, ${sig.metric || ""})`;
  fillBody("#humanSigTable", Object.entries(sig.vs || {}).map(([name, b]) => row([
    td(label(name), "sys-name"), td(signed(b.mean_diff), "num"),
    td(`[${signed(b.ci_low)}, ${signed(b.ci_high)}]`, "num"),
    td(b.p < 0.001 ? "&lt; 0.001" : fmt(b.p), "num"),
    td(b.significant ? "significant" : "not significant", b.significant ? "sig-win" : "sig-ns"),
  ])));
}

function renderHumanTaskList(data) {
  const body = document.querySelector("#humanTaskList tbody");
  body.innerHTML = "";
  for (const t of data.tasks || []) {
    const tr = row([
      td(t.id, "sys-name"), td(t.persona || "—"), td(t.anchor || "—"),
      td(t.primary_dimension || "—"), td(String(t.n_test_choices ?? 0), "num"),
      td(String(t.gold_size ?? 0), "num"),
      td(t.top_pick ? `${t.top_pick.hotel} <span class="muted">(${t.top_pick.votes})</span>` : "—"),
    ]);
    tr.title = t.context || "";
    body.appendChild(tr);
  }
}

function renderHumanStats(data) {
  const c = data.cohort_stats || {};
  const hand = (data.per_choice || {})["WeightedGraphRAG[handset]"] || {};
  statCards("humanStats", [
    { key: "Recruited", val: c.participants_total ?? "—" },
    { key: "Failed attention / too fast", val: `${c.dropped_attention ?? "—"} / ${c.dropped_speeder ?? "—"}` },
    { key: "Kept", val: c.participants_kept ?? "—" },
    { key: "Train / test people", val: `${data.n_train_participants ?? "—"} / ${data.n_test_participants ?? "—"}` },
    { key: "Test choices", val: data.n_test_choices ?? "—" },
    { key: "Hit@10, hand-set", val: hand["R@10"] !== undefined ? `${(hand["R@10"] * 100).toFixed(1)}%` : "—" },
  ]);
}

function applyHumanResults(data) {
  renderHumanStats(data);
  renderHumanTable("humanChoiceTable", data, "per_choice", ["nDCG@10", "R@10", "MRR", "mean_rank"]);
  renderHumanTable("humanTaskTable", data, "per_task", ["P@10", "R@10", "nDCG@10", "gradedNDCG@10", "MRR"]);
  renderHumanSignificance(data);
  renderHumanTaskList(data);
  document.getElementById("humanStatus").textContent =
    `${data.n_observations} usable choices · cohort “${data.cohort}” · ` +
    `${data.candidate_set_size} hotels per task · study material ${data.material_version}`;
}

function renderSeeds(s) {
  const sys = ["handset", "human", "fitted-on-train"];
  const ms = (block) => sys.map((k) => td(`${fmt(block[k].mean)} ± ${fmt(block[k].sd)}`, "num"));
  const n = s.seeds;
  const humanLower = Math.round((1 - s.human_beats_handset_share) * n);
  const fittedHigher = Math.round(s.fitted_beats_handset_share * n);
  fillBody("#seedsTable", [
    row([td("Choice nDCG@10"), ...ms(s.choice_ndcg)]),
    row([td("Difference from hand-set"), td("—", "num"),
      td(`${signed(s.human_minus_handset_choice_ndcg.mean)}; lower in ${humanLower} of ${n}`, "num"),
      td(`${signed(s.fitted_minus_handset_choice_ndcg.mean)}; higher in ${fittedHigher} of ${n}`, "num")]),
    row([td("Per-task graded nDCG@10"), ...ms(s.task_graded_ndcg)]),
    row([td("Average rank of the chosen hotel"), ...sys.map((k) =>
      td(`${fmt(s.mean_rank[k].mean, 2)} ± ${fmt(s.mean_rank[k].sd, 2)}`, "num"))]),
  ]);
  const fw = s.fitted_weights;
  const zero = CRITERIA.filter((k) => s.fitted_weight_exactly_zero_share[k] === 1);
  document.getElementById("seedsNote").innerHTML =
    `${n} random splits, ${Math.round(s.holdout * 100)}% of the ${s.participants} attentive people held out each time. ` +
    `Mean fitted weights: ` + CRITERIA.map((k) => `${k} ${fmt(fw[k].mean)}`).join(" · ") +
    (zero.length ? `. ${zero.join(" and ")} received exactly zero weight in every split.` : ".");
}

function renderWeights(w) {
  const strata = Object.keys(w.weights_per_stratum || {});
  const sizes = w.strata_sizes || {};
  const name = { distance: "Distance-sorted", travel: "Travel-sorted", price: "Price-sorted", rating: "Rating-sorted" };
  document.getElementById("weightsHead").innerHTML =
    `<th>Criterion</th><th class="num">Hand-set</th>` +
    strata.map((s) => `<th class="num">${name[s] || s} (${sizes[s] ?? "?"})</th>`).join("") +
    `<th class="num">Estimated weight [95% range]</th><th class="num">Placebo p</th>`;
  const dims = (w.gates || {}).dimensions || {};
  fillBody("#weightsTable", CRITERIA.map((k) => {
    const ci = (w.bootstrap_ci_95 || {})[k] || [];
    return row([
      td(k[0].toUpperCase() + k.slice(1)), td(fmt(HANDSET[k], 2), "num"),
      ...strata.map((s) => td(fmt(w.weights_per_stratum[s][k]), "num")),
      td(`${fmt(w.weights[k])} [${fmt(ci[0])}, ${fmt(ci[1])}]`, "num"),
      td(fmt((dims[k] || {}).placebo_p), "num"),
    ]);
  }));

  const mark = (ok) => (ok ? `<span class="sig-win">pass</span>` : `<span class="sig-loss">fail</span>`);
  const gateKeys = ["G1_placebo", "G2_likelihood_ratio", "G3_interval", "G4_held_out", "G5_stability"];
  fillBody("#gatesTable", CRITERIA.map((k) => {
    const d = dims[k] || {};
    const checks = d.checks || {};
    return row([
      td(k[0].toUpperCase() + k.slice(1)), ...gateKeys.map((g) => td(mark(checks[g]), "num")),
      td(d.identified ? `<span class="sig-win">usable</span>` : `<span class="sig-ns">not usable</span>`),
    ]);
  }));
  const lr = (w.gates || {}).likelihood_ratio || {};
  const ho = (w.gates || {}).held_out || {};
  document.getElementById("gatesNote").innerHTML =
    `Likelihood ratio χ² = ${fmt(lr.lr, 2)}, ${lr.df} df, p = ${fmt(lr.p)}. ` +
    `Held-out change in log-likelihood per choice ${signed(ho.delta_ll_per_choice, 5)} ` +
    `[${signed((ho.delta_ci95 || [])[0], 4)}, ${signed((ho.delta_ci95 || [])[1], 4)}]. ` +
    `${w.gates?.n_choices ?? ""} choices from ${w.gates?.n_participants ?? ""} participants; ` +
    `${w.cohort?.held_out_participants ?? "—"} people held out. ` +
    `Overall: ${w.shippable ? "weights usable" : "no criterion confirmed, so the weights are refused"}.`;
}

async function loadHumanResults(live = false) {
  const status = document.getElementById("humanStatus");
  const buttons = [document.getElementById("loadHumanBtn"), document.getElementById("runHumanBtn")];
  buttons.forEach((b) => { b.disabled = true; });
  status.textContent = live
    ? "Running the choice-based evaluation (fitting weights on the training people, scoring the test people)…"
    : "Loading saved results…";
  try {
    const data = live
      ? await request("/eval/human/run", { method: "POST" })
      : await request("/eval/human/results");
    if (!data.available) {
      status.textContent = "No saved run yet. Press “Run live” to evaluate against the study data.";
      return;
    }
    applyHumanResults(data);
    if (live) toast("Choice-based evaluation complete");
  } catch (err) {
    status.textContent = `Could not load: ${err.message}`;
    toast(err.message, true);
  } finally {
    buttons.forEach((b) => { b.disabled = false; });
  }
  // The 50-split table and the weight gates come from their own saved files.
  try {
    const seeds = await request("/eval/human/seeds");
    if (seeds.available) renderSeeds(seeds);
  } catch (_) { /* table stays empty */ }
  try {
    const weights = await request("/eval/weights");
    if (weights.available) renderWeights(weights);
  } catch (_) { /* table stays empty */ }
}

// --- init ------------------------------------------------------------------
async function init() {
  setupStepper();
  document.getElementById("loadResultsBtn").addEventListener("click", () => loadResults(false));
  document.getElementById("runSelect").addEventListener("change", () => loadResults(false));
  document.getElementById("runLiveBtn").addEventListener("click", () => loadResults(true));
  document.getElementById("exampleSelect").addEventListener("change", renderExample);
  document.getElementById("inspectBtn").addEventListener("click", inspect);
  document.getElementById("loadHumanBtn").addEventListener("click", () => loadHumanResults(false));
  document.getElementById("runHumanBtn").addEventListener("click", () => loadHumanResults(true));

  try {
    await loadQueryset();
  } catch (err) {
    toast(`Could not load query set: ${err.message}`, true);
  }
  // Fill every step from the saved thesis runs straight away.
  await loadResults(false);
  await loadHumanResults(false);
}

init();

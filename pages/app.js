"use strict";
// ResearchRAG static showcase: renders answers precomputed by the real pipeline (data/demo.json).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TYPES = { single_hop: "single-hop", multi_hop: "multi-hop", numeric_table: "numeric / table", unanswerable: "unanswerable", adversarial: "adversarial" };
let DATA = null;
let activeType = "all";
let current = null;

function citeSpans(text, citations) {
  const by = new Map(citations.map((c) => [c.marker, c]));
  return esc(text).replace(/\[(\d{1,3})\]/g, (m, n) => {
    const c = by.get(Number(n));
    if (!c) return m;
    return `<span class="cite${c.quote_verified ? "" : " unverified"}" tabindex="0" role="button" data-marker="${n}" title="${esc(c.title)}, p.${c.page}">[${n}]</span>`;
  });
}

function renderResult(r, opts = {}) {
  const claims = r.claims.length
    ? `<h3>Claims and verification</h3><ul class="claims">${r.claims.map((c) => `
        <li class="${c.removed ? "removed" : ""}"><span class="verdict v-${c.verdict}">${c.verdict.replace("_", " ")}</span>${esc(c.text)}
        ${c.citations.map((n) => `<span class="cite" tabindex="0" role="button" data-marker="${n}">[${n}]</span>`).join("")}
        ${c.flags.length ? `<span class="pill">${esc(c.flags.join(", "))}</span>` : ""}
        ${c.removed ? `<span class="pill">removed</span>` : ""}</li>`).join("")}</ul>`
    : "";
  const body = r.abstained
    ? `<div class="abstain"><b>Not enough evidence.</b> ${esc(r.abstain_reason)}<br><span class="muted">Searched ${r.searched.passages} passage(s) from: ${esc(r.searched.documents.join("; ") || "no matching papers")}</span></div>`
    : `<div class="answer">${citeSpans(r.answer, r.citations)}</div>`;
  const warnings = r.warnings.length ? `<p class="warn">${r.warnings.map(esc).join(" · ")}</p>` : "";
  const faith = r.faithfulness == null ? "n/a" : r.faithfulness.toFixed(2);
  return `${opts.header || ""}${body}${warnings}${claims}
    <p class="metrics">faithfulness ${faith} · evidence score ${r.searched.evidence_score} · ${r.latency_ms} ms (in-process)</p>`;
}

function showSource(marker) {
  if (!current) return;
  const c = current.citations.find((x) => x.marker === Number(marker));
  const target = current.sourceEl || $("source");
  if (!c) return;
  let passage = esc(c.passage);
  if (c.quote && c.quote_verified) {
    const q = esc(c.quote);
    const i = passage.toLowerCase().indexOf(q.toLowerCase());
    if (i >= 0) passage = `${passage.slice(0, i)}<mark>${passage.slice(i, i + q.length)}</mark>${passage.slice(i + q.length)}`;
  }
  const pdf = c.doc_key ? `<a href="papers/${encodeURIComponent(c.doc_key)}.pdf#page=${c.page}" target="_blank" rel="noopener">Open the PDF at page ${c.page}</a>` : "";
  target.innerHTML = `<h2>Source [${c.marker}]</h2><b>${esc(c.title)}</b><br>
    <span class="muted">page ${c.page} · ${esc(c.section)} · quote ${c.quote_verified ? "verified in passage" : "<span class='v-contradicted'>not found in passage</span>"}</span>
    <blockquote>${passage}</blockquote>${pdf}`;
}

function renderList() {
  const term = $("search").value.trim().toLowerCase();
  const items = DATA.questions.filter((q) => (activeType === "all" || q.type === activeType) && (!term || q.question.toLowerCase().includes(term)));
  $("qlist").innerHTML = items.map((q) => `<li data-id="${q.id}" class="${current && current.id === q.id ? "on" : ""}">${esc(q.question)}
    <span class="meta">${TYPES[q.type]} · ${q.abstained ? "abstained" : "answered"}</span></li>`).join("") || `<li class="muted">No question matches. The live API answers any question; this static page only has precomputed ones.</li>`;
}

function selectQuestion(id) {
  const q = DATA.questions.find((x) => x.id === id);
  if (!q) return;
  current = { ...q, sourceEl: $("source") };
  const expected = q.answerable ? "answerable" : "unanswerable: should abstain";
  const correctAbstain = q.abstained === !q.answerable;
  $("answer-panel").innerHTML = renderResult(q, {
    header: `<p class="q">${esc(q.question)}</p>
      <span class="pill">${TYPES[q.type]}</span><span class="pill">${expected}</span>
      <span class="pill">${correctAbstain ? "abstention decision correct" : q.answerable ? "over-abstained" : "should have abstained"}</span>
      <div class="gold"><b>Gold answer:</b> ${esc(q.gold_answer)}</div>`,
  });
  $("source").innerHTML = `<h2>Source</h2><p class="muted">Click a citation such as [1] to see the passage, the paper and the page.</p>`;
  if (q.citations.length) showSource(q.citations[0].marker);
  renderList();
  history.replaceState(null, "", `#q=${id}`);
}

function renderScenarios() {
  $("slist").innerHTML = DATA.scenarios.map((s, i) => `<li data-idx="${i}">${esc(s.title)}</li>`).join("");
}

function selectScenario(i) {
  const s = DATA.scenarios[i];
  const panel = $("scenario-panel");
  current = { ...s, sourceEl: null };
  panel.innerHTML = `<h2>${esc(s.title)}</h2><p class="muted">${esc(s.description)}</p><p class="q">${esc(s.question)}</p>
    ${renderResult(s)}<div id="scenario-source" class="source"></div>`;
  current.sourceEl = $("scenario-source");
  if (s.citations.length) showSource(s.citations[0].marker);
  [...$("slist").children].forEach((li, j) => li.classList.toggle("on", j === i));
}

function renderEval() {
  const e = DATA.evaluation;
  const rows = Object.entries(e.summary).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.mean.toFixed(3)}</td><td>[${v.low.toFixed(3)}, ${v.high.toFixed(3)}]</td><td>${v.n}</td></tr>`).join("");
  $("eval-panel").innerHTML = `<h2>Offline evaluation (${e.n} questions)</h2>
    <p class="muted">${esc(DATA.components)}. Means with 95% bootstrap confidence intervals, recomputed by the Pages build for commit ${esc(DATA.commit)}.</p>
    <table><thead><tr><th>metric</th><th>mean</th><th>95% CI</th><th>n</th></tr></thead><tbody>${rows}</tbody></table>
    <p>Latency per query (in-process): p50 ${e.latency_ms.p50} ms, p95 ${e.latency_ms.p95} ms. Cost: $0.</p>
    <p><a href="reports/eval.html">Full evaluation report</a> (worst questions with retrieved passages) ·
      <a href="reports/sweep.html">Ablation sweep</a> · <a href="reports/sweep-hnsw.html">HNSW sweep on PostgreSQL</a> ·
      <a href="https://github.com/PiyushDas-2004/ResearchRAG/blob/main/docs/EVALUATION.md">How to read these numbers</a></p>`;
}

function switchTab(name) {
  document.querySelectorAll("[role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  ["ask", "scenarios", "eval", "about"].forEach((t) => { $(`tab-${t}`).hidden = t !== name; });
  if (name === "scenarios" && DATA.scenarios.length && !$("scenario-panel").innerHTML) selectScenario(0);
  if (name === "ask" && current && current.id) current = { ...DATA.questions.find((q) => q.id === current.id), sourceEl: $("source") };
}

async function init() {
  const res = await fetch("data/demo.json");
  DATA = await res.json();
  const types = ["all", ...Object.keys(TYPES)];
  $("chips").innerHTML = types.map((t) => `<button data-type="${t}" class="${t === "all" ? "on" : ""}">${t === "all" ? "all" : TYPES[t]}</button>`).join("");
  $("build-info").textContent = `Built from commit ${DATA.commit} at ${DATA.generated_at}; config ${DATA.config}.`;
  renderList();
  renderScenarios();
  renderEval();
  const fromHash = (location.hash.match(/q=([\w-]+)/) || [])[1];
  selectQuestion(fromHash || "clim-q05");
  if (!current) selectQuestion(DATA.questions[0].id);

  $("search").addEventListener("input", renderList);
  $("chips").addEventListener("click", (e) => {
    const t = e.target.dataset && e.target.dataset.type;
    if (!t) return;
    activeType = t;
    [...$("chips").children].forEach((b) => b.classList.toggle("on", b.dataset.type === t));
    renderList();
  });
  $("qlist").addEventListener("click", (e) => { const li = e.target.closest("li[data-id]"); if (li) selectQuestion(li.dataset.id); });
  $("slist").addEventListener("click", (e) => { const li = e.target.closest("li[data-idx]"); if (li) selectScenario(Number(li.dataset.idx)); });
  document.querySelector(".tabs").addEventListener("click", (e) => { const t = e.target.dataset && e.target.dataset.tab; if (t) switchTab(t); });
  const onCite = (e) => { const m = e.target.dataset && e.target.dataset.marker; if (m) showSource(m); };
  document.body.addEventListener("click", onCite);
  document.body.addEventListener("keydown", (e) => { if (e.key === "Enter") onCite(e); });
}

init().catch((err) => { $("answer-panel").textContent = `Could not load demo data: ${err}`; });

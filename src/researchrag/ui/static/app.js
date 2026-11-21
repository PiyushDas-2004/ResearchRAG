"use strict";
// ResearchRAG demo UI: vanilla JS, no build step. Talks to the JSON/SSE API.
const $ = (id) => document.getElementById(id);
const state = { conversationId: null, traceId: null, citations: new Map() };

function apiKey() { try { return localStorage.getItem("rr-api-key") || ""; } catch { return ""; } }
function headers(extra = {}) { const k = apiKey(); return k ? { ...extra, "X-API-Key": k } : extra; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
async function problemText(res) {
  try { const p = await res.json(); return `${p.title || res.status}: ${p.detail || ""}`; } catch { return `HTTP ${res.status}`; }
}

async function loadDocuments() {
  const res = await fetch("/v1/documents?limit=100", { headers: headers() });
  const list = $("documents");
  if (!res.ok) { list.innerHTML = `<li class="muted">${escapeHtml(await problemText(res))}</li>`; return; }
  const page = await res.json();
  list.innerHTML = page.items.length ? page.items.map((d) =>
    `<li><b>${escapeHtml(d.title || d.filename)}</b><br><span class="muted">${escapeHtml((d.authors || []).slice(0, 3).join(", "))}${d.year ? " · " + d.year : ""} · ${d.page_count} p.</span></li>`
  ).join("") : `<li class="muted">No papers yet. Upload a PDF.</li>`;
}

async function upload(ev) {
  ev.preventDefault();
  const file = $("file").files[0];
  if (!file) return;
  const status = $("upload-status");
  const body = new FormData();
  body.append("file", file);
  status.textContent = "Uploading…";
  const res = await fetch("/v1/documents", { method: "POST", body, headers: headers() });
  if (!res.ok) { status.textContent = await problemText(res); return; }
  const { job } = await res.json();
  for (let i = 0; i < 120; i++) {
    const jr = await fetch(`/v1/jobs/${job.id}`, { headers: headers() });
    const j = await jr.json();
    status.textContent = `Job ${j.status} (${j.stage || "queued"}, ${Math.round(j.progress * 100)}%)`;
    if (j.status === "succeeded") { status.textContent = "Ingested."; loadDocuments(); return; }
    if (j.status === "failed") { status.textContent = `Failed: ${j.error ? j.error.message : "unknown error"}`; return; }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

function renderAnswer(text, citations) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  return escapeHtml(text).replace(/\[(\d{1,3})\]/g, (m, n) => {
    const c = byMarker.get(Number(n));
    if (!c) return m;
    const cls = c.quote_verified ? "cite" : "cite unverified";
    return `<span class="${cls}" tabindex="0" role="button" data-marker="${n}" title="${escapeHtml(c.title)}, p.${c.page}">[${n}]</span>`;
  });
}

function showSource(marker) {
  const c = state.citations.get(Number(marker));
  if (!c) return;
  let passage = escapeHtml(c.passage || "");
  if (c.quote && c.quote_verified) {
    const q = escapeHtml(c.quote);
    const idx = passage.toLowerCase().indexOf(q.toLowerCase());
    if (idx >= 0) passage = passage.slice(0, idx) + "<mark>" + passage.slice(idx, idx + q.length) + "</mark>" + passage.slice(idx + q.length);
  }
  $("source").classList.remove("muted");
  $("source").innerHTML = `<b>[${c.marker}] ${escapeHtml(c.title)}</b><br>Page ${c.page} · ${escapeHtml(c.section)} · quote ${c.quote_verified ? "verified" : "<span class='v-contradicted'>not found in passage</span>"}<blockquote>${passage}</blockquote>`;
}

function renderFinal(r) {
  state.traceId = r.trace_id;
  state.citations = new Map(r.citations.map((c) => [c.marker, c]));
  $("stage").textContent = "";
  if (r.abstained) {
    $("answer-text").innerHTML = "";
    const a = $("abstain");
    a.hidden = false;
    const searched = r.searched || {};
    a.innerHTML = `<b>Not enough evidence.</b> ${escapeHtml(r.abstain_reason || "")}<br><span class="muted">Searched ${searched.passages || 0} passage(s) from: ${escapeHtml((searched.documents || []).join("; ") || "no matching papers")}</span>`;
  } else {
    $("abstain").hidden = true;
    $("answer-text").innerHTML = renderAnswer(r.answer, r.citations);
  }
  const w = $("warnings");
  w.hidden = !r.warnings.length;
  w.textContent = r.warnings.join(" · ");
  $("claims-title").hidden = !r.claims.length;
  $("claims").innerHTML = r.claims.map((c) =>
    `<li class="${c.removed ? "removed" : ""}"><span class="verdict v-${c.verdict}">${c.verdict.replace("_", " ")}</span>${escapeHtml(c.text)} ${c.citations.map((n) => `<span class="cite" tabindex="0" role="button" data-marker="${n}">[${n}]</span>`).join("")}${c.flags.length ? ` <span class="muted">(${escapeHtml(c.flags.join(", "))})</span>` : ""}</li>`
  ).join("");
  const faith = r.faithfulness == null ? "n/a" : r.faithfulness.toFixed(2);
  $("meta").textContent = `faithfulness ${faith} · ${Math.round(r.latency_ms)} ms · trace ${r.trace_id.slice(0, 8)}`;
  $("feedback").hidden = false;
}

async function ask(ev) {
  ev.preventDefault();
  const question = $("question").value.trim();
  if (question.length < 3) return;
  if (!$("followup").checked || !state.conversationId) state.conversationId = "ui-" + Math.random().toString(36).slice(2, 12);
  $("answer").hidden = false; $("abstain").hidden = true; $("warnings").hidden = true; $("feedback").hidden = true;
  $("answer-text").textContent = ""; $("claims").innerHTML = ""; $("claims-title").hidden = true; $("meta").textContent = "";
  $("stage").textContent = "Searching…"; $("ask").disabled = true;
  try {
    const res = await fetch("/v1/query/stream", {
      method: "POST", headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question, conversation_id: state.conversationId }),
    });
    if (!res.ok) { $("stage").textContent = await problemText(res); return; }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, sep); buffer = buffer.slice(sep + 2);
        const event = (raw.match(/^event: (.*)$/m) || [])[1];
        const dataLine = (raw.match(/^data: (.*)$/m) || [])[1];
        if (!event || !dataLine) continue;
        const data = JSON.parse(dataLine);
        if (event === "status") $("stage").textContent = `Stage: ${data.stage}…`;
        else if (event === "token") $("answer-text").textContent += data.text;
        else if (event === "final") renderFinal(data);
        else if (event === "error") $("stage").textContent = `Error: ${data.detail}`;
      }
    }
  } finally { $("ask").disabled = false; }
}

async function sendFeedback(rating) {
  if (!state.traceId) return;
  const res = await fetch("/v1/feedback", {
    method: "POST", headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ trace_id: state.traceId, rating }),
  });
  $("feedback").textContent = res.ok ? "Thanks for the feedback." : await problemText(res);
}

document.addEventListener("DOMContentLoaded", () => {
  $("api-key").value = apiKey();
  $("api-key").addEventListener("change", (e) => { try { localStorage.setItem("rr-api-key", e.target.value); } catch { /* storage unavailable */ } loadDocuments(); });
  $("upload-form").addEventListener("submit", upload);
  $("ask-form").addEventListener("submit", ask);
  $("refresh").addEventListener("click", loadDocuments);
  $("feedback").addEventListener("click", (e) => { const r = e.target.dataset && e.target.dataset.rating; if (r) sendFeedback(r); });
  const onCite = (e) => { const m = e.target.dataset && e.target.dataset.marker; if (m) showSource(m); };
  document.body.addEventListener("click", onCite);
  document.body.addEventListener("keydown", (e) => { if (e.key === "Enter") onCite(e); });
  loadDocuments();
});

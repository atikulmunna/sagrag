// DOM builders. Every piece of server or user text is inserted as a text node,
// never as HTML.

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (name === "class") node.className = value;
    // CSSOM write: allowed by the strict CSP, unlike a style attribute.
    else if (name === "style") node.style.cssText = value;
    else if (name === "dataset") Object.assign(node.dataset, value);
    else if (name.startsWith("on") && typeof value === "function") node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const number = new Intl.NumberFormat();
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 0 });
const when = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

export const fmt = {
  int: (n) => number.format(n),
  pct: (n) => percent.format(n),
  date: (seconds) => (Number.isFinite(seconds) ? when.format(new Date(seconds * 1000)) : ""),
  ms: (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`),
};

// Plain-language explanations for the backend's failure tags.
const ISSUES = {
  no_domain: ["low", "No domain matched the question, so every collection was searched."],
  policy_blocked: ["high", "Every retrieved passage was removed by a content policy."],
  cross_domain_conflict: ["low", "Passages came from more than one domain."],
  no_results: ["high", "No passages were found for this question."],
  low_result_count: ["low", "Only a few passages were found, so the answer rests on little evidence."],
  low_top_score: ["high", "Even the best passage was a weak match for the question."],
  author_gap: ["low", "No passages by the named author mention these terms, so other sources were used."],
  synthesis_timeout: ["high", "The language model took too long; the answer was assembled from the passages instead."],
  synthesis_error: ["high", "The language model failed; the answer was assembled from the passages instead."],
  unsupported_claims: ["low", "Some sentences are not backed by the cited passages."],
  contradicted_claims: ["high", "A cited passage contradicts part of the answer."],
  abstained: ["high", "Too little of the answer was backed by the passages, so it was withheld."],
};
const RETRIEVERS = {
  vector: "Semantic search",
  lexical: "Keyword search",
  structured: "Table and field search",
  lexical_author: "Author search",
};

export function describeIssue(tag) {
  if (ISSUES[tag]) return ISSUES[tag];
  const m = /^(vector|lexical|structured|lexical_author)_(timeout|error|zero_hits)$/.exec(tag);
  if (m) {
    const name = RETRIEVERS[m[1]];
    if (m[2] === "timeout") return ["high", `${name} timed out.`];
    if (m[2] === "error") return ["high", `${name} failed. Check that its service is running.`];
    return ["low", `${name} found nothing.`];
  }
  return ["low", tag.replace(/_/g, " ")];
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const STOP = new Set(
  "a an and are as at be by does do for from how i in is it its of on or say says the this to was what when where which who why with about".split(" "),
);

export function queryTerms(question) {
  return [...new Set((question.toLowerCase().match(/[a-z][a-z'-]{2,}/g) || []).filter((t) => !STOP.has(t)))];
}

// Text with the query's terms wrapped in <mark>, as a fragment of text nodes.
export function highlighted(text, terms) {
  const frag = document.createDocumentFragment();
  if (!terms.length) {
    frag.append(text);
    return frag;
  }
  const re = new RegExp(`\\b(${terms.map(escapeRegExp).join("|")})\\w*`, "gi");
  let last = 0;
  for (const match of text.matchAll(re)) {
    frag.append(text.slice(last, match.index), el("mark", {}, match[0]));
    last = match.index + match[0].length;
  }
  frag.append(text.slice(last));
  return frag;
}

// Character ranges of the verifier's supporting sentences inside a passage
// (whitespace-tolerant, since premise units are re-joined with single spaces).
export function supportRanges(text, supportTexts) {
  const ranges = [];
  for (const s of supportTexts) {
    const pattern = s.trim().split(/\s+/).map(escapeRegExp).join("\\s+");
    if (!pattern) continue;
    const m = new RegExp(pattern).exec(text);
    if (m) ranges.push([m.index, m.index + m[0].length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push(r);
  }
  return merged;
}

function markedPassage(text, terms, ranges) {
  const frag = document.createDocumentFragment();
  let pos = 0;
  for (const [start, end] of ranges) {
    frag.append(highlighted(text.slice(pos, start), terms), el("mark", { class: "support", title: "Supports a sentence in the answer" }, text.slice(start, end)));
    pos = end;
  }
  frag.append(highlighted(text.slice(pos), terms));
  return frag;
}

function band(value) {
  if (value >= 0.7) return "high";
  if (value >= 0.4) return "mid";
  return "low";
}

function meter(label, value, invert = false) {
  const shown = Math.max(0, Math.min(1, value));
  return el(
    "div",
    { class: "score" },
    el("span", { class: "score-label" }, label),
    el("span", { class: "score-value" }, fmt.pct(shown)),
    el(
      "span",
      { class: "meter", role: "img", "aria-label": `${label} ${fmt.pct(shown)}`, dataset: { band: band(invert ? 1 - shown : shown) } },
      el("span", { style: `width: ${(shown * 100).toFixed(0)}%` }),
    ),
  );
}

// A swatch and its label wrap together as one unit.
function legendItem(status, label) {
  return el("span", { class: "legend-item" }, el("span", { class: "swatch", dataset: { status } }), label);
}

export function renderScores(container, response) {
  const cited = (response.provenance || []).length;
  container.replaceChildren(
    meter("Confidence", Number(response.confidence) || 0),
    meter(
      response.hallucination_risk_method === "nli" ? "Sentences not supported" : "Risk of unsupported claims (estimate)",
      Number(response.hallucination_risk) || 0,
      true,
    ),
    el(
      "div",
      { class: "score" },
      el("span", { class: "score-label" }, "Evidence"),
      el("span", { class: "score-value" }, cited === 1 ? "1 cited passage" : `${cited} cited passages`),
    ),
  );
  const claims = Array.isArray(response.claims) ? response.claims : [];
  if (claims.length) {
    container.append(
      el(
        "p",
        { class: "claim-legend" },
        legendItem("supported", "Found in a cited passage"),
        legendItem("unsupported", "Not found in the cited passages"),
        claims.some((c) => c.status === "contradicted") ? legendItem("contradicted", "Contradicted by a cited passage") : null,
      ),
    );
  }
}

function claimNodes(claims, citeIndex) {
  const parts = [];
  claims.forEach((claim, i) => {
    if (i) parts.push(" ");
    const label = { supported: "Supported by a cited passage", unsupported: "Not found in the cited passages", contradicted: "Contradicted by a cited passage" }[claim.status] || "";
    const span = el("span", { class: "claim", title: label, dataset: { status: claim.status || "unknown" } }, claim.text);
    parts.push(span);
    const n = citeIndex.get(claim.support_id);
    if (n) parts.push(el("a", { class: "cite", href: `#note-${n}`, "aria-label": `Source ${n}` }, String(n)));
  });
  return parts;
}

// The answer, with per-sentence support marks when the backend verified claims.
export function renderAnswer(container, response, citeIndex) {
  const claims = Array.isArray(response.claims) && response.claims.length ? response.claims : null;
  if (response.abstained) {
    // Withheld for lack of support: show why, keep the draft one click away.
    const draft = el("div", { class: "draft" }, ...(claims ? claimNodes(claims, citeIndex) : [response.unverified_answer || ""]));
    container.replaceChildren(
      response.answer || "",
      el("details", { class: "draft-toggle" }, el("summary", {}, "Show the unverified draft"), draft),
    );
    return;
  }
  container.replaceChildren(...(claims ? claimNodes(claims, citeIndex) : [response.answer || ""]));
}

export function renderNotes(container, response, question) {
  const provenance = response.provenance || [];
  const byId = new Map((response.results || []).map((r) => [String(r.id), r]));
  const terms = queryTerms(question);
  const index = new Map();
  const supports = new Map();
  for (const claim of Array.isArray(response.claims) ? response.claims : []) {
    if (claim.status === "supported" && claim.support_text) {
      const key = String(claim.support_id);
      supports.set(key, [...(supports.get(key) || []), claim.support_text]);
    }
  }
  if (!provenance.length) {
    container.replaceChildren(el("li", { class: "notes-empty" }, "No passages were cited for this answer."));
    return index;
  }
  const items = provenance.map((p, i) => {
    const n = i + 1;
    index.set(p.id, n);
    const result = byId.get(String(p.id));
    const text = result ? result.text : "";
    const ranges = supportRanges(text, supports.get(String(p.id)) || []);
    // Open long passages when the supporting sentence would be clamped away.
    const clamped = !(ranges.length && ranges[ranges.length - 1][1] > 380);
    const textNode = el("p", { class: "note-text", dataset: { clamped: String(clamped) } }, markedPassage(text, terms, ranges));
    const toggle = el("button", { class: "text-button", type: "button", "aria-expanded": String(!clamped) }, clamped ? "Show whole passage" : "Show less");
    toggle.addEventListener("click", () => {
      const open = textNode.dataset.clamped === "true";
      textNode.dataset.clamped = open ? "false" : "true";
      toggle.setAttribute("aria-expanded", String(open));
      toggle.textContent = open ? "Show less" : "Show whole passage";
    });
    const found = result && result.agent ? `found by ${(RETRIEVERS[result.agent] || result.agent).toLowerCase()}` : null;
    return el(
      "li",
      { class: "note", id: `note-${n}` },
      el("span", { class: "note-num", "aria-hidden": "true" }, String(n)),
      el(
        "div",
        { class: "note-body" },
        el("div", { class: "note-source" }, p.source),
        el(
          "div",
          { class: "note-meta" },
          `Characters ${fmt.int(p.offset_start)} to ${fmt.int(p.offset_end)}`,
          found ? `, ${found}` : "",
        ),
        text ? textNode : null,
        text && text.length > 420 ? toggle : null,
      ),
    );
  });
  container.replaceChildren(...items);
  return index;
}

function retrieverTable(stats) {
  const diag = (stats && stats.agent_diagnostics) || {};
  const rows = Object.entries(diag).filter(([, d]) => d && d.attempted);
  if (!rows.length) return el("p", {}, "No retrievers ran.");
  return el(
    "div",
    { class: "table-wrap" },
    el(
      "table",
      { class: "ledger retrievers" },
      el("thead", {}, el("tr", {}, ...["Retriever", "Searches", "With results", "Empty", "Timed out", "Failed"].map((h, i) => el("th", { scope: "col", class: i ? "num" : null }, h)))),
      el(
        "tbody",
        {},
        ...rows.map(([name, d]) =>
          el(
            "tr",
            {},
            el("td", {}, RETRIEVERS[name] || name),
            ...[d.attempted, d.ok, d.zero_hits, d.timeout, d.error].map((v) => el("td", { class: "num" }, fmt.int(v || 0))),
          ),
        ),
      ),
    ),
  );
}

const STAGE_NAMES = {
  plan: "Plan the search",
  retrieve: "Search the corpus",
  rerank: "Rank passages",
  graph: "Check the knowledge graph",
  judge: "Weigh the evidence",
  synthesize: "Write the answer",
  verify: "Check each sentence",
};

export function stageName(stage) {
  return STAGE_NAMES[stage] || stage;
}

export function renderWorkings(container, response) {
  const failures = response.retrieval_failures || [];
  const issues = failures.length
    ? failures.map((tag) => {
        const [severity, text] = describeIssue(tag);
        return el("li", { dataset: { severity } }, text);
      })
    : [el("li", { dataset: { severity: "none" } }, "No retrieval problems were detected.")];
  const plan = response.plan || {};
  const queries = plan.queries || [];
  const timings = Object.entries(response.timings_ms || {});
  const judgeNotes = response.judge && typeof response.judge.notes === "string" ? response.judge.notes : "";
  container.replaceChildren(
    el("section", {}, el("h3", {}, "Problems found"), el("ul", { class: "issues" }, ...issues)),
    queries.length
      ? el(
          "section",
          {},
          el("h3", {}, response.domain ? `Searches run in the ${response.domain} domain` : "Searches run"),
          el("ol", { class: "plan-queries" }, ...queries.map((q) => el("li", {}, q))),
        )
      : null,
    el("section", {}, el("h3", {}, "Retrievers"), retrieverTable(response.retrieval_stats)),
    judgeNotes ? el("section", {}, el("h3", {}, "Evidence review"), el("p", {}, judgeNotes)) : null,
    timings.length
      ? el(
          "section",
          {},
          el("h3", {}, "Time spent"),
          el("p", {}, timings.map(([stage, ms]) => `${stageName(stage)}: ${fmt.ms(ms)}`).join("; ") + "."),
        )
      : null,
  );
}

function excerpt(text, limit) {
  const t = String(text || "");
  if (t.length <= limit) return t;
  const cut = t.lastIndexOf(" ", limit);
  return `${t.slice(0, cut > limit * 0.6 ? cut : limit).trimEnd()}…`;
}

export function renderAuditRows(tbody, rows) {
  if (!rows.length) {
    tbody.replaceChildren(el("tr", {}, el("td", { class: "empty", colspan: 4 }, "No questions yet. Ask one to see it here.")));
    return;
  }
  tbody.replaceChildren(
    ...rows.map((r) =>
      el(
        "tr",
        {},
        el("td", { class: "when" }, fmt.date(r.created_at)),
        el("td", {}, el("div", { class: "q" }, r.query || ""), el("div", { class: "a" }, excerpt(r.answer, 240))),
        el("td", {}, r.domain && r.domain !== "domain_unknown" ? r.domain : "Any"),
        el("td", { class: "num" }, r.confidence === null || r.confidence === undefined ? "" : fmt.pct(r.confidence)),
      ),
    ),
  );
}

export function renderFeedbackRows(tbody, rows) {
  if (!rows.length) {
    tbody.replaceChildren(el("tr", {}, el("td", { class: "empty", colspan: 4 }, "No feedback yet.")));
    return;
  }
  tbody.replaceChildren(
    ...rows.map((r) =>
      el(
        "tr",
        {},
        el("td", { class: "when" }, fmt.date(r.created_at)),
        el("td", { class: "q" }, r.query || ""),
        el("td", {}, r.rating >= 4 ? "Helpful" : r.rating <= 2 ? "Not helpful" : `${r.rating} of 5`),
        el("td", {}, r.comment || ""),
      ),
    ),
  );
}

import { ApiError, loadSettings, request, saveSettings, streamQuery } from "./api.js";
import {
  el,
  fmt,
  renderAnswer,
  renderAuditRows,
  renderFeedbackRows,
  renderNotes,
  renderScores,
  renderWorkings,
  stageName,
} from "./render.js";

const EXAMPLES = [
  "What does Seneca say about fear and how to handle it?",
  "Explain the Stoic dichotomy of control with a concrete example.",
  "How does Stoicism view grief and loss?",
  "What does Marcus Aurelius say about anxiety?",
];
const STAGES = ["plan", "retrieve", "rerank", "judge", "synthesize"];

const $ = (id) => document.getElementById(id);
const ui = {
  form: $("ask-form"),
  question: $("question"),
  domain: $("domain"),
  askButton: $("ask-button"),
  formError: $("form-error"),
  intro: $("intro"),
  examples: $("examples"),
  result: $("result"),
  rail: $("rail"),
  sheet: $("sheet"),
  asked: $("asked-question"),
  answer: $("answer"),
  footer: $("sheet-footer"),
  scores: $("scores"),
  feedback: $("feedback"),
  feedbackStatus: $("feedback-status"),
  notes: $("notes"),
  workings: $("workings"),
  workingsBody: $("workings-body"),
  live: $("live-status"),
  notice: $("service-notice"),
  settings: $("settings"),
  settingsForm: $("settings-form"),
  auditBody: document.querySelector("#audit-table tbody"),
  feedbackBody: document.querySelector("#feedback-table tbody"),
  activityError: $("activity-error"),
};

const state = {
  settings: loadSettings(),
  controller: null,
  question: "",
  response: null,
};

// ---------- theme ----------

function applyTheme(theme) {
  document.documentElement.dataset.theme = ["dark", "system"].includes(theme) ? theme : "light";
}

// ---------- tabs ----------

function selectTab(tab) {
  for (const t of document.querySelectorAll(".tab")) {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !on;
  }
  if (tab.id === "tab-activity") loadActivity();
}

function wireTabs() {
  const tabs = [...document.querySelectorAll(".tab")];
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      next.focus();
      selectTab(next);
    });
  });
}

// ---------- pipeline rail ----------

function resetRail() {
  ui.rail.replaceChildren(
    ...STAGES.map((stage) =>
      el(
        "li",
        { dataset: { stage, state: "pending" } },
        el("span", { class: "step-num", "aria-hidden": "true" }),
        el("span", { class: "step-label" }, stageName(stage)),
        el("span", { class: "step-time" }),
      ),
    ),
  );
}

function railItem(stage) {
  let item = ui.rail.querySelector(`[data-stage="${stage}"]`);
  if (!item && (stage === "graph" || stage === "verify")) {
    // Optional steps only appear when the backend has them switched on.
    item = el(
      "li",
      { dataset: { stage, state: "pending" } },
      el("span", { class: "step-num", "aria-hidden": "true" }),
      el("span", { class: "step-label" }, stageName(stage)),
      el("span", { class: "step-time" }),
    );
    if (stage === "graph") ui.rail.insertBefore(item, ui.rail.querySelector('[data-stage="judge"]'));
    else ui.rail.append(item);
  }
  return item;
}

function updateRail(event) {
  if (event.stage === "cache") {
    for (const li of ui.rail.children) {
      li.dataset.state = "skipped";
      li.removeAttribute("aria-current");
    }
    ui.rail.append(el("li", { dataset: { state: "done" } }, el("span", { class: "step-num", "aria-hidden": "true" }), "Answered from cache"));
    return;
  }
  const item = railItem(event.stage);
  if (!item) return;
  item.dataset.state = event.state;
  if (event.state === "running") item.setAttribute("aria-current", "step");
  else item.removeAttribute("aria-current");
  if (event.state === "done" && Number.isFinite(event.ms)) item.querySelector(".step-time").textContent = fmt.ms(event.ms);
}

function failRunningStep() {
  for (const li of ui.rail.querySelectorAll('[data-state="running"]')) li.dataset.state = "failed";
}

// ---------- ask ----------

function showFormError(message) {
  ui.formError.textContent = message;
  ui.formError.hidden = !message;
}

function setBusy(busy) {
  ui.sheet.setAttribute("aria-busy", String(busy));
  ui.askButton.textContent = busy ? "Stop" : "Ask";
  ui.askButton.dataset.state = busy ? "stop" : "ask";
}

function startResult(question) {
  ui.intro.hidden = true;
  ui.result.hidden = false;
  ui.asked.textContent = question;
  ui.answer.replaceChildren();
  ui.answer.dataset.placeholder = "Searching the corpus…";
  ui.footer.hidden = true;
  ui.workings.hidden = true;
  ui.workings.open = false;
  ui.notes.replaceChildren(el("li", { class: "notes-empty" }, "Passages appear here once the answer is ready."));
  ui.feedbackStatus.textContent = "";
  for (const b of ui.feedback.querySelectorAll("button")) {
    b.disabled = false;
    b.removeAttribute("aria-pressed");
  }
  resetRail();
}

function finishResult(response) {
  state.response = response;
  const index = renderNotes(ui.notes, response, state.question);
  renderAnswer(ui.answer, response, index);
  delete ui.answer.dataset.placeholder;
  renderScores(ui.scores, response);
  renderWorkings(ui.workingsBody, response);
  ui.footer.hidden = false;
  ui.workings.hidden = false;
  const n = (response.provenance || []).length;
  ui.live.textContent = `Answer ready, with ${n === 1 ? "1 cited passage" : `${n} cited passages`}.`;
}

async function ask(question) {
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  state.question = question;
  state.response = null;
  showFormError("");
  startResult(question);
  setBusy(true);
  const payload = { user_id: state.settings.user, query: question };
  if (ui.domain.value) payload.preferences = { domain: ui.domain.value };
  let started = false;
  try {
    for await (const { event, data } of streamQuery(state.settings, payload, controller.signal)) {
      started = true;
      if (event === "stage") updateRail(data);
      else if (event === "meta") ui.answer.dataset.placeholder = "Writing the answer…";
      else if (event === "token") {
        delete ui.answer.dataset.placeholder;
        ui.answer.append(data.text || "");
      } else if (event === "final") finishResult(data);
      else if (event === "error") throw new ApiError("The query failed on the server. Check the backend logs for details.");
    }
    if (!state.response) throw new ApiError("The connection closed before the answer was complete. Try again.");
  } catch (err) {
    if (err.name === "AbortError") {
      ui.answer.dataset.placeholder = "Stopped.";
      failRunningStep();
    } else {
      showFormError(err.message || "Something went wrong.");
      if (!started) {
        // Rejected before anything streamed: an empty answer sheet adds nothing.
        ui.result.hidden = true;
        ui.intro.hidden = false;
      } else {
        failRunningStep();
        ui.answer.replaceChildren();
        ui.answer.dataset.placeholder = "No answer.";
      }
    }
  } finally {
    if (state.controller === controller) {
      state.controller = null;
      setBusy(false);
    }
  }
}

function wireAsk() {
  ui.form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.controller) {
      state.controller.abort();
      return;
    }
    const question = ui.question.value.trim();
    if (!question) {
      showFormError("Type a question first.");
      ui.question.focus();
      return;
    }
    ask(question);
  });
  ui.question.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      ui.form.requestSubmit();
    }
  });
  ui.examples.replaceChildren(
    ...EXAMPLES.map((q) =>
      el(
        "li",
        {},
        el("button", {
          type: "button",
          onclick: () => {
            ui.question.value = q;
            ui.form.requestSubmit();
          },
        }, q),
      ),
    ),
  );
  ui.answer.addEventListener("click", (e) => {
    const link = e.target.closest(".cite");
    if (!link) return;
    for (const note of ui.notes.querySelectorAll(".note")) note.dataset.active = String(`#${note.id}` === link.getAttribute("href"));
  });
}

// ---------- feedback ----------

function wireFeedback() {
  ui.feedback.addEventListener("click", async (e) => {
    const button = e.target.closest("button[data-rating]");
    if (!button || !state.response) return;
    const buttons = ui.feedback.querySelectorAll("button");
    buttons.forEach((b) => (b.disabled = true));
    try {
      await request(state.settings, "/v1/feedback", {
        method: "POST",
        body: {
          user_id: state.settings.user,
          query: state.question,
          rating: Number(button.dataset.rating),
          query_id: state.response.query_id,
        },
      });
      button.setAttribute("aria-pressed", "true");
      ui.feedbackStatus.textContent = "Feedback saved.";
    } catch (err) {
      buttons.forEach((b) => (b.disabled = false));
      ui.feedbackStatus.textContent = err.message;
    }
  });
}

// ---------- activity ----------

async function loadActivity() {
  ui.activityError.hidden = true;
  try {
    const [audit, feedback] = await Promise.all([
      request(state.settings, "/v1/audit?limit=25"),
      request(state.settings, "/v1/feedback/list?limit=25"),
    ]);
    renderAuditRows(ui.auditBody, audit.results || []);
    renderFeedbackRows(ui.feedbackBody, feedback.results || []);
  } catch (err) {
    ui.activityError.textContent = err.message;
    ui.activityError.hidden = false;
  }
}

// ---------- settings ----------

function fillSettingsForm() {
  const f = ui.settingsForm;
  f.base.value = state.settings.base;
  f.key.value = state.settings.key;
  f.user.value = state.settings.user;
  for (const r of f.theme) r.checked = r.value === state.settings.theme;
}

function wireSettings() {
  $("open-settings").addEventListener("click", () => {
    fillSettingsForm();
    ui.settings.showModal();
  });
  ui.settings.addEventListener("close", () => {
    if (ui.settings.returnValue !== "save") return;
    const f = ui.settingsForm;
    state.settings = {
      base: f.base.value.trim(),
      key: f.key.value.trim(),
      user: f.user.value.trim() || "reader",
      theme: f.theme.value || "light",
    };
    saveSettings(state.settings);
    applyTheme(state.settings.theme);
    checkService();
    loadDomains();
  });
}

// ---------- service status & domains ----------

async function checkService() {
  try {
    const res = await fetch(`${state.settings.base.replace(/\/+$/, "") || ""}/ready`);
    const body = await res.json();
    const down = Object.entries(body.checks || {}).filter(([, c]) => !c.ok).map(([name]) => name);
    ui.notice.hidden = !down.length;
    ui.notice.textContent = down.length ? `Some services are not responding: ${down.join(", ")}. Answers may be incomplete.` : "";
  } catch {
    ui.notice.hidden = false;
    ui.notice.textContent = "Can't reach the API. Check that the backend is running, or set its address in Settings.";
  }
}

async function loadDomains() {
  try {
    const { domains = [] } = await request(state.settings, "/v1/domains");
    const current = ui.domain.value;
    ui.domain.replaceChildren(el("option", { value: "" }, "All domains"), ...domains.map((d) => el("option", { value: d }, d)));
    ui.domain.value = domains.includes(current) ? current : "";
  } catch {
    // The selector simply stays at "All domains".
  }
}

// ---------- boot ----------

applyTheme(state.settings.theme);
wireTabs();
wireAsk();
wireFeedback();
wireSettings();
$("refresh-activity").addEventListener("click", loadActivity);
checkService();
loadDomains();

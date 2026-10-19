// API client: settings, JSON requests with readable errors, and an SSE reader
// for POST /v1/query/stream (EventSource cannot send a POST body).

const SETTINGS_KEY = "sagrag.settings";
const API_KEY_KEY = "sagrag.apiKey";

function readStorage(storage, key) {
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(storage, key, value) {
  try {
    if (value) storage.setItem(key, value);
    else storage.removeItem(key);
  } catch {
    // Private mode or blocked storage: settings last for this page only.
  }
}

export function loadSettings() {
  let saved = {};
  try {
    saved = JSON.parse(readStorage(localStorage, SETTINGS_KEY) || "{}");
  } catch {
    saved = {};
  }
  return {
    base: typeof saved.base === "string" ? saved.base : "",
    user: typeof saved.user === "string" && saved.user ? saved.user : "reader",
    // Light unless the reader picks dark (or "match system") in Settings.
    theme: ["dark", "system"].includes(saved.theme) ? saved.theme : "light",
    // The key stays in sessionStorage: it is a secret and should not outlive the tab.
    key: readStorage(sessionStorage, API_KEY_KEY) || "",
  };
}

export function saveSettings(settings) {
  const { key, ...rest } = settings;
  writeStorage(localStorage, SETTINGS_KEY, JSON.stringify(rest));
  writeStorage(sessionStorage, API_KEY_KEY, key);
}

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.status = status;
  }
}

function apiOrigin(settings) {
  return settings.base ? settings.base.replace(/\/+$/, "") : window.location.origin;
}

function headers(settings, json = true) {
  const h = {};
  if (json) h["content-type"] = "application/json";
  if (settings.key) h["x-api-key"] = settings.key;
  return h;
}

async function errorFrom(response) {
  let detail = "";
  try {
    const body = await response.json();
    if (typeof body.detail === "string") detail = body.detail;
    else if (Array.isArray(body.detail) && body.detail[0]) {
      const first = body.detail[0];
      const where = Array.isArray(first.loc) ? first.loc.filter((p) => p !== "body").join(".") : "";
      detail = where ? `${where}: ${first.msg}` : first.msg;
    }
  } catch {
    // Non-JSON error body.
  }
  const status = response.status;
  if (status === 401) return new ApiError("The API needs a valid key. Add one in Settings.", status);
  if (status === 422) return new ApiError(detail ? `The request was rejected: ${detail}.` : "The request was rejected.", status);
  if (status === 429) {
    const wait = response.headers.get("retry-after");
    return new ApiError(wait ? `Too many requests. Try again in ${wait} seconds.` : "Too many requests. Try again in a minute.", status);
  }
  return new ApiError(`The API returned an error (${status}). Check the backend logs.`, status);
}

function networkError(settings) {
  return new ApiError(`Can't reach the API at ${apiOrigin(settings)}. Check that the backend is running.`);
}

export async function request(settings, path, { method = "GET", body, signal } = {}) {
  let response;
  try {
    response = await fetch(apiOrigin(settings) + path, {
      method,
      headers: headers(settings, body !== undefined),
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw networkError(settings);
  }
  if (!response.ok) throw await errorFrom(response);
  return response.json();
}

function parseEvent(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    return null;
  }
}

// Yields {event, data} objects as the server sends them.
export async function* streamQuery(settings, payload, signal) {
  let response;
  try {
    response = await fetch(apiOrigin(settings) + "/v1/query/stream", {
      method: "POST",
      headers: headers(settings),
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw networkError(settings);
  }
  if (!response.ok) throw await errorFrom(response);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value.replace(/\r\n/g, "\n");
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const parsed = parseEvent(buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
        if (parsed) yield parsed;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

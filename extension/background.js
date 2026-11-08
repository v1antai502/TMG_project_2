// Service worker: the only part that talks to the model service.
// Content scripts send headlines here; scores are cached for the browser session.

const DEFAULTS = { enabled: true, threshold: 0.5, showSafe: false, apiUrl: "http://127.0.0.1:8000", apiKey: "" };
const BATCH = 64;

async function getSettings() {
  return { ...DEFAULTS, ...(await chrome.storage.sync.get(Object.keys(DEFAULTS))) };
}

// title -> probability, kept in session storage so it survives the worker being stopped
async function cacheGet(titles) {
  const keys = titles.map((t) => "p:" + t);
  const found = await chrome.storage.session.get(keys);
  const out = {};
  for (const t of titles) if (found["p:" + t] !== undefined) out[t] = found["p:" + t];
  return out;
}

async function cachePut(scores) {
  const entries = {};
  for (const [t, p] of Object.entries(scores)) entries["p:" + t] = p;
  try {
    await chrome.storage.session.set(entries);
  } catch (e) {
    // session storage is full (about 10 MB): start over rather than fail
    await chrome.storage.session.clear();
  }
}

function headers(apiKey, json = false) {
  const h = json ? { "Content-Type": "application/json" } : {};
  if (apiKey) h["X-API-Key"] = apiKey;
  return h;
}

async function scoreTitles(titles) {
  const { apiUrl, apiKey } = await getSettings();
  const unique = [...new Set(titles)];
  const scores = await cacheGet(unique);
  const missing = unique.filter((t) => scores[t] === undefined);

  for (let i = 0; i < missing.length; i += BATCH) {
    const chunk = missing.slice(i, i + BATCH);
    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: headers(apiKey, true),
      body: JSON.stringify({ headlines: chunk.map((title, j) => ({ id: String(j), title })) }),
    });
    if (!response.ok) throw new Error(`service answered ${response.status}`);
    const { results } = await response.json();
    const fresh = {};
    for (const r of results) fresh[chunk[Number(r.id)]] = r.prob_clickbait;
    Object.assign(scores, fresh);
    await cachePut(fresh);
  }
  return scores;
}

// The popup also asks for permission for a custom service address (e.g. a computer on the
// same Wi-Fi when testing on a phone); that origin is not a news site, so it is skipped below.
const isServiceOrigin = async (origin) => {
  const { apiUrl } = await getSettings();
  return origin.startsWith(new URL(apiUrl).origin);
};

// Sites switched on from the popup. The popup only asks Chrome for permission; when it is
// granted (even if the popup has closed meanwhile) the content script is registered here
// for that site and injected into its open tabs. Removing the permission switches it off.
const SERVICE_ORIGINS = ["http://127.0.0.1:8000/*", "http://localhost:8000/*"];
const scriptId = (origin) => "site-" + origin.replace(/[^a-z0-9.-]/gi, "_");

chrome.permissions.onAdded.addListener(async ({ origins = [] }) => {
  for (const origin of origins) {
    if (SERVICE_ORIGINS.includes(origin) || (await isServiceOrigin(origin))) continue;
    const id = scriptId(origin);
    const existing = await chrome.scripting.getRegisteredContentScripts({ ids: [id] });
    if (!existing.length) {
      await chrome.scripting.registerContentScripts([
        { id, matches: [origin], js: ["content.js"], css: ["content.css"], runAt: "document_idle", persistAcrossSessions: true },
      ]);
    }
    for (const tab of await chrome.tabs.query({ url: origin })) {
      try {
        await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ["content.css"] });
        await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
      } catch (e) {
        // tab closed or not scriptable; the registered script covers the next load
      }
    }
  }
});

chrome.permissions.onRemoved.addListener(async ({ origins = [] }) => {
  const ids = origins.filter((o) => !SERVICE_ORIGINS.includes(o)).map(scriptId);
  if (ids.length) await chrome.scripting.unregisterContentScripts({ ids }).catch(() => {});
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "score") {
    scoreTitles(message.titles)
      .then((scores) => sendResponse({ ok: true, scores }))
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true; // answer asynchronously
  }
  if (message.type === "count" && sender.tab) {
    chrome.action.setBadgeBackgroundColor({ tabId: sender.tab.id, color: "#d93025" });
    chrome.action.setBadgeText({ tabId: sender.tab.id, text: message.flagged ? String(message.flagged) : "" });
    return false;
  }
  if (message.type === "health") {
    getSettings()
      .then(({ apiUrl, apiKey }) => fetch(`${apiUrl}/health`, { headers: headers(apiKey) }))
      .then((r) => r.json())
      .then((health) => sendResponse({ ok: true, health }))
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  }
  return false;
});

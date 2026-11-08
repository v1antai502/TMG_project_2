const DEFAULTS = { enabled: true, threshold: 0.5, showSafe: false, apiUrl: "http://127.0.0.1:8000", apiKey: "" };
const $ = (id) => document.getElementById(id);

async function load() {
  const s = { ...DEFAULTS, ...(await chrome.storage.sync.get(Object.keys(DEFAULTS))) };
  $("enabled").checked = s.enabled;
  $("showSafe").checked = s.showSafe;
  $("threshold").value = s.threshold;
  $("thresholdValue").textContent = Number(s.threshold).toFixed(2);
  $("apiUrl").value = s.apiUrl;
  $("apiKey").value = s.apiKey;
}

function save(key, value) {
  chrome.storage.sync.set({ [key]: value });
}

$("enabled").addEventListener("change", (e) => save("enabled", e.target.checked));
$("showSafe").addEventListener("change", (e) => save("showSafe", e.target.checked));
$("threshold").addEventListener("input", (e) => {
  $("thresholdValue").textContent = Number(e.target.value).toFixed(2);
  save("threshold", Number(e.target.value));
});
// A service address other than this computer needs the user's permission for that origin.
$("saveService").addEventListener("click", () => {
  const apiUrl = $("apiUrl").value.trim().replace(/\/+$/, "") || DEFAULTS.apiUrl;
  const apiKey = $("apiKey").value.trim();
  let origin;
  try {
    origin = new URL(apiUrl).origin;
  } catch (e) {
    $("status").className = "status err";
    $("status").textContent = "Địa chỉ không hợp lệ.";
    return;
  }
  // request first, while the click still counts as a user gesture
  chrome.permissions.request({ origins: [origin + "/*"] }).then(async (granted) => {
    if (!granted) return;
    await chrome.storage.sync.set({ apiUrl, apiKey });
    showStatus();
  });
});

async function showStatus() {
  const status = $("status");
  const response = await chrome.runtime.sendMessage({ type: "health" });
  if (response && response.ok && response.health.status === "ok") {
    status.className = "status ok";
    status.textContent = `Dịch vụ đang chạy (${response.health.device}).`;
  } else {
    status.className = "status err";
    status.textContent = "Không kết nối được dịch vụ mô hình. Hãy chạy: python -m service.app";
  }
}

// Same id scheme as background.js, which registers the script once permission is granted.
const scriptId = (origin) => "site-" + origin.replace(/[^a-z0-9.-]/gi, "_");

async function showStats() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  let url = null;
  try {
    url = new URL(tab.url);
  } catch (e) {
    // no URL (e.g. a new tab page)
  }
  const web = url && (url.protocol === "https:" || url.protocol === "http:");
  const origin = web ? `${url.protocol}//${url.host}/*` : null;
  const custom = origin ? (await chrome.scripting.getRegisteredContentScripts({ ids: [scriptId(origin)] })).length > 0 : false;

  let stats = null;
  try {
    stats = await chrome.tabs.sendMessage(tab.id, { type: "stats" });
  } catch (e) {
    // no content script on this page
  }
  if (stats) {
    $("flagged").textContent = stats.flagged;
    $("scored").textContent = `${stats.scored}/${stats.headlines} tiêu đề`;
  } else {
    $("flagged").textContent = "–";
    $("scored").textContent = web ? "" : "Trang này không được hỗ trợ";
  }

  $("enableBox").hidden = !web || !!stats || custom;
  $("disableBox").hidden = !custom;
  if (web) $("enableHost").textContent = $("disableHost").textContent = url.host;

  // Chrome shows its own permission prompt; background.js does the rest when it is granted
  $("enableSite").onclick = () => chrome.permissions.request({ origins: [origin] }).then((granted) => {
    if (granted) setTimeout(showStats, 1500);
  });
  $("disableSite").onclick = () => chrome.permissions.remove({ origins: [origin] }).then(() => {
    chrome.tabs.reload(tab.id);
    window.close();
  });
}

load();
showStatus();
showStats();

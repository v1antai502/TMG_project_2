// Content script: finds headlines on the page, asks the background worker for scores,
// and marks the ones above the threshold. Re-scans when the page loads more articles.
(() => {
  // the script can be injected twice on a site switched on from the popup
  if (window.__clickbaitDetectorLoaded) return;
  window.__clickbaitDetectorLoaded = true;

  const DEFAULTS = { enabled: true, threshold: 0.5, showSafe: false };
  // Most Vietnamese news sites end article links in an article number:
  // "-5134398.html", "-185261020111903145.htm", "-c56309005.epi" (Báo Mới),
  // "-post1562469.html", "-post123456.vnp", ".990623.html", "-c48a1234567.html" (24h)
  const ARTICLE_URL = /([-.][a-z]{0,6}\d{5,}(\.[a-z]{2,5})?\/?|-c\d+a\d+\.html)$/;
  // A few use only words ("/de-moi-nguoi-dan-duoc-thu-huong.html"). On those pages a
  // headline link inside a heading with a long word slug counts too.
  const SLUG_URL = /\/[a-z0-9]+(?:-[a-z0-9]+){5,}(\.(html?|aspx?|php))?\/?$/;
  const MIN_NUMBERED = 5;
  const MIN_CHARS = 20;
  const MAX_CHARS = 220;
  const MIN_WORDS = 5;

  let settings = { ...DEFAULTS };
  const scores = new Map(); // headline text -> probability
  let scanTimer = null;
  let serviceError = null;

  // Visible text of a link, without counters such as a comment count glued to the end.
  function linkText(a) {
    let text = "";
    const walker = document.createTreeWalker(a, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const parent = node.parentElement;
      if (parent && parent !== a && /^\s*\d+\s*$/.test(parent.textContent)) continue;
      if (parent && parent.closest(".cbd-badge")) continue;
      text += node.textContent;
    }
    return text.replace(/\s+/g, " ").trim();
  }

  let slugMode = false; // decided per scan in findHeadlines

  function headlineOf(a) {
    let url;
    try {
      url = new URL(a.href, location.href);
    } catch (e) {
      return null;
    }
    const numbered = ARTICLE_URL.test(url.pathname);
    if (!numbered && !(slugMode && SLUG_URL.test(url.pathname) && a.closest("h1, h2, h3, h4"))) return null;
    const visible = linkText(a);
    if (visible.length < 10) return null; // image-only link
    // Kênh14 shortens long titles with "…" but keeps the full one in the title attribute
    const attr = (a.getAttribute("title") || "").replace(/\s+/g, " ").trim();
    const text = attr.length >= MIN_CHARS && visible.endsWith("…") ? attr : visible;
    if (text.length < MIN_CHARS || text.length > MAX_CHARS) return null;
    if (text.split(" ").length < MIN_WORDS) return null;
    return { key: url.pathname, text };
  }

  // One headline per article: the link inside a heading if there is one, otherwise the
  // shortest text (a summary paragraph often links to the same article).
  function findHeadlines() {
    // use the word-slug rule only where numbered article links are (almost) absent;
    // elsewhere it would also pick up section and topic links
    slugMode = false;
    let numbered = 0;
    for (const a of document.querySelectorAll("a[href]")) {
      if (ARTICLE_URL.test(a.pathname || "") && ++numbered >= MIN_NUMBERED) break;
    }
    slugMode = numbered < MIN_NUMBERED;

    const byArticle = new Map();
    for (const a of document.querySelectorAll("a[href]")) {
      const h = headlineOf(a);
      if (!h) continue;
      const inHeading = !!a.closest("h1, h2, h3, h4, h5");
      const best = byArticle.get(h.key);
      const better =
        !best ||
        (inHeading && !best.inHeading) ||
        (inHeading === best.inHeading && h.text.length < best.text.length);
      if (better) byArticle.set(h.key, { text: h.text, inHeading });
    }
    // every link showing that headline gets marked (some pages repeat a story)
    const links = [];
    for (const a of document.querySelectorAll("a[href]")) {
      const h = headlineOf(a);
      if (h && byArticle.get(h.key)?.text === h.text) links.push({ a, text: h.text });
    }
    return links;
  }

  function badgeFor(a) {
    const next = a.nextElementSibling;
    return next && next.classList.contains("cbd-badge") ? next : null;
  }

  function mark(a, text) {
    const prob = scores.get(text);
    let badge = badgeFor(a);
    a.classList.remove("cbd-flagged");
    if (prob === undefined || !settings.enabled) {
      badge?.remove();
      return false;
    }
    const flagged = prob >= settings.threshold;
    if (!flagged && !settings.showSafe) {
      badge?.remove();
      return false;
    }
    if (!badge) {
      badge = document.createElement("span");
      a.insertAdjacentElement("afterend", badge);
    }
    const pct = Math.round(prob * 100);
    badge.className = "cbd-badge " + (flagged ? "cbd-badge--bait" : "cbd-badge--safe");
    badge.textContent = flagged ? `⚠ Clickbait ${pct}%` : `✓ ${pct}%`;
    badge.title = flagged
      ? `Mô hình đánh giá tiêu đề này là clickbait (xác suất ${pct}%).`
      : `Mô hình đánh giá tiêu đề này không phải clickbait (xác suất clickbait ${pct}%).`;
    if (flagged) a.classList.add("cbd-flagged");
    return flagged;
  }

  function render() {
    const seen = new Set();
    let flagged = 0;
    for (const { a, text } of findHeadlines()) {
      if (mark(a, text) && !seen.has(text)) {
        flagged += 1;
        seen.add(text);
      }
    }
    chrome.runtime.sendMessage({ type: "count", flagged: settings.enabled ? flagged : 0 });
    return flagged;
  }

  async function scan() {
    if (!settings.enabled) {
      render();
      return;
    }
    const titles = [...new Set(findHeadlines().map((h) => h.text))].filter((t) => !scores.has(t));
    if (titles.length) {
      const response = await chrome.runtime.sendMessage({ type: "score", titles });
      if (response && response.ok) {
        serviceError = null;
        for (const [t, p] of Object.entries(response.scores)) scores.set(t, p);
      } else {
        serviceError = response ? response.error : "no response";
        console.warn("[clickbait detector]", serviceError);
      }
    }
    render();
  }

  function scheduleScan() {
    clearTimeout(scanTimer);
    scanTimer = setTimeout(scan, 800);
  }

  new MutationObserver((mutations) => {
    // ignore the badges this script adds
    if (mutations.some((m) => [...m.addedNodes].some((n) => !(n.classList && n.classList.contains("cbd-badge"))))) {
      scheduleScan();
    }
  }).observe(document.body, { childList: true, subtree: true });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;
    for (const [key, { newValue }] of Object.entries(changes)) if (key in DEFAULTS) settings[key] = newValue;
    scan();
  });

  // the popup asks for this page's numbers
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "stats") {
      const headlines = [...new Set(findHeadlines().map((h) => h.text))];
      const scored = headlines.filter((t) => scores.has(t));
      const flagged = scored.filter((t) => scores.get(t) >= settings.threshold);
      sendResponse({ headlines: headlines.length, scored: scored.length, flagged: flagged.length, error: serviceError });
    }
    return false;
  });

  chrome.storage.sync.get(Object.keys(DEFAULTS)).then((stored) => {
    settings = { ...DEFAULTS, ...stored };
    scan();
  });
})();

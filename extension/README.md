# Clickbait Detector: browser extension (Chrome / Edge, Manifest V3)

Marks clickbait headlines on Vietnamese news sites. The extension finds the headlines on a
page and sends them to a small model service running on your own computer
(`service/app.py`), which scores them with the final PhoBERT model.

Built in: 48 Vietnamese news sites, including the eight the model was trained on
(VnExpress, Thanh Niên, Tuổi Trẻ, VietNamNet, 24h, Kênh14, Saostar, Báo Mới) and
Dân trí, Znews, Tiền Phong, Người Lao Động, VTV, Nhân Dân, VietnamPlus, CafeF, VnEconomy
and others (full list in `manifest.json`). Any other site can be switched on from the popup.

```
news page ──(headlines)──> content.js ──> background.js ──HTTP──> service (127.0.0.1:8000) ──> PhoBERT
          <──(badges)────                <──(probabilities)──
```

## 1. Start the model service

The service needs the trained model in `models/phobert_final/` (`best.pt`, `tokenizer/`,
`config.json`) and the Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
python -m service.app
```

Wait for `model loaded from ...` and `Uvicorn running on http://127.0.0.1:8000`, then
check it in a second terminal:

```bash
curl http://127.0.0.1:8000/health
```

It answers `{"status":"ok", ...}`. Leave the service running while you browse.

## 2. Load the extension

Chrome:
1. Open `chrome://extensions`.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and choose this `extension/` folder.

Edge:
1. Open `edge://extensions`.
2. Turn on **Developer mode** (left side).
3. Click **Load unpacked** and choose this `extension/` folder.

Pin it from the puzzle-piece menu so the icon stays in the toolbar.

## 3. Use it

Open a supported site, for example https://vnexpress.net. After a second or two:
- clickbait headlines get a red **⚠ Clickbait 94%** badge and a wavy red underline;
- the toolbar icon shows how many headlines on the page were flagged.

Hover a badge for an explanation. New headlines loaded while scrolling are scored too.

Click the toolbar icon to:
- see whether the service is reachable and how many headlines were scored;
- on a site that is not built in, click **Bật trên trang web này** (switch on for this site):
  Chrome asks for permission for that one site, then the page is scanned straight away and
  on every later visit. **Tắt trên trang web này** switches it off again;
- switch marking on or off;
- show safe headlines as well (green **✓** badges);
- move the threshold: higher means fewer false alarms but more missed clickbait.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Popup says it cannot reach the model service | Start it with `python -m service.app`; check `http://127.0.0.1:8000/health` |
| No badges at all | Reload the page after starting the service; check the site is in the supported list |
| Console shows `service answered 422` | A headline was empty or over 500 characters; it is skipped, the rest still work |
| You changed a file in `extension/` | Click the reload icon on the extension's card in `chrome://extensions`, then reload the page |

To see what the extension is doing: on the news page open DevTools (F12) → Console, and
look for messages starting with `[clickbait detector]`. For the background worker, click
**service worker** on the extension's card in `chrome://extensions`.

## Files

| File | Role |
|---|---|
| `manifest.json` | Extension definition: supported sites, permissions, scripts |
| `content.js` | Runs on news pages: finds headlines, requests scores, adds badges |
| `content.css` | Badge styles |
| `background.js` | Calls the model service, caches scores for the browser session |
| `popup.html`, `popup.js` | Toolbar popup: status, counts and settings |
| `icons/` | Toolbar and store icons |

## Privacy

Only headline text is sent, and only to the address in the popup's settings (by default
the service on your own computer, `127.0.0.1`). The service listens on this computer only
and does not store anything beyond an in-memory cache.

## Adding a site

For yourself: open the site and click **Bật trên trang web này** in the popup.

For everyone (built in):
1. Add `https://site/*` and `https://*.site/*` to `content_scripts.matches` in `manifest.json`.
2. Check that headlines are found: most sites end article links in a number
   (`-12345.html`, `-post12345.html`, `.chn`, `.epi`, `.vnp`); pages with almost no such
   links fall back to long word-only links inside headings (`SLUG_URL` in `content.js`).
3. Reload the extension.

How headlines are found, briefly: a link counts if its URL looks like an article and its
text has 5+ words and 20-220 characters; one headline is kept per article (the link inside
a heading, otherwise the shortest text). A few topic or event links can slip through; they
are scored like any headline and usually come out as non-clickbait.

The model was trained on eight outlets (VnExpress, Thanh Niên, Tuổi Trẻ, VietNamNet, 24h,
Kênh14, Saostar, Báo Mới); accuracy on the other sites is untested.

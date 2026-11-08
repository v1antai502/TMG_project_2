# Clickbait Detector: installation guide

A browser extension for Chrome and Edge that marks clickbait headlines on Vietnamese news
sites (VnExpress, Tuổi Trẻ, Thanh Niên, Dân trí, Znews and 43 more). A small program on
your computer, the model service, reads each headline and decides; nothing is sent to the
internet.

## What you need

- Windows 10 or 11 (macOS and Linux also work, see the end)
- Google Chrome or Microsoft Edge
- Python 3.11: https://www.python.org/downloads/ (during installation, tick
  **Add python.exe to PATH**)
- About 3 GB of free disk space and 4 GB of RAM
- An internet connection the first time only, to install packages

## Step 1: Unzip

Unzip `clickbait-detector-desktop.zip` to a folder you will keep, for example
`C:\ClickbaitDetector`. Do not run it from inside the zip file.

## Step 2: Start the model service

Double-click `start_service.bat`.

- **First time:** it installs what it needs. This takes 5-10 minutes; wait until you see
  `Uvicorn running on http://127.0.0.1:8000`.
- **Later:** it starts in about 10 seconds.

Keep this window open while you browse. Closing it switches the detector off.

If Windows shows "Windows protected your PC", click **More info**, then **Run anyway**.

Check it works: open http://127.0.0.1:8000/health in your browser. You should see
`"status":"ok"`.

## Step 3: Add the extension to the browser

Chrome:
1. Go to `chrome://extensions`.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and choose the `extension` folder inside the folder from Step 1.

Edge:
1. Go to `edge://extensions`.
2. Turn on **Developer mode** (left side).
3. Click **Load unpacked** and choose the `extension` folder.

Click the puzzle-piece icon in the toolbar and pin **Clickbait Detector**.

## Step 4: Try it

Open https://vnexpress.net. After a few seconds:

- clickbait headlines get a red **⚠ Clickbait 94%** label and a wavy red underline;
- the extension icon shows how many headlines were flagged on the page.

Click the extension icon to:

- check that it is connected to the model service;
- switch it on or off, or show safe headlines too (green ✓);
- change the threshold (higher = fewer false alarms, but more clickbait missed);
- switch it on for a news site that is not in the list: **Bật trên trang web này**.

## Something not working?

| What you see | What to do |
|---|---|
| The popup says it cannot connect to the model service | Start `start_service.bat` and wait for `Uvicorn running ...` |
| No labels on the page | Reload the page after the service has started |
| `start_service.bat` says Python was not found | Install Python 3.11 with **Add python.exe to PATH** ticked, then run it again |
| Installing packages failed | Check your internet connection, delete the `.venv` folder, run `start_service.bat` again |
| `only one usage of each socket address` | The service is already running in another window; use that one |

## Feedback

Please note, for each problem or wrong label: the site, the headline, and what you
expected. Screenshots help.

## Uninstall

1. Remove the extension in `chrome://extensions` or `edge://extensions`.
2. Delete the folder from Step 1.

## macOS / Linux

In a terminal, in the unzipped folder: `bash start_service.sh`, then Steps 3-4 as above.

# Chess Lab for iPhone

The Vision Lab as an iPhone app. You give it a screenshot or photo of a chessboard. It finds the board, reads the
pieces and asks Stockfish for the best move, then shows it as a green arrow. Everything runs on the phone, so
it works offline and nothing is uploaded.

It's a web app (PWA): open the link in Safari, then **Share → Add to Home Screen**. From then on it opens
full-screen like a normal app and works without a connection.

## What's inside

| File | Port of / role |
|---|---|
| `vision.js` | `chesslab/vision/*`: board detection, piece recognition, orientation, turn detection, calibration (no OpenCV) |
| `rules.js` | `chesslab/chesscore/*`: FEN building and the legality checks, using chess.js |
| `pipeline.js` | `chesslab/pipeline.py`: refuses instead of guessing, with the same thresholds |
| `vision-worker.js` | runs the vision in a Web Worker so the UI stays smooth |
| `engine.js` | Stockfish 19 lite (single-threaded WASM, `vendor/`) over UCI |
| `app.js`, `index.html`, `styles.css` | the app |
| `sw.js`, `manifest.webmanifest` | offline copy and home-screen install |
| `templates/` | the default piece set (lichess style); `build.py --with-calibrated` adds your desktop sets for local use |

For other piece styles (chess.com and so on), calibrate on the phone from a phone screenshot of the starting
position: Options → Calibrate. Those sets are stored on the phone (IndexedDB).

## Rebuilding

After changing app files:

```bash
.venv\Scripts\python iphone_app\build.py
node iphone_app\tests\run.mjs
```

`build.py` exports the piece sets, icons and test fixtures, and bumps the offline-cache version so installed
copies update. `tests/run.mjs` runs the JavaScript pipeline on the sample screenshots, with the same ground truth
as `tests/test_vision.py`.

## Licences

Stockfish.js: GPL-3.0 (`vendor/stockfish-LICENSE-GPL3.txt`). chess.js: BSD-2-Clause (`vendor/chess.js-LICENSE.txt`).
Piece artwork: cburnett, CC BY-SA 3.0.

> Analysis only, same as the desktop app: use it on your own games, puzzles and bots, not on live games
> against people.

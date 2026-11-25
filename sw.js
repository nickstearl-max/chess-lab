// Offline copy of the app. build.py rewrites VERSION whenever any app file changes, so an
// installed app swaps to the new files the next time it's opened online.

const VERSION = "41933dffdad8";
const CACHE = `chesslab-${VERSION}`;
const CORE = [
  "./", "index.html", "styles.css", "app.js", "engine.js", "pipeline.js", "rules.js", "vision.js",
  "vision-worker.js", "pieces.js", "manifest.webmanifest", "icons/icon.svg", "icons/icon-180.png",
  "icons/icon-192.png", "icons/icon-512.png", "vendor/chess.js", "vendor/stockfish-19-lite-single.js",
  "vendor/stockfish-19-lite-single.wasm", "templates/index.json",
];

self.addEventListener("install", event => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await cache.addAll(CORE);
    const index = await (await cache.match("templates/index.json")).json();
    await Promise.allSettled(index.map(s => cache.add(s.file)));   // one missing set shouldn't block install
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", event => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key !== CACHE) await caches.delete(key);
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET" || new URL(event.request.url).origin !== location.origin) return;
  event.respondWith((async () => {
    const cached = await caches.match(event.request, { ignoreSearch: true });
    if (cached) return cached;
    const res = await fetch(event.request);
    if (res.ok) (await caches.open(CACHE)).put(event.request, res.clone());
    return res;
  })());
});

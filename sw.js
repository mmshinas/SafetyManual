// sw.js — Offline-first cache for the e-manual PWA.
// Bump CACHE_NAME whenever index.html/app.js/style.css OR the manual data
// (data/*.json, data/manual.pdf) change, so clients pick up the new files.
const CACHE_NAME = "manual-search-v7";

const ASSETS = [
  "./",
  "index.html",
  "style.css",
  "app.js",
  "manifest.json",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "data/config.json",
  "data/chunks.json",
  "data/index.json",
  "data/manual.pdf",
  "vendor/pdf.min.js",
  "vendor/pdf.worker.min.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Cache-first: once installed, the app never needs the network again.
// A background revalidation keeps things fresh if the network happens to be up.
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      const networkFetch = fetch(event.request)
        .then((response) => {
          if (response && response.ok) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
          }
          return response;
        })
        .catch(() => null);

      return cached || networkFetch || caches.match("index.html");
    })
  );
});

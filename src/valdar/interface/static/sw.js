/* Le service worker : juste ce qu'il faut pour que l'appli s'ouvre d'un coup et qu'elle
 * s'installe sur l'écran d'accueil. Rien n'est mis en cache de ce que dit Valdar : ses
 * paroles ne traînent pas dans le téléphone.
 */
const COQUILLE = "valdar-coquille-1";
const FICHIERS = ["tel.html", "tel.css", "tel.js", "face.js", "visage.svg",
                  "manifeste.json", "icone.svg", "icone-192.png", "icone-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(COQUILLE).then((c) => c.addAll(FICHIERS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((noms) =>
    Promise.all(noms.filter((n) => n !== COQUILLE).map((n) => caches.delete(n)))
  ).then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.pathname.startsWith("/api/")) return;   // jamais ses mots
  e.respondWith(
    fetch(e.request).then((r) => {
      if (r.ok && FICHIERS.some((f) => u.pathname.endsWith(f))) {
        const copie = r.clone();
        caches.open(COQUILLE).then((c) => c.put(e.request, copie));
      }
      return r;
    }).catch(() => caches.match(e.request, { ignoreSearch: true }))
  );
});

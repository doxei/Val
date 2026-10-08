/* Interface de Valdar : onglets, discussion en direct, état du cœur, foyer, mémoire… */
(function () {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const hm = (t) => new Date(t * 1000).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  const day = (t) => new Date(t * 1000).toLocaleDateString("fr-FR", { weekday: "short", day: "numeric", month: "short" });

  async function get(url) { const r = await fetch(url); return r.json(); }
  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify(body || {}) });
    const d = await r.json();
    if (d && d.ok === false && d.error) toast(d.error);
    if (d && d.message) toast(d.message);
    return d;
  }
  let toastTimer = 0;
  function toast(text) {
    const t = $("#toast"); t.textContent = text; t.classList.add("show");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), 5000);
  }

  // ------------------------------------------------------------------ onglets
  let tab = localStorageGet("tab") || "discussion";
  function localStorageGet(k) { try { return localStorage.getItem("valdar." + k); } catch (e) { return null; } }
  function localStorageSet(k, v) { try { localStorage.setItem("valdar." + k, v); } catch (e) { /* rien */ } }
  function show(name) {
    tab = name; localStorageSet("tab", name);
    document.body.className = document.body.className.replace(/\btab-\S+/g, "").trim() + " tab-" + name;
    $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    (LOADERS[name] || (() => {}))();
  }
  $$("#tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));

  // ------------------------------------------------------------------ horloge
  setInterval(() => {
    $("#clock").textContent = new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  }, 1000);

  // ------------------------------------------------------------------ discussion
  const chat = $("#chat");
  let current = null;              // bulle de Valdar en cours d'écriture
  function nearBottom() { return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 160; }
  function bubble(kind, who, text, meta) {
    const stick = nearBottom();
    const el = document.createElement("div");
    el.className = "msg " + kind;
    el.innerHTML = `<div class="who">${esc(who)}</div><div class="body"></div>`;
    el.querySelector(".body").textContent = text;
    if (meta) {
      const m = document.createElement("span"); m.className = "meta"; m.textContent = meta;
      el.querySelector(".body").appendChild(m);
    }
    chat.appendChild(el);
    while (chat.children.length > 400) chat.removeChild(chat.firstChild);
    if (stick) chat.scrollTop = chat.scrollHeight;
    return el;
  }
  let subTimer = 0;
  function subtitle(text) {
    $("#subtitle").textContent = text;
    clearTimeout(subTimer);
    subTimer = setTimeout(() => { $("#subtitle").textContent = ""; }, 12000);
  }
  function onUser(d) {
    const who = d.who || "toi";
    const meta = d.source === "voix" ? `à l'oral${d.confidence ? " · " + Math.round(d.confidence * 100) + " % sûr" : ""}` : "";
    bubble("user", who, d.text, meta);
    current = bubble("valdar typing", "Valdar", "");
    Face.thinking(true);
  }
  function onToken(d) {
    if (!current) current = bubble("valdar typing", "Valdar", "");
    Face.thinking(false);
    const body = current.querySelector(".body");
    const stick = nearBottom();
    body.textContent += d.text;
    subtitle(body.textContent.slice(-220));
    if (stick) chat.scrollTop = chat.scrollHeight;
  }
  function onReply(d) {
    Face.thinking(false);
    if (current) {
      const body = current.querySelector(".body");
      if (!body.textContent.trim()) body.textContent = d.text;
      current.classList.remove("typing");
      if (d.tools && d.tools.length) {
        const m = document.createElement("span"); m.className = "meta";
        m.textContent = "outils : " + d.tools.join(", "); body.appendChild(m);
      }
      current = null;
    } else {
      bubble("valdar", "Valdar", d.text);
    }
    subtitle(d.text.slice(-220));
  }
  const EVENT_LABEL = { sursaut: "sursaut", douleur: "douleur", reflexe: "réflexe",
    vigie: "imprimante", vigie_triage: "imprimante", pensee: "pensée", nuit: "nuit",
    interoception: "charge", chien: "chien", rappel: "rappel", erreur: "erreur" };
  function onEvent(d) {
    if (d.kind === "pensee") return;       // ses pensées vont dans l'onglet Pensées
    if (d.kind === "sursaut") Face.startle();
    bubble("event", EVENT_LABEL[d.kind] || d.kind, d.text);
  }

  const input = $("#input");
  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, window.innerHeight * 0.4) + "px"; }
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = ""; autosize();
    await post("/api/message", { text });
  });

  // ------------------------------------------------------------------ boutons du visage
  $("#btn-hush").onclick = () => post("/api/control", { action: "tais-toi" });
  let muted = false, silenced = false;
  $("#btn-mute").onclick = () => post("/api/control", { action: muted ? "voix" : "muet" });
  $("#btn-silence").onclick = () => post("/api/control", { action: silenced ? "parle" : "silence" });
  $("#btn-focus").onclick = () => document.body.classList.toggle("focus");
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select")) return;
    if (e.key === "f" || e.key === "F") document.body.classList.toggle("focus");
    if (e.key === "Escape") post("/api/control", { action: "tais-toi" });
  });

  // ------------------------------------------------------------------ état (1 fois / s)
  let lastState = null;
  function gauge(label, v, signed, fmt) {
    const val = fmt ? fmt(v) : (signed ? (v >= 0 ? "+" : "") + v.toFixed(2) : Math.round(v * 100) + " %");
    let style;
    if (signed) {
      const x = Math.max(-1, Math.min(1, v));
      style = x >= 0 ? `left:50%;width:${x * 50}%` : `left:${50 + x * 50}%;width:${-x * 50}%`;
    } else style = `width:${Math.max(0, Math.min(1, v)) * 100}%`;
    return `<div class="gauge${signed ? " signed" : ""}"><span class="lab">${esc(label)}</span><span class="track"><span class="fill" style="${style}"></span></span><span class="val">${esc(val)}</span></div>`;
  }
  const VAR_LABEL = { dopamine: "dopamine", noradrenaline: "noradrénaline", serotonin: "sérotonine",
    oxytocin: "ocytocine", cortisol: "cortisol", energy: "énergie" };
  const ORGAN_LABEL = { corde: "la corde (euphorie)", ventre: "le ventre", coeur: "le cœur",
    chaleur: "la chaleur", tete: "la tête", gorge: "la gorge" };
  function onState(st) {
    lastState = st;
    Face.state(st);
    $("#chip-emotion").textContent = st.awake ? st.emotion : "il dort";
    $("#chip-mood").textContent = "humeur " + st.mood;
    const ears = $("#chip-ears");
    if (!st.ears) { ears.textContent = "pas d'oreilles"; ears.classList.remove("on"); }
    else { ears.textContent = st.ears.engaged ? "il t'écoute" : "dis « Valdar »"; ears.classList.toggle("on", st.ears.engaged); }
    $("#chip-who").textContent = st.who ? "parle avec " + st.who.name : "";
    const pain = $("#chip-pain");
    pain.classList.toggle("hidden", !st.pain || !st.pain.length);
    if (st.pain && st.pain.length) pain.textContent = st.pain[0];
    const load = $("#chip-load");
    load.classList.toggle("hidden", !st.load || st.load.level === "normal");
    if (st.load) load.textContent = "tête " + st.load.level;
    muted = !!st.muted; silenced = !!st.silenced;
    $("#btn-mute").classList.toggle("active-toggle", muted);
    $("#btn-mute").textContent = muted ? "Rendre la voix" : "Muet";
    $("#btn-silence").classList.toggle("active-toggle", silenced);
    if (tab === "coeur") renderHeart(st);
  }
  function renderHeart(st) {
    $("#heart-summary").textContent = st.awake
      ? `${st.emotion} (intensité ${Math.round(st.intensity * 100)} %), humeur ${st.mood}, cœur à ${st.bpm} bpm`
      : `il dort (cœur à ${st.bpm} bpm)`;
    $("#felt").innerHTML = (st.felt || []).map((t) => `<li>${esc(t)}</li>`).join("");
    $("#pad").innerHTML = gauge("plaisir", st.pad.P, true) + gauge("activation", st.pad.A, true) + gauge("dominance", st.pad.D, true);
    let senses = "";
    if (st.load) senses += gauge("charge mentale", st.load.value, false);
    if (st.ambient) senses += gauge("bruit de la pièce", (st.ambient.db + 70) / 70, false, () => st.ambient.mood + " (" + st.ambient.db + " dB)");
    $("#senses").innerHTML = senses;
    $("#vars").innerHTML = Object.entries(st.variables).map(([k, v]) => gauge(VAR_LABEL[k] || k, v, false)).join("");
    $("#organs").innerHTML = Object.entries(st.organs).map(([k, v]) => gauge(ORGAN_LABEL[k] || k, Math.abs(v) <= 1 && v < 0 ? v : v, v < 0)).join("");
    const NEED = { contact: "besoin de contact", novelty: "besoin de nouveauté" };
    $("#needs").innerHTML = Object.entries(st.needs).map(([k, v]) => gauge(NEED[k] || k, v, false)).join("");
  }

  // ------------------------------------------------------------------ foyer
  let enrollTimer = 0;
  function onEnroll(d) {
    const bar = $("#enroll-bar");
    bar.classList.remove("hidden");
    $("#enroll-text").textContent = d.done
      ? `${d.kind === "voix" ? "Voix" : "Visage"} appris (${d.prints} empreintes).`
      : `${d.kind === "voix" ? "J'écoute la voix" : "Je regarde le visage"}…`;
    $("#enroll-fill").style.width = Math.round((d.progress || 0) * 100) + "%";
    if (d.done) {
      toast($("#enroll-text").textContent);
      clearTimeout(enrollTimer);
      enrollTimer = setTimeout(() => bar.classList.add("hidden"), 4000);
      if (tab === "foyer") loadFoyer();
    }
  }
  $("#enroll-cancel").onclick = () => { post("/api/control", { action: "annuler" }); $("#enroll-bar").classList.add("hidden"); };
  async function loadFoyer() {
    const d = await get("/api/foyer");
    const now = Date.now() / 1000;
    const seen = Object.fromEntries(d.vus || []);
    $("#members").innerHTML = d.membres.map((m) => {
      const here = seen[m.id] && now - seen[m.id] < 60;
      const aff = m.affection >= 0.4 ? "il l'adore" : m.affection >= 0.15 ? "il l'apprécie" : m.affection <= -0.15 ? "relation tendue" : "il apprend à le connaître";
      return `<div class="member${here ? " here" : ""}">
        <div class="name">${esc(m.nom)}${m.mineur ? ' <span class="tag">enfant</span>' : ""}${m.role === "owner" ? ' <span class="tag valide">toi</span>' : ""}</div>
        <div class="lien">${esc(m.lien || "")}</div>
        <div class="prints"><span class="${m.voix ? "ok" : ""}">voix : ${m.voix || "—"}</span><span class="${m.visage ? "ok" : ""}">visage : ${m.visage || "—"}</span></div>
        <div class="small">${m.rencontres} rencontre(s) · ${aff}${m.sujets.length ? " · " + m.sujets.map((s) => esc(s[0])).join(", ") : ""}</div>
        ${m.lecons.length ? `<div class="small">${m.lecons.map(esc).join(" · ")}</div>` : ""}
        <div class="btns">
          <button data-a="voix" data-id="${esc(m.id)}" ${d.voix_ok ? "" : "disabled title=\"modèle de voix absent\""}>Apprendre sa voix</button>
          <button data-a="visage" data-id="${esc(m.id)}" ${d.visage_ok ? "" : "disabled title=\"caméra absente\""}>Apprendre son visage</button>
          <button data-a="oublier_empreintes" data-id="${esc(m.id)}">Effacer empreintes</button>
          ${m.role === "owner" ? "" : `<button class="danger" data-a="retirer" data-id="${esc(m.id)}">Retirer</button>`}
        </div></div>`;
    }).join("");
    $$("#members button").forEach((b) => b.onclick = async () => {
      if (b.dataset.a === "retirer" && !confirmIt("Retirer cette personne et effacer ses empreintes ?")) return;
      await post("/api/foyer", { action: b.dataset.a, id: b.dataset.id });
      loadFoyer();
    });
    $("#animals").innerHTML = d.animaux.length ? d.animaux.map((a) =>
      `<div class="row"><span class="grow">${esc(a.nom)} <span class="small">(${esc(a.espece)})</span></span><button data-n="${esc(a.nom)}">Retirer</button></div>`).join("") : '<p class="hint">Aucun animal.</p>';
    $$("#animals button").forEach((b) => b.onclick = async () => { await post("/api/foyer", { action: "retirer_animal", nom: b.dataset.n }); loadFoyer(); });
    const cam = d.camera;
    $("#camera").innerHTML = !cam ? '<p class="hint">Caméra désactivée ou absente (onglet Réglages).</p>'
      : cam.error ? `<p class="hint">Erreur caméra : ${esc(cam.error)}</p>`
      : cam.scene ? `<p>${cam.scene.persons} personne(s) · chien : ${cam.scene.dog ? (cam.scene.dog_on_table ? "<b>sur la table !</b>" : "oui") : "non"}</p>
         <p class="small">${(d.vus || []).map(([p, t]) => esc(p) + " vu à " + hm(t)).join(" · ") || "personne de reconnu pour l'instant"}</p>`
      : '<p class="hint">La caméra démarre…</p>';
  }
  function confirmIt(msg) { return window.confirm(msg); }
  $("#add-member").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    await post("/api/foyer", { action: "ajouter", nom: f.nom.value, lien: f.lien.value, mineur: f.mineur.checked });
    f.reset(); loadFoyer();
  });
  $("#add-animal").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    await post("/api/foyer", { action: "animal", nom: f.nom.value, espece: f.espece.value });
    f.reset(); loadFoyer();
  });

  // ------------------------------------------------------------------ mémoire
  let memTimer = 0;
  const STATUT = { valide: "valide", hypothese: "hypothèse", quarantaine: "quarantaine" };
  const ORIGINE = { dit: "on me l'a dit", lu: "lu", deduit: "ma déduction" };
  async function loadMem() {
    const q = $("#mem-q").value, st = $("#mem-statut").value;
    const d = await get(`/api/memoire?q=${encodeURIComponent(q)}&statut=${encodeURIComponent(st)}`);
    $("#mem-list").innerHTML = d.croyances.length ? d.croyances.map((c) => `
      <div class="item"><div class="grow">
        <span class="tag ${esc(c.statut)}">${esc(STATUT[c.statut] || c.statut)}</span><span class="tag">${esc(ORIGINE[c.origine] || c.origine)}</span>${esc(c.text)}
        <div class="small">confiance ${Math.round(c.confidence * 100)} % · ${c.pour} pour / ${c.contre} contre${c.motif ? " · " + esc(c.motif) : ""}${c.refutation ? " · réfuté si : " + esc(c.refutation) : ""}</div>
      </div><div class="actions">
        ${c.statut === "quarantaine" ? `<button data-a="lever" data-id="${c.id}">Lever</button>` : `<button data-a="quarantaine" data-id="${c.id}">Douteux</button>`}
        <button class="danger" data-a="oublier" data-id="${c.id}">Oublier</button>
      </div></div>`).join("") : '<p class="hint">Rien.</p>';
    $$("#mem-list button").forEach((b) => b.onclick = async () => { await post("/api/memoire", { action: b.dataset.a, id: b.dataset.id }); loadMem(); });
  }
  $("#mem-q").addEventListener("input", () => { clearTimeout(memTimer); memTimer = setTimeout(loadMem, 300); });
  $("#mem-statut").addEventListener("change", loadMem);
  $("#mem-add").addEventListener("submit", async (e) => { e.preventDefault(); await post("/api/memoire", { action: "ajouter", texte: e.target.texte.value }); e.target.reset(); loadMem(); });

  // ------------------------------------------------------------------ pensées
  async function loadThoughts() {
    const d = await get("/api/pensees");
    $("#night").textContent = d.nuit ? "Dernière nuit : " + d.nuit : "";
    $("#ideas").innerHTML = d.idees.length ? d.idees.map((i) => `
      <div class="item"><div class="grow"><span class="tag">${esc(i.statut)}</span>${esc(i.texte)}<div class="small">${esc(i.pourquoi)} · ${day(i.t)} ${hm(i.t)}</div></div>
      <div class="actions"><button data-s="acceptee" data-id="${i.id}">Bonne idée</button><button data-s="refusee" data-id="${i.id}">Non</button></div></div>`).join("") : '<p class="hint">Pas encore d\'idée.</p>';
    $$("#ideas button").forEach((b) => b.onclick = async () => { await post("/api/idee", { id: b.dataset.id, statut: b.dataset.s }); loadThoughts(); });
    $("#thoughts").innerHTML = d.pensees.map((p) => `<div class="item"><div class="grow">${esc(p.texte)}${p.curiosite ? `<div class="small">curieux de : ${esc(p.curiosite)}</div>` : ""}<div class="small">${day(p.t)} ${hm(p.t)}</div></div></div>`).join("") || '<p class="hint">Il n\'a pas encore pensé dans son coin.</p>';
  }

  // ------------------------------------------------------------------ atelier
  async function loadAtelier() {
    const d = await get("/api/atelier");
    $("#printer").textContent = d.imprimante || "pas d'imprimante branchée";
    $("#vigie").textContent = d.vigie || "";
    $("#checklist").textContent = d.checklist || "aucune checklist en cours";
    $("#reminders").innerHTML = d.rappels.length ? d.rappels.map((r) => `<div class="row"><span class="grow">${esc(r.texte)}</span><span class="small">${day(r.due)} ${hm(r.due)}</span></div>`).join("") : '<p class="hint">Aucun rappel.</p>';
    const low = new Set(d.bas.map((x) => x.name || x.nom));
    $("#stock").innerHTML = d.stock.length ? `<div class="list">${d.stock.map((s) => {
      const name = s.name || s.nom, qty = s.qty ?? s.quantite, unit = s.unit || s.unite || "", loc = s.loc || s.localisation || "";
      return `<div class="item"><div class="grow">${esc(name)}${low.has(name) ? ' <span class="tag quarantaine">bas</span>' : ""}<div class="small">${esc(loc)}</div></div><div>${esc(qty)} ${esc(unit)}</div></div>`;
    }).join("")}</div>` : '<p class="hint">Stock vide.</p>';
  }

  // ------------------------------------------------------------------ savoir
  let knowTimer = 0;
  const TIER = { safe: "libre", safety: "sécurité", elevated: "toi seulement", dangerous: "avec confirmation" };
  async function loadKnow() {
    const q = $("#know-q").value;
    const d = await get(`/api/connaissances?q=${encodeURIComponent(q)}`);
    $("#kiwix").textContent = d.kiwix || "";
    $("#know-res").innerHTML = d.resultats.map((h) => `<div class="item"><div class="grow"><b>${esc(h.titre || h.doc)}</b> <span class="small">${esc(h.doc)} · ${h.score}</span><div>${esc(h.texte)}</div></div></div>`).join("");
    $("#docs").innerHTML = `<p class="small">${d.total} passages dans ${d.docs.length} documents</p>` + d.docs.map((x) => `<span class="tag">${esc(x.doc)} (${x.passages})</span>`).join(" ");
    const t = await get("/api/outils");
    $("#tools").innerHTML = `<div class="list">${t.outils.map((o) => `<div class="item"><div class="grow"><b>${esc(o.nom)}</b> <span class="tag">${esc(TIER[o.niveau] || o.niveau)}</span><span class="tag">${esc(o.famille)}</span><div class="small">${esc(o.description)}</div></div></div>`).join("")}</div>`;
  }
  $("#know-q").addEventListener("input", () => { clearTimeout(knowTimer); knowTimer = setTimeout(loadKnow, 350); });

  // ------------------------------------------------------------------ journal
  async function loadJournal() {
    const d = await get("/api/journal");
    let lastDay = "";
    $("#journal").innerHTML = d.tours.slice().reverse().map((t) => {
      const dd = day(t.t);
      const head = dd !== lastDay ? `<h3>${esc(dd)}</h3>` : "";
      lastDay = dd;
      return head + `<div class="item"><div class="grow"><b>${esc(t.speaker)}</b> <span class="small">${hm(t.t)}${t.quarantaine ? " · quarantaine : " + esc(t.quarantaine) : ""}</span><div>${esc(t.text)}</div></div></div>`;
    }).join("") || '<p class="hint">Rien encore.</p>';
  }

  // ------------------------------------------------------------------ réglages
  async function loadSettings() {
    const d = await get("/api/reglages");
    const groups = {};
    d.reglages.forEach((s) => (groups[s.group] = groups[s.group] || []).push(s));
    $("#settings").innerHTML = Object.entries(groups).map(([g, items]) => `<div class="card"><h3>${esc(g)}</h3>${items.map((s) => control(s, d.modeles)).join("")}</div>`).join("");
    $$("#settings [data-path]").forEach((el) => {
      const send = async () => {
        let v = el.type === "checkbox" ? el.checked : el.value;
        const r = await post("/api/reglages", { path: el.dataset.path, value: v });
        if (r.ok) toast(r.restart ? "Gardé — au prochain démarrage." : "Appliqué.");
        const out = el.parentElement.querySelector(".num"); if (out && r.ok) out.textContent = r.value;
      };
      if (el.type === "range") {
        el.addEventListener("input", () => { const o = el.parentElement.querySelector(".num"); if (o) o.textContent = el.value; });
        el.addEventListener("change", send);
      } else el.addEventListener("change", send);
    });
  }
  function control(s, models) {
    const v = s.value;
    const help = s.help ? `<span class="help">${esc(s.help)}</span>` : "";
    const rs = s.restart ? '<span class="restart">au redémarrage</span>' : "";
    let ctl;
    if (s.type === "bool") ctl = `<label class="check"><input type="checkbox" data-path="${esc(s.path)}" ${v ? "checked" : ""}> activé</label>`;
    else if (s.type === "model") {
      const opts = (models.includes(v) ? models : [v, ...models]).map((m) => `<option ${m === v ? "selected" : ""}>${esc(m)}</option>`).join("");
      ctl = `<select data-path="${esc(s.path)}">${opts}</select>`;
    } else if (s.type === "choice") ctl = `<select data-path="${esc(s.path)}">${s.options.map((o) => `<option ${o === v ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
    else if (s.type === "int" || s.type === "float") ctl = `<input type="range" data-path="${esc(s.path)}" min="${s.min}" max="${s.max}" step="${s.step}" value="${v}"><span class="num">${v}</span>`;
    else if (s.type === "list") ctl = `<input data-path="${esc(s.path)}" value="${esc((v || []).join(", "))}">`;
    else ctl = `<input data-path="${esc(s.path)}" value="${esc(v)}">`;
    return `<div class="setting"><div>${esc(s.label)}${rs}${help}</div><div class="ctl">${ctl}</div></div>`;
  }

  const LOADERS = { foyer: loadFoyer, memoire: loadMem, pensees: loadThoughts, atelier: loadAtelier,
    savoir: loadKnow, journal: loadJournal, reglages: loadSettings,
    coeur: () => lastState && renderHeart(lastState) };
  setInterval(() => { if (tab === "foyer") loadFoyer(); if (tab === "atelier") loadAtelier(); }, 5000);

  // ------------------------------------------------------------------ direct (SSE)
  function connect() {
    const es = new EventSource("/api/events");
    es.addEventListener("state", (e) => onState(JSON.parse(e.data)));
    es.addEventListener("mouth", (e) => Face.mouth(JSON.parse(e.data).level));
    es.addEventListener("user", (e) => onUser(JSON.parse(e.data)));
    es.addEventListener("token", (e) => onToken(JSON.parse(e.data)));
    es.addEventListener("reply", (e) => onReply(JSON.parse(e.data)));
    es.addEventListener("heard", () => { /* la bulle « user » suit */ });
    es.addEventListener("spontaneous", (e) => { const d = JSON.parse(e.data); bubble("valdar", "Valdar", d.text, "de lui-même"); subtitle(d.text); });
    es.addEventListener("event", (e) => onEvent(JSON.parse(e.data)));
    es.addEventListener("enroll", (e) => onEnroll(JSON.parse(e.data)));
    es.addEventListener("scene", () => {});
    es.onerror = () => { $("#chip-emotion").textContent = "déconnecté…"; };
  }
  (async function boot() {
    try {
      const h = await get("/api/historique");
      for (const m of h.log) {
        if (m.kind === "user") bubble("user", m.data.who || "toi", m.data.text);
        else if (m.kind === "reply") bubble("valdar", "Valdar", m.data.text);
        else if (m.kind === "spontaneous") bubble("valdar", "Valdar", m.data.text, "de lui-même");
        else if (m.kind === "event" && m.data.kind !== "pensee") bubble("event", EVENT_LABEL[m.data.kind] || m.data.kind, m.data.text);
      }
      chat.scrollTop = chat.scrollHeight;
      onState(await get("/api/state"));
    } catch (e) { /* le direct prendra le relais */ }
    connect();
    show(tab);
  })();
})();

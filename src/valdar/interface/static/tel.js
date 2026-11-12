/* L'appli téléphone de Valdar.
 *
 * Le jeton arrive dans l'adresse après le #, qui n'est jamais envoyé au serveur : la page
 * le lit, le garde dans le téléphone, et le remet à chaque appel dans un en-tête.
 *
 * Le direct passe par le flux d'événements du PC (SSE) : ses phrases s'écrivent au fur et
 * à mesure, et chaque phrase dite arrive avec un petit son à jouer.
 *
 * Le micro : on enregistre le son brut, on le ramène à 16 kHz et on fabrique le WAV ici.
 * Comme ça le PC n'a aucun décodeur à installer, et rien ne part chez personne d'autre.
 */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const CLE = "valdar.jeton";

  // ---------------------------------------------------------------- le jeton
  let jeton = "";
  const frag = new URLSearchParams((location.hash || "").replace(/^#/, ""));
  if (frag.get("j")) {
    jeton = frag.get("j");
    try { localStorage.setItem(CLE, jeton); } catch (e) { /* navigation privée */ }
    history.replaceState(null, "", location.pathname);      // on efface le jeton de la barre
  } else {
    try { jeton = localStorage.getItem(CLE) || ""; } catch (e) { jeton = ""; }
  }

  const entetes = (extra) => Object.assign({ "X-Valdar-Jeton": jeton }, extra || {});
  async function api(chemin, options) {
    const r = await fetch(chemin, Object.assign({ cache: "no-store" }, options, {
      headers: entetes((options || {}).headers),
    }));
    if (r.status === 401) { throw new Error("appairage à refaire : rescanne le QR code"); }
    return r;
  }

  // ---------------------------------------------------------------- le fil
  const fil = $("fil");
  let bulleEnCours = null;

  function colle(classe, texte) {
    const d = document.createElement("div");
    d.className = classe;
    d.textContent = texte;
    fil.appendChild(d);
    fil.scrollTop = fil.scrollHeight;
    return d;
  }
  function note(texte, rate) { return colle("note" + (rate ? " rate" : ""), texte); }
  function toast(texte) {
    const t = $("toast");
    t.textContent = texte;
    t.classList.add("on");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.remove("on"), 3200);
  }
  function attend(on) {
    if (!on) { if (attend._d) { attend._d.remove(); attend._d = null; } return; }
    if (attend._d) return;
    attend._d = colle("bulle lui points", "");
    attend._d.innerHTML = "<span>.</span><span>.</span><span>.</span>";
  }

  // ---------------------------------------------------------------- sa voix
  const file = [];
  let joue = false;
  const audio = new Audio();
  audio.preload = "auto";
  let ctxAnalyse = null, analyse = null, source = null;

  function bouche() {           // la bouche du visage suit vraiment le son joué
    if (!analyse || !window.Face) return;
    const buf = new Uint8Array(analyse.frequencyBinCount);
    analyse.getByteFrequencyData(buf);
    let s = 0;
    for (let i = 2; i < 24; i++) s += buf[i];
    window.Face.mouth(Math.min(1, s / (22 * 160)));
    if (!audio.paused) requestAnimationFrame(bouche);
    else window.Face.mouth(0);
  }

  function brancheAnalyse() {
    if (ctxAnalyse || !window.AudioContext) return;
    try {
      ctxAnalyse = new AudioContext();
      source = ctxAnalyse.createMediaElementSource(audio);
      analyse = ctxAnalyse.createAnalyser();
      analyse.fftSize = 256;
      source.connect(analyse);
      analyse.connect(ctxAnalyse.destination);
    } catch (e) { ctxAnalyse = null; analyse = null; }
  }

  async function suivant() {
    if (joue || !file.length) return;
    joue = true;
    const id = file.shift();
    try {
      brancheAnalyse();
      if (ctxAnalyse && ctxAnalyse.state === "suspended") await ctxAnalyse.resume();
      audio.src = "/api/tel/voix/" + id + "?j=" + encodeURIComponent(jeton);
      await audio.play();
      requestAnimationFrame(bouche);
      await new Promise((ok) => { audio.onended = ok; audio.onerror = ok; });
    } catch (e) {
      toast("touche l'écran une fois pour autoriser le son");
    }
    joue = false;
    suivant();
  }

  // Dernier recours : la voix du système, si la sienne n'est pas chargée sur le PC.
  let voixSysteme = false, voixPrete = null;
  function litSysteme(texte) {
    if (!voixSysteme || !window.speechSynthesis || !texte) return;
    const u = new SpeechSynthesisUtterance(texte);
    u.lang = "fr-FR";
    if (voixPrete) u.voice = voixPrete;
    u.onboundary = () => window.Face && window.Face.mouth(0.5 + Math.random() * 0.4);
    u.onend = () => window.Face && window.Face.mouth(0);
    window.speechSynthesis.speak(u);
  }
  if (window.speechSynthesis) {
    const choisis = () => {
      const v = window.speechSynthesis.getVoices().filter((x) => /^fr/i.test(x.lang));
      voixPrete = v.find((x) => /thomas|paul|homme|male/i.test(x.name)) || v[0] || null;
    };
    choisis();
    window.speechSynthesis.onvoiceschanged = choisis;
  }

  // ---------------------------------------------------------------- le direct
  let flux = null, vivant = false;
  function ecouteLePc() {
    if (flux) flux.close();
    flux = new EventSource("/api/events?j=" + encodeURIComponent(jeton));
    flux.onopen = () => { vivant = true; etat(); };
    flux.onerror = () => { vivant = false; etat("hors de portée — je réessaie…"); };
    flux.addEventListener("token", (e) => {
      const d = JSON.parse(e.data || "{}");
      attend(false);
      if (!bulleEnCours) bulleEnCours = colle("bulle lui parle", "");
      bulleEnCours.textContent += d.text || "";
      fil.scrollTop = fil.scrollHeight;
    });
    flux.addEventListener("reply", (e) => {
      const d = JSON.parse(e.data || "{}");
      attend(false);
      if (!bulleEnCours) bulleEnCours = colle("bulle lui", d.text || "");
      bulleEnCours.classList.remove("parle");
      if ((d.text || "").trim() && bulleEnCours.textContent.trim() !== (d.text || "").trim()
          && !bulleEnCours.textContent.trim()) bulleEnCours.textContent = d.text;
      if (d.tools && d.tools.length) colle("outils", "outils : " + d.tools.join(", "));
      if (d.error) note(d.error, true);
      if (voixSysteme && !file.length) litSysteme(d.text);
      bulleEnCours = null;
    });
    flux.addEventListener("tel_audio", (e) => {
      const d = JSON.parse(e.data || "{}");
      if (d.id) { file.push(d.id); suivant(); }
    });
    flux.addEventListener("spontaneous", (e) => {
      const d = JSON.parse(e.data || "{}");
      if (d.text) colle("bulle lui", d.text);
    });
    flux.addEventListener("state", (e) => {
      const d = JSON.parse(e.data || "{}");
      if (window.Face) window.Face.state(d);
      etat();
    });
  }

  let dernierEtat = {};
  function etat(message) {
    const e = $("etat");
    if (message) { e.textContent = message; return; }
    if (!vivant) { e.textContent = "hors de portée"; return; }
    const v = dernierEtat.voix || {};
    const bouts = [];
    if (dernierEtat.ressenti) bouts.push(dernierEtat.ressenti);
    else if (dernierEtat.humeur) bouts.push(dernierEtat.humeur);
    bouts.push(v.voix ? "sa voix" : (voixSysteme ? "voix du téléphone" : "sans voix"));
    if (dernierEtat.corps === false) bouts.push("PC fermé");
    e.textContent = bouts.join(" · ");
  }

  async function rafraichit() {
    try {
      const r = await api("/api/tel/etat");
      dernierEtat = await r.json();
      const v = dernierEtat.voix || {};
      voixSysteme = !v.voix && !!v.secours_navigateur;
      etat();
    } catch (e) { etat(e.message); }
  }

  // ---------------------------------------------------------------- écrire
  $("bas").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const champ = $("texte");
    const texte = champ.value.trim();
    if (!texte) return;
    champ.value = "";
    colle("bulle moi", texte);
    attend(true);
    try {
      const r = await api("/api/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: texte, voix: true }),
      });
      const d = await r.json();
      if (!d.ok) { attend(false); note(d.error || "il n'a pas pris le message", true); }
    } catch (e) { attend(false); note(e.message, true); }
  });

  // ---------------------------------------------------------------- parler
  let flot = null, ctxMicro = null, morceaux = [], enregistre = false, tailleTotale = 0;
  const bMicro = $("b-micro");

  function wav(chunks, total, rate) {
    const pcm = new Float32Array(total);
    let i = 0;
    for (const c of chunks) { pcm.set(c, i); i += c.length; }
    const cible = 16000;
    const n = Math.max(1, Math.round(pcm.length * cible / rate));
    const out = new Int16Array(n);
    for (let k = 0; k < n; k++) {                 // ré-échantillonnage linéaire
      const x = k * rate / cible, a = Math.floor(x), b = Math.min(a + 1, pcm.length - 1);
      const v = pcm[a] + (pcm[b] - pcm[a]) * (x - a);
      out[k] = Math.max(-32768, Math.min(32767, v * 32767));
    }
    const buf = new ArrayBuffer(44 + out.length * 2), vue = new DataView(buf);
    const txt = (p, s) => { for (let j = 0; j < s.length; j++) vue.setUint8(p + j, s.charCodeAt(j)); };
    txt(0, "RIFF"); vue.setUint32(4, 36 + out.length * 2, true); txt(8, "WAVEfmt ");
    vue.setUint32(16, 16, true); vue.setUint16(20, 1, true); vue.setUint16(22, 1, true);
    vue.setUint32(24, cible, true); vue.setUint32(28, cible * 2, true);
    vue.setUint16(32, 2, true); vue.setUint16(34, 16, true);
    txt(36, "data"); vue.setUint32(40, out.length * 2, true);
    new Int16Array(buf, 44).set(out);
    return new Blob([buf], { type: "audio/wav" });
  }

  async function ouvreMicro() {
    if (ctxMicro) return true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("le micro n'est pas disponible ici (il faut du https)");
      return false;
    }
    try {
      flot = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (e) {
      toast("micro refusé : autorise-le dans le navigateur");
      return false;
    }
    ctxMicro = new AudioContext();
    const src = ctxMicro.createMediaStreamSource(flot);
    const noeud = ctxMicro.createScriptProcessor(4096, 1, 1);
    noeud.onaudioprocess = (e) => {
      if (!enregistre) return;
      const d = e.inputBuffer.getChannelData(0);
      morceaux.push(new Float32Array(d));
      tailleTotale += d.length;
      if (tailleTotale > ctxMicro.sampleRate * 30) arrete();     // 30 s, pas plus
    };
    src.connect(noeud);
    noeud.connect(ctxMicro.destination);
    return true;
  }

  async function demarre() {
    if (enregistre) return;
    if (!(await ouvreMicro())) return;
    if (ctxMicro.state === "suspended") await ctxMicro.resume();
    morceaux = []; tailleTotale = 0; enregistre = true;
    bMicro.classList.add("ecoute");
    if (window.navigator.vibrate) navigator.vibrate(12);
  }

  async function arrete() {
    if (!enregistre) return;
    enregistre = false;
    bMicro.classList.remove("ecoute");
    if (tailleTotale < ctxMicro.sampleRate * 0.3) { toast("trop court"); return; }
    const blob = wav(morceaux, tailleTotale, ctxMicro.sampleRate);
    morceaux = [];
    const mien = colle("bulle moi", "…");
    attend(true);
    try {
      const r = await api("/api/tel/ecouter", { method: "POST", body: blob,
        headers: { "Content-Type": "audio/wav" } });
      const d = await r.json();
      if (d.ok) { mien.textContent = d.texte || "…"; }
      else { mien.remove(); attend(false); note(d.error || "je n'ai pas compris", true); }
    } catch (e) { mien.remove(); attend(false); note(e.message, true); }
  }

  ["pointerdown"].forEach((n) => bMicro.addEventListener(n, (e) => { e.preventDefault(); demarre(); }));
  ["pointerup", "pointercancel", "pointerleave"].forEach((n) =>
    bMicro.addEventListener(n, (e) => { e.preventDefault(); arrete(); }));

  // ---------------------------------------------------------------- ses yeux
  let camOn = false, camFlot = null, timerCam = null;
  const bCam = $("b-cam"), video = $("cam"), pige = $("pige");

  async function camera(on) {
    if (on === camOn) return;
    if (!on) {
      camOn = false;
      clearInterval(timerCam); timerCam = null;
      if (camFlot) camFlot.getTracks().forEach((t) => t.stop());
      camFlot = null; video.srcObject = null;
      document.body.classList.remove("camera");
      bCam.setAttribute("aria-pressed", "false");
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("la caméra demande du https"); return;
    }
    try {
      camFlot = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 } }, audio: false });
    } catch (e) { toast("caméra refusée"); return; }
    video.srcObject = camFlot;
    camOn = true;
    document.body.classList.add("camera");
    bCam.setAttribute("aria-pressed", "true");
    const periode = Math.max(500, (dernierEtat.camera_secondes || 2.5) * 1000);
    timerCam = setInterval(envoieImage, periode);
    envoieImage();
  }

  async function envoieImage() {
    if (!camOn || video.videoWidth < 2) return;
    const l = 480, h = Math.round(l * video.videoHeight / video.videoWidth);
    pige.width = l; pige.height = h;
    pige.getContext("2d").drawImage(video, 0, 0, l, h);
    const blob = await new Promise((ok) => pige.toBlob(ok, "image/jpeg", 0.7));
    if (!blob) return;
    try {
      const r = await api("/api/tel/voir", { method: "POST", body: blob,
        headers: { "Content-Type": "image/jpeg" } });
      const d = await r.json();
      if (d.qui && d.qui.id) etat("il te voit : " + d.qui.id);
      else if (d.erreur) { toast(d.erreur); camera(false); }
    } catch (e) { /* une image perdue n'est pas un drame */ }
  }
  bCam.addEventListener("click", () => camera(!camOn));

  // ---------------------------------------------------------------- départ
  (async function demarrage() {
    await window.chargerVisage("visage-hote");
    if (!jeton) {
      note("Pas encore appairé. Sur le PC : python -m valdar tel — puis scanne le QR code.",
           true);
      return;
    }
    await rafraichit();
    ecouteLePc();
    setInterval(rafraichit, 20000);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) { rafraichit(); if (!vivant) ecouteLePc(); }
    });
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("sw.js").catch(() => {});
    }
  })();
})();

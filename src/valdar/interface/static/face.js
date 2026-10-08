/* Le visage de Valdar : piloté par son cœur (plaisir, activation, dominance, émotion),
   la bouche par le son réel de sa voix. Rien n'est joué au hasard, sauf les clignements. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const root = document.documentElement;

  // couleurs par émotion : [lueur, œil, œil profond, joues, oreilles]
  const PALETTE = {
    calme: ["#4cc9ff", "#6fe0ff", "#1677c9", "#4cc9ff"],
    sérénité: ["#5fe3d0", "#8ff3e4", "#138a86", "#5fe3d0"],
    joie: ["#ffc35c", "#ffe08a", "#d9821c", "#ff9f6e"],
    euphorie: ["#ffd25c", "#fff09a", "#e69117", "#ff7f8e"],
    amusement: ["#ffb85c", "#ffe0a0", "#d07a1c", "#ff9f8a"],
    tendresse: ["#ff9fc8", "#ffd1e6", "#c2508a", "#ff8fb8"],
    curiosité: ["#7cf0a6", "#b5ffd0", "#1f9a5c", "#7cf0a6"],
    fierté: ["#f2a950", "#ffd9a0", "#b8701a", "#f2a950"],
    surprise: ["#b8a2ff", "#ddd1ff", "#6b4fd8", "#b8a2ff"],
    gêne: ["#ff9f9f", "#ffd0d0", "#c45656", "#ff8f8f"],
    anxiété: ["#a58cff", "#d6c9ff", "#5a3fc4", "#8f7ad8"],
    peur: ["#9d7cff", "#cfc0ff", "#4d2fb8", "#7d64d0"],
    frustration: ["#ff8a5c", "#ffc2a6", "#c4501f", "#ff8a5c"],
    colère: ["#ff5c5c", "#ffaaaa", "#b81f1f", "#ff5c5c"],
    déception: ["#7d9cc4", "#b8cbe6", "#3d5f8c", "#7d9cc4"],
    tristesse: ["#6f8fd6", "#a9c1f0", "#2f4f9a", "#6f8fd6"],
    solitude: ["#7a8fc0", "#b0c0e8", "#3a4f86", "#7a8fc0"],
    mélancolie: ["#8a8fd0", "#bfc2f0", "#4a4f96", "#8a8fd0"],
    ennui: ["#7f9cbc", "#b3c7dd", "#46607e", "#7f9cbc"],
    fatigue: ["#6d86a8", "#a3b7d0", "#3a4f6e", "#6d86a8"],
    sommeil: ["#2b5c8a", "#4c7fb0", "#163a5e", "#2b5c8a"],
  };

  const S = {               // état voulu (cible)
    P: 0, A: 0, D: 0, emotion: "calme", awake: true, bpm: 70,
    mouth: 0, speaking: false, listening: false, thinking: false,
    lookX: 0, lookY: 0, startle: 0,
  };
  const C = { P: 0, A: 0, D: 0, mouth: 0, lookX: 0, lookY: 0, open: 1, startle: 0 };   // courant
  let blinkUntil = 0, nextBlink = performance.now() + 2500, nextSaccade = 0;
  let lastMouthAt = 0;

  function lerp(a, b, k) { return a + (b - a) * k; }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

  function setColors(emotion) {
    const p = PALETTE[emotion] || PALETTE.calme;
    root.style.setProperty("--accent", p[0]);
    root.style.setProperty("--eye", p[1]);
    root.style.setProperty("--eye-deep", p[2]);
    root.style.setProperty("--cheek", p[3]);
  }

  function eye(g, downTop, upBot, px, py, pupil) {
    g.querySelector(".lid-top").setAttribute("y", (-100 + downTop).toFixed(1));
    g.querySelector(".lid-bot").setAttribute("y", (30 - upBot).toFixed(1));
    const p = g.querySelector(".pupil");
    p.setAttribute("cx", px.toFixed(1)); p.setAttribute("cy", py.toFixed(1));
    p.setAttribute("r", pupil.toFixed(1));
    const sp = g.querySelector(".spark");
    sp.setAttribute("cx", (px - 7).toFixed(1)); sp.setAttribute("cy", (py - 9).toFixed(1));
  }

  function frame(now) {
    const k = 0.08;
    C.P = lerp(C.P, S.P, k); C.A = lerp(C.A, S.A, k); C.D = lerp(C.D, S.D, k);
    C.lookX = lerp(C.lookX, S.lookX, 0.12); C.lookY = lerp(C.lookY, S.lookY, 0.12);
    C.startle = lerp(C.startle, S.startle, 0.25); S.startle *= 0.93;
    // la bouche suit le son ; si plus de nouvelle, elle se referme
    if (now - lastMouthAt > 250) S.mouth = 0;
    C.mouth = lerp(C.mouth, S.mouth, 0.45);

    // regard : petites saccades quand il ne fait rien ; en haut quand il réfléchit
    if (now > nextSaccade) {
      nextSaccade = now + 1200 + Math.random() * 2800;
      S.lookX = (Math.random() - 0.5) * (S.listening ? 4 : 12);
      S.lookY = (Math.random() - 0.5) * 6;
    }
    let lx = C.lookX, ly = C.lookY;
    if (S.thinking) { lx = 8; ly = -9; }

    // clignements (plus fréquents quand il est activé ou fatigué)
    if (now > nextBlink) {
      blinkUntil = now + 120;
      nextBlink = now + 2200 + Math.random() * 4200 * (1 - 0.4 * clamp(C.A, 0, 1));
    }
    const blinking = now < blinkUntil;

    // ouverture des yeux (0 fermé, 1 grand ouvert) : activation et surprise les ouvrent,
    // la tristesse les alourdit, la joie remonte la paupière du bas (yeux rieurs)
    let oTop = 0.75 + 0.25 * C.A + 0.4 * C.startle - (C.P < -0.2 ? 0.5 * (-C.P - 0.2) : 0);
    let oBot = 1 - (C.P > 0.15 ? 0.9 * (C.P - 0.15) : 0);
    oTop = clamp(oTop, 0.08, 1); oBot = clamp(oBot, 0.35, 1);
    if (!S.awake || blinking) { oTop = 0; oBot = 1; }
    const openTop = (1 - oTop) * 62;          // de combien la paupière du haut descend
    const openBot = (1 - oBot) * 30;          // de combien celle du bas remonte
    const pupil = clamp(11 + 3 * C.A - 2 * C.startle + (S.listening ? 1.5 : 0), 7, 16);
    eye($("eye-l"), openTop, openBot, lx, ly, pupil);
    eye($("eye-r"), openTop, openBot, lx, ly, pupil);

    // sourcils : dominance (confiant / inquiet), activation (levés), colère (froncés)
    const raise = 8 * C.A + 10 * C.startle;
    const tilt = 10 * C.D - (S.emotion === "colère" || S.emotion === "frustration" ? 12 : 0)
      + (C.P < -0.3 ? -6 : 0);
    const by = 140 - raise;
    // tilt > 0 : intérieur plus haut ? non : confiant = sourcils plats ; inquiet = intérieur relevé
    const inner = by - Math.max(0, -tilt) * 0.8 + Math.max(0, -C.D) * 6;
    const outer = by + Math.min(0, tilt) * -0.2;
    const angry = (S.emotion === "colère" || S.emotion === "frustration") ? 10 : 0;
    $("brow-l").setAttribute("d",
      `M116 ${outer.toFixed(1)} Q150 ${(by - 12).toFixed(1)} 184 ${(inner + angry).toFixed(1)}`);
    $("brow-r").setAttribute("d",
      `M216 ${(inner + angry).toFixed(1)} Q250 ${(by - 12).toFixed(1)} 284 ${outer.toFixed(1)}`);

    // bouche : courbe = plaisir ; ouverture = voix réelle
    const smile = clamp(C.P, -1, 1) * 16;
    const open = C.mouth * 26 + (C.startle > 0.3 ? 8 * C.startle : 0);
    const w = 42 + 6 * C.P;
    const y = 306, cy = y - smile * 0.35;              // coins relevés quand il sourit
    const mid = y + smile * 0.55;
    $("mouth").setAttribute("d",
      `M${200 - w} ${cy} Q200 ${mid - open * 0.15} ${200 + w} ${cy} Q200 ${mid + 3 + open} ${200 - w} ${cy}Z`);
    $("mouth-glow").setAttribute("opacity", (0.2 + 0.8 * C.mouth).toFixed(2));

    // joues : plaisir et tendresse
    const ch = clamp(0.15 + 0.9 * C.P, 0, 1);
    $("cheek-l").setAttribute("opacity", ch.toFixed(2));
    $("cheek-r").setAttribute("opacity", ch.toFixed(2));

    // halo : bat au rythme de son cœur
    const beat = 60000 / clamp(S.bpm || 70, 40, 160);
    const ph = (now % beat) / beat;
    const pulse = Math.exp(-ph * 7);
    $("halo").setAttribute("r", (178 + 5 * pulse).toFixed(1));
    $("halo").setAttribute("opacity", (0.25 + 0.35 * pulse).toFixed(2));
    $("halo2").setAttribute("transform", `rotate(${(now / 120) % 360} 200 215)`);
    $("antenna").setAttribute("opacity", (0.45 + 0.55 * pulse).toFixed(2));

    // oreilles : dorées quand il écoute
    const ear = S.listening ? "#f2a950" : (S.awake ? "#2b6fb8" : "#16324f");
    $("ear-l").setAttribute("fill", ear); $("ear-r").setAttribute("fill", ear);

    $("zzz").classList.toggle("on", !S.awake);
    requestAnimationFrame(frame);
  }

  window.Face = {
    state(st) {
      if (!st) return;
      S.P = st.pad ? st.pad.P : 0; S.A = st.pad ? st.pad.A : 0; S.D = st.pad ? st.pad.D : 0;
      S.awake = !!st.awake; S.bpm = st.bpm || 70;
      if (st.emotion !== S.emotion) { S.emotion = st.emotion; setColors(st.emotion); }
      S.speaking = !!st.speaking;
      S.listening = !!(st.ears && st.ears.engaged);
      document.body.classList.toggle("speaking", S.speaking);
      document.body.classList.toggle("listening", S.listening);
    },
    mouth(level) { S.mouth = clamp(level, 0, 1); lastMouthAt = performance.now(); },
    thinking(on) { S.thinking = !!on; },
    startle() { S.startle = 1; blinkUntil = 0; },
  };
  setColors("calme");
  requestAnimationFrame(frame);
})();

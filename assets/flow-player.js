// Step-through diagram player. Reads window.FLOWS (see langchain/flows-data.js).
(function () {
  const FLOWS = window.FLOWS;
  const $ = (id) => document.getElementById(id);
  const DIRS = { r: [1, 0], l: [-1, 0], t: [0, -1], b: [0, 1] };
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  const isFlow = (id) => FLOWS.some((f) => f.id === id);
  let flow = FLOWS[0], step = 0, timer = null;

  function anchor(n, side, off = 0) {
    switch (side) {
      case "r": return [n.x + n.w, n.y + n.h / 2 + off];
      case "l": return [n.x, n.y + n.h / 2 + off];
      case "t": return [n.x + n.w / 2 + off, n.y];
      default:  return [n.x + n.w / 2 + off, n.y + n.h];
    }
  }
  function edgePath(f, e) {
    const a = f.nodes.find((n) => n.id === e.from[0]), b = f.nodes.find((n) => n.id === e.to[0]);
    const p1 = anchor(a, e.from[1], e.from[2] || 0), p2 = anchor(b, e.to[1], e.to[2] || 0);
    const d1 = DIRS[e.from[1]], d2 = DIRS[e.to[1]];
    const q2 = [p2[0] + d2[0] * 6, p2[1] + d2[1] * 6];
    const k = Math.max(30, Math.hypot(q2[0] - p1[0], q2[1] - p1[1]) * 0.4);
    const c1 = [p1[0] + d1[0] * k, p1[1] + d1[1] * k], c2 = [q2[0] + d2[0] * k, q2[1] + d2[1] * k];
    return `M${p1} C${c1} ${c2} ${q2}`;
  }

  function renderDiagram() {
    const f = flow, [w, h] = f.vb;
    let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(f.title)} diagram"><defs>` +
      ["arrow", "arrow-on", "arrow-seen"].map((c) =>
        `<marker id="m-${c}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="${c}" d="M0,0 L10,5 L0,10 z"/></marker>`).join("") + `</defs>`;
    for (const e of f.edges) s += `<path id="edge-${e.id}" class="edge${e.dashed ? " dashed" : ""}" d="${edgePath(f, e)}" marker-end="url(#m-arrow)"/>`;
    for (const l of f.labels || []) s += `<text class="${l.zone ? "zone" : "el"}" x="${l.x}" y="${l.y}"${l.zone ? "" : ' text-anchor="middle"'}>${esc(l.t)}</text>`;
    for (const n of f.nodes) {
      const cx = n.x + n.w / 2, sub = !!n.sub;
      s += `<g class="node" id="node-${n.id}"><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${n.io ? n.h / 2 : 10}"/>` +
        `<text class="nl" x="${cx}" y="${n.y + n.h / 2 + (sub ? -3 : 5)}" text-anchor="middle">${esc(n.label)}</text>` +
        (sub ? `<text class="ns" x="${cx}" y="${n.y + n.h / 2 + 15}" text-anchor="middle">${esc(n.sub)}</text>` : "") + `</g>`;
    }
    $("diagram").innerHTML = s + "</svg>";
  }

  function setStep(i) {
    step = Math.max(0, Math.min(flow.steps.length - 1, i));
    const st = flow.steps[step], seenN = new Set(), seenE = new Set();
    flow.steps.slice(0, step).forEach((p) => { p.on.forEach((x) => seenN.add(x)); p.edges.forEach((x) => seenE.add(x)); });
    for (const n of flow.nodes) {
      const g = $("node-" + n.id);
      g.classList.toggle("on", st.on.includes(n.id));
      g.classList.toggle("seen", !st.on.includes(n.id) && seenN.has(n.id));
    }
    for (const e of flow.edges) {
      const p = $("edge-" + e.id), on = st.edges.includes(e.id), seen = !on && seenE.has(e.id);
      p.classList.toggle("on", on);
      p.classList.toggle("seen", seen);
      p.setAttribute("marker-end", `url(#m-${on ? "arrow-on" : seen ? "arrow-seen" : "arrow"})`);
    }
    $("stepTitle").textContent = st.title;
    $("stepText").textContent = st.text;
    $("stepCode").textContent = st.code;
    $("counter").textContent = `Step ${step + 1} of ${flow.steps.length}`;
    $("prev").disabled = step === 0;
    $("next").disabled = step === flow.steps.length - 1;
    [...$("dots").children].forEach((d, k) => {
      d.className = "dot" + (k === step ? " cur" : k < step ? " done" : "");
      d.setAttribute("aria-current", k === step ? "step" : "false");
    });
  }

  function stopPlay() { clearInterval(timer); timer = null; $("play").textContent = "▶ Play"; }
  function togglePlay() {
    if (timer) return stopPlay();
    if (step === flow.steps.length - 1) setStep(0);
    $("play").textContent = "❚❚ Pause";
    timer = setInterval(() => { if (step >= flow.steps.length - 1) return stopPlay(); setStep(step + 1); }, 3600);
  }

  function showFlow(id) {
    stopPlay();
    flow = FLOWS.find((f) => f.id === id) || FLOWS[0];
    [...$("tabs").children].forEach((t) => t.setAttribute("aria-selected", String(t.dataset.id === flow.id)));
    $("flowTitle").textContent = flow.title;
    $("flowIntro").textContent = flow.intro;
    $("oneLiner").textContent = flow.oneLiner;
    $("dots").innerHTML = flow.steps.map((s, k) => `<button class="dot" type="button" aria-label="Step ${k + 1}: ${esc(s.title)}">${k + 1}</button>`).join("");
    [...$("dots").children].forEach((d, k) => d.addEventListener("click", () => { stopPlay(); setStep(k); }));
    renderDiagram();
    setStep(0);
    if (window.NotesNav) window.NotesNav.mark("#" + flow.id);
  }

  $("tabs").innerHTML = FLOWS.map((f, k) =>
    `<button class="tab" role="tab" type="button" data-id="${f.id}"><span class="n">${k + 1}</span>${esc(f.tab)}</button>`).join("");
  [...$("tabs").children].forEach((t) => t.addEventListener("click", () => {
    if (location.hash === "#" + t.dataset.id) showFlow(t.dataset.id); else location.hash = t.dataset.id;
  }));
  $("prev").addEventListener("click", () => { stopPlay(); setStep(step - 1); });
  $("next").addEventListener("click", () => { stopPlay(); setStep(step + 1); });
  $("play").addEventListener("click", togglePlay);
  document.addEventListener("keydown", (e) => {
    if (e.target.closest && e.target.closest("input, textarea")) return;
    if (e.key === "ArrowRight") { stopPlay(); setStep(step + 1); }
    if (e.key === "ArrowLeft") { stopPlay(); setStep(step - 1); }
  });
  window.addEventListener("hashchange", () => {
    const id = location.hash.slice(1);
    if (!isFlow(id)) return;            // other anchors (like #stack) just scroll
    showFlow(id);
    $("player").scrollIntoView({ block: "start", behavior: "smooth" });
  });
  const start = location.hash.slice(1);
  showFlow(isFlow(start) ? start : FLOWS[0].id);
})();

// "How it works" renderer. Reads window.HOW (see concepts/how-it-works-data.js).
// Diagram types:
//   flow: { type: "flow", chain?: true, rows: [{ label?, alt?: true, steps: ["Label|sub|kind", ...], loop?: "text" }] }
//         kinds: io (input/output), model, store, check (decision/check), step (default)
//         chain: draw ↓ between rows. alt: steps are alternatives ("or"), not a sequence.
//   seq:  { type: "seq", actors: ["App", "LLM"], msgs: [[from, to, "label"], ...] }  (from === to → note on that actor)
(function () {
  const H = window.HOW;
  if (!H) return;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

  window.PAGE_SUBS = [["#legend", "How to read the diagrams"]].concat(H.cats.map((c) => ["#" + c.id, c.title]));

  function node(spec) {
    // fields are separated by "|"; write "\\|" in the data for a literal pipe (e.g. LCEL code)
    const [label, sub, kind] = String(spec).split(/(?<!\\)\|/).map((s) => s.replace(/\\\|/g, "|"));
    return `<span class="fn k-${kind || "step"}"><b>${esc(label)}</b>${sub ? `<small>${esc(sub)}</small>` : ""}</span>`;
  }
  const ARROW = '<span class="farr" aria-hidden="true"><svg viewBox="0 0 24 12"><path d="M0 6h20"/><path d="M15 1l6 5-6 5"/></svg></span>';

  function flow(d) {
    const rows = d.rows.map((r) => {
      const joiner = r.alt ? '<span class="for">or</span>' : ARROW;
      // keep each arrow glued to the node after it, so wrapped lines start with "→ node"
      const steps = r.steps.map((s, i) => (i === 0 ? node(s) : `<span class="fpair">${joiner}${node(s)}</span>`)).join("");
      return `<div class="frow${r.alt ? " alt" : ""}">
        ${r.label ? `<span class="frow-label">${esc(r.label)}</span>` : ""}
        <div class="fsteps">${steps}</div>
        ${r.loop ? `<div class="floop"><span>↻ ${esc(r.loop)}</span></div>` : ""}
      </div>`;
    });
    return `<div class="fd flow">${rows.join(d.chain ? '<div class="fdown" aria-hidden="true">↓</div>' : "")}</div>`;
  }

  function seq(d) {
    const n = d.actors.length;
    let num = 0;
    // every element gets an explicit grid row + column, so auto-placement can't shuffle them
    const msgs = d.msgs.map(([a, b, label], i) => {
      const row = i + 2;
      if (a === b) {
        return `<div class="snote" style="grid-row:${row};grid-column:${a + 1}/${a + 2}"><span>${esc(label)}</span></div>`;
      }
      num++;
      const lo = Math.min(a, b), hi = Math.max(a, b), k = hi - lo + 1;
      return `<div class="smsg ${b > a ? "right" : "left"}" style="grid-row:${row};grid-column:${lo + 1}/${hi + 2};--k:${k}">
        <span class="slabel"><i>${num}</i>${esc(label)}</span><span class="sline"></span>
      </div>`;
    }).join("");
    const lastRow = d.msgs.length + 2;
    const lifelines = d.actors.map((_, i) => `<span class="slife" style="left:calc(${(i + 0.5) * 100 / n}%)"></span>`).join("");
    const actors = (cls, row) => d.actors.map((a, i) => `<div class="sactor${cls}" style="grid-row:${row};grid-column:${i + 1}">${esc(a)}</div>`).join("");
    return `<div class="fd seqwrap"><div class="seq" style="--n:${n}">
      ${lifelines}
      ${actors("", 1)}
      ${msgs}
      ${actors(" end", lastRow)}
    </div></div>`;
  }

  const diagram = (d) => (d.type === "seq" ? seq(d) : flow(d));

  $("hw-chips").innerHTML = H.cats.map((c) => `<a class="chip" href="#${c.id}">${esc(c.title)}</a>`).join("");

  $("hw-list").innerHTML = H.cats.map((c) => {
    const items = H.items.filter((it) => it.cat === c.id).map((it) => {
      const ds = Array.isArray(it.diagram) ? it.diagram : [it.diagram];
      const hay = [it.q, it.tldr, ...it.steps, ...(it.notes || []), it.code || "", c.title,
        ...ds.flatMap((d) => d.type === "seq" ? d.actors.concat(d.msgs.map((m) => m[2])) : d.rows.flatMap((r) => r.steps.concat(r.label || "", r.loop || "")))].join(" ").toLowerCase();
      return `<article class="hw" id="${it.id}" data-search="${esc(hay)}">
        <header class="hw-head">
          <h3>${esc(it.q)}</h3>
          <a class="hw-anchor" href="#${it.id}" aria-label="Link to this explanation">#</a>
        </header>
        <p class="tldr"><span class="label">In one line</span>${esc(it.tldr)}</p>
        ${ds.map((d) => (d.title ? `<span class="label dtitle">${esc(d.title)}</span>` : "") + diagram(d)).join("")}
        <div class="hw-body">
          <div class="hw-steps">
            <span class="label">Step by step</span>
            <ol>${it.steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>
          </div>
          ${it.notes && it.notes.length ? `<div class="hw-notes"><span class="label">Key details</span><ul>${it.notes.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>` : ""}
        </div>
        ${it.code ? `<pre>${esc(it.code)}</pre>` : ""}
      </article>`;
    }).join("");
    return `<section class="section hwcat" id="${c.id}">
      <div class="section-head"><h2>${esc(c.title)}</h2>${c.blurb ? `<p>${esc(c.blurb)}</p>` : ""}</div>
      <div class="hw-items">${items}</div>
    </section>`;
  }).join("");

  // Search
  const q = $("q");
  function apply() {
    const words = q.value.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    document.querySelectorAll(".hwcat").forEach((sec) => {
      let any = false;
      sec.querySelectorAll(".hw").forEach((card) => {
        const ok = words.every((w) => card.dataset.search.includes(w));
        card.hidden = !ok;
        if (ok) { any = true; shown++; }
      });
      sec.hidden = !any;
    });
    $("count").textContent = `${shown} of ${H.items.length}`;
    $("empty").hidden = shown > 0;
  }
  q.addEventListener("input", apply);
  document.addEventListener("keydown", (e) => { if (e.key === "/" && document.activeElement !== q) { e.preventDefault(); q.focus(); } });
  $("stats").innerHTML = `<b>${H.items.length}</b> explanations · <b>${H.cats.length}</b> topics · every one with a diagram`;
  apply();
})();

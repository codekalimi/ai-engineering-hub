// "How it works" renderer. Reads window.HOW (see concepts/how-it-works-data.js).
// Used two ways:
//   - in the browser by concepts/how-it-works.html (the one-page cheat sheet, with search)
//   - in Node by concepts/gen-how-pages.js, which writes one static page per explanation
// Diagram types:
//   flow: { type: "flow", chain?: true, rows: [{ label?, alt?: true, steps: ["Label|sub|kind", ...], loop?: "text" }] }
//         kinds: io (input/output), model, store, check (decision/check), step (default)
//         chain: draw ↓ between rows. alt: steps are alternatives ("or"), not a sequence.
//   seq:  { type: "seq", actors: ["App", "LLM"], msgs: [[from, to, "label"], ...] }  (from === to → note on that actor)
(function (root) {
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

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
  const diagrams = (it) => (Array.isArray(it.diagram) ? it.diagram : [it.diagram]);

  // Everything below the one-line answer: diagram(s), steps, key details (the code is in codeLevels)
  function body(it) {
    return `${diagrams(it).map((d) => (d.title ? `<span class="label dtitle">${esc(d.title)}</span>` : "") + diagram(d)).join("")}
        <div class="hw-body">
          <div class="hw-steps">
            <span class="label">Step by step</span>
            <ol>${it.steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>
          </div>
          ${it.notes && it.notes.length ? `<div class="hw-notes"><span class="label">Key details</span><ul>${it.notes.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>` : ""}
        </div>`;
  }

  const legend = () => `<div class="fd legend">
          <span class="legend-item"><span class="fn k-io"><b>Input / output</b></span></span>
          <span class="legend-item"><span class="fn k-model"><b>Model / compute</b></span></span>
          <span class="legend-item"><span class="fn k-store"><b>Stored data</b></span></span>
          <span class="legend-item"><span class="fn k-check"><b>Check / decision</b></span></span>
          <span class="legend-item"><span class="fn"><b>Processing step</b></span></span>
          <span class="legend-item">${ARROW}then</span>
          <span class="legend-item"><span class="for">or</span>alternatives</span>
        </div>`;

  // ---- Behind the scenes: runnable Python + its real output + how to watch the real system ----
  const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");

  // Dim Python comments (a "#" outside quotes) and make "Step N" stand out, so code maps to the steps
  function pyHighlight(code) {
    return code.split("\n").map((line) => {
      let q = null, cut = -1;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (q) { if (ch === "\\") i++; else if (ch === q) q = null; }
        else if (ch === '"' || ch === "'") q = ch;
        else if (ch === "#") { cut = i; break; }
      }
      if (cut < 0) return esc(line);
      const comment = esc(line.slice(cut)).replace(/(Steps? \d+(?:[-–,+ ]+\d+)*(?: \([^)]*\))?)/g, "<b>$1</b>");
      return esc(line.slice(0, cut)) + `<span class="c">${comment}</span>`;
    }).join("\n");
  }

  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

  // How a Level 1/2 snippet was tested, in one line
  function testedLine(r, versions) {
    if (!r) return `<p class="lv-test none">Not run yet: run <code>node concepts/run-levels.js</code>.</p>`;
    if (r.status === "checked") return `<p class="lv-test checked"><b>Checked, not run:</b> ${esc(r.why)}.</p>`;
    if (r.status === "parsed") return `<p class="lv-test ok"><b>✓ Validated:</b> parsed as YAML.</p>`;
    const libs = r.libs && r.libs.length ? r.libs.join(", ") : `Python ${(versions || {}).python || "3"}`;
    const fakes = r.fakes && r.fakes.length ? ` · with ${r.fakes.join("; ")}` : "";
    return `<p class="lv-test ok"><b>✓ Ran</b> with ${esc(libs)}${esc(fakes)}.</p>`;
  }

  function outputBlock(out, label) {
    if (!out) return "";
    const lines = out.split("\n").length;
    const pre = `<pre class="bts-out">${esc(out)}</pre>`;
    return lines <= 14
      ? `<span class="label">${label}</span>${pre}`
      : `<details class="lv-out"><summary>${label} (${plural(lines, "line")})</summary>${pre}</details>`;
  }

  const LEVELS = [
    ["use", "1", "Use it", "What you'd write at work"],
    ["under", "2", "Under the hood", "What the library does for you, with its real classes"],
    ["scratch", "3", "From scratch", "The idea in plain Python, no libraries"],
  ];

  // The code for one explanation at three zoom levels.
  // d = { levels: {use, under}, levelsOut: {use, under}, versions, bts: {code, see}, btsOut }
  function codeLevels(d, opts) {
    opts = opts || {};
    const lv = d.levels || {}, out = d.levelsOut || {};
    if (!lv.use && !lv.under && !d.bts) return "";
    const block = ([key, n, title, sub]) => {
      if (key === "scratch") {
        if (!d.bts) return "";
        return `<div class="clv clv-3" id="${opts.collapsed ? "" : "level-3"}">
          <div class="lv-head"><span class="lv-n">${n}</span><div><h3>${title}</h3><p>${sub}</p></div></div>
          <p class="lv-note">A tiny version of the mechanism. The <b>Step</b> comments match the numbered steps above. Save it as a <code>.py</code> file and run <code>python3 file.py</code>: nothing to install.</p>
          <div class="bts-code"><span class="bts-file">python3 · standard library only</span><pre data-copy>${pyHighlight(d.bts.code)}</pre></div>
          <p class="lv-test ok"><b>✓ Ran</b> with plain Python 3.9+.</p>
          ${outputBlock(d.btsOut, "What it prints")}
        </div>`;
      }
      const snip = lv[key];
      if (!snip) return "";
      const code = snip.code.replace(/^\n/, "");
      const file = snip.lang === "yaml" ? "yaml" : "python";
      return `<div class="clv clv-${n}" id="${opts.collapsed ? "" : "level-" + n}">
        <div class="lv-head"><span class="lv-n">${n}</span><div><h3>${title}</h3><p>${sub}</p></div></div>
        ${snip.note ? `<p class="lv-note">${inline(snip.note)}</p>` : ""}
        <div class="bts-code"><span class="bts-file">${file}</span><pre data-copy>${pyHighlight(code)}</pre></div>
        ${testedLine(out[key], d.versions)}
        ${out[key] && out[key].out ? outputBlock(out[key].out, "What it printed") : ""}
        ${snip.source && snip.source.length ? `<p class="lv-src"><span class="label">Read the real code</span> ${snip.source.map((x) => `<code>${esc(x)}</code>`).join(" · ")}</p>` : ""}
      </div>`;
    };
    const see = d.bts && d.bts.see ? `<div class="lv-see"><span class="label">Watch it in a real system</span>
          <ul class="bts-see">${d.bts.see.map((s) => `<li>${inline(s)}</li>`).join("")}</ul>
          ${opts.guide ? `<p class="bts-more">More ways to look inside any AI tool: <a href="${opts.guide}">How to see behind the scenes</a></p>` : ""}</div>` : "";
    const nav = `<div class="lv-nav">${LEVELS.filter(([k]) => (k === "scratch" ? d.bts : lv[k])).map(([k, n, t]) => opts.collapsed
      ? `<span class="lv-chip"><b>${n}</b> ${t}</span>` : `<a class="lv-chip" href="#level-${n}"><b>${n}</b> ${t}</a>`).join('<span class="lv-arrow">→</span>')}</div>`;
    const inner = `<p class="bts-intro">The same mechanism at three zoom levels: the code you write, what the library does with it, and the idea with no library at all.</p>
        ${nav}
        ${LEVELS.map(block).join("")}
        ${see}`;
    if (opts.collapsed) {
      return `<details class="bts bts-details"><summary><span class="label">The code</span> use it → under the hood → from scratch, each tested</summary>${inner}</details>`;
    }
    return `<section class="bts" id="behind-the-scenes"><div class="bts-head"><span class="label">The code</span><h2>From the API down to the idea</h2></div>${inner}</section>`;
  }

  const R = { esc, node, flow, seq, diagram, diagrams, body, legend, codeLevels, pyHighlight };
  if (typeof module !== "undefined" && module.exports) { module.exports = R; return; }
  root.HowRender = R;

  // ---- The one-page cheat sheet (concepts/how-it-works.html) ----
  const H = root.HOW;
  const $ = (id) => document.getElementById(id);
  if (!H || !$("hw-list")) return;
  const META = root.HOW_META || {};
  const LV_OUT = root.HOW_LEVELS_OUTPUT || { results: {}, versions: {} };

  window.PAGE_SUBS = [["#legend", "How to read the diagrams"]];

  $("hw-chips").innerHTML = H.cats.map((c) => `<a class="chip" href="#${c.id}">${esc(c.title)}</a>`).join("");

  $("hw-list").innerHTML = H.cats.map((c) => {
    const items = H.items.filter((it) => it.cat === c.id).map((it) => {
      const page = META[it.id] ? `how/${META[it.id].slug}.html` : "";
      const lvl = (root.HOW_LEVELS || {})[it.id] || {};
      const hay = [it.q, it.tldr, ...it.steps, ...(it.notes || []), c.title,
        (lvl.use || {}).code || "", (lvl.under || {}).code || "",
        ...diagrams(it).flatMap((d) => d.type === "seq" ? d.actors.concat(d.msgs.map((m) => m[2])) : d.rows.flatMap((r) => r.steps.concat(r.label || "", r.loop || "")))].join(" ").toLowerCase();
      return `<article class="hw" id="${it.id}" data-search="${esc(hay)}">
        <header class="hw-head">
          <h3>${page ? `<a href="${page}">${esc(it.q)}</a>` : esc(it.q)}</h3>
          <span class="hw-links">${page ? `<a class="hw-open" href="${page}">Open page →</a>` : ""}<a class="hw-anchor" href="#${it.id}" aria-label="Link to this explanation">#</a></span>
        </header>
        <p class="tldr"><span class="label">In one line</span>${esc(it.tldr)}</p>
        ${body(it)}
        ${codeLevels({ levels: (root.HOW_LEVELS || {})[it.id], levelsOut: LV_OUT.results[it.id], versions: LV_OUT.versions,
                       bts: (root.HOW_INTERNALS || {})[it.id], btsOut: (root.HOW_INTERNALS_OUTPUT || {})[it.id] },
                     { collapsed: true, guide: "behind-the-scenes.html" })}
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
  $("stats").innerHTML = `<b>${H.items.length}</b> explanations · <b>${H.cats.length}</b> topics · every one with a diagram and its own page`;
  apply();
})(typeof window !== "undefined" ? window : globalThis);

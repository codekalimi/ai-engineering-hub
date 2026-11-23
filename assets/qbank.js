// Question-bank engine for the role interview pages. Reads window.QBANK (see career/data/*.js).
(function () {
  const D = window.QBANK;
  if (!D) return;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  const LVL = { must: "Must prepare", maybe: "Possible" };
  const hash = (s) => { let h = 5381; for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0; return (h >>> 0).toString(36); };
  const KEY = "qbank.confident." + D.id;
  const store = {
    get() { try { return new Set(JSON.parse(localStorage.getItem(KEY) || "[]")); } catch (e) { return new Set(); } },
    set(v) { try { localStorage.setItem(KEY, JSON.stringify([...v])); } catch (e) {} },
  };
  const confident = store.get();

  // Spoken answers come from data/<role>-spoken.js, matched by exact question text
  const spoken = new Map(window.QBANK_SPOKEN || []);
  D.questions.forEach((q) => { q.say = spoken.get(q.q) || ""; });
  // [text in brackets] = where to drop in your own example
  const fmtSay = (s) => esc(s).replace(/\[([^\]]+)\]/g, '<mark class="yours">[$1]</mark>');

  // Index entries for the left sidebar (read by site.js)
  window.PAGE_SUBS = [["#rounds", "Interview rounds"]].concat(D.topics.map((t) => ["#" + t.id, t.title]));

  // Rounds
  $("rounds-list").innerHTML = D.rounds.map((r, i) => `
    <li>
      <span class="k">Round ${i + 1}${r.len ? " · " + esc(r.len) : ""}</span>
      <h3>${esc(r.name)}</h3>
      <p>${esc(r.what)}</p>
      ${r.tip ? `<p class="tip"><b>Tip</b> ${esc(r.tip)}</p>` : ""}
    </li>`).join("");

  // Stats
  const total = D.questions.length, must = D.questions.filter((q) => q.lvl === "must").length;
  $("stats").innerHTML = `<b>${total}</b> questions · <b>${must}</b> must prepare · <b>${total - must}</b> possible · <b>${D.topics.length}</b> topics`;

  // Question cards by topic
  $("bank").innerHTML = D.topics.map((t) => {
    const qs = D.questions.filter((q) => q.t === t.id);
    const cards = qs.map((q) => {
      const key = hash(q.q);
      const hay = [q.q, q.test, ...q.a, ...q.fu, q.code, q.say, t.title].join(" ").toLowerCase();
      return `<details class="qcard" id="q-${key}" data-key="${key}" data-lvl="${q.lvl}" data-search="${esc(hay)}">
        <summary>
          <span class="lvl ${q.lvl}">${LVL[q.lvl]}</span>
          <span class="qt">${esc(q.q)}</span>
          <span class="done" title="You marked this as confident">✓</span>
        </summary>
        <div class="qbody">
          ${q.test ? `<p class="testing"><b>What they're testing</b> ${esc(q.test)}</p>` : ""}
          <ul class="pts">${q.a.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
          ${q.code ? `<pre>${esc(q.code)}</pre>` : ""}
          ${q.say ? `<div class="spoken"><span class="label">Interview answer · say it like this</span><p>${fmtSay(q.say)}</p></div>` : ""}
          ${q.fu.length ? `<div class="fu"><b>Likely follow-ups</b><ul>${q.fu.map((f) => `<li>${esc(f)}</li>`).join("")}</ul></div>` : ""}
          <button class="conf" type="button" aria-pressed="false">I can answer this</button>
        </div>
      </details>`;
    }).join("");
    return `<section class="section qtopic" id="${t.id}">
      <div class="section-head"><h2>${esc(t.title)}</h2>${t.blurb ? `<p>${esc(t.blurb)}</p>` : ""}</div>
      <div class="qlist">${cards}</div>
    </section>`;
  }).join("");

  // Confidence tracking
  function paintConf(card) {
    const on = confident.has(card.dataset.key);
    card.classList.toggle("is-conf", on);
    const b = card.querySelector(".conf");
    b.setAttribute("aria-pressed", String(on));
    b.textContent = on ? "✓ Confident (click to undo)" : "I can answer this";
  }
  function paintProgress() {
    const n = [...document.querySelectorAll(".qcard")].filter((c) => confident.has(c.dataset.key)).length;
    $("progress-text").textContent = `${n} of ${total} confident`;
    $("progress-bar").style.width = (total ? (n / total) * 100 : 0) + "%";
  }
  document.querySelectorAll(".qcard").forEach(paintConf);
  $("bank").addEventListener("click", (e) => {
    const b = e.target.closest(".conf");
    if (!b) return;
    const card = b.closest(".qcard"), k = card.dataset.key;
    if (confident.has(k)) confident.delete(k); else confident.add(k);
    store.set(confident);
    paintConf(card); paintProgress(); apply();
  });
  $("reset").addEventListener("click", () => {
    if ($("reset").dataset.armed !== "1") {
      $("reset").dataset.armed = "1"; $("reset").textContent = "Click again to reset";
      setTimeout(() => { $("reset").dataset.armed = ""; $("reset").textContent = "Reset progress"; }, 3000);
      return;
    }
    confident.clear(); store.set(confident);
    document.querySelectorAll(".qcard").forEach(paintConf); paintProgress(); apply();
    $("reset").dataset.armed = ""; $("reset").textContent = "Reset progress";
  });

  // Filters
  let filter = "all";
  const q = $("q");
  function apply() {
    const words = q.value.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    document.querySelectorAll(".qtopic").forEach((sec) => {
      let any = false;
      sec.querySelectorAll(".qcard").forEach((card) => {
        const okF = filter === "all" || (filter === "todo" ? !confident.has(card.dataset.key) : card.dataset.lvl === filter);
        const ok = okF && words.every((w) => card.dataset.search.includes(w));
        card.hidden = !ok;
        if (ok) { any = true; shown++; }
      });
      sec.hidden = !any;
    });
    $("count").textContent = `Showing ${shown}`;
    $("empty").hidden = shown > 0;
  }
  q.addEventListener("input", apply);
  document.querySelectorAll(".qf").forEach((b) => b.addEventListener("click", () => {
    filter = b.dataset.f;
    document.querySelectorAll(".qf").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    apply();
  }));
  $("toggleAll").addEventListener("click", () => {
    const open = $("toggleAll").textContent.startsWith("Show");
    document.querySelectorAll(".qcard").forEach((d) => { if (!d.hidden) d.open = open; });
    $("toggleAll").textContent = open ? "Hide all answers" : "Show all answers";
  });
  document.addEventListener("keydown", (e) => { if (e.key === "/" && document.activeElement !== q) { e.preventDefault(); q.focus(); } });

  function openFromHash() {
    const el = location.hash && document.getElementById(location.hash.slice(1));
    if (el && el.tagName === "DETAILS") { el.open = true; el.scrollIntoView({ block: "center" }); }
  }
  window.addEventListener("hashchange", openFromHash);
  paintProgress(); apply(); openFromHash();
})();

// AI Engineering Notes: builds the left index on every page.
// To add a page: create the HTML file, then add an entry to NAV below.
(function () {
  const NAV = [
    {
      group: "LangChain",
      pages: [
        {
          id: "langchain/under-the-hood", title: "Under the hood", href: "langchain/under-the-hood.html",
          subs: [
            ["#lcel", "1. The pipe"], ["#messages", "2. Messages & memory"], ["#tools", "3. Tool calling"],
            ["#agent", "4. Agent loop"], ["#rag", "5. RAG"], ["#callbacks", "6. Callbacks & tracing"],
            ["#langgraph", "7. LangGraph"], ["#stack", "The whole picture"],
          ],
        },
        {
          id: "langchain/templates", title: "Mental templates", href: "langchain/templates.html",
          subs: [
            ["#core", "The core template"], ["#addons", "The four add-ons"], ["#rag-template", "RAG template"],
            ["#agent-template", "Agent template"], ["#graph-template", "LangGraph template"],
            ["#compare", "Next to scikit-learn & APIs"], ["#recall", "10-second recall"],
          ],
        },
        {
          id: "langchain/components", title: "Components", href: "langchain/components.html",
          subs: [
            ["#start", "Where each part sits"], ["#models", "Models"], ["#messages", "Messages"], ["#prompts", "Prompts"],
            ["#output", "Structured output"], ["#compose", "Composing chains"], ["#rag", "Documents & RAG"],
            ["#tools", "Tools & agents"], ["#graph", "LangGraph"], ["#observe", "Debugging & tracing"],
          ],
        },
        {
          id: "langchain/interview", title: "Interview prep", href: "langchain/interview.html",
          subs: [
            ["#most-asked", "Most-asked questions"], ["#core", "Runnable & chat models"],
            ["#prompts-output", "Messages, prompts & output"], ["#rag-classes", "RAG classes"],
            ["#tools-agents", "Tools & agents"], ["#graph-classes", "LangGraph"], ["#observability", "Tracing & evaluation"],
          ],
        },
      ],
    },
    {
      group: "Career",
      pages: [
        {
          id: "career/ai-roles", title: "AI roles & skills", href: "career/ai-roles.html",
          subs: [
            ["#roles-overview", "Compare the roles"], ["#level", "Senior vs Staff vs Architect"], ["#ai-engineer", "AI Engineer"], ["#fde", "Forward Deployed AI Engineer"],
            ["#ai-infra", "AI Infrastructure Engineer"], ["#ai-architect", "AI Architect"], ["#skill-matrix", "Skill matrix"],
            ["#your-edge", "Your backend edge"], ["#roadmap", "90-day plan"],
          ],
        },
      ],
    },
  ];

  const shell = document.querySelector(".shell");
  if (!shell) return;
  const root = shell.dataset.root || "";
  const current = shell.dataset.page || "";
  const side = document.getElementById("sidebar");
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
  };
  const caret = '<span class="caret" aria-hidden="true">▾</span>';

  let html = `<a class="brand" href="${root}index.html">AI Engineering Notes<small>Visual notes for interviews</small></a>
    <a class="nav-home${current === "home" ? " current" : ""}" href="${root}index.html">Home</a>`;
  NAV.forEach((g, gi) => {
    const gid = "grp-" + gi;
    html += `<div class="nav-group">
      <button class="group-head" type="button" aria-expanded="true" aria-controls="${gid}">${esc(g.group)}${caret}</button>
      <div class="group-body" id="${gid}">`;
    g.pages.forEach((p, pi) => {
      const isCur = p.id === current, sid = `subs-${gi}-${pi}`;
      html += `<div class="nav-page${isCur ? " current" : ""}">
        <div class="row">
          <a href="${root}${p.href}"${isCur ? ' aria-current="page"' : ""}>${esc(p.title)}</a>
          <button class="chev" type="button" aria-expanded="${isCur}" aria-controls="${sid}" aria-label="Show sections of ${esc(p.title)}">${caret}</button>
        </div>
        <ul class="subs" id="${sid}"${isCur ? "" : " hidden"}>
          ${p.subs.map(([h, t]) => `<li><a href="${isCur ? h : root + p.href + h}" data-hash="${isCur ? h : ""}">${esc(t)}</a></li>`).join("")}
        </ul>
      </div>`;
    });
    html += `</div></div>`;
  });
  side.innerHTML = html;

  // Expand / collapse groups and page sections
  side.addEventListener("click", (e) => {
    const btn = e.target.closest("button[aria-controls]");
    if (btn) {
      const open = btn.getAttribute("aria-expanded") !== "true";
      btn.setAttribute("aria-expanded", String(open));
      document.getElementById(btn.getAttribute("aria-controls")).hidden = !open;
      return;
    }
    if (e.target.closest("a") && isMobile()) closeMobile();
  });

  // Show / hide the whole index
  const toggle = document.getElementById("navToggle");
  const scrim = document.querySelector(".scrim");
  const isMobile = () => window.matchMedia("(max-width: 900px)").matches;
  const syncToggle = () => {
    const shown = isMobile() ? shell.classList.contains("nav-open") : !shell.classList.contains("collapsed");
    toggle.setAttribute("aria-expanded", String(shown));
    toggle.querySelector(".t").textContent = shown ? "Hide index" : "Index";
  };
  const closeMobile = () => { shell.classList.remove("nav-open"); syncToggle(); };
  if (store.get("notes.navCollapsed") === "1") shell.classList.add("collapsed");
  toggle.addEventListener("click", () => {
    if (isMobile()) shell.classList.toggle("nav-open");
    else {
      shell.classList.toggle("collapsed");
      store.set("notes.navCollapsed", shell.classList.contains("collapsed") ? "1" : "0");
    }
    syncToggle();
  });
  if (scrim) scrim.addEventListener("click", closeMobile);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && shell.classList.contains("nav-open")) closeMobile(); });
  window.matchMedia("(max-width: 900px)").addEventListener("change", () => { shell.classList.remove("nav-open"); syncToggle(); });
  syncToggle();

  // Highlight the section you're on
  const subLinks = [...side.querySelectorAll(".nav-page.current .subs a")];
  const mark = (hash) => subLinks.forEach((a) => a.classList.toggle("active", a.dataset.hash === hash));
  window.addEventListener("hashchange", () => mark(location.hash));
  if (location.hash) mark(location.hash);
  if (shell.dataset.spy && "IntersectionObserver" in window) {
    const targets = subLinks.map((a) => document.getElementById(a.dataset.hash.slice(1))).filter(Boolean);
    const visible = new Map();
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => visible.set(en.target.id, en.isIntersecting));
      const first = targets.find((t) => visible.get(t.id));
      if (first) mark("#" + first.id);
    }, { rootMargin: "0px 0px -60% 0px" });
    targets.forEach((t) => io.observe(t));
  }
  window.NotesNav = { mark };
})();

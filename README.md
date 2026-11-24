# AI Engineering Notes

Plain HTML/CSS/JS study notes. Open `index.html` in a browser. There's nothing to install or build.

## Structure

```
ai-engineering-notes/
├── index.html                  Home page (topic cards)
├── assets/
│   ├── site.css                Shared styles (colors, layout, sidebar, components)
│   ├── site.js                 Builds the left index — the NAV list lives here
│   └── flow-player.js          Step-through diagram engine
├── langchain/
│   ├── under-the-hood.html     Interactive flows page
│   ├── flows-data.js           The flow diagrams + steps (edit content here)
│   ├── templates.html          Mental templates page
│   ├── components.html         Component reference (data: components-data.js)
│   └── interview.html          Class/method interview prep (data: interview-data.js)
├── concepts/
│   ├── how-it-works.html       All 52 "how does X work" explanations on one page, with search
│   ├── how-it-works-data*.js   The content (two files)
│   ├── how-meta.js             Each explanation's page name, short title and related links
│   ├── gen-how-pages.js        Generates how/*.html and the topic list in the left index
│   └── how/                    One generated page per explanation (don't edit by hand)
└── career/
    ├── ai-roles.html           Roles, skills, levels, 90-day plan
    ├── *-interview.html        One question bank per role (all share assets/qbank.js)
    └── data/                   Questions per role: ai-engineer.js, fde.js, ai-infra.js, ai-architect.js
```

## Add a "How it works" explanation

Each explanation appears twice: on the one-page cheat sheet and on its own page in `concepts/how/`.
The separate pages are generated, so after any change to the data run (Node.js only, nothing to install):

```
node concepts/gen-how-pages.js
```

It checks the data first, then rewrites `concepts/how/*.html` and the topic list in `assets/site.js`.

1. Add an `I(...)` entry in `concepts/how-it-works-data-2.js` (or the first file):

```js
I("c-retrieval", "hw-my-topic", "How does X work?", "One-line answer.",
  F([{ label: "Optional row label", steps: ["Input||io", "Step|small text", "LLM||model", "Store||store", "Check?||check", "Output||io"],
       loop: "optional: what repeats" }]),
  ["Step 1 explanation", "Step 2 ..."],
  ["Key detail", "..."],
  `optional code`),
```

2. Add a line to `concepts/how-meta.js` (page file name, short title for the index, 3-4 related ids):

```js
M("my-topic", "my-topic", "My topic", "rag rerank embeddings");
```

3. Run `node concepts/gen-how-pages.js`.

Step syntax: `"Label|small text|kind"`, kinds `io`, `model`, `store`, `check` (default: plain step).
Write `\\|` for a literal pipe. Options: `alt: true` on a row = alternatives ("or"), `{ chain: true }` on F = ↓ between rows.
Sequence diagrams: `S(["App", "LLM"], [[0, 1, "request"], [1, 0, "response"], [1, 1, "a note"]])`.

## Add an interview question

Open the role's file in `career/data/` and add a line to `questions`:

```js
Q("t-rag", M, "Your question?", "What they're testing (optional)",
  ["Answer point 1", "Answer point 2"],
  ["Likely follow-up"],          // optional
  `optional code snippet`),      // optional
```

`M` = must prepare, `P` = possible. The topic id must match one in `topics`.

Then add its spoken "Interview answer" to the matching `<role>-spoken.js` file, using the
exact same question text: `` [`Your question?`, `How you'd say it out loud...`], ``
Put `[square brackets]` around parts where you should insert your own example. They're highlighted on the page.
Search, filters, progress tracking and the sidebar pick it up automatically.

## Add a new page

1. Copy `langchain/templates.html` into a topic folder, e.g. `rag/evaluation.html`.
2. In the copy, change `<title>`, `data-page="rag/evaluation"`, the breadcrumb and the content.
   Keep `data-root="../"` for pages one folder deep.
3. Give each section an `id`, e.g. `<section class="section" id="metrics">`.
4. Add the page to `NAV` in `assets/site.js`:

```js
{ group: "RAG", pages: [
  { id: "rag/evaluation", title: "Evaluation", href: "rag/evaluation.html",
    subs: [["#metrics", "Metrics"], ["#datasets", "Datasets"]] },
]}
```

5. Optionally add a card for it on `index.html`.

## Add a new step-through flow

Add an object to `FLOWS` in `langchain/flows-data.js`: nodes (boxes with x/y), edges
(arrows between box sides) and steps (which boxes/arrows light up, plus the text and code).
A new tab appears automatically. Add it to the `subs` list in `site.js` too.

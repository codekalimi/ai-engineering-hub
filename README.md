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
└── langchain/
    ├── under-the-hood.html     Interactive flows page
    ├── flows-data.js           The flow diagrams + steps (edit content here)
    └── templates.html          Mental templates page
```

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

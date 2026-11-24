// Runs every Level 1 ("Use it") and Level 2 ("Under the hood") snippet with the REAL libraries,
// and saves what each one printed into how-levels-output.js. It doubles as a test suite.
//
// Setup once (needs uv: https://docs.astral.sh/uv/):
//   uv venv --python 3.12 .venv-levels
//   VIRTUAL_ENV=.venv-levels uv pip install -r concepts/levels-test/requirements.txt
// Run (from the site folder):
//   LEVELS_PYTHON=.venv-levels/bin/python node concepts/run-levels.js            all snippets
//   LEVELS_PYTHON=.venv-levels/bin/python node concepts/run-levels.js agentloop  one topic (prints output)
//
// What is faked (see levels-test/prelude.py): the chat model API (a scripted model replies from the
// snippet's `script`), embedding APIs (a small local embedding model) and Anthropic's HTTP endpoint (the real
// SDK runs; replies come from `anthropic`). Snippets with `check` set need a GPU, hours of training
// or a live server: for those only the syntax and the imports are checked.
const fs = require("fs");
const os = require("os");
const path = require("path");
const vm = require("vm");
const { spawnSync } = require("child_process");

const PY = process.env.LEVELS_PYTHON || "python3";
const PRELUDE_DIR = path.join(__dirname, "levels-test");
const FILES = ["how-levels-1.js", "how-levels-2.js", "how-levels-3.js"];
const OUT_FILE = path.join(__dirname, "how-levels-output.js");

const ctx = { window: {} };
vm.createContext(ctx);
for (const f of FILES) {
  const p = path.join(__dirname, f);
  if (fs.existsSync(p)) vm.runInContext(fs.readFileSync(p, "utf8"), ctx, { filename: f });
}
const ALL = ctx.window.HOW_LEVELS || {};
const only = process.argv[2] ? "hw-" + process.argv[2].replace(/^hw-/, "") : null;
const ids = Object.keys(ALL).filter((id) => !only || id === only);
if (!ids.length) { console.error(only ? `No snippets for ${only}` : "No snippets found"); process.exit(1); }

// Library versions, so each page can say exactly what the code was tested with
const PKGS = { langchain: "langchain", langchain_core: "langchain-core", langgraph: "langgraph",
  langchain_text_splitters: "langchain-text-splitters", langchain_community: "langchain-community",
  langchain_classic: "langchain-classic", langchain_mcp_adapters: "langchain-mcp-adapters", mcp: "mcp",
  anthropic: "anthropic", transformers: "transformers", torch: "torch", tiktoken: "tiktoken", peft: "peft",
  trl: "trl", datasets: "datasets", faiss: "faiss-cpu", sentence_transformers: "sentence-transformers", numpy: "numpy" };
const ver = spawnSync(PY, ["-c", `import json, importlib.metadata as m
out = {}
for d in ${JSON.stringify([...new Set(Object.values(PKGS))])}:
    try: out[d] = m.version(d)
    except Exception: pass
import sys; out["python"] = sys.version.split()[0]
print(json.dumps(out))`], { encoding: "utf8" });
if (ver.status !== 0) { console.error(`Can't run ${PY}. Set LEVELS_PYTHON (see the top of this file).\n${ver.stderr}`); process.exit(1); }
const versions = JSON.parse(ver.stdout);

function libsUsed(code) {
  const mods = new Set();
  for (const m of code.matchAll(/^\s*(?:from|import)\s+([a-zA-Z_]\w*)/gm)) if (PKGS[m[1]]) mods.add(PKGS[m[1]]);
  return [...mods].filter((d) => versions[d]).map((d) => `${d} ${versions[d]}`);
}
function fakesUsed(code, snip) {
  const f = [];
  if (/init_chat_model\(/.test(code)) f.push("a scripted chat model in place of the API");
  if (/init_embeddings\(/.test(code)) f.push("a local embedding model (all-MiniLM-L6-v2) in place of the embedding API");
  if (/anthropic\.Anthropic\(/.test(code)) f.push("a mocked Anthropic HTTP endpoint (the real SDK validated the request)");
  if (snip.subs) f.push("a smaller model with the same code: " + Object.entries(snip.subs).map(([a, b]) => `${b} for ${a}`).join(", "));
  return f;
}

const results = {};
if (fs.existsSync(OUT_FILE)) {
  const c = { window: {} };
  vm.createContext(c);
  vm.runInContext(fs.readFileSync(OUT_FILE, "utf8"), c);
  Object.assign(results, (c.window.HOW_LEVELS_OUTPUT || {}).results || {});
}

const RUNNER = `
import sys, runpy, warnings, ast
warnings.filterwarnings("ignore")
mode, path = sys.argv[1], sys.argv[2]
if mode == "run":
    sys.path.insert(0, ${JSON.stringify(PRELUDE_DIR)})
    import prelude  # noqa: F401  (fakes for the network-bound parts)
    runpy.run_path(path, run_name="__main__")
elif mode == "yaml":
    import yaml
    doc = yaml.safe_load(open(path))           # parse; print what a reviewer would check
    print(f"valid YAML: {doc['kind']} '{doc['metadata']['name']}'")
    for k, v in doc["spec"].items():
        print(f"  {k}: {v}")
else:
    skip = set(filter(None, sys.argv[3].split(","))) if len(sys.argv) > 3 else set()
    tree = ast.parse(open(path).read())        # syntax check
    for node in tree.body:                     # and every top-level import must resolve
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top = (node.module if isinstance(node, ast.ImportFrom) else node.names[0].name).split(".")[0]
            if top not in skip:
                exec(compile(ast.Module([node], []), path, "exec"), {})
    print("syntax and imports OK")
`;

let failed = 0;
for (const id of ids) {
  for (const level of ["use", "under"]) {
    const snip = ALL[id][level];
    if (!snip) continue;
    const code = snip.code.replace(/^\n/, "");
    let runCode = code;
    for (const [a, b] of Object.entries(snip.subs || {})) runCode = runCode.split(a).join(b);
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "how-levels-"));
    for (const [name, body] of Object.entries(snip.files || {})) fs.writeFileSync(path.join(dir, name), body.replace(/^\n/, ""));
    const isYaml = snip.lang === "yaml";
    const file = path.join(dir, isYaml ? "snippet.yaml" : "snippet.py");
    fs.writeFileSync(file, runCode);
    const mode = isYaml ? "yaml" : snip.check ? "check" : "run";
    const r = spawnSync(PY, ["-c", RUNNER, mode, file, (snip.noImport || []).join(",")], {
      cwd: dir, encoding: "utf8", timeout: 600000, maxBuffer: 1 << 24,
      env: { ...process.env, FAKE_SCRIPT: JSON.stringify(snip.script || []), FAKE_ANTHROPIC: JSON.stringify(snip.anthropic || []),
        PYTHONWARNINGS: "ignore", TRANSFORMERS_VERBOSITY: "error", HF_HUB_DISABLE_PROGRESS_BARS: "1",
        TOKENIZERS_PARALLELISM: "false", PYTHONIOENCODING: "utf-8", TQDM_DISABLE: "1" },
    });
    fs.rmSync(dir, { recursive: true, force: true });
    if (r.status !== 0) {
      failed++;
      console.error(`FAIL ${id} [${level}]\n${String(r.stderr || r.error || "").trim().split("\n").slice(-12).join("\n")}\n`);
      continue;
    }
    const out = r.stdout.replace(/\x1b\[[0-9;]*m/g, "").replace(/\s+$/, "");
    results[id] = results[id] || {};
    results[id][level] = isYaml
      ? { status: "parsed", out, libs: [] }
      : snip.check
      ? { status: "checked", why: snip.check, libs: libsUsed(code) }
      : { status: "ran", out, libs: libsUsed(code), fakes: fakesUsed(code, snip) };
    if (only) console.log(`---- ${id} [${level}] ${snip.check ? "(checked only)" : ""}\n${out}\n`);
    else process.stdout.write(".");
  }
}
if (!only) process.stdout.write("\n");
if (failed) { console.error(`${failed} snippet(s) failed; output file not updated.`); process.exit(1); }

const sorted = Object.fromEntries(Object.keys(results).sort().map((k) => [k, results[k]]));
fs.writeFileSync(OUT_FILE, `// Generated by concepts/run-levels.js - how each Level 1/2 snippet was tested, and what it printed.
(function (root) {
  root.HOW_LEVELS_OUTPUT = ${JSON.stringify({ tested: new Date().toISOString().slice(0, 10), versions, results: sorted }, null, 2)};
})(typeof window !== "undefined" ? window : globalThis);
`);
const n = Object.values(sorted).reduce((a, r) => a + Object.keys(r).length, 0);
console.log(`OK: ${ids.length} topic(s) this run; ${n} snippets recorded in concepts/how-levels-output.js`);

// "Behind the scenes" for each How-it-works explanation, part 1: Inside the LLM, Retrieval & RAG.
// Each entry: a small from-scratch Python program (standard library only, runs with plain
// python3) whose comments point at the numbered steps, plus ways to watch the real system.
// Its printed output is captured by: node concepts/run-internals.js
// In `see`, text between backticks is shown as code.
(function (root) {
  const P = String.raw;
  const B = (root.HOW_INTERNALS = root.HOW_INTERNALS || {});
  const X = (id, see, code) => { B["hw-" + id] = { see, code: code.replace(/^\n/, "") }; };

  // ===================================================================== INSIDE THE LLM
  X("generate", [
    'Hugging Face: run the loop yourself: `logits = model(input_ids).logits[:, -1, :]`, pick `logits.argmax()`, append it to `input_ids`, repeat. That is exactly what `model.generate()` does for you.',
    'Print the top candidates at each step: `model.generate(..., output_scores=True, return_dict_in_generate=True)` returns the scores for every generated position.',
    'With an API, ask for `logprobs=True, top_logprobs=5` (OpenAI-style APIs) to see which tokens the model almost picked.',
  ], P`
# A toy language model: the next-token scores come from a lookup table instead of a transformer.
import math, random
random.seed(0)
VOCAB = ["<eos>", "the", "cat", "sat", "on", "mat", "."]

# Step 1: tokenizer = text -> ids
def encode(text): return [VOCAB.index(w) for w in text.split()]
def decode(ids): return " ".join(VOCAB[i] for i in ids)

# Steps 2-4: the "model" returns one score (logit) per vocabulary token for the NEXT position
BIGRAMS = {"the": {"cat": 1.5, "mat": 3.0}, "cat": {"sat": 3.0}, "sat": {"on": 3.0},
           "on": {"the": 3.0}, "mat": {".": 3.0}, ".": {"<eos>": 5.0}}
def model(ids):
    last = VOCAB[ids[-1]]
    return [BIGRAMS.get(last, {}).get(tok, -2.0) for tok in VOCAB]

# Step 5: sampling turns scores into probabilities and picks one token
def sample(logits, temperature=0.7):
    exps = [math.exp(l / temperature) for l in logits]
    total = sum(exps)
    return random.choices(range(len(VOCAB)), weights=[e / total for e in exps])[0]

# Step 6: the loop - append the new token and run again
ids = encode("the cat")
for step in range(10):                       # max_tokens
    next_id = sample(model(ids))
    print(f"step {step}: {decode(ids)!r:28} -> {VOCAB[next_id]!r}")
    if VOCAB[next_id] == "<eos>":            # stop token
        break
    ids.append(next_id)
print("final:", decode(ids))
`);

  X("tokenize", [
    'OpenAI tokenizers: `import tiktoken; enc = tiktoken.get_encoding("o200k_base"); ids = enc.encode("unbelievable")`, then `[enc.decode([i]) for i in ids]` shows each piece.',
    'Hugging Face: `AutoTokenizer.from_pretrained(name).tokenize("unbelievable")` shows the pieces; `.encode(...)` shows the IDs.',
    'Compare the same sentence across two tokenizers (and in another language, or code) to see why token counts and costs differ.',
  ], P`
# Train a tiny BPE tokenizer, then use it.
from collections import Counter

corpus = "low lower lowest newer newest wider widest"
# Step 1: start from single characters (real BPE starts from bytes). "_" marks the end of a word.
words = Counter(tuple(w) + ("_",) for w in corpus.split())

def pair_counts(words):
    counts = Counter()
    for word, freq in words.items():
        for a, b in zip(word, word[1:]):
            counts[a, b] += freq
    return counts

def merge(word, pair):
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

# Step 2: repeatedly merge the most frequent adjacent pair into a new token
merges = []
for _ in range(8):
    best, n = pair_counts(words).most_common(1)[0]
    merges.append(best)
    words = Counter({merge(w, best): f for w, f in words.items()})
    print(f"merge {best[0]!r} + {best[1]!r} (seen {n}x)")

# Step 3: tokenize new text by replaying the merges in order; each piece gets an ID
def tokenize(word):
    parts = tuple(word) + ("_",)
    for pair in merges:
        parts = merge(parts, pair)
    return list(parts)

vocab = {}
for w in ["lowest", "newest", "slower"]:
    pieces = tokenize(w)
    ids = [vocab.setdefault(p, len(vocab)) for p in pieces]
    print(f"{w!r:10} -> {pieces} -> ids {ids}")
`);

  X("attention", [
    'Hugging Face: load the model with `attn_implementation="eager"`, call `model(**inputs, output_attentions=True)` and read `.attentions`: one tensor per layer, shaped [batch, heads, tokens, tokens].',
    'Visualize it with the `bertviz` package (`head_view(attentions, tokens)`) to see which words each head looks at.',
    'Read a minimal implementation such as Karpathy’s nanoGPT `model.py` (CausalSelfAttention): about 30 lines, the same maths as below.',
  ], P`
# Single-head causal attention over 3 tokens, written out by hand.
import math

tokens = ["The", "cat", "sat"]
# Step 1: each token has a query, key and value vector (normally token vector x learned matrices)
Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
K = [[1.0, 0.0], [0.2, 1.0], [0.9, 0.8]]
V = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
d = len(Q[0])

def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [x / sum(e) for x in e]

for i, tok in enumerate(tokens):
    # Step 2: score = query . key / sqrt(d)
    # Step 3: causal mask - only tokens at or before position i are scored
    scores = [sum(q * k for q, k in zip(Q[i], K[j])) / math.sqrt(d) for j in range(i + 1)]
    # Step 4: softmax turns the scores into weights that add up to 1
    w = softmax(scores)
    # Step 5: the output is the weighted mix of the value vectors
    out = [sum(w[j] * V[j][c] for j in range(i + 1)) for c in range(d)]
    looks = ", ".join(f"{tokens[j]} {w[j]:.2f}" for j in range(i + 1))
    print(f"{tok:>4} attends to [{looks}] -> output {[round(x, 2) for x in out]}")
# Step 6: multi-head attention = this, several times in parallel with different Q/K/V matrices
`);

  X("kvcache", [
    'Hugging Face: `out = model(**inputs, use_cache=True)` returns `out.past_key_values`; pass it back with the next token to skip recomputing the prefix.',
    'Time `model.generate(..., use_cache=False)` against the default: generation gets much slower as the output grows.',
    'vLLM prints the KV-cache size (in tokens) at startup and exposes KV-cache usage on its `/metrics` endpoint.',
  ], P`
# Count the work with and without a KV cache. "Work" = how many tokens get K and V computed.
prompt = ["The", "capital", "of", "France", "is"]
new_tokens = ["Paris", ",", "a", "city"]

def kv(token):                    # stands in for: token vector x weight matrices W_k and W_v
    return (f"K({token})", f"V({token})")

# Without a cache: every step recomputes K and V for the whole sequence
work_no_cache, seq = 0, list(prompt)
for tok in new_tokens:
    work_no_cache += len(seq)
    seq.append(tok)

# Step 1 (prefill): the whole prompt at once, saved in the cache
cache = [kv(t) for t in prompt]
work_cache = len(prompt)
print(f"prefill: cached {len(cache)} tokens")
for tok in new_tokens:
    # Step 2: the newest token computes only its own K and V
    k, v = kv(tok)
    work_cache += 1
    # Step 3: its query attends over the cached K/V (read, not recomputed)
    # Step 4: append, so the cache grows by one token per step
    cache.append((k, v))
    print(f"decode {tok!r:8} reads {len(cache) - 1} cached K/V, computes 1, cache now {len(cache)}")
print(f"K/V computations without cache: {work_no_cache}, with cache: {work_cache}")

# Memory per token = 2 (K and V) x layers x KV heads x head size x bytes
layers, kv_heads, head_dim, fp16 = 80, 8, 128, 2
per_token = 2 * layers * kv_heads * head_dim * fp16
print(f"70B-style model: {per_token / 1e6:.2f} MB per token, {per_token * 8192 / 1e9:.1f} GB for one 8k context")
`);

  X("sampling", [
    'Ask an API for `logprobs=True, top_logprobs=5` and print the alternatives for each output token; then change `temperature` and watch them shift.',
    'Hugging Face: `model.generate(..., do_sample=True, temperature=0.7, top_p=0.9)`; set `do_sample=False` for greedy decoding.',
    'Run the same prompt 5 times at temperature 0 and at 1.0 to feel the difference in variety.',
  ], P`
import math, random
random.seed(1)

# Step 1: raw scores (logits) for the next token after "The weather today is"
logits = {"sunny": 3.0, "cloudy": 2.5, "rainy": 2.0, "cold": 1.0, "purple": -1.0}

def softmax(scores, temperature):
    # Step 2: divide by the temperature. Step 3: softmax -> probabilities that add up to 1
    exps = {t: math.exp(s / temperature) for t, s in scores.items()}
    total = sum(exps.values())
    return {t: e / total for t, e in exps.items()}

for temp in (0.2, 1.0, 2.0):
    probs = softmax(logits, temp)
    print(f"T={temp}: " + "  ".join(f"{t} {p:.2f}" for t, p in probs.items()))

# Step 4: top-p keeps the smallest set of tokens whose probabilities reach p
def top_p(probs, p=0.9):
    kept, total = {}, 0.0
    for tok, pr in sorted(probs.items(), key=lambda kv: -kv[1]):
        kept[tok] = pr
        total += pr
        if total >= p:
            break
    return kept

kept = top_p(softmax(logits, 1.0), 0.9)
print("top-p 0.9 keeps:", list(kept))
# Step 5: draw one token at random, weighted by probability
draws = [random.choices(list(kept), weights=list(kept.values()))[0] for _ in range(10)]
print("10 draws:", draws)
print("temperature 0 (greedy) always picks:", max(logits, key=logits.get))
`);

  X("training-overview", [
    'Compare a base model and its instruct version on Hugging Face (e.g. a `-Base` and an `-Instruct` model of the same family) with the same prompt: the base one just continues the text.',
    'Print `tokenizer.apply_chat_template(messages, tokenize=False)` to see the exact text an instruct model was fine-tuned on.',
    'Read the training logs of an open run (loss going down over steps): pretraining and SFT both minimize the same next-token loss.',
  ], P`
# One tiny model, trained twice: pretraining on raw text, then SFT on an instruction example.
# The model is a table of scores: next token = softmax(W[previous token]).
import math

def train(W, pairs, epochs, label, lr=0.5):
    for epoch in range(epochs):
        loss = 0.0
        for prev, nxt in pairs:
            row = W.setdefault(prev, {t: 0.0 for t in VOCAB})
            exps = {t: math.exp(s) for t, s in row.items()}
            z = sum(exps.values())
            loss += -math.log(exps[nxt] / z)              # cross-entropy: surprise at the right token
            for t in VOCAB:                               # gradient step on the scores
                row[t] -= lr * (exps[t] / z - (1.0 if t == nxt else 0.0))
        if epoch in (0, epochs - 1):
            print(f"{label} epoch {epoch + 1:2}: loss {loss / len(pairs):.3f}")

def pairs_of(text):
    toks = text.split()
    return list(zip(toks, toks[1:]))

def predict(W, prev):
    row = W.get(prev, {})
    return max(row, key=row.get) if row else None

web = ("paris is the capital of france . the capital of france is paris . "
       "what is the capital of spain ? what is the capital of italy ?")
sft = "what is the capital of france ? answer : paris ."
VOCAB = sorted(set(web.split()) | set(sft.split()))
W = {}
# Step 1: pretraining - learn to continue text
train(W, pairs_of(web), 30, "pretrain")
# Step 2: a base model continues text; after "?" it expects another question
print("base model: after '?' comes", repr(predict(W, "?")))
# Step 3: SFT on instruction -> answer examples
train(W, pairs_of(sft) * 5, 30, "sft     ")
print("after SFT:  after '?' comes", repr(predict(W, "?")))
# Steps 4-5 (preference tuning, RL) - see the RLHF/DPO page
`);

  X("reasoning", [
    'Anthropic API: enable extended thinking with `thinking={"type": "enabled", "budget_tokens": 4000}`; the response contains `thinking` blocks before the `text` block.',
    'OpenAI-style APIs report hidden reasoning in usage, e.g. `usage.completion_tokens_details.reasoning_tokens`: compare it with the visible answer length.',
    'Run one hard question at low and high effort/budget and compare accuracy, latency and tokens.',
  ], P`
# Thinking = writing intermediate steps before answering. RL rewards chains that reach right answers.
import random
random.seed(3)

def think_and_answer(a, b, budget, slip=0.15):
    # computes a x b via partial products; each written step may contain a small slip
    tens, ones = b // 10 * 10, b % 10
    steps = [(f"{a} x {tens}", a * tens), (f"{a} x {ones}", a * ones)]
    thoughts, total = [], 0
    for i, (text, value) in enumerate(steps):
        if i >= budget:                          # out of thinking budget: guess
            value = a * b // 20 * 10
            text += " (guessed)"
        elif random.random() < slip:
            value += random.choice([-10, 10])    # an arithmetic slip
        thoughts.append(f"{text} = {value}")
        total += value
    return thoughts, total

# Step 1 + 3: think first (often hidden), then answer. Step 4: the budget trades cost for quality
for budget in (0, 2):
    thoughts, answer = think_and_answer(17, 24, budget)
    print(f"budget {budget}: thinking {thoughts} -> answer {answer}")

# Step 2: RL with a verifiable reward - sample many chains, reward the ones that are right
correct = 17 * 24
samples = [think_and_answer(17, 24, 2) for _ in range(8)]
rewards = [1 if ans == correct else 0 for _, ans in samples]
print("rewards for 8 sampled chains:", rewards)
print("training raises the probability of the rewarded chains, so careful reasoning becomes the habit")
`);

  X("structured", [
    'LangChain: `model.with_structured_output(MySchema, include_raw=True)` returns the raw message, the parsed object and any parsing error, so you see what came back.',
    'Print `MySchema.model_json_schema()` (Pydantic) to see the JSON Schema that is sent to the API.',
    'Self-hosted: vLLM and llama.cpp support grammar/JSON-schema guided decoding; compare output with and without it on a small model.',
  ], P`
# Constrained generation masks out every token the schema does not allow, before each pick.
import json, random
random.seed(0)

VOCAB = ['{', '}', '"sentiment"', ':', '"pos"', '"neg"', 'Sure!', 'Here', ',']
# Step 1: the schema {"sentiment": "pos" | "neg"} becomes a grammar: what may come at each position
GRAMMAR = [['{'], ['"sentiment"'], [':'], ['"pos"', '"neg"'], ['}']]

def model_scores(so_far):
    # the unconstrained model would love to chat first
    return {t: (5.0 if t in ('Sure!', 'Here') else random.uniform(0, 2)) for t in VOCAB}

out = []
while len(out) < len(GRAMMAR):
    # Step 2: work out which tokens could legally come next
    legal = GRAMMAR[len(out)]
    scores = model_scores(out)
    wanted = max(scores, key=scores.get)
    # Step 3: every other token gets probability zero (score -inf) before picking
    masked = {t: (s if t in legal else float("-inf")) for t, s in scores.items()}
    pick = max(masked, key=masked.get)
    print(f"model wanted {wanted:12} allowed {legal} -> {pick}")
    out.append(pick)

# Step 4: guaranteed to parse
text = "".join(out)
print("output:", text, "->", json.loads(text))
`);

  // ===================================================================== RETRIEVAL & RAG
  X("embeddings", [
    'sentence-transformers: `m = SentenceTransformer("all-MiniLM-L6-v2"); v = m.encode(["my dog is sick", "take your puppy to the vet"])`, then `util.cos_sim(v[0], v[1])`.',
    'Print `len(v[0])` (the dimension) and a few values: an embedding is just a list of floats.',
    'Make a small similarity table for 5 sentences, including a paraphrase and an unrelated one, to build intuition for what scores mean.',
  ], P`
# A real embedding model is a trained transformer. Here each word has a hand-made vector over
# 3 "meaning" dimensions [pets, money, weather]; a text's vector is the average of its words.
import math

WORDS = {"dog": [1, 0, 0], "puppy": [.9, 0, .1], "cat": [.9, 0, 0], "vet": [.8, .2, 0],
         "bank": [0, 1, 0], "loan": [0, .9, 0], "interest": [0, .8, .1], "rate": [0, .6, .2],
         "rain": [0, 0, 1], "storm": [0, .1, .9], "forecast": [0, .1, .8]}

def embed(text):
    # Step 1: text -> a fixed-length vector
    vecs = [WORDS[w] for w in text.lower().split() if w in WORDS]
    return [sum(c) / len(vecs) for c in zip(*vecs)]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

# Step 3: documents are embedded once and stored
docs = ["Take your puppy to the vet", "Bank loan interest rate rises", "Storm and rain forecast"]
index = [(d, embed(d)) for d in docs]

# Step 4: the query is embedded with the same model; rank by cosine similarity
for query in ["my dog is sick", "what will the rate be on my loan"]:
    q = embed(query)
    print(f"query: {query!r}")
    for d, v in sorted(index, key=lambda dv: cosine(q, dv[1]), reverse=True):
        print(f"   {cosine(q, v):.2f}  {d}")
`);

  X("hnsw", [
    'FAISS: `index = faiss.IndexHNSWFlat(dim, 32)`; change `index.hnsw.efSearch` and measure recall against `IndexFlatL2` (exact search) and query time.',
    'pgvector: `CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)`, then `EXPLAIN ANALYZE` your query to confirm the index is used; tune `SET hnsw.ef_search = 100`.',
    'hnswlib: `p.set_ef(k)`; plot recall vs latency as ef grows to see the trade-off.',
  ], P`
# A small HNSW-style search: a sparse top layer for long jumps, a dense bottom layer for precision.
import math, random
random.seed(7)

points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(400)]
query = (73.0, 21.0)

# Step 1 (insert): link every point to its nearest neighbours; a few points are also promoted
# to an upper layer. (Real HNSW builds this incrementally as vectors arrive.)
def link(ids, m):
    return {i: sorted((j for j in ids if j != i), key=lambda j: math.dist(points[i], points[j]))[:m] for i in ids}
bottom_ids = list(range(len(points)))
top_ids = bottom_ids[::20]                     # ~5% of the points
layers = [link(top_ids, 4), link(bottom_ids, 8)]

memo = {}                                      # count how many distances we compute
def d(i):
    if i not in memo:
        memo[i] = math.dist(points[i], query)
    return memo[i]

# Steps 2-3: greedy walk on the top layer, then drop down and continue from there
cur = top_ids[0]
for name, graph in zip(["top", "bottom"], layers):
    improved = True
    while improved:
        improved = False
        for n in graph[cur]:
            if d(n) < d(cur):
                cur, improved = n, True
    print(f"{name:6} layer: closest so far = point {cur} at distance {d(cur):.1f}")

# Step 4: on the bottom layer, explore a candidate list of size ef and keep the best k
ef, k = 16, 3
cands, seen, frontier = [cur], {cur}, [cur]
while frontier:
    for n in layers[1][frontier.pop()]:
        if n not in seen:
            seen.add(n)
            cands = sorted(cands + [n], key=d)[:ef]
            if n in cands:
                frontier.append(n)

exact = sorted(bottom_ids, key=lambda i: math.dist(points[i], query))[:k]
print("HNSW top-3: ", sorted(cands, key=d)[:k], f"({len(memo)} distance checks)")
print("exact top-3:", exact, f"({len(points)} distance checks)")
`);

  X("rag", [
    'LangChain: `from langchain_core.globals import set_debug; set_debug(True)` prints every step’s input and output, including the final prompt with the retrieved chunks.',
    'LangSmith: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`; each run shows the retriever’s documents and the exact prompt sent to the model.',
    'Always print the retrieved chunks next to the answer while developing: most bad answers are bad retrieval.',
  ], P`
# A whole RAG pipeline in miniature. Retrieval is word overlap instead of embeddings and the
# "LLM" is a stub, so you can see exactly what gets built and passed along.
import re

DOCS = {
    "hr-policy.md": "Employees get 24 days of paid leave per year. Unused leave carries over up to 5 days.",
    "it-policy.md": "Laptops are replaced every 3 years. Report lost devices to IT within 24 hours.",
    "travel.md": "Book flights through the travel portal. Economy class for flights under 6 hours.",
}
def words(t): return set(re.findall(r"[a-z0-9]+", t.lower()))

# Step 1 (indexing, ahead of time): split into chunks and index them
chunks = [(src, s.strip()) for src, text in DOCS.items() for s in text.split(". ") if s.strip()]
index = [(src, text, words(text)) for src, text in chunks]
print(f"indexed {len(index)} chunks")

# Step 2: a chat follow-up would be rewritten into a standalone question here
question = "How many days of paid leave do I get?"
# Step 3: retrieval - score every chunk
q = words(question)
scored = sorted(((len(q & w), src, text) for src, text, w in index), reverse=True)
# Step 4: "rerank" - keep the best 2 that match at all
top = [(src, text) for score, src, text in scored[:2] if score > 0]
# Step 5: build the prompt: numbered, labelled context plus the instruction
context = "\n".join(f"[{i + 1}] ({src}) {text}" for i, (src, text) in enumerate(top))
prompt = f"Answer only from the context. Cite like [1].\n\nContext:\n{context}\n\nQuestion: {question}"
print("---- prompt sent to the LLM ----\n" + prompt + "\n--------------------------------")

# Step 6: the LLM answers from the context with a citation (stubbed)
def fake_llm(prompt):
    return context.splitlines()[0].split(") ", 1)[1] + " [1]"
print("answer:", fake_llm(prompt))
`);

  X("hybrid", [
    'LangChain: combine a `BM25Retriever` and a vector-store retriever in an `EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])`, which fuses with RRF.',
    'Elasticsearch/OpenSearch, Weaviate, Qdrant and pgvector setups all offer hybrid queries; log both ranked lists before fusion to see what each one contributes.',
    'Test with an exact code or ID (like an error code): keyword search finds it, embeddings often don’t.',
  ], P`
import math, re

docs = {
    "d1": "reset your password from the account settings page",
    "d2": "error code E-4021 means the payment token expired",
    "d3": "how to change login credentials and recover access",
    "d4": "payment failed because the card was declined",
}
tok = lambda t: re.findall(r"[a-z0-9-]+", t.lower())

# Step 1: BM25 - matching terms; rare terms weigh more; long documents are normalized
def bm25(query, k1=1.5, b=0.75):
    N = len(docs)
    avg = sum(len(tok(t)) for t in docs.values()) / N
    scores = {}
    for doc_id, text in docs.items():
        words, s = tok(text), 0.0
        for term in tok(query):
            df = sum(term in tok(t) for t in docs.values())
            if df == 0:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            tf = words.count(term)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(words) / avg))
        scores[doc_id] = s
    return [d for d in sorted(scores, key=scores.get, reverse=True) if scores[d] > 0]

# Step 2: vector search - pretend an embedding model returned this ranking
# (it understands "sign in problems" ~ password/credentials, but not the exact error code)
query = "sign in problems after E-4021"
vector_ranking = ["d3", "d1", "d4", "d2"]

# Step 3: RRF ignores the scores and combines ranks: 1 / (k + rank)
def rrf(*rankings, k=60):
    fused = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            fused[d] = fused.get(d, 0) + 1 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])

keyword_ranking = bm25(query)
print("BM25 ranking:  ", keyword_ranking)
print("vector ranking:", vector_ranking)
# Step 4: documents found by both, or near the top of either, rise
for d, s in rrf(keyword_ranking, vector_ranking):
    print(f"  {d} {s:.4f}  {docs[d]}")
`);

  X("rerank", [
    'sentence-transformers: `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict([(query, doc) for doc in candidates])` returns one relevance score per pair.',
    'Hosted rerankers (Cohere, Voyage, Jina and others) take a query plus a list of documents and return them re-sorted with scores; log the order before and after.',
    'Measure it: recall@5 or MRR on a small labelled set, with and without the reranker.',
  ], P`
import re
words = lambda t: re.findall(r"[a-z]+", t.lower())

query = "flights from Delhi to Mumbai"
candidates = [
    "Flights from Mumbai to Delhi every morning",
    "Mumbai and Delhi hotel deals",
    "Delhi to Mumbai flights depart every hour",
]

# Step 1: bi-encoder - each text becomes a vector ON ITS OWN (here: a set of words), ahead of time
def embed(text): return set(words(text))
def bi_score(q, d): return len(q & d) / len(q | d)
doc_vectors = [embed(d) for d in candidates]
q_vec = embed(query)
print("bi-encoder (fast, but query and document never meet):")
for d, v in sorted(zip(candidates, doc_vectors), key=lambda x: -bi_score(q_vec, x[1])):
    print(f"  {bi_score(q_vec, v):.2f}  {d}")

# Step 2-3: cross-encoder - reads the query and document TOGETHER, so it can see that
# the direction of travel matters. (A real one is a transformer that outputs a score.)
def cross_score(q, d):
    qw, dw = words(q), words(d)
    overlap = len(set(qw) & set(dw)) / len(set(qw))
    right_direction = {"delhi", "mumbai"} <= set(dw) and dw.index("delhi") < dw.index("mumbai")
    return round(0.4 * overlap + 0.4 * right_direction + 0.2 * ("flights" in dw), 2)

# Step 4: run it only on the top candidates - one model call per pair
print("cross-encoder rerank:")
for d in sorted(candidates, key=lambda d: -cross_score(query, d)):
    print(f"  {cross_score(query, d):.2f}  {d}")
`);

  X("chunking", [
    'LangChain: `RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_text(text)`; print each chunk with `len(chunk)` and look at where it cut.',
    'Use `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, ...)` for code: its separators are class and def boundaries.',
    'Open 10 random chunks from your real index and ask: would a person understand this chunk on its own?',
  ], P`
# A small version of LangChain's RecursiveCharacterTextSplitter.
text = ("RAG systems split documents into chunks.\n\nEach chunk is embedded separately. "
        "Chunks that are too big blur the meaning; chunks that are too small lose context.\n\n"
        "Overlap repeats a little text between neighbouring chunks.")
# Step 1: an ordered list of separators, from biggest to smallest
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def split(text, size, seps=SEPARATORS):
    # Step 2: split on the first separator
    sep, rest = seps[0], seps[1:]
    pieces = [p for p in (text.split(sep) if sep else list(text)) if p]
    small = []
    for p in pieces:
        # Step 3: pieces still too big are split again with the next separator
        small.extend([p] if len(p) <= size else split(p, size, rest))
    # Step 4: merge small neighbours back together, up to the chunk size
    chunks, cur = [], ""
    for p in small:
        candidate = cur + sep + p if cur else p
        if len(candidate) <= size:
            cur = candidate
        else:
            chunks.append(cur)
            cur = p
    return chunks + ([cur] if cur else [])

def add_overlap(chunks, n_words=3):
    # Step 5: start each chunk with the last few words of the previous one
    return [chunks[0]] + [" ".join(prev.split()[-n_words:]) + " ... " + c for prev, c in zip(chunks, chunks[1:])]

for i, c in enumerate(add_overlap(split(text, 70)), 1):
    print(f"chunk {i} ({len(c)} chars): {c!r}")
`);

  X("conv-rag", [
    'LangSmith or `set_debug(True)`: find the rewrite step and read the standalone question it produced; that is what was actually searched.',
    'Log both the raw follow-up and the rewritten query for every request, and review the ones with bad answers.',
    'Test with pronoun follow-ups ("what about it?", "and for contractors?"): they break retrieval when there is no rewrite step.',
  ], P`
import re
words = lambda t: set(re.findall(r"[a-z]+", t.lower()))

history = [
    ("user", "What is our parental leave policy?"),
    ("assistant", "Employees get 26 weeks of paid parental leave."),
]
follow_up = "Does it apply to contractors too?"
# Step 1: on its own, "it" means nothing to a search engine

# Step 2: a small, fast model rewrites the follow-up into a standalone question
def rewrite(history, question):
    prompt = ("Rewrite the last question so it makes sense without the chat history.\n"
              + "\n".join(f"{role}: {text}" for role, text in history)
              + f"\nuser: {question}\nStandalone question:")
    print("---- rewrite prompt ----\n" + prompt + "\n------------------------")
    return "Does the parental leave policy apply to contractors?"   # what the small LLM returns

standalone = rewrite(history, follow_up)

# Step 3: retrieval runs on the rewritten question
DOCS = ["Contractors are paid monthly via the vendor portal.",
        "Parental leave: 26 weeks paid for employees. Contractors are not eligible."]
best = lambda q: max(DOCS, key=lambda d: len(words(q) & words(d)))
print("search with raw follow-up ->", best(follow_up))
print("search with rewrite       ->", best(standalone))
# Step 4: the answer is generated from the retrieved context plus the conversation
`);

  X("parent-doc", [
    'LangChain: `ParentDocumentRetriever` (child splitter + parent splitter + a docstore); print the child hits and the parent ids they map to.',
    'A simple DIY version: store `parent_id` in each chunk’s metadata, search chunks, then fetch parents by id and de-duplicate.',
    'Compare answers using only the matched chunk vs the full parent for questions whose answer spans several sentences.',
  ], P`
import re
words = lambda t: set(re.findall(r"[a-z0-9]+", t.lower()))

# Step 1: split each document twice - big parent sections, small child chunks that point to a parent
parents = {
    "P1": "Refunds. Customers can request a refund within 30 days. Refunds go back to the original "
          "payment method. Digital downloads are refundable only if not yet downloaded.",
    "P2": "Shipping. Orders ship within 2 business days. Express shipping costs extra. "
          "International orders may take up to 3 weeks.",
}
children = [{"text": s, "parent": pid} for pid, text in parents.items() for s in text.split(". ")]
print(f"{len(parents)} parents -> {len(children)} child chunks")

# Step 2: only the small chunks are searched, so matching is precise
question = "Can I get a refund for an e-book I already downloaded?"
best = max(children, key=lambda c: len(words(question) & words(c["text"])))
print("best child chunk:", repr(best["text"]), "-> parent", best["parent"])

# Step 3-4: return the parent section instead: it holds the rule the child chunk is missing
print("context sent to the LLM:", parents[best["parent"]])
`);

  X("text-to-sql", [
    'Log the generated SQL for every question, next to the answer; show it to users too, so they can check it.',
    'Run `EXPLAIN` on generated queries in development to catch slow full-table scans before they reach production.',
    'Connect with a read-only database role and a statement timeout, then try a prompt that asks it to delete something, and confirm the database refuses.',
  ], P`
# Uses SQLite from the standard library.
import re, sqlite3

db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE orders (id INTEGER, customer TEXT, amount REAL, created TEXT);
INSERT INTO orders VALUES (1,'acme',120,'2026-10-02'), (2,'globex',80,'2026-10-15'),
                          (3,'acme',200,'2026-11-01'), (4,'initech',50,'2026-11-03');
""")
ALLOWED_TABLES = {"orders"}

# Step 1: the prompt carries the schema (real systems add column meanings and example queries)
schema = db.execute("SELECT sql FROM sqlite_master WHERE name='orders'").fetchone()[0]
question = "Which customer spent the most in total?"
prompt = f"Schema:\n{schema}\n\nWrite one SQLite query for: {question}"

# Step 2: the model writes SQL (stubbed: the first attempt has a mistake, the second is fixed)
attempts = iter([
    "SELECT customer, SUM(amount) AS total FROM order GROUP BY customer ORDER BY total DESC",
    "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY total DESC",
])

def validate(sql):
    # Step 3: one read-only SELECT, allowed tables only, always a row limit
    if not re.match(r"(?is)^\s*select\b", sql) or ";" in sql.rstrip(";"):
        raise ValueError("only a single SELECT is allowed")
    for table in re.findall(r"(?i)\b(?:from|join)\s+(\w+)", sql):
        if table.lower() not in ALLOWED_TABLES:
            raise ValueError(f"table '{table}' is not allowed (allowed: {sorted(ALLOWED_TABLES)})")
    return sql + " LIMIT 100"

sql = next(attempts)
for attempt in (1, 2):
    try:
        safe = validate(sql)
        rows = db.execute(safe).fetchall()   # Step 4: in production, a read-only user on a replica
        print(f"attempt {attempt} ran: {safe}")
        break
    except Exception as err:
        # Step 5: the error goes back to the model so it can fix the query
        print(f"attempt {attempt} rejected: {err} -> error sent back to the model")
        sql = next(attempts)

# Step 6: summarize, and show the SQL so users can check it
print("rows:", rows)
print(f"answer: {rows[0][0]} spent the most ({rows[0][1]:.0f}).")
`);
})(typeof window !== "undefined" ? window : globalThis);

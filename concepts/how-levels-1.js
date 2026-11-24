// Level 1 ("Use it": what you'd write at work) and Level 2 ("Under the hood": what the library does
// for you, written with its real lower-level classes) for each How-it-works explanation.
// Part 1: Inside the LLM, Retrieval & RAG.  Level 3 lives in how-internals-*.js.
//
// A snippet: { code, note?, source?: [files to read next], script?: [scripted chat-model replies],
//              anthropic?: [mocked Anthropic replies], subs?: {"real": "smaller stand-in for tests"},
//              files?: {name: contents}, check?: "why it is only syntax/import-checked" }
// Every snippet is executed by concepts/run-levels.js with the real libraries.
(function (root) {
  const P = String.raw;
  const B = (root.HOW_LEVELS = root.HOW_LEVELS || {});
  const L = (id, use, under) => { B["hw-" + id] = { use, under }; };
  const QWEN = "Qwen/Qwen2.5-0.5B-Instruct";

  // ===================================================================== INSIDE THE LLM
  L("generate", {
    note: "One call. The provider runs the whole generation loop on its GPUs and bills you per input and output token.",
    script: ["A KV cache stores each token's keys and values, so every new token only computes its own."],
    code: P`
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-opus-5-5", max_tokens=1000)
reply = model.invoke("Explain a KV cache in one sentence.")

print(reply.content)
print(reply.usage_metadata)      # output tokens drive latency and cost: they are generated one by one
`,
  }, {
    note: "This is what model.generate() (and every inference server) does: the token-by-token loop, written out with a real 0.5B model.",
    source: ["transformers/generation/utils.py (GenerationMixin.generate)"],
    code: P`
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "${QWEN}"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

ids = tok("The capital of France is", return_tensors="pt").input_ids   # Step 1: text -> token ids
cache = None
for _ in range(8):
    with torch.no_grad():
        # Steps 2-4: embeddings -> transformer layers -> one score (logit) per vocabulary token
        out = model(input_ids=ids if cache is None else ids[:, -1:], cache=cache, use_cache=True)
    cache = out.cache                                   # KV cache: next step only feeds the newest token
    probs = out.logits[0, -1].softmax(-1)
    next_id = int(probs.argmax())                       # Step 5: pick one (greedy here; see Sampling)
    print(f"{tok.decode(next_id)!r:10} p={probs[next_id]:.2f}  (vocabulary: {probs.numel():,} tokens)")
    if next_id == tok.eos_token_id:
        break
    ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)   # Step 6: append and repeat
print(tok.decode(ids[0]))
`,
  });

  L("tokenize", {
    note: "Count tokens before sending big inputs. Each model family has its own tokenizer; for Claude use the API's count endpoint: client.messages.count_tokens(model=..., messages=...).",
    code: P`
import tiktoken

enc = tiktoken.get_encoding("o200k_base")          # the tokenizer used by recent OpenAI models
for text in ["unbelievable", "The cat sat on the mat.", "def add(a, b): return a + b"]:
    ids = enc.encode(text)
    print(f"{len(text):2} chars -> {len(ids)} tokens: {[enc.decode([i]) for i in ids]}")
print("Hindi:", len(enc.encode("नमस्ते दुनिया")), "tokens for 13 characters")
`,
  }, {
    note: "Inside a Hugging Face tokenizer: a byte-level BPE vocabulary plus the ordered merge rules learned in training.",
    source: ["tokenizer.json in any model repo (the 'model' section holds vocab and merges)"],
    code: P`
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("${QWEN}")
spec = json.loads(tok.backend_tokenizer.to_str())["model"]
# Step 2's result: a vocabulary plus merge rules, applied in order
print(spec["type"], "- vocabulary", len(spec["vocab"]), "pieces,", len(spec["merges"]), "merge rules")

pieces = tok.tokenize("unbelievable tokenization")   # Step 3: split into known pieces
print(pieces)                                         # 'Ġ' marks a leading space
print(tok.convert_tokens_to_ids(pieces))              # ...and each piece becomes an id
# Step 1: byte-level, so any text works - even emoji - and nothing is "unknown"
print(tok.tokenize("🙂"), "->", tok.decode(tok.encode("🙂")))
`,
  });

  L("attention", {
    note: "You rarely write attention yourself: models call PyTorch's fused kernel, which picks FlashAttention-style implementations when the hardware supports them.",
    code: P`
import torch
import torch.nn.functional as F

torch.manual_seed(0)
batch, heads, tokens, head_dim = 1, 2, 4, 8
q, k, v = (torch.randn(batch, heads, tokens, head_dim) for _ in range(3))

out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # one fused call
print(out.shape)   # same shape as q: one new vector per token per head
# In a model config you only choose the implementation, e.g.
# AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa")
`,
  }, {
    note: "The same computation by hand, as it happens inside every layer: projections, scaled scores, causal mask, softmax, weighted sum, heads.",
    source: ["transformers/models/qwen2/modeling_qwen2.py (eager_attention_forward)"],
    code: P`
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, d_model, H = 1, 4, 16, 2
hd = d_model // H
x = torch.randn(B, T, d_model)                                   # 4 token vectors
W_q, W_k, W_v, W_o = (torch.nn.Linear(d_model, d_model, bias=False) for _ in range(4))

def heads(t):                                                    # Step 6: split into H heads
    return t.view(B, T, H, hd).transpose(1, 2)

q, k, v = heads(W_q(x)), heads(W_k(x)), heads(W_v(x))            # Step 1: query, key, value
scores = q @ k.transpose(-2, -1) / math.sqrt(hd)                 # Step 2: dot products / sqrt(d)
future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(future, float("-inf"))               # Step 3: causal mask
weights = scores.softmax(dim=-1)                                 # Step 4: rows add up to 1
out = weights @ v                                                # Step 5: weighted sum of values

print("head 0 weights (row = token, column = token it looks at):")
print(weights[0, 0].detach().numpy().round(2))
same = torch.allclose(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-6)
print("same as the fused kernel:", same)
y = W_o(out.transpose(1, 2).reshape(B, T, d_model))              # join the heads, project back
print("layer output:", tuple(y.shape))
`,
  });

  L("kvcache", {
    note: "The cache is on by default; you mostly feel it as speed. Here a real 0.5B model generates the same 30 tokens with and without it.",
    code: P`
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "${QWEN}"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
inputs = tok("Write a short poem about the sea.", return_tensors="pt")

for use_cache in (True, False):
    start = time.time()
    model.generate(**inputs, max_new_tokens=30, do_sample=False, use_cache=use_cache)
    print(f"use_cache={use_cache}: {time.time() - start:.1f}s")
`,
  }, {
    note: "Looking inside the cache object the model fills: one key and value tensor per layer, growing by one position per generated token.",
    source: ["transformers/cache_utils.py (DynamicCache)"],
    code: P`
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

name = "${QWEN}"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()
ids = tok("The capital of France is", return_tensors="pt").input_ids

cache = DynamicCache()
with torch.no_grad():
    out = model(input_ids=ids, cache=cache, use_cache=True)      # Step 1: prefill the whole prompt
    print("after prefill:", cache.get_seq_length(), "tokens cached in", len(cache.layers), "layers")
    print("layer 0 keys [batch, kv_heads, tokens, head_dim]:", tuple(cache.layers[0].keys.shape))
    for _ in range(3):
        nxt = out.logits[:, -1:].argmax(-1)                      # Step 2: only the newest token goes in
        out = model(input_ids=nxt, cache=cache, use_cache=True)  # Step 3: it attends over the cache
        print(f"decode {tok.decode(nxt[0])!r:9} input length {nxt.shape[1]} -> cache {cache.get_seq_length()}")  # Step 4

cfg = model.config
head_dim = cfg.hidden_size // cfg.num_attention_heads
per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2   # K+V, 2 bytes each
print(f"{per_token / 1024:.0f} KB per token -> {per_token * 32768 / 1e9:.2f} GB for one 32k-token context")
`,
  });

  L("sampling", {
    note: "Open models and many APIs take temperature/top_p. Current Claude models don't accept them any more: you steer those with the prompt and effort instead.",
    code: P`
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

name = "${QWEN}"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
messages = [{"role": "user", "content": "Suggest a name for a coffee shop. Reply with the name only."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

set_seed(0)
for temperature in (0.2, 1.5):
    for _ in range(3):
        out = model.generate(**inputs, do_sample=True, temperature=temperature, top_p=0.9, max_new_tokens=12)
        reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"temperature {temperature}: {reply.strip()}")
`,
  }, {
    note: "What do_sample, temperature and top_p do to the model's real next-token scores.",
    source: ["transformers/generation/logits_process.py (TemperatureLogitsWarper, TopPLogitsWarper)"],
    code: P`
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "${QWEN}"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()
with torch.no_grad():
    logits = model(**tok("The most popular fruit in the world is the", return_tensors="pt")).logits[0, -1].float()   # Step 1

def top5(probs):
    p, i = torch.topk(probs, 5)
    return ", ".join(f"{tok.decode(int(t))!r} {v:.2f}" for v, t in zip(p, i))

for T in (0.5, 1.0, 1.5):                       # Step 2: divide by temperature; Step 3: softmax
    print(f"T={T}: {top5((logits / T).softmax(-1))}")

probs = logits.softmax(-1)
sorted_p, order = probs.sort(descending=True)
keep = (sorted_p.cumsum(0) - sorted_p) < 0.9    # Step 4: top-p - smallest set reaching 90%
print(f"top-p 0.9 keeps {int(keep.sum())} of {probs.numel():,} tokens")
filtered = torch.zeros_like(probs)
filtered[order[keep]] = sorted_p[keep]
torch.manual_seed(0)
picks = torch.multinomial(filtered / filtered.sum(), 6, replacement=True)   # Step 5: weighted draw
print("6 draws:", [tok.decode(int(t)) for t in picks])
`,
  });

  L("training-overview", {
    note: "The stages you actually run, with Hugging Face TRL. Pretraining is done by the model lab; you start from their checkpoint.",
    check: "needs a GPU and hours of training; syntax and imports were checked",
    code: P`
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

base = "Qwen/Qwen2.5-0.5B"                       # Steps 1-2: a pretrained base model from the lab

# Step 3: supervised fine-tuning on instruction -> response conversations
sft = SFTTrainer(model=base, train_dataset=load_dataset("trl-lib/Capybara", split="train"),
                 args=SFTConfig(output_dir="qwen-sft"))
sft.train()
sft.save_model()

# Step 4: preference tuning on (prompt, chosen, rejected) pairs
dpo = DPOTrainer(model="qwen-sft", train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
                 args=DPOConfig(output_dir="qwen-dpo"))
dpo.train()
`,
  }, {
    note: "The one number every stage minimizes: next-token cross-entropy. Here it is computed by hand on a real model and matched with the model's own loss.",
    source: ["transformers/loss/loss_utils.py (ForCausalLMLoss)"],
    code: P`
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "${QWEN}"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()
batch = tok("Paris is the capital of France.", return_tensors="pt")

with torch.no_grad():
    out = model(**batch, labels=batch.input_ids)          # what every trainer calls each step
logits = out.logits[0, :-1].float()                       # the prediction made at each position...
targets = batch.input_ids[0, 1:]                          # ...is scored against the NEXT token
manual = F.cross_entropy(logits, targets)
print(f"model's loss {out.loss:.4f} = hand-computed {manual:.4f}")

for t, l in zip(targets, F.cross_entropy(logits, targets, reduction="none")):
    print(f"  {tok.decode(int(t))!r:12} surprise {l:.2f}")   # training pushes these down
`,
  });

  L("reasoning", {
    note: "Thinking is adaptive on current Claude models: you choose how hard it works with effort. Server-side fallback reruns the request on another model if this one declines.",
    anthropic: [{ content: [
      { type: "thinking", thinking: "1001 is odd and its digit sum is 2, so not 3. Try 7: 7 x 143 = 1001. And 143 = 11 x 13.", signature: "sig" },
      { type: "text", text: "No. 1001 = 7 x 11 x 13." }],
      usage: { input_tokens: 18, output_tokens: 140, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 } }],
    code: P`
import anthropic

client = anthropic.Anthropic()
response = client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    thinking={"type": "adaptive", "display": "summarized"},   # show a summary of the reasoning
    output_config={"effort": "high"},                          # low | medium | high | xhigh | max
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",                                        # rerun on a fallback model if declined
    messages=[{"role": "user", "content": "Is 1001 a prime number?"}],
)
for block in response.content:
    if block.type == "thinking":
        print("THINKING:", block.thinking)
    elif block.type == "text":
        print("ANSWER:  ", block.text)
print("output tokens (thinking included):", response.usage.output_tokens)
`,
  }, {
    note: "How the habit of reasoning is trained (Step 2): sample several chains, score them with a checker, push up the ones that beat the group average. This is the core of GRPO, as in TRL's GRPOTrainer.",
    source: ["trl/trainer/grpo_trainer.py"],
    code: P`
import torch

question = "What is 17 x 24?"
answers = ["408", "398", "408", "418", "408", "408"]   # final answers of 6 sampled reasoning chains
correct = "408"                                        # verifiable: code can check it

rewards = torch.tensor([1.0 if a == correct else 0.0 for a in answers])
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)   # better than the group -> positive
print("rewards:   ", rewards.tolist())
print("advantages:", [round(a, 2) for a in advantages.tolist()])

# log-probability the model gave each whole chain (normally summed over its tokens)
logprobs = torch.tensor([-42.0, -40.0, -45.0, -39.0, -44.0, -41.0], requires_grad=True)
loss = -(advantages * logprobs).mean()        # policy gradient
loss.backward()
step = (-logprobs.grad).tolist()              # gradient descent moves log-probs this way
print("push each chain:", ["up" if s > 0 else "down" for s in step])
`,
  });

  L("structured", {
    note: "Describe the shape you want as a Pydantic class; you get a validated object back, not text to parse.",
    script: [{ tool_calls: [{ name: "Ticket", args: { category: "billing", urgent: true, summary: "Customer was charged twice and needs a fix today." } }] }],
    code: P`
from typing import Literal

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class Ticket(BaseModel):
    category: Literal["billing", "technical", "other"]
    urgent: bool
    summary: str = Field(description="One sentence")

model = init_chat_model("anthropic:claude-opus-5-5")
extractor = model.with_structured_output(Ticket)
ticket = extractor.invoke("My card was charged twice and I need this fixed today!")
print(type(ticket).__name__, ticket.model_dump())
`,
  }, {
    note: "What with_structured_output does: your class becomes a JSON schema sent as a tool, the model's tool call is parsed and validated. With strict tools or structured-output mode the server also constrains the tokens (Steps 2-3).",
    source: ["langchain_core/language_models/chat_models.py (with_structured_output)", "langchain_core/output_parsers/openai_tools.py"],
    script: [{ tool_calls: [{ name: "Ticket", args: { category: "billing", urgent: true, summary: "Charged twice." } }] }],
    code: P`
import json
from typing import Literal

from pydantic import BaseModel, ValidationError
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool

class Ticket(BaseModel):
    category: Literal["billing", "technical", "other"]
    urgent: bool
    summary: str

schema = convert_to_openai_tool(Ticket)["function"]["parameters"]     # Step 1: class -> JSON schema
print("schema sent:", json.dumps(schema))

model = init_chat_model("anthropic:claude-opus-5-5").bind_tools([Ticket])
ai = model.invoke("My card was charged twice and I need this fixed today!")
print("model returned a tool call:", ai.tool_calls[0]["args"])

ticket = PydanticToolsParser(tools=[Ticket], first_tool_only=True).invoke(ai)   # parse + validate
print("parsed:", repr(ticket))
try:
    Ticket(category="refund", urgent="maybe", summary="x")             # what validation catches
except ValidationError as err:
    print("rejected:", err.error_count(), "errors")
`,
  });

  // ===================================================================== RETRIEVAL & RAG
  L("embeddings", {
    note: "Embed once, store, then search by meaning. In production the store is pgvector, Qdrant, Pinecone and so on: same interface.",
    code: P`
from langchain.embeddings import init_embeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryVectorStore(embeddings)
store.add_texts(["Refunds are processed within 5 working days.",
                 "Reset your password from the account settings page.",
                 "We ship to 40 countries worldwide."])

for doc, score in store.similarity_search_with_score("how long does a refund take", k=2):
    print(f"{score:.2f}  {doc.page_content}")
`,
  }, {
    note: "What the store does on add and on search: embed the texts, embed the query with the same model, rank by cosine similarity.",
    source: ["langchain_core/vectorstores/in_memory.py (InMemoryVectorStore)"],
    code: P`
import math
from langchain.embeddings import init_embeddings

embeddings = init_embeddings("openai:text-embedding-3-small")
texts = ["Refunds are processed within 5 working days.",
         "Reset your password from the account settings page.",
         "We ship to 40 countries worldwide."]

vectors = embeddings.embed_documents(texts)        # Steps 1 + 3: one vector per text, stored
query = embeddings.embed_query("how long does a refund take")   # Step 4: same model for the query
print(f"{len(vectors)} vectors of {len(query)} numbers each")

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

for text, vec in sorted(zip(texts, vectors), key=lambda tv: -cosine(query, tv[1])):
    print(f"{cosine(query, vec):.2f}  {text}")
`,
  });

  L("hnsw", {
    note: "An HNSW index trades a little recall for a lot of speed; efSearch is the dial. pgvector equivalent: CREATE INDEX ... USING hnsw (embedding vector_cosine_ops); SET hnsw.ef_search = 100;",
    code: P`
import time
import faiss
import numpy as np

rng = np.random.default_rng(0)
d = 64
data = rng.random((20000, d), dtype=np.float32)
queries = rng.random((200, d), dtype=np.float32)

index = faiss.IndexHNSWFlat(d, 32)           # M = 32 links per vector
index.add(data)
exact = faiss.IndexFlatL2(d)                 # brute force, for the true answers
exact.add(data)
_, truth = exact.search(queries, 10)

for ef in (8, 32, 128):
    index.hnsw.efSearch = ef                 # size of the candidate list at query time
    start = time.time()
    _, found = index.search(queries, 10)
    ms = (time.time() - start) * 1000
    recall = np.mean([len(set(f) & set(t)) / 10 for f, t in zip(found, truth)])
    print(f"efSearch={ef:3}: recall@10 {recall:.2f} in {ms:.0f} ms for {len(queries)} queries")
`,
  }, {
    note: "Inside the index FAISS built: why the upper layers are sparse. Each vector draws its top layer at random, so each layer has about 1/M of the vectors of the one below.",
    source: ["faiss/impl/HNSW.cpp (random_level, search)"],
    code: P`
import faiss
import numpy as np

rng = np.random.default_rng(0)
d, M, n = 64, 32, 20000
index = faiss.IndexHNSWFlat(d, M)
index.add(rng.random((n, d), dtype=np.float32))

levels = faiss.vector_to_array(index.hnsw.levels)      # per vector: how many layers it lives on
print("top layer:", index.hnsw.max_level, "| search starts at vector", index.hnsw.entry_point)
for layer in range(levels.max() - 1, -1, -1):
    print(f"layer {layer}: {(levels > layer).sum():6} vectors")

# Step 1: the level draw used on insert: floor(-ln(U) * mL), with mL = 1 / ln(M)
mL = 1 / np.log(M)
drawn = np.floor(-np.log(rng.random(n)) * mL)
print("same rule, simulated:", [int((drawn >= layer).sum()) for layer in range(int(drawn.max()), -1, -1)])
# Steps 2-4 (greedy descent, then a beam of size efSearch on layer 0) run inside index.search()
`,
  });

  L("rag", {
    note: "Retrieve, put the chunks in the prompt, answer with citations. Most production RAG is this function plus good data and evals.",
    script: ["You get 24 days of paid leave per year [hr-policy.md]."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.vectorstores import InMemoryVectorStore

store = InMemoryVectorStore(init_embeddings("openai:text-embedding-3-small"))
store.add_texts(
    ["Employees get 24 days of paid leave per year.", "Laptops are replaced every 3 years.",
     "Book flights through the travel portal."],
    metadatas=[{"source": "hr-policy.md"}, {"source": "it-policy.md"}, {"source": "travel.md"}],
)
model = init_chat_model("anthropic:claude-opus-5-5")

def answer(question):
    docs = store.similarity_search(question, k=2)
    context = "\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)
    reply = model.invoke([("system", f"Answer only from this context and cite the source.\n\n{context}"),
                          ("human", question)])
    return reply.content, [d.metadata["source"] for d in docs]

print(answer("How many days of paid leave do I get?"))
`,
  }, {
    note: "The pieces behind that function: Documents with metadata, splitting, indexing, a metadata filter for permissions, and the exact messages the model receives.",
    source: ["langchain_core/documents/base.py", "langchain_core/prompts/chat.py"],
    script: ["You get 24 days of paid leave per year [hr-policy.md]."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1 (indexing, ahead of time): documents keep their metadata through every step
raw = [Document("Employees get 24 days of paid leave per year. Unused leave carries over up to 5 days.",
                metadata={"source": "hr-policy.md", "team": "all"}),
       Document("Executive bonus pool is 2M this year.", metadata={"source": "finance.md", "team": "finance"})]
chunks = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=0).split_documents(raw)
store = InMemoryVectorStore(init_embeddings("openai:text-embedding-3-small"))
store.add_documents(chunks)
print(f"{len(raw)} documents -> {len(chunks)} chunks:", [c.page_content for c in chunks])

# Step 3: retrieval, filtered to what this user may see
question = "How many days of paid leave do I get?"
docs = store.similarity_search(question, k=2, filter=lambda d: d.metadata["team"] == "all")

# Step 5: the prompt template turns context + question into messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the context. Cite the source in brackets.\n\n{context}"),
    ("human", "{question}"),
])
messages = prompt.invoke({"context": "\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs),
                          "question": question}).to_messages()
for m in messages:
    print(f"--- {m.type} ---\n{m.content}")
print("--- answer ---\n" + init_chat_model("anthropic:claude-opus-5-5").invoke(messages).content)   # Step 6
`,
  });

  L("hybrid", {
    note: "Keyword search (BM25) and vector search, fused with reciprocal rank fusion. Search engines like Elasticsearch, OpenSearch, Weaviate and Qdrant offer the same as one query.",
    code: P`
from langchain.embeddings import init_embeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import InMemoryVectorStore

texts = ["Reset your password from the account settings page.",
         "Error code E-4021 means the payment token expired.",
         "How to change login credentials and recover access.",
         "Payment failed because the card was declined."]

keyword = BM25Retriever.from_texts(texts, k=3)
vector = InMemoryVectorStore.from_texts(texts, init_embeddings("openai:text-embedding-3-small")).as_retriever(search_kwargs={"k": 3})
hybrid = EnsembleRetriever(retrievers=[keyword, vector], weights=[0.5, 0.5])

for doc in hybrid.invoke("login problems after error E-4021"):
    print(doc.page_content)
`,
  }, {
    note: "What EnsembleRetriever does: run each retriever, then weighted RRF on the ranks (weight / (rank + 60)). Computed by hand and compared.",
    source: ["langchain_classic/retrievers/ensemble.py (weighted_reciprocal_rank)"],
    code: P`
from langchain.embeddings import init_embeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import InMemoryVectorStore

texts = ["Reset your password from the account settings page.",
         "Error code E-4021 means the payment token expired.",
         "How to change login credentials and recover access.",
         "Payment failed because the card was declined."]
query = "login problems after error E-4021"
keyword = BM25Retriever.from_texts(texts, k=3)
vector = InMemoryVectorStore.from_texts(texts, init_embeddings("openai:text-embedding-3-small")).as_retriever(search_kwargs={"k": 3})

rankings = [keyword.invoke(query), vector.invoke(query)]      # Steps 1-2: two separate rankings
for name, docs in zip(["BM25", "vector"], rankings):
    print(f"{name:6}:", [texts.index(d.page_content) for d in docs])

scores = {}                                                    # Step 3: RRF on ranks, not scores
for weight, docs in zip([0.5, 0.5], rankings):
    for rank, d in enumerate(docs, start=1):
        scores[d.page_content] = scores.get(d.page_content, 0) + weight / (rank + 60)
by_hand = sorted(scores, key=scores.get, reverse=True)
library = [d.page_content for d in EnsembleRetriever(retrievers=[keyword, vector], weights=[0.5, 0.5]).invoke(query)]
print("fused:", [texts.index(t) for t in by_hand], "| same as EnsembleRetriever:", by_hand == library)
`,
  });

  L("rerank", {
    note: "A real cross-encoder (an 80 MB model) reranking candidates that a fast first-stage search returned.",
    code: P`
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
query = "How do I stop my subscription?"
candidates = ["Subscription plans start at 10 dollars a month.",          # shares words, wrong answer
              "Stop words are removed before indexing.",                  # shares words, unrelated
              "To cancel, open Settings and choose End membership."]      # no shared words, right answer

for hit in reranker.rank(query, candidates):
    print(f"{hit['score']:6.2f}  {candidates[hit['corpus_id']]}")
`,
  }, {
    note: "What CrossEncoder does: tokenize the query and document TOGETHER as one input, run a transformer classifier, read one relevance score per pair.",
    source: ["sentence_transformers/cross_encoder/CrossEncoder.py (predict)"],
    code: P`
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "cross-encoder/ms-marco-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()

query = "How do I stop my subscription?"
candidates = ["Subscription plans start at 10 dollars a month.",
              "Stop words are removed before indexing.",
              "To cancel, open Settings and choose End membership."]
batch = tok([query] * len(candidates), candidates, padding=True, return_tensors="pt")   # Step 3: pairs
print("one input:", tok.decode(batch["input_ids"][0], skip_special_tokens=False).replace(" [PAD]", ""))

with torch.no_grad():
    scores = model(**batch).logits.squeeze(-1)          # one number per (query, document) pair
for s, c in sorted(zip(scores.tolist(), candidates), reverse=True):
    print(f"{s:6.2f}  {c}")
# Step 4: one forward pass per candidate - which is why you only rerank the top 20-100
`,
  });

  L("chunking", {
    note: "Split documents before embedding them. Keep metadata, and record where each chunk starts so citations can point back.",
    code: P`
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

doc = Document(
    "RAG systems split documents into chunks.\n\nEach chunk is embedded separately. Chunks that are too "
    "big blur the meaning; chunks that are too small lose context.\n\nOverlap repeats a little text.",
    metadata={"source": "guide.md"},
)
splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20, add_start_index=True)
for c in splitter.split_documents([doc]):
    print(c.metadata, repr(c.page_content))
`,
  }, {
    note: "The splitter's real knobs: its separator list (Step 1), language-aware separators for code, and measuring size in tokens instead of characters.",
    source: ["langchain_text_splitters/character.py (RecursiveCharacterTextSplitter._split_text)"],
    code: P`
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

print("default separators:", RecursiveCharacterTextSplitter()._separators)           # Step 1
print("python separators: ", RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)[:4])

code = '''class Cart:
    def add(self, item):
        self.items.append(item)

    def total(self):
        return sum(i.price for i in self.items)

def checkout(cart):
    return cart.total()
'''
py = RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=90, chunk_overlap=0)
for c in py.split_text(code):                                      # Steps 2-4: split at def/class first
    print("---\n" + c)

by_tokens = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="o200k_base", chunk_size=12, chunk_overlap=3)
print([c for c in by_tokens.split_text("Chunk size can be measured in tokens, which is what the embedding model actually limits.")])
`,
  });

  L("conv-rag", {
    note: "Rewrite the follow-up into a standalone question with a fast model, then retrieve with that.",
    script: ["Does the parental leave policy apply to contractors?", "No - contractors are not eligible for parental leave."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore

retriever = InMemoryVectorStore.from_texts(
    ["Contractors are paid monthly via the vendor portal.",
     "Parental leave: 26 weeks paid for employees. Contractors are not eligible."],
    init_embeddings("openai:text-embedding-3-small")).as_retriever(search_kwargs={"k": 1})
fast = init_chat_model("anthropic:claude-haiku-5-5")
main = init_chat_model("anthropic:claude-opus-5-5")

rewrite = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"), ("human", "{question}"),
    ("human", "Rewrite my last question as a standalone search query. Reply with the query only."),
]) | fast | StrOutputParser()

history = [("human", "What is our parental leave policy?"), ("ai", "Employees get 26 weeks of paid leave.")]
question = "Does it apply to contractors too?"
query = rewrite.invoke({"history": history, "question": question})
docs = retriever.invoke(query)
print("searched for:", query, "| found:", docs[0].page_content)
print(main.invoke(history + [("system", f"Context: {docs[0].page_content}"), ("human", question)]).content)
`,
  }, {
    note: "Why the rewrite matters, shown with the real retriever: the same follow-up, searched raw and rewritten.",
    source: ["langchain_core/prompts/chat.py (MessagesPlaceholder)"],
    code: P`
from langchain.embeddings import init_embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore

retriever = InMemoryVectorStore.from_texts(
    ["Contractors are paid monthly via the vendor portal.",
     "Parental leave: 26 weeks paid for employees. Contractors are not eligible."],
    init_embeddings("openai:text-embedding-3-small")).as_retriever(search_kwargs={"k": 1})

history = [("human", "What is our parental leave policy?"), ("ai", "Employees get 26 weeks of paid leave.")]
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"), ("human", "{question}"),
    ("human", "Rewrite my last question as a standalone search query. Reply with the query only."),
])
# Step 2: exactly what the small rewriting model receives
for m in prompt.invoke({"history": history, "question": "Does it apply to contractors too?"}).to_messages():
    print(f"{m.type:5} {m.content}")

# Step 3: retrieval with and without the rewrite
for q in ["Does it apply to contractors too?", "Does the parental leave policy apply to contractors?"]:
    print(f"{q!r:56} -> {retriever.invoke(q)[0].page_content}")
`,
  });

  L("parent-doc", {
    note: "Search small chunks, return the big section they came from.",
    code: P`
from langchain.embeddings import init_embeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

retriever = ParentDocumentRetriever(
    vectorstore=InMemoryVectorStore(init_embeddings("openai:text-embedding-3-small")),   # small chunks
    docstore=InMemoryStore(),                                                           # full parents
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=0),
)
retriever.add_documents([
    Document("Refunds. Customers can request a refund within 30 days. Refunds go back to the original "
             "payment method. Digital downloads are refundable only if not yet downloaded."),
    Document("Shipping. Orders ship within 2 business days. International orders may take 3 weeks."),
])
print(retriever.invoke("refund for an e-book I already downloaded")[0].page_content)
`,
  }, {
    note: "The same thing by hand with the real stores: children carry their parent's id; search children; fetch parents by id.",
    source: ["langchain_classic/retrievers/multi_vector.py", "langchain_classic/retrievers/parent_document_retriever.py"],
    code: P`
import uuid
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

parents = [Document("Refunds. Customers can request a refund within 30 days. Refunds go back to the original "
                    "payment method. Digital downloads are refundable only if not yet downloaded."),
           Document("Shipping. Orders ship within 2 business days. International orders may take 3 weeks.")]
vectors, docstore = InMemoryVectorStore(init_embeddings("openai:text-embedding-3-small")), InMemoryStore()
splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=0)

for parent in parents:                                    # Step 1: split twice, link child -> parent
    pid = str(uuid.uuid4())
    docstore.mset([(pid, parent)])
    children = splitter.split_documents([parent])
    for c in children:
        c.metadata["doc_id"] = pid
    vectors.add_documents(children)                       # Step 2: only children are embedded

hit = vectors.similarity_search("refund for an e-book I already downloaded", k=1)[0]
print("matched child:", repr(hit.page_content))
print("returned parent:", docstore.mget([hit.metadata["doc_id"]])[0].page_content)   # Steps 3-4
`,
  });

  L("text-to-sql", {
    note: "An agent with one read-only SQL tool. The first query fails, the error goes back, the model fixes it: the loop from the diagram, live.",
    script: [
      { tool_calls: [{ name: "run_sql", args: { query: "SELECT customer, SUM(amount) AS total FROM order GROUP BY customer ORDER BY total DESC" } }] },
      { tool_calls: [{ name: "run_sql", args: { query: "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY total DESC" } }] },
      "acme spent the most: 320 in total. SQL: SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY total DESC",
    ],
    code: P`
import sqlite3
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

with sqlite3.connect("shop.db") as db:
    db.executescript("""CREATE TABLE orders (id INTEGER, customer TEXT, amount REAL);
        INSERT INTO orders VALUES (1,'acme',120), (2,'globex',80), (3,'acme',200);""")

@tool
def run_sql(query: str) -> str:
    """Run one read-only SQLite SELECT on table orders(id, customer, amount). Returns rows or an error."""
    if not query.lstrip().lower().startswith("select"):
        return "Error: only SELECT queries are allowed"
    try:
        with sqlite3.connect("file:shop.db?mode=ro", uri=True) as conn:      # read-only connection
            return str(conn.execute(query + " LIMIT 100").fetchall())
    except sqlite3.Error as err:
        return f"Error: {err}"                       # goes back to the model so it can fix the query

agent = create_agent(init_chat_model("anthropic:claude-opus-5-5"), [run_sql],
                     system_prompt="Answer questions about orders with SQL. Show the SQL you used.")
result = agent.invoke({"messages": [("user", "Which customer spent the most?")]})
for m in result["messages"][1:]:
    print(f"{m.type:5}", m.tool_calls[0]["args"]["query"] if getattr(m, "tool_calls", None) else m.content)
`,
  }, {
    note: "The safety rails the tool relies on, at the database level: read-only mode, a time limit and the query plan.",
    source: ["Python docs: sqlite3 (set_progress_handler, PRAGMA query_only)"],
    code: P`
import sqlite3
import time

conn = sqlite3.connect(":memory:")
conn.executescript("""CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, amount REAL);
    INSERT INTO orders VALUES (1,'acme',120), (2,'globex',80), (3,'acme',200);""")
schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()   # Step 1: for the prompt
print("schema for the prompt:", schema[0][0])

conn.execute("PRAGMA query_only = ON")                       # Step 4: this connection can't write
try:
    conn.execute("DELETE FROM orders")
except sqlite3.OperationalError as err:
    print("write blocked:", err)

deadline = time.time() + 2.0                                 # Step 3: a time limit
conn.set_progress_handler(lambda: time.time() > deadline, 10_000)

sql = "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY total DESC"
print("plan:", conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall())   # catch full scans on big tables
print("rows:", conn.execute(sql + " LIMIT 100").fetchall())
`,
  });
})(typeof window !== "undefined" ? window : globalThis);

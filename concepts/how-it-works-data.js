// Content for how-it-works.html.
// I(category, id, question, oneLine, diagram(s), steps[], keyDetails[], code?)
// Flow step syntax: "Label|small text|kind"  kinds: io, model, store, check, step (default)
(function () {
  const F = (rows, opts) => Object.assign({ type: "flow", rows }, opts || {});
  const S = (actors, msgs, opts) => Object.assign({ type: "seq", actors, msgs }, opts || {});
  const I = (cat, id, q, tldr, diagram, steps, notes, code) => ({ cat, id, q, tldr, diagram, steps, notes: notes || [], code: code || "" });

  window.HOW = {
    cats: [
      { id: "c-llm", title: "Inside the LLM", blurb: "What happens between your prompt and the model's reply." },
      { id: "c-retrieval", title: "Retrieval & RAG", blurb: "How the right information gets into the prompt." },
      { id: "c-agents", title: "Tools & agents", blurb: "How models take actions and work in loops." },
      { id: "c-frameworks", title: "LangChain & LangGraph", blurb: "What the frameworks do when your code runs." },
      { id: "c-serving", title: "Serving & infrastructure", blurb: "How models are run fast and cheaply at scale." },
      { id: "c-training", title: "Training & fine-tuning", blurb: "How models learn, and how you adapt them." },
      { id: "c-quality", title: "Quality, safety & operations", blurb: "How you measure, protect and run AI systems." },
    ],
    items: [
      // =====================================================================
      // INSIDE THE LLM
      // =====================================================================
      I("c-llm", "hw-generate", "How does an LLM generate text?",
        "It predicts one token at a time: text becomes tokens, the transformer scores every possible next token, one is picked, appended, and the loop repeats.",
        F([{ steps: ["Prompt text||io", "Tokenizer|text → token IDs", "Embeddings|IDs → vectors", "Transformer layers|attention + MLP, × N|model", "Logits|a score per vocab token", "Sampling|temperature, top-p|check", "Next token||io"],
             loop: "Append the new token and run again, until a stop token or max_tokens" }]),
        [
          "The tokenizer splits the text into tokens (sub-word pieces) and maps each one to an integer ID.",
          "Each ID is looked up in an embedding table to get a vector, and position information is added.",
          "The vectors pass through dozens of transformer layers. In each layer, attention mixes in information from earlier tokens, then an MLP transforms it.",
          "The final layer outputs a logit (a score) for every token in the vocabulary, often 100,000+ of them.",
          "Sampling turns the scores into probabilities and picks one token.",
          "That token is appended to the input and the loop runs again. Thanks to the KV cache, each new step only processes the newest token.",
        ],
        [
          "Generation is sequential, which is why output length drives latency.",
          "The model has no memory between requests: everything it knows about the conversation is in the input tokens.",
          "Generation stops at an end-of-sequence token, a stop string, or the max_tokens limit.",
        ]),

      I("c-llm", "hw-tokenize", "How does tokenization (BPE) work?",
        "Byte-pair encoding builds a vocabulary of frequent character sequences. Text is split into the longest known pieces, so common words are one token and rare words become several.",
        F([
          { label: "Building the tokenizer (once)", steps: ["Huge text corpus||store", "Start from bytes|single characters", "Merge the most frequent pair|e.g. t + h → th", "Repeat ~100k times|until vocab size reached", "Vocabulary|piece ↔ ID table|store"] },
          { label: "Using it", steps: ["\"unbelievable\"||io", "Split into known pieces|un · believ · able", "Token IDs|e.g. [348, 22016, 481]|io"] },
        ]),
        [
          "Training starts from raw bytes, so any text can be represented and there are no unknown words.",
          "It repeatedly merges the most frequent adjacent pair into a new token, building up common chunks like \"ing\" or \" the\".",
          "At run time, text is split into the vocabulary's pieces and each piece becomes an ID.",
          "Every model family has its own tokenizer, so the same text has different token counts on different models.",
        ],
        [
          "English averages about 4 characters per token; code, numbers and many other languages need more.",
          "The model sees pieces, not letters. That's why it can miscount the letters in a word.",
          "Count tokens with the provider's tokenizer or token-counting endpoint before sending large inputs.",
          "The example IDs above are illustrative; real IDs depend on the tokenizer.",
        ]),

      I("c-llm", "hw-attention", "How does attention work?",
        "Each token makes a query, a key and a value. It compares its query with earlier tokens' keys to decide how much to pay attention to each, then takes a weighted mix of their values.",
        F([{ steps: ["Token vectors||io", "Project|→ Q, K, V", "Scores|Q · Kᵀ / √d", "Causal mask|hide future tokens|check", "Softmax|scores → weights", "Weighted sum|weights × V|model", "Updated vectors||io"] }]),
        [
          "Each token vector is multiplied by three learned matrices, giving a query (what I'm looking for), a key (what I contain) and a value (what I pass on).",
          "Each token's query is compared with every token's key using a dot product, scaled by √d to keep values stable.",
          "A causal mask blocks attention to future tokens, so generation stays left to right.",
          "Softmax turns the scores into weights that add up to 1.",
          "The output for each token is the weighted sum of the values: information from relevant tokens is mixed in.",
          "Multi-head attention runs this many times in parallel with different projections, so different heads can track different relationships.",
        ],
        [
          "Prefill cost grows with the square of the sequence length, which is why very long prompts are expensive.",
          "Grouped-query attention (GQA) shares keys and values across heads, shrinking the KV cache.",
          "FlashAttention computes exactly the same result with far less memory traffic.",
        ],
        `import numpy as np

def attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    mask = np.triu(np.ones_like(scores), k=1) * -1e9   # causal mask
    w = np.exp(scores + mask)
    w = w / w.sum(axis=-1, keepdims=True)              # softmax
    return w @ V`),

      I("c-llm", "hw-kvcache", "How does the KV cache work?",
        "Keys and values for tokens already processed are stored, so each new token only computes its own and reads the rest from the cache.",
        F([
          { label: "Prefill: once per request", steps: ["Whole prompt||io", "Compute K, V for every token|in parallel|model", "KV cache|per layer, per token|store"] },
          { label: "Decode: every new token", steps: ["Newest token||io", "Compute its Q, K, V|1 token only|model", "Attend over the cache|read past K, V|store", "Next token||io"], loop: "Append this token's K, V to the cache and repeat" },
        ]),
        [
          "During prefill, the whole prompt is processed in parallel, and every token's keys and values are saved in GPU memory.",
          "During decode, each new token computes only its own query, key and value.",
          "Its query attends over the cached keys and values instead of recomputing the whole prefix.",
          "Its own key and value are appended, so the cache grows by one token per step.",
        ],
        [
          "Size ≈ 2 × layers × KV heads × head dim × bytes × tokens: about 0.3 MB per token for a 70B GQA model in FP16.",
          "The cache, not the weights, usually limits how many requests fit on a GPU at once.",
          "Prompt (prefix) caching reuses the cached keys and values of an identical prefix across requests.",
        ]),

      I("c-llm", "hw-sampling", "How does sampling (temperature, top-k, top-p) work?",
        "Scores are divided by the temperature, turned into probabilities, cut down to the most likely tokens, and then one token is drawn at random.",
        F([{ steps: ["Logits|raw scores", "÷ temperature|low = sharper|check", "Softmax|→ probabilities", "Top-k / top-p|keep likely tokens|check", "Random draw|weighted by probability|model", "Chosen token||io"] }]),
        [
          "Logits are raw scores for every token in the vocabulary.",
          "Dividing by a temperature below 1 exaggerates the differences (more focused); above 1 flattens them (more random). Temperature 0 means always taking the top token (greedy).",
          "Softmax converts the scores into probabilities that add up to 1.",
          "Top-k keeps only the k most likely tokens; top-p keeps the smallest set whose probabilities add up to p.",
          "One token is drawn at random from what's left, weighted by probability.",
        ],
        [
          "Extraction and classification: temperature around 0. Creative writing: around 0.7–1.",
          "Tune temperature or top-p, rarely both.",
          "Even greedy decoding can vary slightly between runs because of batching and floating-point order on the server.",
        ],
        `import numpy as np

def sample(logits, temperature=0.7, top_p=0.9):
    if temperature == 0:
        return int(np.argmax(logits))
    p = np.exp((logits - logits.max()) / temperature)
    p /= p.sum()
    order = np.argsort(-p)
    keep = order[: np.searchsorted(np.cumsum(p[order]), top_p) + 1]
    return int(np.random.choice(keep, p=p[keep] / p[keep].sum()))`),

      I("c-llm", "hw-training-overview", "How are LLMs trained, from pretraining to assistant?",
        "First the model learns language by predicting the next token on huge amounts of text; then it learns to follow instructions from examples; then it's tuned towards the answers people prefer.",
        F([{ steps: ["Web, books, code|trillions of tokens|store", "Pretraining|predict the next token|model", "Base model|knows language, not manners", "Supervised fine-tuning|instruction → good answer", "Preference tuning|RLHF / DPO|model", "Assistant model||io"] }]),
        [
          "Pretraining: the model reads trillions of tokens, and every wrong next-token prediction adjusts its billions of weights slightly. This is by far the most expensive stage.",
          "The result is a base model: excellent at continuing text, but not at following instructions.",
          "Supervised fine-tuning (SFT) trains it on curated instruction–response pairs, so it behaves like an assistant.",
          "Preference tuning: people (or AI) compare pairs of answers, and the model is trained to prefer the better ones (RLHF or DPO).",
          "Safety training, and reinforcement learning on checkable tasks such as maths and code, are added on top.",
        ],
        [
          "Most knowledge comes from pretraining, which is why models have a knowledge cutoff date.",
          "Fine-tuning shapes behaviour much more than it adds facts.",
          "See the Training & fine-tuning section for SFT, RLHF/DPO and LoRA in detail.",
        ]),

      I("c-llm", "hw-reasoning", "How do reasoning (thinking) models work?",
        "They're trained to write out intermediate reasoning before the final answer. Spending more tokens on thinking lets them solve harder multi-step problems.",
        F([{ steps: ["Question||io", "Thinking tokens|plan, try, check, revise|model", "Final answer||io"], loop: "A bigger thinking budget allows more attempts and self-checks" }]),
        [
          "The model first generates a chain of reasoning (often hidden or summarized): breaking the problem down, trying approaches, checking results.",
          "This behaviour is trained with reinforcement learning on tasks with verifiable answers, like maths and code, so reasoning that leads to correct answers gets rewarded.",
          "Only after thinking does it write the answer the user sees.",
          "Many APIs expose a thinking budget or effort level to trade quality against latency and cost.",
        ],
        [
          "Thinking tokens are usually billed as output tokens.",
          "The biggest gains are on maths, coding, planning and complex analysis; they're small on simple lookups.",
          "Prompting \"think step by step\" matters far less with these models.",
        ]),

      I("c-llm", "hw-structured", "How does structured output (JSON mode, constrained decoding) work?",
        "Either the model is trained to emit schema-shaped JSON (as in tool calling), or the server restricts which tokens may be chosen at each step so the output must match your schema.",
        F([{ steps: ["JSON schema|e.g. from Pydantic", "Grammar|what may come next", "Model logits||model", "Mask invalid tokens|only valid ones allowed|check", "Sample a valid token", "Valid JSON||io"], loop: "At every step, recompute which tokens keep the output valid" }]),
        [
          "Your schema (often generated from a Pydantic class) is turned into a grammar that describes every valid output.",
          "At each generation step, the server works out which tokens could legally come next.",
          "All other tokens get their probability set to zero before sampling.",
          "The output is therefore guaranteed to parse and match the schema's structure.",
        ],
        [
          "Provider structured-output modes and tool calling build on this; LangChain's with_structured_output chooses the right mechanism.",
          "Valid structure doesn't mean correct content. Still validate the values.",
          "Open-source servers implement it with libraries such as Outlines or XGrammar.",
        ]),

      // =====================================================================
      // RETRIEVAL & RAG
      // =====================================================================
      I("c-retrieval", "hw-embeddings", "How do embeddings and semantic search work?",
        "An embedding model turns text into a vector so that similar meanings land close together. Search embeds the query and returns the nearest document vectors.",
        F([
          { label: "Index (ahead of time)", steps: ["Documents||io", "Embedding model|text → vector|model", "Vectors|e.g. 1,536 numbers each|store"] },
          { label: "Search (per query)", steps: ["Query||io", "Same embedding model||model", "Compare|cosine similarity|check", "Closest documents||io"] },
        ]),
        [
          "An embedding model (a transformer trained for this purpose) reads a text and outputs a fixed-length vector.",
          "It's trained so related texts, such as a question and its answer, get similar vectors (contrastive training).",
          "Documents are embedded once and stored.",
          "A query is embedded with the same model, and the documents with the highest cosine similarity are returned.",
        ],
        [
          "Never mix vectors from different embedding models.",
          "Embeddings capture meaning but blur exact identifiers, which is why hybrid search exists.",
          "More dimensions can mean better accuracy but more storage; some models let you shorten vectors.",
        ],
        `import numpy as np

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))`),

      I("c-retrieval", "hw-hnsw", "How does a vector database find nearest neighbours fast (HNSW)?",
        "Instead of comparing the query with every vector, it walks a layered graph of neighbour links: long jumps on sparse upper layers, then finer steps on denser lower layers.",
        F([{ steps: ["Query vector||io", "Top layer|few nodes, long links", "Greedy hop|move to the closer neighbour|check", "Drop a layer|more nodes, shorter links", "Bottom layer|all vectors|store", "Top-k results||io"], loop: "Repeat the greedy search on each layer down to the bottom" }]),
        [
          "When a vector is inserted, it's linked to its nearest neighbours; a few vectors are also placed on higher, sparser layers.",
          "A search starts at an entry point on the top layer and keeps moving to whichever neighbour is closer to the query.",
          "When no neighbour is closer, it drops to the next layer and continues from that point.",
          "On the bottom layer, it explores a list of candidates (size ef) and returns the best k.",
        ],
        [
          "It's approximate: very fast, with recall usually 95–99%+, tunable through the M and ef parameters.",
          "Exact brute-force search is fine up to tens of thousands of vectors.",
          "Other index types: IVF (cluster first, then search nearby clusters), often combined with product quantization to compress vectors.",
        ]),

      I("c-retrieval", "hw-rag", "How does RAG work end to end?",
        "Documents are split, embedded and stored ahead of time. For each question, the most relevant chunks are retrieved and put into the prompt, so the model answers from them.",
        F([
          { label: "Indexing (ahead of time)", steps: ["Sources|PDF, wiki, tickets|io", "Load & clean", "Chunk|~500–1000 tokens", "Embed||model", "Vector store|+ metadata|store"] },
          { label: "Answering (per question)", steps: ["Question||io", "Rewrite|standalone query", "Retrieve|vector + keyword|store", "Rerank|keep top 5", "Prompt|context + question", "LLM|answer with citations|model", "Answer||io"] },
        ]),
        [
          "Indexing runs ahead of time: load the sources, clean them, split them into chunks, embed each chunk, and store the vectors with metadata (source, date, permissions).",
          "At question time, a chat follow-up is first rewritten into a standalone search query.",
          "Retrieval finds candidate chunks using vector and/or keyword search, filtered by the user's permissions.",
          "A reranker picks the most relevant few.",
          "The chunks and question are combined into a prompt that tells the model to answer only from the context and cite its sources.",
          "The model generates the answer, and the citations link back to the source documents.",
        ],
        [
          "Most quality problems are retrieval problems, so always log what was retrieved.",
          "Evaluate retrieval (was the right chunk found?) separately from generation (was the answer faithful?).",
          "Keep the index fresh with document IDs and content hashes.",
        ]),

      I("c-retrieval", "hw-hybrid", "How does hybrid search with reciprocal rank fusion (RRF) work?",
        "Keyword search and vector search run separately, and their two ranked lists are merged by rewarding documents that rank highly in either one.",
        F([
          { label: "Search 1: keywords", steps: ["Query||io", "BM25|exact terms, rare words count more|store", "Ranked list A"] },
          { label: "Search 2: meaning", steps: ["Query||io", "Vector search|semantic similarity|store", "Ranked list B"] },
          { label: "Merge", steps: ["Lists A + B", "RRF score|Σ 1 / (60 + rank)|check", "Merged top-k||io"] },
        ]),
        [
          "BM25 scores documents by matching terms, giving rare terms more weight and normalizing for document length.",
          "Vector search scores documents by semantic similarity.",
          "The two kinds of score can't be compared, so RRF ignores scores and uses ranks: each document gets 1 / (k + rank) from each list, with k usually 60.",
          "Documents found by both lists, or ranked near the top of either, rise to the top.",
        ],
        [
          "Hybrid search fixes the classic embedding weakness: product codes, names, error IDs.",
          "Some databases support hybrid search natively; otherwise run both searches and fuse in code.",
          "Weighted fusion of normalized scores is an alternative to RRF.",
        ],
        `def rrf(*ranked_lists, k=60):
    scores = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

rrf(["d3", "d1", "d7"], ["d1", "d9", "d3"])   # -> ['d1', 'd3', 'd9', 'd7']`),

      I("c-retrieval", "hw-rerank", "How does reranking work (bi-encoder vs cross-encoder)?",
        "Retrieval embeds the query and documents separately: fast but coarse. A reranker reads the query and each candidate together: slower but much more accurate, so it only runs on a short list.",
        F([
          { label: "Bi-encoder (retrieval): fast", steps: ["Query, document|each on its own|io", "Encoded separately||model", "Compare vectors|cosine|check", "Rough score"] },
          { label: "Cross-encoder (reranking): accurate", steps: ["Query + document|read together|io", "Transformer||model", "Relevance score|0–1|check"] },
          { label: "Pipeline", steps: ["Retrieve 50|cheap bi-encoder|store", "Rerank the 50|cross-encoder|model", "Keep top 5||io"] },
        ]),
        [
          "A bi-encoder embeds documents ahead of time, so search is just a vector comparison and scales to millions of documents.",
          "But the query and document never see each other, so subtle relevance can be missed.",
          "A cross-encoder takes the query and one document as a single input and outputs a relevance score, capturing fine-grained interactions.",
          "Because it must run once per candidate at query time, it's only used to rerank the top 20–100 results.",
        ],
        [
          "Typical effect: noticeably better precision in the top few results.",
          "Cost: tens to hundreds of milliseconds per query.",
          "An LLM can also act as a reranker, at a higher cost.",
        ]),

      I("c-retrieval", "hw-chunking", "How does recursive text splitting work?",
        "It splits on the biggest natural boundary first (paragraphs) and only falls back to smaller ones (lines, words, characters) for pieces that are still too big, then adds overlap.",
        F([{ steps: ["Document||io", "Split on blank lines|paragraphs", "Too big?|> chunk_size|check", "Split on line breaks", "Still too big?||check", "Split on spaces / characters", "Merge small pieces|up to chunk_size", "Add overlap|repeat last N chars", "Chunks||io"] }]),
        [
          "The splitter has an ordered list of separators: by default paragraph break, line break, space, then single characters.",
          "It splits on the first separator and checks the size of each piece.",
          "Pieces that are still too large are split again with the next separator.",
          "Small neighbouring pieces are merged back together up to chunk_size, so chunks aren't tiny.",
          "Each chunk repeats the end of the previous one (overlap), so an idea crossing a boundary isn't lost.",
        ],
        [
          "Sizes are in characters unless you use a token-based length function.",
          "Structure-aware splitters (markdown headings, code syntax) often do better.",
          "Store the chunk's position and parent document ID, for citations and for expanding context later.",
        ]),

      I("c-retrieval", "hw-conv-rag", "How does RAG handle follow-up questions in a chat?",
        "A follow-up like \"what about the second one?\" can't be searched as it is, so an LLM first rewrites it into a standalone question using the chat history, and that's what gets retrieved.",
        F([{ steps: ["Chat history + follow-up|\"what about the second one?\"|io", "Rewrite|→ standalone question|model", "Retrieve|with the rewritten query|store", "Answer|history + retrieved context|model", "Answer||io"] }]),
        [
          "The user's new message often only makes sense with the earlier turns.",
          "A small, fast model rewrites it into a self-contained question, using the history.",
          "Retrieval runs on the rewritten question.",
          "The final answer is generated from the retrieved context plus the conversation.",
        ],
        [
          "Skip the rewrite for the first message of a conversation to save latency.",
          "Log the rewritten query; it's a common hidden failure point.",
          "Multi-query (several rewrites, results merged) can improve recall further.",
        ]),

      I("c-retrieval", "hw-parent-doc", "How does parent-document (small-to-big) retrieval work?",
        "Search over small, precise chunks, but hand the model the larger section they came from, so you get precise matching and enough context.",
        F([
          { label: "Index", steps: ["Document||io", "Large parent sections|e.g. 2,000 tokens|store", "Small child chunks|e.g. 300 tokens, linked to parent", "Embed children||model"] },
          { label: "Query", steps: ["Question||io", "Search child chunks|precise match|store", "Look up their parents", "Send parents to the LLM||model"] },
        ]),
        [
          "Each document is split twice: into large parent sections and small child chunks, each child pointing to its parent.",
          "Only the small chunks are embedded, so similarity search matches precisely.",
          "When a child matches, the system retrieves its parent section instead.",
          "The model gets the full surrounding context, which improves answers that need more than one sentence.",
        ],
        [
          "De-duplicate parents when several children from the same section match.",
          "A variant is \"sentence window\": retrieve a sentence and include a few sentences around it.",
        ]),

      I("c-retrieval", "hw-text-to-sql", "How does text-to-SQL work?",
        "The model gets the database schema and the question, writes a SQL query, your code runs it safely, and the model turns the result rows into an answer.",
        F([{ steps: ["Question|\"revenue by region last quarter?\"|io", "Schema + examples|tables, columns, sample queries|store", "LLM writes SQL||model", "Validate|read-only, allowed tables, LIMIT|check", "Run on a read replica||store", "LLM explains result rows||model", "Answer + the SQL||io"], loop: "If the query errors, send the error back and let the model fix it" }]),
        [
          "The prompt includes the relevant tables and columns, their meaning, and a few example queries.",
          "The model writes a SQL query for the question.",
          "Code validates it: read-only, only allowed tables, a row limit, and a timeout.",
          "It runs against a read replica or a restricted user, never with write access.",
          "On an error, the error message goes back to the model to fix the query.",
          "The model summarizes the result, and showing the SQL lets users check it.",
        ],
        [
          "Use this instead of RAG when the answer needs aggregation (totals, counts, trends).",
          "For big schemas, retrieve only the relevant tables first.",
          "Never let the model's SQL run with permissions the user doesn't have.",
        ]),

      // =====================================================================
      // TOOLS & AGENTS
      // =====================================================================
      I("c-agents", "hw-toolcall", "How does tool (function) calling work?",
        "The model never runs code. It returns a structured request to call a tool; your app runs the tool and sends the result back for the model to use.",
        S(["Your app", "LLM", "Tool (your code)"], [
          [0, 1, "messages + tool schemas"],
          [1, 1, "decides a tool would help"],
          [1, 0, "tool_call: get_weather(city=\"Paris\")"],
          [0, 2, "run get_weather(\"Paris\")"],
          [2, 0, "\"18°C, sunny\""],
          [0, 1, "ToolMessage(result, tool_call_id)"],
          [1, 0, "\"It's 18°C and sunny in Paris.\""],
        ]),
        [
          "Your app sends the conversation plus the tool schemas: name, description and argument schema for each tool.",
          "The model decides a tool would help and replies with a structured tool call instead of text.",
          "Your code (or the framework) runs the matching function with those arguments.",
          "The result goes back to the model as a tool message, linked by the tool call ID.",
          "The model uses the result to write its answer, or asks for another tool.",
        ],
        [
          "The tool's name, description and argument schema are effectively part of the prompt: write them for the model.",
          "Models can request several tool calls in one turn (parallel tool calls).",
          "Validate arguments in code; never execute them blindly.",
        ]),

      I("c-agents", "hw-agentloop", "How does an agent loop work?",
        "An agent is a loop: the model decides whether to call a tool; if it does, the tool runs and the result goes back to the model; when it stops calling tools, its reply is the answer.",
        F([
          { steps: ["Task||io", "LLM decides|tool call or answer?|model", "Tool call?||check", "Run the tool|your code", "Add result to messages||store"], loop: "Repeat until the LLM answers without a tool call, or a step limit is hit" },
          { label: "When there's no tool call", steps: ["No tool call||check", "Final answer||io"] },
        ]),
        [
          "The agent starts with the task, the system prompt and the available tools.",
          "Each turn, the model either requests tool calls or writes a final answer.",
          "If it requests tools, they run and their results are appended to the conversation.",
          "The model is called again with the updated conversation.",
          "When it replies without a tool call, that reply is the final answer.",
        ],
        [
          "Always set a step limit (LangGraph's recursion_limit) plus timeouts and cost budgets.",
          "Require human approval for irreversible actions.",
          "Return tool errors as text so the model can recover instead of retrying blindly.",
        ]),

      I("c-agents", "hw-mcp", "How does MCP (Model Context Protocol) work?",
        "MCP standardizes how AI apps discover and call tools and data: an MCP server describes its tools, and any MCP client can list and call them.",
        S(["AI app (host)", "MCP client", "MCP server", "Your system"], [
          [1, 2, "initialize: exchange capabilities"],
          [1, 2, "tools/list"],
          [2, 1, "tools: name, description, input schema"],
          [0, 0, "LLM sees the tools, picks one"],
          [1, 2, "tools/call(name, arguments)"],
          [2, 3, "API or database call"],
          [3, 2, "data"],
          [2, 1, "result content"],
          [1, 0, "result → back to the LLM"],
        ]),
        [
          "The host app (an IDE assistant, a chat app, your agent) runs an MCP client for each server it connects to.",
          "On connect, client and server exchange capabilities, and the client lists the server's tools.",
          "The tool descriptions are handed to the LLM like any other tool schemas.",
          "When the LLM picks a tool, the client sends tools/call to the server.",
          "The server runs the real work (API, database, files) and returns the result.",
        ],
        [
          "Servers can also expose resources (readable data) and prompts (templates).",
          "Transports: stdio for local servers, HTTP for remote ones, with OAuth for authentication.",
          "Treat third-party servers as untrusted code: they can inject instructions or exfiltrate data.",
        ]),

      I("c-agents", "hw-supervisor", "How does a multi-agent supervisor work?",
        "A supervisor agent looks at the task and hands each piece to a specialist agent with its own tools and instructions, collecting results until the job is done.",
        F([
          { steps: ["User task||io", "Supervisor LLM|which specialist next?|model"] },
          { label: "Routes to one of", alt: true, steps: ["Research agent|search tools", "Coding agent|code tools", "Writer agent|drafting"] },
          { steps: ["Result back to supervisor||store", "Task complete?||check", "Final answer||io"], loop: "The supervisor keeps routing until the task is complete" },
        ], { chain: true }),
        [
          "The supervisor is an LLM whose tools are the other agents.",
          "It decides which specialist should handle the next part of the task.",
          "Each specialist runs its own loop with its own prompt and tools, and returns a result.",
          "The supervisor reviews the result and either routes again or produces the final answer.",
        ],
        [
          "Use it when tasks need clearly different skills or tools, or one agent's context would get too big.",
          "Costs: more tokens, latency and failure points, and harder debugging.",
          "Start with one agent and good tools; split only when evals show it helps.",
        ]),

      I("c-agents", "hw-agent-memory", "How does agent memory work (short-term vs long-term)?",
        "Short-term memory is the current conversation's state, saved per thread. Long-term memory is facts saved across conversations, searched and added to the prompt when relevant.",
        F([
          { label: "Short-term: one conversation", steps: ["New message||io", "Load thread state|by thread_id|store", "Agent runs||model", "Save checkpoint||store"] },
          { label: "Long-term: across conversations", steps: ["Something worth remembering|preference, fact", "Write memory|namespace: user|store", "A later conversation||io", "Search memories|by meaning|store", "Add to prompt||model"] },
        ]),
        [
          "Short-term memory is the message history and working state of one conversation, checkpointed per thread_id so it can resume.",
          "Long-term memory stores durable facts, such as preferences or past decisions, in a store organized by namespace (e.g. per user).",
          "Memories are written either by the agent as it works (a \"save memory\" tool) or by a background job afterwards.",
          "In a later conversation, relevant memories are searched and added to the prompt.",
        ],
        [
          "Trim or summarize short-term history to control token cost.",
          "Be deliberate about what's saved long-term, to avoid storing noise or sensitive data.",
          "Let users see and delete what's remembered about them.",
        ]),

      I("c-agents", "hw-hitl", "How does human-in-the-loop approval work (interrupt and resume)?",
        "The graph pauses at a node, saves its state, and returns the pending action to your app. When a person decides, the app resumes the same thread with their answer.",
        S(["User / UI", "Your app", "LangGraph", "Checkpointer"], [
          [1, 2, "invoke(input, thread_id)"],
          [2, 2, "node calls interrupt(draft)"],
          [2, 3, "save state"],
          [2, 1, "returns __interrupt__: draft"],
          [1, 0, "show draft, ask for approval"],
          [0, 1, "\"approve\""],
          [1, 2, "invoke(Command(resume=\"approve\"), thread_id)"],
          [3, 2, "load saved state"],
          [2, 1, "node continues → final result"],
        ]),
        [
          "Your app invokes the graph with a thread_id.",
          "A node calls interrupt() with the action that needs approval, e.g. an email draft.",
          "The state is saved by the checkpointer, and invoke returns the interrupt payload.",
          "Your UI shows it to a person, who approves, edits or rejects it.",
          "Your app invokes again with Command(resume=decision) and the same thread_id.",
          "The graph loads the saved state and continues; interrupt() returns the person's decision.",
        ],
        [
          "It requires a checkpointer, and the wait can last seconds or days.",
          "On resume, the node runs again from its start: keep side effects after the interrupt call.",
        ]),

      I("c-agents", "hw-delegated-auth", "How does an agent act on behalf of a user safely (delegated OAuth)?",
        "The user grants the app limited, scoped access through OAuth, and the agent calls APIs with that user's token, so it can never see or do more than the user could.",
        S(["User", "Your app", "Identity provider", "Agent", "Target API"], [
          [0, 1, "sign in"],
          [1, 2, "request consent for scopes"],
          [2, 0, "consent screen: \"read calendar\""],
          [2, 1, "access token (scoped, short-lived)"],
          [1, 3, "run task with the user's token"],
          [3, 4, "API call + token"],
          [4, 3, "only data this user may see"],
        ]),
        [
          "The user signs in and approves specific scopes, like reading their calendar.",
          "The identity provider issues a scoped, short-lived access token for that user.",
          "The agent runs with that token rather than a powerful service account.",
          "The target API enforces the user's own permissions on every call.",
        ],
        [
          "This contains the damage from prompt injection: the agent can't exceed the user's access.",
          "Request the narrowest scopes needed, and keep write scopes behind approval.",
          "Log every action with the user's identity for audit.",
        ]),
    ],
  };
})();

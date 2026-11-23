// AI Engineer interview question bank. Q(topic, level, question, whatTheyTest, answerPoints[], followUps[], code)
(function () {
  const M = "must", P = "maybe";
  const Q = (t, lvl, q, test, a, fu, code) => ({ t, lvl, q, test: test || "", a, fu: fu || [], code: code || "" });

  window.QBANK = {
    id: "ai-engineer",
    rounds: [
      { name: "Recruiter / hiring manager screen", len: "30 min", what: "Your background, why AI, and a walkthrough of one project.", tip: "Have a 2-minute story of your best AI system, with numbers." },
      { name: "Coding", len: "45–60 min", what: "Practical Python: data handling, async calls, error handling. Sometimes LLM-flavoured: a tool loop, a chunker, retrieval.", tip: "Talk while you code, and handle errors and edge cases out loud." },
      { name: "LLM fundamentals deep dive", len: "45 min", what: "Tokens, context, sampling, embeddings, RAG, agents, evals, cost.", tip: "Answer with trade-offs, not definitions." },
      { name: "LLM system design", len: "60 min", what: "Design a RAG assistant or an agent end to end.", tip: "Requirements → architecture → evals → cost → failure modes → security." },
      { name: "Take-home or live build", len: "2–4 h", what: "Build a small RAG app or agent, often with evals.", tip: "A clear README with your decisions, eval results and next steps matters more than extra features." },
      { name: "Behavioral / leadership", len: "45 min", what: "Ownership, ambiguity, conflict. At Staff level: setting direction and influence.", tip: "Use STAR, and always end with a measured result." },
    ],
    topics: [
      { id: "t-fund", title: "LLM fundamentals", blurb: "The basics every AI engineer is expected to explain clearly and quickly." },
      { id: "t-prompt", title: "Prompting & structured output", blurb: "Getting reliable, parseable, safe behaviour from a model." },
      { id: "t-rag", title: "RAG", blurb: "The most common system-design topic for AI engineers." },
      { id: "t-agents", title: "Agents & tools", blurb: "When to use agents, and how to keep them safe and bounded." },
      { id: "t-evals", title: "Evals & quality", blurb: "How you prove it works. Senior candidates are judged heavily on this." },
      { id: "t-prod", title: "Production: cost, latency, reliability", blurb: "Where your backend experience shows." },
      { id: "t-ft", title: "Fine-tuning & model choice", blurb: "Knowing when not to fine-tune is half the answer." },
      { id: "t-design", title: "System design", blurb: "Structured outlines for the design round. Practise saying them out loud in 30–40 minutes." },
      { id: "t-coding", title: "Practical coding", blurb: "Short exercises that show you understand what frameworks do for you." },
      { id: "t-behav", title: "Behavioral & leadership", blurb: "Prepare a story for each. Every story ends with a number." },
    ],
    questions: [
      // ---------- Fundamentals ----------
      Q("t-fund", M, "What is a token, and why does it matter for cost and limits?", "Whether you understand how pricing, context limits and latency work.", [
        "A token is a chunk of text the model reads and writes: on average about 3–4 characters, or roughly ¾ of an English word. Code and many other languages use more tokens per word.",
        "Pricing is per input token and per output token. Output tokens usually cost several times more than input tokens.",
        "Context limits are counted in tokens and include everything: system prompt, history, retrieved documents and the answer.",
        "Latency grows with output length, because the model generates one token at a time.",
      ], ["How would you estimate the monthly cost of a feature?", "How do you count tokens before sending a request?"]),
      Q("t-fund", M, "What is the context window, and what happens when you exceed it?", "", [
        "It's the maximum number of tokens the model can consider in one request: input plus output.",
        "If you exceed it, the API returns an error, or a framework silently truncates, usually dropping the oldest history.",
        "Even within the limit, long contexts cost more and add latency, and models can miss facts buried in the middle (\"lost in the middle\").",
        "Strategies: trim or summarize history, retrieve only relevant chunks, put key instructions at the start or end.",
      ]),
      Q("t-fund", M, "Explain temperature and top-p. What would you set for extraction versus creative writing?", "", [
        "The model outputs a probability distribution over the next token, and sampling picks from it.",
        "Temperature rescales that distribution: low values make the top choices dominate (focused), high values flatten it (varied).",
        "Top-p (nucleus sampling) samples only from the smallest set of tokens whose probabilities add up to p.",
        "Extraction and classification: temperature 0–0.2. Creative writing: around 0.7–1. Usually tune one of the two, not both.",
        "Even at temperature 0, outputs can vary slightly between runs because of batching and floating-point non-determinism on the server.",
      ]),
      Q("t-fund", M, "Why do LLMs hallucinate, and how do you reduce it?", "Practical mitigation, not just a definition.", [
        "They're trained to produce plausible text, not verified facts. When knowledge is missing or the prompt is ambiguous, plausible but wrong text comes out.",
        "Ground answers in retrieved sources (RAG), and instruct the model to answer only from the context and to say \"I don't know\" otherwise.",
        "Ask for citations and check them; use structured output so claims are easy to verify.",
        "Use low temperature for factual tasks, and measure the hallucination rate with a faithfulness eval.",
        "For high-stakes outputs, add a verification step or human review.",
      ], ["How would you detect hallucinations automatically?"]),
      Q("t-fund", M, "What are embeddings, and how is similarity measured?", "", [
        "An embedding is a vector that represents the meaning of a text, so texts with similar meaning end up close together.",
        "Similarity is usually cosine similarity, the angle between vectors. With normalized vectors, the dot product gives the same ranking.",
        "Used for semantic search, RAG, clustering, deduplication and classification.",
        "You must use the same embedding model for indexing and for queries; vectors from different models aren't comparable.",
      ], [], `import numpy as np

def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))`),
      Q("t-fund", M, "How does tool (function) calling actually work under the hood?", "That you know the model never executes anything.", [
        "You send tool schemas (name, description, a JSON schema for the arguments) with the request.",
        "The model is trained to output a structured tool call when a tool fits, and the API returns that call instead of text.",
        "Your code runs the function and sends the result back as a tool message with the matching call id.",
        "The model then continues: it either calls another tool or writes the final answer.",
      ]),
      Q("t-fund", P, "Explain the transformer and attention at a high level.", "Conceptual grounding, not maths.", [
        "Text becomes token embeddings. In each layer, every token \"looks at\" other tokens through attention and mixes in relevant information.",
        "Attention scores how relevant each earlier token is (query–key match) and takes a weighted sum of their values.",
        "Many stacked layers and attention heads build up meaning, and the last layer predicts the next token.",
        "Decoder-only models such as GPT, Claude and Llama use causal attention: each token can only see earlier tokens.",
      ]),
      Q("t-fund", P, "What is the KV cache, and why does it matter for latency and cost?", "", [
        "During generation, the keys and values for past tokens are stored, so each new token doesn't recompute attention over the whole prefix.",
        "It makes decoding much faster, but it uses GPU memory that grows with sequence length and batch size.",
        "That memory limit is why long contexts reduce how many requests a server can batch.",
        "Provider prompt caching reuses the work for a repeated prompt prefix across requests, which cuts cost and time to first token.",
      ]),
      Q("t-fund", P, "What are reasoning (thinking) models, and when would you use one?", "", [
        "Models that generate internal reasoning before answering, trading extra tokens and latency for better results on multi-step problems.",
        "Good for complex analysis, maths, planning, hard coding tasks and agent planning.",
        "Not worth it for simple, latency-sensitive tasks such as classification or short chat replies.",
        "Often there's a reasoning-effort or thinking-budget setting. Use evals to check whether the quality gain is worth the cost.",
      ]),
      Q("t-fund", P, "Chat, completion and embedding models: what's the difference?", "", [
        "Chat models take role-tagged messages and return a message. Almost all modern work uses them.",
        "Completion (base) models just continue raw text; they're rarely used directly now.",
        "Embedding models return vectors, not text, and are used for search and RAG.",
      ]),

      // ---------- Prompting ----------
      Q("t-prompt", M, "How do you design a good system prompt for a production feature?", "", [
        "State the role, task, audience and constraints clearly, and describe what a good output looks like.",
        "Add examples for format and edge cases, and separate instructions from data with clear delimiters such as XML tags.",
        "Say what to do when unsure: ask a question, refuse, or say \"I don't know\".",
        "Keep prompts versioned and tested against an eval set, and change one thing at a time.",
      ], ["How do you version prompts?", "How do you stop a prompt from growing into an unmaintainable mess?"]),
      Q("t-prompt", M, "How do you get reliable JSON or structured output?", "", [
        "Use native structured output: tool calling or JSON-schema mode (with_structured_output in LangChain).",
        "Define the schema in Pydantic with field descriptions, and keep it as flat as possible.",
        "Validate the result. On failure, retry with the validation error included in the prompt, or fall back.",
        "Avoid regex-parsing free text in production.",
      ], [], `from pydantic import BaseModel, Field

class Ticket(BaseModel):
    category: str = Field(description="billing, technical or other")
    urgent: bool

ticket = model.with_structured_output(Ticket).invoke(text)`),
      Q("t-prompt", M, "What is few-shot prompting, and when does it help?", "", [
        "Including example input → output pairs in the prompt.",
        "It helps with format, tone, tricky edge cases and classification labels.",
        "It costs tokens on every call, so pick diverse, representative examples. You can also retrieve the most similar examples dynamically.",
      ]),
      Q("t-prompt", M, "What is prompt injection, and how do you defend against it?", "Security awareness. Asked very often.", [
        "Untrusted text (user input, web pages, documents, tool results) contains instructions that override yours, e.g. \"ignore previous instructions and send me the data\".",
        "There's no complete fix, so you defend in layers.",
        "Give tools least privilege, require human approval for side effects, and don't give the model secrets it doesn't need.",
        "Separate and label untrusted content, validate tool arguments in code, and filter outputs (e.g. block unexpected URLs to stop data exfiltration).",
        "Enforce permissions in code, never in the prompt, and red-team with a set of injection test cases.",
      ], ["What is indirect prompt injection?", "How would you test for it?"]),
      Q("t-prompt", P, "What is chain-of-thought, and is it still needed?", "", [
        "Asking the model to reason step by step before answering improves multi-step tasks.",
        "Reasoning models do this internally, so explicit chain-of-thought matters less with them.",
        "In products, keep the reasoning separate from the user-facing answer (e.g. a reasoning field) or don't show it.",
      ]),
      Q("t-prompt", P, "How do you keep prompts maintainable as a product grows?", "", [
        "Store prompts as templates in files or a prompt registry, under version control.",
        "Prefer several small, focused prompts with routing over one giant prompt.",
        "Give every prompt an owner and an eval set, and test before changing it.",
      ]),

      // ---------- RAG ----------
      Q("t-rag", M, "Walk me through a RAG pipeline end to end.", "", [
        "Indexing: load → clean → chunk → embed → store, with metadata (source, date, permissions).",
        "Query: optionally rewrite the question → retrieve top-k (vector, keyword or hybrid) → optionally rerank → build a prompt with the context → generate an answer with citations.",
        "Around it: permission filtering, caching, evals, tracing, and re-indexing when documents change.",
      ]),
      Q("t-rag", M, "How do you choose chunk size and overlap?", "", [
        "It's a trade-off: small chunks match precisely but lose context; large chunks keep context but dilute the embedding and use more tokens.",
        "Start at roughly 500–1000 tokens or characters depending on content, with 10–20% overlap, and split on structure (headings, paragraphs).",
        "Tune with retrieval evals (recall@k on real questions), not by intuition.",
      ]),
      Q("t-rag", M, "Your RAG app gives wrong answers. How do you debug it?", "A systematic method. One of the most common questions.", [
        "Collect failing examples, and trace each one: was the right chunk retrieved?",
        "If not, it's a retrieval problem: chunking, embedding model, k, hybrid search, query rewriting, metadata filters or reranking.",
        "If yes, it's a generation problem: the prompt, context order or size, model choice, or instructions to cite and abstain.",
        "Fix it, then re-run the eval set to confirm the fix and check for regressions.",
      ]),
      Q("t-rag", M, "What is hybrid search, and why use it?", "", [
        "It combines keyword search (BM25) with vector search and merges the results, e.g. with reciprocal rank fusion.",
        "Vectors catch meaning; keywords catch exact names, product codes, IDs and rare terms that embeddings blur.",
        "On real enterprise data it usually improves recall noticeably.",
      ]),
      Q("t-rag", M, "What is reranking?", "", [
        "Retrieve a larger candidate set cheaply (e.g. 50), then score each candidate against the query with a cross-encoder or reranker model and keep the top few.",
        "Cross-encoders read the query and document together, so they're more accurate than embedding similarity, but slower.",
        "It improves precision at the cost of extra latency and cost per query.",
      ]),
      Q("t-rag", M, "How do you make sure users only see documents they're allowed to see?", "", [
        "Store access metadata with each chunk: tenant, groups, document ACLs.",
        "Filter at retrieval time in the vector-store query, based on the authenticated user. Never rely on the prompt for this.",
        "Keep ACLs in sync when permissions change. For strict isolation, use a separate index per tenant.",
        "Include users with different roles in your eval set.",
      ]),
      Q("t-rag", M, "How do you keep the index fresh when documents change?", "", [
        "Give each document a stable ID and a content hash. On change, delete its old chunks by ID and re-embed.",
        "Use event-driven ingestion (webhooks, change data capture) or scheduled syncs, with a queue and idempotent workers.",
        "Track an index version, and re-embed everything when you change the embedding model.",
      ]),
      Q("t-rag", P, "What are query rewriting, multi-query and HyDE?", "", [
        "Query rewriting: an LLM turns a vague or follow-up question into a standalone search query. This is essential in chat, where \"what about the second one?\" means nothing on its own.",
        "Multi-query: generate several versions of the query, retrieve for each and merge the results, which improves recall.",
        "HyDE: generate a hypothetical answer and embed that, because answers look more like documents than questions do.",
      ]),
      Q("t-rag", P, "When is RAG not the right approach?", "", [
        "When the answer needs aggregation over all the data (counts, totals, trends): use SQL or text-to-SQL.",
        "When the knowledge is small and stable: put it straight into the prompt, with prompt caching.",
        "When you need a change in behaviour or style rather than knowledge: improve prompts or fine-tune.",
      ]),
      Q("t-rag", P, "Context windows are huge now. Is RAG still needed?", "", [
        "Usually yes: cost and latency grow with context size, attention to detail drops in very long contexts, and permissions and freshness are easier with retrieval.",
        "For a small corpus, putting everything in the prompt with caching can be simpler. Measure both on your eval set.",
      ]),
      Q("t-rag", P, "How do you handle tables, PDFs and images in RAG?", "", [
        "Use layout-aware parsing. Keep tables as structured text (markdown or HTML) or summarize them, and OCR scanned pages.",
        "Store captions or summaries of images, or use multimodal models.",
        "Keep page numbers in the metadata so you can cite them.",
      ]),

      // ---------- Agents ----------
      Q("t-agents", M, "What is an agent, and how is it different from a chain or a workflow?", "", [
        "A workflow follows a path you wrote. An agent lets the LLM decide the next step (which tool, when to stop) in a loop.",
        "Agents handle open-ended tasks, but they're less predictable, slower and more expensive.",
        "Prefer a workflow when the steps are known and use an agent where flexibility is needed. Real systems are often a mix.",
      ]),
      Q("t-agents", M, "How do you stop an agent from looping forever or doing something harmful?", "", [
        "Set a maximum number of steps (recursion limit), timeouts, and token or cost budgets.",
        "Provide a few well-scoped tools with clear descriptions, and return informative errors.",
        "Require human approval for irreversible actions, give tools least-privilege permissions, and keep audit logs.",
        "Watch traces for loops and recurring failure patterns.",
      ]),
      Q("t-agents", M, "How would you design the tools for an agent?", "", [
        "Use a few well-named tools with clear docstrings, written for the model.",
        "Use strictly typed arguments, validate them, and return concise, structured results rather than huge payloads.",
        "Make actions idempotent where possible, and return errors as text the model can act on.",
        "Separate read tools from write tools, and gate the write tools.",
      ]),
      Q("t-agents", M, "How does memory work for agents, short-term and long-term?", "", [
        "Short-term memory is the state of one conversation: the message history, checkpointed per thread_id.",
        "Long-term memory is facts that persist across sessions (preferences, past decisions), stored in a database or vector store and retrieved when relevant.",
        "Manage the size: trim, summarize, and decide explicitly what gets written to long-term memory.",
      ]),
      Q("t-agents", P, "Single agent or multi-agent: when would you use several agents?", "", [
        "Use several agents when tasks need different skills, tools or prompts, or when one agent's context would get too large. For example, a supervisor routing to research, coding and review agents.",
        "The costs are more tokens, more latency, more failure points and harder debugging.",
        "Start with one agent and good tools, and split only when evals show it helps.",
      ]),
      Q("t-agents", P, "What is MCP (Model Context Protocol)?", "", [
        "An open protocol that connects AI applications to tools and data sources through a standard client–server interface.",
        "You write a tool server once and use it from any MCP-compatible app or agent.",
        "Things to watch: authentication, permission scoping, and how far you trust third-party servers.",
      ]),
      Q("t-agents", P, "What is ReAct?", "", [
        "A pattern where the model alternates reasoning and acting: think, call a tool, observe the result, repeat.",
        "Native tool calling implements the same loop today, but the term still comes up in interviews.",
      ]),

      // ---------- Evals ----------
      Q("t-evals", M, "How do you evaluate an LLM application?", "Whether you treat quality as engineering. Key for senior roles.", [
        "Build a dataset of realistic inputs (from logs, users and edge cases) with expected outputs or grading criteria.",
        "Score with a mix: exact or programmatic checks, LLM-as-judge with a rubric, and human review of samples.",
        "For RAG, evaluate retrieval (recall@k, hit rate) and generation (faithfulness, relevance, correctness) separately.",
        "Run it on every prompt, model or config change in CI, track results over time, and monitor production with online metrics and user feedback.",
      ], ["How big should the eval set be?", "How do you know you can trust an LLM judge?"]),
      Q("t-evals", M, "What is LLM-as-a-judge, and what are its pitfalls?", "", [
        "Using a model to grade outputs against a rubric or a reference answer.",
        "Pitfalls: bias towards the first option or longer answers, preference for its own style, and inconsistency.",
        "Mitigations: a clear rubric with examples, pairwise comparisons in both orders, a strong judge model, narrow yes/no criteria, and calibration against human labels.",
      ]),
      Q("t-evals", M, "How do you know a prompt change made things better and didn't break other cases?", "", [
        "Run the regression eval set before and after, and compare case by case, not only the averages.",
        "Look at the cases the change broke, not just the ones it fixed.",
        "For important changes, A/B test or shadow in production while watching guardrail metrics.",
      ]),
      Q("t-evals", P, "Which metrics would you track in production?", "", [
        "Quality: user feedback, task success, escalation rate, sampled judge scores.",
        "Operations: latency (time to first token, total), error rate, tokens and cost per request, cache hit rate.",
        "Safety: refusal rate, flagged outputs, detected injection attempts.",
      ]),

      // ---------- Production ----------
      Q("t-prod", M, "How do you reduce LLM cost?", "", [
        "Use the smallest model that passes your evals, and route easy requests to cheaper models.",
        "Use prompt caching for repeated prefixes, and response or semantic caching for repeated questions.",
        "Cut tokens: shorter prompts, fewer or shorter retrieved chunks, limits on output length.",
        "Use batch APIs for offline work, and track cost per feature and per user to find hotspots.",
      ]),
      Q("t-prod", M, "How do you reduce latency?", "", [
        "Stream tokens to improve perceived latency (time to first token).",
        "Use smaller or faster models, cut sequential LLM calls, and run independent calls in parallel.",
        "Use prompt caching, shorter outputs and cached retrieval results, and keep connections warm.",
        "Measure p50 and p95 per step with tracing to find the real bottleneck.",
      ]),
      Q("t-prod", M, "How do you handle rate limits and provider outages?", "", [
        "Retry with exponential backoff and jitter on 429 and 5xx errors, and respect retry-after headers.",
        "Limit concurrency with queues, fall back to another model or provider, and use a circuit breaker.",
        "Degrade gracefully in the UI, and use idempotency keys for actions.",
      ], [], `import asyncio, random

async def call_with_retry(fn, attempts=5, base=0.5):
    for i in range(attempts):
        try:
            return await fn()
        except RateLimitError:
            if i == attempts - 1:
                raise
            await asyncio.sleep(base * 2 ** i + random.random() * base)`),
      Q("t-prod", M, "How do you stream LLM responses to a web client?", "", [
        "The server calls the model with streaming and forwards chunks using Server-Sent Events (or WebSockets).",
        "Handle client disconnects (cancel the upstream call), timeouts and errors mid-stream.",
        "Assemble the full response on the server for logging and evals.",
      ], [], `from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/chat")
async def chat(q: str):
    async def events():
        async for chunk in chain.astream({"question": q}):
            yield f"data: {chunk}\\n\\n"
    return StreamingResponse(events(), media_type="text/event-stream")`),
      Q("t-prod", P, "How do you handle PII and sensitive data with LLM APIs?", "", [
        "Know the provider's data retention and training policies, and use zero-retention or enterprise agreements where needed.",
        "Redact or tokenize PII before sending when possible, and restore it afterwards.",
        "Apply access controls, encryption and audit logging. Don't log raw prompts that contain PII, and respect data-residency rules.",
      ]),
      Q("t-prod", P, "How do you version and deploy prompt and model changes safely?", "", [
        "Keep prompts in version control or a registry, and reference them by version.",
        "Gate changes with evals in CI, roll out to a percentage of traffic, monitor, and keep rollback quick.",
        "Pin model versions, and re-run evals before upgrading.",
      ]),

      // ---------- Fine-tuning ----------
      Q("t-ft", M, "When would you fine-tune instead of prompting or using RAG?", "", [
        "Prompting first. RAG for knowledge. Fine-tuning for behaviour: a consistent format or style, a domain tone, or a narrow task run cheaper and faster on a smaller model.",
        "Not for facts that change often; that's what RAG is for.",
        "It needs good labeled data (hundreds to thousands of examples), an eval set, and upkeep when base models change.",
      ]),
      Q("t-ft", M, "How do you choose a model for a feature?", "", [
        "Set quality, latency and cost targets, and build an eval set.",
        "Test the candidates on it: a frontier API model, smaller API models, open-weight models.",
        "Also weigh context size, tool-calling quality, data and privacy constraints, availability and price.",
        "Pick the cheapest model that meets the bar, and re-evaluate as new models come out.",
      ]),
      Q("t-ft", P, "What are LoRA and QLoRA?", "", [
        "LoRA trains small low-rank adapter matrices instead of all the weights, using far less memory and storage.",
        "QLoRA does the same on a 4-bit quantized base model, so large models can be fine-tuned on a single GPU.",
        "Adapters can be swapped at serving time, and multi-LoRA serving runs many adapters on one base model.",
      ]),
      Q("t-ft", P, "Hosted API or self-hosted open-weight model?", "", [
        "API: best quality, no ops, fast to start. You pay per token, and data leaves your network (under the agreement terms).",
        "Self-hosted: full control, data stays inside, cheaper at high steady volume. It needs GPU operations skills, and quality is often lower.",
        "Decide on quality evals, volume, latency and compliance. Many teams use both behind a gateway.",
      ]),

      // ---------- System design ----------
      Q("t-design", M, "Design a customer support assistant over the help centre and past tickets.", "End-to-end thinking, from requirements to operations.", [
        "Clarify: users, channels, languages, volume, latency target, hand-off to humans, and whether it only answers or also takes actions.",
        "Ingestion: help articles and resolved tickets → clean, de-duplicate, chunk, embed, with metadata (product, date, permissions), on a scheduled sync.",
        "Serving: API → rewrite the query using the conversation → hybrid retrieval and reranking → answer with citations → escalate when confidence is low.",
        "Actions: start with read-only tools (order status); put refunds behind human approval.",
        "Quality: an eval set from real tickets; track resolution and escalation rates; feedback buttons.",
        "Operations: tracing, cost per conversation, caching common questions, a fallback model, PII redaction.",
      ], ["How do you measure whether it actually helps?", "How would you add a refund action safely?"]),
      Q("t-design", M, "Design document Q&A over 1M internal documents with per-user permissions.", "", [
        "Ingestion: connectors → queue → workers that parse and OCR → chunk → embed in batches → vector DB with ACL metadata. Idempotent by document ID and content hash.",
        "Retrieval: authenticate the user → resolve their groups → filtered hybrid search → rerank → top 5–8 chunks.",
        "Generation: a grounded prompt, citations with links, and refusing when the context doesn't support an answer.",
        "Scale: shard the index, use async workers, plan embedding throughput and a re-indexing strategy.",
        "Run evals per department, monitor for stale ACLs, and keep an audit log of who saw what.",
      ]),
      Q("t-design", M, "Design an agent that takes actions for users, e.g. rescheduling appointments.", "", [
        "Tools with strict schemas that wrap existing APIs, using the user's own auth token, not a super-user account.",
        "A LangGraph-style loop with state, checkpointing and a step limit.",
        "A confirmation step before any write: interrupt, show the proposed change, wait for approval.",
        "An audit trail, idempotency keys, and undo where possible.",
        "Evals with scripted scenarios, including adversarial and injection cases.",
      ]),
      Q("t-design", P, "Design an LLM gateway for several product teams.", "", [
        "A single API in front of all providers: per-team auth, routing by model alias, fallbacks and retries.",
        "Rate limits and budgets per team, and usage metering (tokens, cost) into a warehouse.",
        "Caching, PII-redaction hooks, logging and tracing, and prompt and response policies.",
        "Model upgrades happen centrally, behind eval gates.",
      ]),
      Q("t-design", P, "Design a system that summarizes 10,000 call transcripts per day.", "", [
        "A batch pipeline: queue the transcripts, split long calls, and use map-reduce summarization (summarize parts, then combine).",
        "Use batch APIs or cheaper models, with structured output (topics, sentiment, action items).",
        "Store the results, evaluate a daily sample, and use retries and a dead-letter queue.",
      ]),

      // ---------- Coding ----------
      Q("t-coding", M, "Implement a simple tool-calling loop without a framework.", "That you understand what frameworks do for you.", [
        "Send the messages and tool schemas. If the reply contains tool calls, run each one, append the results as tool messages, and repeat.",
        "Stop when there are no tool calls, or when you hit the maximum number of steps.",
      ], [], `def run_agent(messages, tools, max_steps=8):
    for _ in range(max_steps):
        reply = llm(messages, tools=tools)
        messages.append(reply)
        if not reply.tool_calls:
            return reply.text
        for call in reply.tool_calls:
            try:
                result = TOOLS[call.name](**call.args)
            except Exception as e:
                result = f"Error: {e}"      # let the model see and recover
            messages.append(tool_result(call.id, result))
    raise RuntimeError("step limit reached")`),
      Q("t-coding", M, "Write a function that splits text into overlapping chunks.", "", [
        "Mention that real splitters prefer natural boundaries (paragraphs, sentences) and measure tokens, not characters.",
      ], [], `def chunk(text, size=1000, overlap=200):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks`),
      Q("t-coding", P, "Call an LLM for 1,000 inputs concurrently without hitting rate limits.", "", [
        "Cap concurrency with a semaphore, retry with backoff, and keep results in input order (gather does this).",
        "Handle errors per item so one failure doesn't lose the whole batch.",
      ], [], `import asyncio

async def run_all(items, limit=10):
    sem = asyncio.Semaphore(limit)

    async def one(x):
        async with sem:
            return await call_with_retry(lambda: llm.ainvoke(x))

    return await asyncio.gather(*(one(x) for x in items), return_exceptions=True)`),
      Q("t-coding", P, "Implement top-k retrieval by cosine similarity over an in-memory list.", "", [
        "Normalize, compute similarities with one matrix multiplication, and take the top k indices.",
      ], [], `import numpy as np

def top_k(query_vec, doc_vecs, k=4):
    D = np.asarray(doc_vecs)
    q = np.asarray(query_vec)
    sims = D @ q / (np.linalg.norm(D, axis=1) * np.linalg.norm(q))
    return np.argsort(-sims)[:k]`),

      // ---------- Behavioral ----------
      Q("t-behav", M, "Tell me about an AI system you built and shipped.", "Ownership, depth and impact.", [
        "Use STAR (situation, task, action, result) and keep it to 2–3 minutes.",
        "Cover the problem and users, the architecture, one hard trade-off you made, how you evaluated it, and the measured result (accuracy, latency, cost, business metric).",
        "Be ready for deep dives into any component you mention.",
      ]),
      Q("t-behav", M, "Tell me about a time an AI feature failed or behaved badly in production.", "", [
        "Show calm debugging: how you detected it (monitoring or feedback), triaged with traces, fixed the root cause, and added an eval case so it can't regress.",
        "Mention what you changed in the process, not only in the code.",
      ]),
      Q("t-behav", M, "How do you explain LLM limitations to product managers or stakeholders?", "", [
        "Set expectations with measured accuracy on realistic cases, not with demos.",
        "Show failure examples, and agree on acceptable error rates and fallbacks (human review, confidence thresholds).",
        "Frame decisions as quality vs cost vs latency, with numbers.",
      ]),
      Q("t-behav", P, "(Staff) How have you set technical direction for AI work across teams?", "", [
        "Give an example of a shared component or standard (gateway, eval framework, RAG service) that you proposed in a design doc, got buy-in for, and drove adoption of.",
        "Show influence without authority: pilots, data, documentation, office hours.",
        "End with the measured outcome: time saved, quality improved, cost reduced.",
      ]),
      Q("t-behav", P, "How do you keep up with how fast AI changes?", "", [
        "Filter instead of following everything: release notes for the models and tools you use, a few trusted sources, papers when they're relevant to your work.",
        "Test new things against your own eval set before adopting them.",
      ]),
    ],
  };
})();

// Page info for each "How it works" explanation: its own page's file name, a short title
// for the left index, and 3-4 related explanations (ids without the "hw-" prefix).
// Used by the cheat sheet (how-it-works.html) and by gen-how-pages.js.
// After editing, regenerate: node concepts/gen-how-pages.js
(function (root) {
  const META = {};
  const M = (id, slug, short, related) => { META["hw-" + id] = { slug, short, related: related.split(" ").map((r) => "hw-" + r) }; };

  // Inside the LLM
  M("generate", "text-generation", "Text generation", "tokenize kvcache sampling inference-server");
  M("tokenize", "tokenization", "Tokenization (BPE)", "generate embeddings chunking prompt-caching");
  M("attention", "attention", "Attention", "kvcache generate paged-attention tensor-parallel");
  M("kvcache", "kv-cache", "KV cache", "attention paged-attention prompt-caching cont-batching");
  M("sampling", "sampling", "Sampling", "generate structured spec-decoding llm-judge");
  M("training-overview", "training-pipeline", "Training pipeline", "sft rlhf-dpo distributed-training reasoning");
  M("reasoning", "reasoning-models", "Reasoning models", "training-overview rlhf-dpo agentloop sampling");
  M("structured", "structured-output", "Structured output", "sampling toolcall lcel guardrails");

  // Retrieval & RAG
  M("embeddings", "embeddings", "Embeddings & semantic search", "hnsw rag hybrid rerank");
  M("hnsw", "hnsw-vector-search", "HNSW vector search", "embeddings hybrid rag semantic-cache");
  M("rag", "rag", "RAG end to end", "chunking hybrid rerank rag-eval");
  M("hybrid", "hybrid-search-rrf", "Hybrid search + RRF", "embeddings rerank hnsw rag");
  M("rerank", "reranking", "Reranking", "hybrid rag embeddings rag-eval");
  M("chunking", "chunking", "Recursive chunking", "parent-doc rag tokenize embeddings");
  M("conv-rag", "conversational-rag", "RAG in a chat", "rag agent-memory rag-chain checkpoint");
  M("parent-doc", "parent-document-retrieval", "Parent-document retrieval", "chunking rag rerank");
  M("text-to-sql", "text-to-sql", "Text-to-SQL", "toolcall structured guardrails rag");

  // Tools & agents
  M("toolcall", "tool-calling", "Tool calling", "agentloop mcp structured delegated-auth");
  M("agentloop", "agent-loop", "Agent loop", "toolcall langgraph-run hitl supervisor");
  M("mcp", "mcp", "MCP", "toolcall delegated-auth agentloop prompt-injection");
  M("supervisor", "multi-agent-supervisor", "Multi-agent supervisor", "agentloop langgraph-run reducers agent-memory");
  M("agent-memory", "agent-memory", "Agent memory", "checkpoint conv-rag embeddings reducers");
  M("hitl", "human-in-the-loop", "Human-in-the-loop", "checkpoint agentloop langgraph-run guardrails");
  M("delegated-auth", "delegated-oauth", "Delegated OAuth", "mcp toolcall prompt-injection guardrails");

  // LangChain & LangGraph
  M("lcel", "lcel", "LCEL execution", "rag-chain streaming callbacks structured");
  M("rag-chain", "rag-chain", "RAG chain wiring", "lcel rag conv-rag");
  M("langgraph-run", "langgraph-execution", "LangGraph execution", "reducers checkpoint agentloop hitl");
  M("reducers", "reducers", "Reducers", "langgraph-run checkpoint supervisor");
  M("checkpoint", "checkpointers", "Checkpointers", "hitl langgraph-run agent-memory reducers");
  M("streaming", "sse-streaming", "Streaming (SSE)", "lcel inference-server gateway callbacks");
  M("callbacks", "callbacks-tracing", "Callbacks & tracing", "observability lcel eval-pipeline");

  // Serving & infrastructure
  M("inference-server", "inference-server", "Inference server", "cont-batching paged-attention kvcache autoscaling");
  M("cont-batching", "continuous-batching", "Continuous batching", "inference-server paged-attention autoscaling");
  M("paged-attention", "paged-attention", "PagedAttention", "kvcache cont-batching prompt-caching");
  M("quantization", "quantization", "Quantization", "lora tensor-parallel inference-server distillation");
  M("spec-decoding", "speculative-decoding", "Speculative decoding", "sampling generate distillation inference-server");
  M("tensor-parallel", "tensor-parallelism", "Tensor parallelism", "distributed-training quantization attention inference-server");
  M("prompt-caching", "prompt-caching", "Prompt caching", "kvcache paged-attention semantic-cache gateway");
  M("gateway", "llm-gateway", "LLM gateway", "semantic-cache guardrails observability autoscaling");
  M("semantic-cache", "semantic-caching", "Semantic caching", "prompt-caching embeddings gateway");
  M("autoscaling", "autoscaling", "Autoscaling inference", "cont-batching inference-server gateway");

  // Training & fine-tuning
  M("sft", "sft", "Supervised fine-tuning", "lora rlhf-dpo distillation training-overview");
  M("lora", "lora", "LoRA", "sft quantization distributed-training");
  M("rlhf-dpo", "rlhf-dpo", "RLHF vs DPO", "sft llm-judge training-overview reasoning");
  M("distillation", "distillation", "Distillation", "sft spec-decoding quantization");
  M("distributed-training", "distributed-training", "Distributed training", "tensor-parallel lora training-overview");

  // Quality, safety & operations
  M("eval-pipeline", "eval-pipeline", "Eval pipeline", "llm-judge rag-eval observability callbacks");
  M("llm-judge", "llm-as-judge", "LLM-as-a-judge", "eval-pipeline rag-eval rlhf-dpo");
  M("rag-eval", "rag-evaluation", "RAG evaluation", "rag eval-pipeline rerank llm-judge");
  M("prompt-injection", "prompt-injection", "Prompt injection", "guardrails delegated-auth mcp rag");
  M("guardrails", "guardrails", "Guardrails", "prompt-injection structured gateway hitl");
  M("observability", "monitoring", "Production monitoring", "callbacks eval-pipeline gateway guardrails");

  if (typeof module !== "undefined" && module.exports) module.exports = META;
  else root.HOW_META = META;
})(typeof window !== "undefined" ? window : globalThis);

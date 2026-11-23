// AI Infrastructure Engineer interview question bank. Q(topic, level, question, whatTheyTest, answerPoints[], followUps[], code)
(function () {
  const M = "must", P = "maybe";
  const Q = (t, lvl, q, test, a, fu, code) => ({ t, lvl, q, test: test || "", a, fu: fu || [], code: code || "" });

  window.QBANK = {
    id: "ai-infra",
    rounds: [
      { name: "Recruiter / hiring manager screen", len: "30 min", what: "Your infra background, scale you've run, on-call experience.", tip: "Lead with numbers: GPUs, requests per second, cost saved." },
      { name: "Coding", len: "45–60 min", what: "Python or Go with a systems flavour: concurrency, a rate limiter, an LRU cache, parsing logs or metrics.", tip: "State the complexity and how it behaves under concurrency." },
      { name: "Infra system design", len: "60 min", what: "Design an inference platform, a batch inference system, or a training cluster.", tip: "Do the capacity maths out loud: tokens per second, GPUs, memory." },
      { name: "Deep dive", len: "45–60 min", what: "GPU memory, KV cache, batching, serving engines, Kubernetes scheduling.", tip: "Expect \"why?\" three levels deep. Say when you're reasoning rather than remembering." },
      { name: "Troubleshooting scenario", len: "45 min", what: "Latency spike, out-of-memory errors, low GPU utilization, a hanging training job.", tip: "Mitigate first, then find the root cause, then prevent it." },
      { name: "Behavioral / on-call", len: "45 min", what: "Incidents you owned, prioritizing platform requests, working with researchers.", tip: "Blameless framing; end with what changed afterwards." },
    ],
    topics: [
      { id: "t-gpu", title: "GPU & inference fundamentals", blurb: "Memory maths and how LLM inference actually uses the hardware." },
      { id: "t-serving", title: "Model serving", blurb: "Engines, autoscaling, cold starts, rollouts." },
      { id: "t-k8s", title: "Kubernetes & GPU clusters", blurb: "Scheduling, sharing and observing GPUs." },
      { id: "t-train", title: "Training & fine-tuning infrastructure", blurb: "Distributed training and keeping long jobs alive." },
      { id: "t-platform", title: "AI platform & gateway", blurb: "The layer between product teams and GPUs." },
      { id: "t-debug", title: "Debugging & reliability", blurb: "Scenario questions. Structure beats recall." },
      { id: "t-design", title: "System design", blurb: "Full design prompts with outlines." },
      { id: "t-behav", title: "Behavioral", blurb: "Incidents, performance wins, prioritization." },
    ],
    questions: [
      // ---------- GPU fundamentals ----------
      Q("t-gpu", M, "How much GPU memory do you need to serve a 70B-parameter model?", "Memory maths. Asked in almost every infra loop.", [
        "Weights = parameters × bytes per parameter. 70B × 2 bytes (FP16/BF16) ≈ 140 GB. At 8-bit ≈ 70 GB; at 4-bit ≈ 35–40 GB.",
        "Add the KV cache (grows with batch size × sequence length), activations and framework overhead, often 20% or more on top.",
        "So a 70B model in FP16 needs at least two 80 GB GPUs with tensor parallelism, and more for large batches or long contexts.",
      ], ["How would you fit it on one GPU?", "How does quantization affect quality?"]),
      Q("t-gpu", M, "What is the KV cache, and how big does it get?", "", [
        "It stores the attention keys and values for every past token, so decoding doesn't recompute them.",
        "Size ≈ 2 (K and V) × layers × KV heads × head dimension × bytes per value × tokens.",
        "For a 70B model with grouped-query attention (80 layers, 8 KV heads, head dim 128) at FP16, that's about 0.3 MB per token, so 100 concurrent requests × 4k tokens ≈ 130 GB.",
        "It's the main limit on batch size and context length. Mitigations: paged attention, KV-cache quantization, GQA/MQA models, prefix caching.",
      ], [], `def kv_cache_bytes(layers, kv_heads, head_dim, tokens, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * bytes_per * tokens

kv_cache_bytes(80, 8, 128, 1) / 1e6   # ≈ 0.33 MB per token`),
      Q("t-gpu", M, "Explain prefill versus decode.", "", [
        "Prefill processes the whole prompt in parallel. It's compute-bound and determines time to first token (TTFT).",
        "Decode generates one token at a time. It's memory-bandwidth-bound and determines time per output token (TPOT).",
        "They stress the hardware differently, which is why some systems run them on separate pools (disaggregated serving) or chunk prefill so it doesn't stall ongoing decodes.",
      ]),
      Q("t-gpu", M, "What is continuous batching, and why does it matter?", "", [
        "Instead of waiting for a whole batch to finish, the server adds new requests and removes finished ones at every decoding step.",
        "It keeps the GPU busy despite very different request lengths, giving large throughput gains over static batching.",
        "It's standard in vLLM, TGI, TensorRT-LLM and SGLang.",
      ]),
      Q("t-gpu", M, "What is PagedAttention?", "", [
        "vLLM's technique for storing the KV cache in fixed-size blocks, like operating-system memory pages, instead of one contiguous buffer per request.",
        "It removes fragmentation and over-reservation, so many more requests fit in memory, and it lets requests share common prefixes.",
      ]),
      Q("t-gpu", M, "Which latency metrics matter for LLM serving?", "", [
        "TTFT (time to first token), TPOT or ITL (time per output token, inter-token latency), end-to-end latency, and throughput (tokens per second, requests per second).",
        "Report p50, p95 and p99 under realistic input and output length distributions.",
        "Goodput: the throughput you get while still meeting the latency SLO.",
      ]),
      Q("t-gpu", M, "What is quantization, and what are the trade-offs?", "", [
        "Storing weights (and sometimes activations and the KV cache) in fewer bits: FP8, INT8, INT4 (AWQ, GPTQ).",
        "Less memory and bandwidth means bigger batches, faster decoding and fewer GPUs.",
        "Quality can drop, especially at 4-bit, so check with evals. Some formats need specific hardware or kernels.",
      ]),
      Q("t-gpu", M, "Explain data, tensor and pipeline parallelism.", "", [
        "Data parallel: a full model copy on each GPU, each processing different data. Simple, and scales throughput.",
        "Tensor parallel: each layer's matrices are split across GPUs. Needs a fast interconnect (NVLink), so it's used within a node.",
        "Pipeline parallel: different layers on different GPUs or nodes. Adds pipeline \"bubbles\"; used across nodes.",
        "Large models combine them. Mixture-of-experts models add expert parallelism.",
      ]),
      Q("t-gpu", P, "What is speculative decoding?", "", [
        "A small draft model (or extra prediction heads) proposes several tokens; the large model verifies them in one pass and accepts the correct prefix.",
        "It speeds up decoding when the acceptance rate is high, without changing the output distribution.",
      ]),
      Q("t-gpu", P, "Why is LLM decoding memory-bandwidth-bound?", "", [
        "Each new token requires reading all the weights from GPU memory, but does little compute per byte at small batch sizes.",
        "Batching more requests reuses each weight read across requests and raises arithmetic intensity. That's why batching increases throughput.",
      ]),
      Q("t-gpu", P, "What are mixture-of-experts (MoE) models, and how do they affect serving?", "", [
        "Each token is routed to a few expert sub-networks, so total parameters are large but active parameters per token are much smaller.",
        "Compute per token is lower, but all experts must sit in memory. Serving uses expert parallelism and has to handle uneven load across experts.",
      ]),

      // ---------- Serving ----------
      Q("t-serving", M, "Compare vLLM, TGI, TensorRT-LLM/Triton and SGLang.", "", [
        "vLLM: popular open-source engine with PagedAttention, continuous batching, an OpenAI-compatible API and wide model support.",
        "TGI: Hugging Face's server, with similar features and tight Hugging Face integration.",
        "TensorRT-LLM (often behind Triton): NVIDIA-optimized compiled engines; top performance on NVIDIA hardware, but more build complexity.",
        "SGLang: fast runtime with strong prefix caching (RadixAttention) and structured generation.",
        "Choose by benchmarking your models and traffic on your hardware, and by how much operational complexity you can take on.",
      ]),
      Q("t-serving", M, "How would you autoscale LLM inference on Kubernetes?", "", [
        "CPU-based autoscaling is useless here. Scale on queue depth, in-flight requests, KV-cache utilization or latency (custom metrics, e.g. with KEDA).",
        "Model load time is long, so keep warm replicas, cache weights on local NVMe or use fast loading, and pre-pull images.",
        "GPU node autoscaling (Cluster Autoscaler, Karpenter) is slow and capacity can be scarce, so plan reservations or a buffer.",
      ]),
      Q("t-serving", M, "How do you reduce cold-start time for model servers?", "", [
        "Pre-pull container images, keep weights on local disk or fast shared storage, and stream weights while loading (safetensors-style formats).",
        "Keep a minimum number of warm replicas; keep weights out of the container image so images stay small.",
      ]),
      Q("t-serving", M, "How do you load test an LLM endpoint?", "", [
        "Use realistic traffic: input and output length distributions from production, ramping concurrency, streaming on.",
        "Measure TTFT, TPOT, throughput and errors at each concurrency level, and find the point where the latency SLO breaks.",
        "Tools: the serving engine's own benchmark scripts, or Locust or k6 with streaming support.",
      ]),
      Q("t-serving", P, "How do you serve many fine-tuned variants of one base model efficiently?", "", [
        "Multi-LoRA serving: load the base model once and batch requests for many LoRA adapters (vLLM, LoRAX).",
        "Route requests by adapter ID, and keep popular adapters cached on the GPU.",
      ]),
      Q("t-serving", P, "How does structured (guided) decoding work in the server?", "", [
        "At each step, token choices are constrained to a grammar or JSON schema by masking invalid tokens; vLLM and SGLang support this through libraries such as Outlines or XGrammar.",
        "It guarantees valid output format, with a small overhead.",
      ]),
      Q("t-serving", P, "How do you roll out a new model version safely?", "", [
        "Shadow traffic, then a percentage canary; compare quality evals and latency and error metrics; roll back automatically on regression.",
        "Version everything together: weights, engine version, configuration.",
      ]),

      // ---------- Kubernetes ----------
      Q("t-k8s", M, "How are GPUs exposed and scheduled in Kubernetes?", "", [
        "The NVIDIA device plugin (often installed via the GPU Operator) advertises nvidia.com/gpu resources, and pods request them in their limits.",
        "Use separate GPU node pools with taints and tolerations, and select GPU types with node selectors or affinity.",
        "GPUs aren't overcommitted by default. Sharing is done with MIG (hardware partitions) or time-slicing.",
      ], [], `resources:
  limits:
    nvidia.com/gpu: 2
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
nodeSelector:
  gpu-type: h100`),
      Q("t-k8s", M, "What is MIG, and when would you use time-slicing instead?", "", [
        "MIG (Multi-Instance GPU) splits A100/H100-class GPUs into isolated instances with dedicated memory and compute. Good for small models that need isolation.",
        "Time-slicing shares a GPU without memory isolation. Fine for development or light, bursty workloads; risky for production latency.",
      ]),
      Q("t-k8s", M, "A GPU node shows low utilization, but latency is high. How do you investigate?", "", [
        "Find where the time goes: tokenization or CPU, network, queueing, data loading, or batches that are too small.",
        "Check metrics: GPU utilization and memory (DCGM exporter), KV-cache usage, batch size, request queue length.",
        "Common causes: max-concurrent-sequence limits set too low, a CPU bottleneck, single-threaded preprocessing, or tensor parallelism across PCIe instead of NVLink.",
      ]),
      Q("t-k8s", P, "How do you schedule multi-node distributed training jobs on Kubernetes?", "", [
        "Gang scheduling, so all workers start together (Kueue, Volcano, the Kubeflow Training Operator, or Ray on Kubernetes).",
        "Topology-aware placement for fast interconnect (InfiniBand or RDMA), and quotas, priorities and preemption for fair sharing.",
      ]),
      Q("t-k8s", P, "How do you handle GPU scarcity and cost?", "", [
        "Reserved or committed capacity for the baseline, on-demand or spot for bursts, and spot for training with frequent checkpoints.",
        "Bin-pack, match the GPU type to the model, quantize, share GPUs for small models, and schedule batch work off-peak.",
        "Track cost per 1M tokens and GPU utilization per team.",
      ]),
      Q("t-k8s", P, "How do you observe GPU workloads?", "", [
        "DCGM exporter → Prometheus/Grafana: utilization, memory, temperature, ECC and Xid errors.",
        "Serving metrics from the engine (queue, KV-cache usage, tokens per second), per-request traces, and alerts on errors and saturation.",
      ]),

      // ---------- Training ----------
      Q("t-train", M, "Explain DDP versus FSDP (or ZeRO).", "", [
        "DDP: every GPU holds a full model copy, and gradients are all-reduced each step. Simple, but the model must fit on one GPU.",
        "FSDP and ZeRO shard parameters, gradients and optimizer state across GPUs and gather them when needed. That allows much larger models at the cost of more communication.",
      ]),
      Q("t-train", M, "A multi-node training job keeps failing after several hours. How do you make it resilient?", "", [
        "Checkpoint frequently to durable storage (asynchronous, sharded), and restart automatically from the latest checkpoint (elastic training with torchrun).",
        "Health-check nodes for bad GPUs (Xid errors), then cordon and replace them. Set NCCL timeouts and logging to find hangs.",
        "Add retry policies at the scheduler level.",
      ]),
      Q("t-train", P, "Why does interconnect bandwidth matter for training?", "", [
        "Every step synchronizes gradients or parameters across GPUs (all-reduce, all-gather). Slow links leave GPUs waiting.",
        "NVLink within a node, InfiniBand or RoCE across nodes. NCCL implements the collective operations.",
      ]),
      Q("t-train", P, "How much memory does full fine-tuning need compared with LoRA?", "", [
        "Full fine-tuning with Adam in mixed precision needs roughly 16 bytes per parameter (weights, gradients, optimizer states) before activations, so a 7B model needs about 110 GB or more.",
        "LoRA trains only small adapters. QLoRA uses a 4-bit base model, so a 7B model can be fine-tuned on a single 24 GB GPU.",
      ]),
      Q("t-train", P, "What does an ML platform need for reproducible training?", "", [
        "Versioned data and code, experiment tracking (MLflow, Weights & Biases), pinned container images, recorded seeds and configs, and a model registry with lineage.",
      ]),

      // ---------- Platform ----------
      Q("t-platform", M, "Design an internal LLM gateway used by 30 teams.", "", [
        "An OpenAI-compatible API with per-team auth (API keys or OIDC); model aliases routed to providers or self-hosted endpoints.",
        "Rate limits and budgets per team; retries, fallbacks and circuit breakers; streaming passed through.",
        "Usage metering (tokens, cost) into a warehouse, logging with PII controls, and caching.",
        "One place for model upgrades, policies and eval gates. Highly available, with little added latency.",
      ]),
      Q("t-platform", M, "Self-hosting versus an API: how do you calculate the break-even point?", "", [
        "Self-hosted cost: GPU hours (including idle time and redundancy) plus engineering and operations time, divided by measured throughput at your latency SLO, gives cost per 1M tokens.",
        "API cost: price per 1M input and output tokens at your traffic mix.",
        "Break-even needs steady, high utilization; bursty traffic favours APIs. Also weigh quality differences and compliance.",
      ]),
      Q("t-platform", P, "How would you implement token-based rate limiting?", "", [
        "A token bucket per tenant, measured in LLM tokens per minute: estimate input tokens up front and reconcile actual output tokens afterwards.",
        "Keep shared state in Redis, and return 429 with a retry-after header.",
      ], [], `import time

class TokenBucket:
    def __init__(self, rate_per_sec, capacity):
        self.rate, self.capacity = rate_per_sec, capacity
        self.tokens, self.last = capacity, time.monotonic()

    def allow(self, cost=1):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False`),
      Q("t-platform", P, "What belongs in an MLOps / LLMOps pipeline?", "", [
        "Data versioning, training and fine-tuning pipelines, eval gates, a model registry, automated deployment with canaries, monitoring and feedback loops, and rollback.",
      ]),

      // ---------- Debugging ----------
      Q("t-debug", M, "The inference service's p99 latency suddenly doubled. Walk me through debugging it.", "Structured incident handling.", [
        "Scope it: all models or one, all regions or one? Correlate with deploys, traffic mix (longer prompts?) and node changes.",
        "Break down the latency: queue time vs TTFT vs decode. Check KV-cache saturation and preemptions, batch sizes and GPU health.",
        "Mitigate first (scale out, roll back, shed load), then find the root cause, write a postmortem and add alerts.",
      ]),
      Q("t-debug", M, "Pods are being OOM-killed, or you see CUDA out-of-memory errors. What do you check?", "", [
        "First tell host OOM (container memory limit) apart from GPU OOM.",
        "GPU: max model length, max concurrent sequences, the GPU-memory-utilization setting, KV-cache size, quantization, tensor-parallel size, fragmentation.",
        "Host: buffers while loading weights, tokenizer workers, memory limits set too low.",
      ]),
      Q("t-debug", P, "How do you design model serving for high availability?", "", [
        "Multiple replicas across zones, readiness checks that confirm the model is loaded, graceful draining of in-flight streams, a fallback to a provider API, and spare capacity.",
      ]),
      Q("t-debug", P, "What SLOs would you define for an inference platform?", "", [
        "Availability (successful responses), p95 TTFT and p95 TPOT per model tier, an error-budget policy, and separate tiers for interactive and batch traffic.",
      ]),

      // ---------- Design ----------
      Q("t-design", M, "Design a platform that serves open-weight LLMs to internal teams at 1,000 requests per second.", "Capacity maths, plus a full platform view.", [
        "Requirements: which models, context lengths, latency SLOs, traffic shape, tenants, budget.",
        "Capacity maths: required tokens per second = requests per second × output tokens; benchmark per-GPU throughput at the SLO → number of GPUs plus headroom.",
        "Stack: gateway (auth, routing, limits) → per-model vLLM deployments on GPU node pools → autoscaling on queue and KV-cache metrics.",
        "Weights in object storage with a local cache, prefix caching, and quantized variants for cheaper tiers.",
        "Observability, canary releases, multiple zones, fallback to an API, and cost per team.",
      ], ["How many GPUs would you need? Walk through the numbers.", "What changes if half the traffic needs a 128k context?"]),
      Q("t-design", M, "Design a batch inference system for 100 million documents (e.g. embeddings or classification).", "", [
        "A queue or partitioned dataset; workers with large static batches for maximum throughput; spot GPUs with checkpointed progress; idempotent outputs keyed by document ID.",
        "Size the cluster from the throughput maths, add backpressure, retries and a dead-letter queue, and write results in bulk to the warehouse or vector DB.",
      ]),
      Q("t-design", P, "Design a fine-tuning service that lets teams train LoRA adapters on their own data.", "", [
        "An API to submit jobs with a dataset reference; validation; a job queue with quotas; training on a GPU pool (QLoRA); an eval step; a registry of adapters; deployment to multi-LoRA serving.",
      ]),

      // ---------- Behavioral ----------
      Q("t-behav", M, "Tell me about a production incident you owned.", "", [
        "STAR: how it was detected, how you mitigated it, the root cause, and what you changed to prevent it. Blameless, and with measured improvement.",
      ]),
      Q("t-behav", M, "Tell me about a time you significantly improved performance or cut cost.", "", [
        "Baseline numbers → what you measured → what you changed → the result, and the trade-offs you accepted.",
      ]),
      Q("t-behav", P, "How do you prioritize platform work requested by many teams?", "", [
        "Impact versus effort, needs shared across teams, SLAs, a transparent roadmap, and saying no with data.",
      ]),
    ],
  };
})();

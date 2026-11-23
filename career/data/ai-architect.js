// AI Architect interview question bank. Q(topic, level, question, whatTheyTest, answerPoints[], followUps[], code)
(function () {
  const M = "must", P = "maybe";
  const Q = (t, lvl, q, test, a, fu, code) => ({ t, lvl, q, test: test || "", a, fu: fu || [], code: code || "" });

  window.QBANK = {
    id: "ai-architect",
    rounds: [
      { name: "Hiring manager screen", len: "30–45 min", what: "The scope of architecture work you've done, and how you influence teams.", tip: "Describe scope in teams, systems and budget, not just technologies." },
      { name: "Architecture case study", len: "60–90 min", what: "Design an enterprise AI platform or use case on a whiteboard.", tip: "Clarify the business goal and constraints first; show options before deciding." },
      { name: "Past-architecture deep dive", len: "60 min", what: "Walk through a system you designed: decisions, trade-offs, what you'd change.", tip: "Bring one diagram and an ADR-style summary. Mention a regret honestly." },
      { name: "Trade-off & governance scenarios", len: "45 min", what: "Security, compliance, build versus buy, vendor choice.", tip: "Name the risk, the control, and who owns it." },
      { name: "Stakeholder presentation", len: "30–45 min", what: "Present a recommendation to executives, sometimes prepared in advance.", tip: "Outcome first, options with cost and risk, a clear recommendation, the ask." },
      { name: "Behavioral / leadership", len: "45 min", what: "Influence without authority, pushing back, driving adoption.", tip: "Show how you brought people along, with evidence." },
    ],
    topics: [
      { id: "t-strategy", title: "Strategy & decisions", blurb: "Build versus buy, model strategy, and when not to use GenAI." },
      { id: "t-patterns", title: "Reference architectures", blurb: "The patterns you should be able to draw from memory." },
      { id: "t-security", title: "Security & threat modelling", blurb: "LLM-specific risks and the controls that address them." },
      { id: "t-gov", title: "Data, governance & compliance", blurb: "Policies that work in practice, and their effect on design." },
      { id: "t-cost", title: "Cost, scale & operations", blurb: "Unit economics, quality standards and resilience across many teams." },
      { id: "t-case", title: "Architecture case studies", blurb: "Full scenarios with structured answers." },
      { id: "t-lead", title: "Leadership & communication", blurb: "Architects are hired for judgment and influence." },
    ],
    questions: [
      // ---------- Strategy ----------
      Q("t-strategy", M, "How do you decide whether to build or buy an AI capability?", "Business judgment, not technology preference.", [
        "Is it core differentiation? If not, buy (SaaS or a copilot product) and integrate it.",
        "Weigh fit with data and security constraints, integration needs, total cost of ownership (licences vs team + infrastructure + maintenance), time to value, lock-in and vendor roadmap risk.",
        "A common outcome: buy commodity capabilities (chat assistant, transcription), and build differentiating workflows on a shared platform.",
      ]),
      Q("t-strategy", M, "How do you choose an LLM provider and model strategy for an enterprise?", "", [
        "Evaluate on your own use cases with evals: quality, latency, cost, context length, tool use.",
        "Non-functional needs: data handling and residency, enterprise agreements, availability and SLAs, alignment with your cloud.",
        "Avoid lock-in: a gateway abstraction, portable prompts and evals, at least two approved providers.",
        "Tier the models: frontier models for complex tasks, smaller cheaper models for high-volume simple ones.",
      ], ["How often would you revisit this decision?"]),
      Q("t-strategy", M, "When is generative AI the wrong solution?", "", [
        "Deterministic rules, exact calculations, and decisions with zero error tolerance.",
        "Problems that search, SQL or classic ML solve more cheaply and reliably.",
        "When the data doesn't exist, or the underlying process itself is broken.",
      ]),
      Q("t-strategy", P, "How do you prioritize a portfolio of AI use cases?", "", [
        "Score each by business value, feasibility (data, integration), risk and time to value. Start with high-value, low-risk internal use cases.",
        "Fund the shared platform capabilities that several use cases need.",
      ]),
      Q("t-strategy", P, "A central AI platform team, or AI engineers embedded in product teams?", "", [
        "Usually hub-and-spoke: a central platform (gateway, evals, RAG service, governance) plus embedded engineers who build the use cases.",
        "Centralize what is shared and risky; leave product decisions with the product teams.",
      ]),

      // ---------- Patterns ----------
      Q("t-patterns", M, "Draw a reference architecture for an enterprise GenAI platform.", "Whether you see all the layers and the cross-cutting concerns.", [
        "Experience layer: apps, chat, APIs, agents.",
        "Orchestration: an agent framework (e.g. LangGraph), prompt management, tools exposed through MCP or an API catalogue.",
        "Knowledge: ingestion pipelines, vector and keyword search, permission-aware retrieval.",
        "Model access: an LLM gateway with routing, limits, fallbacks, caching and redaction, in front of hosted APIs and self-hosted models.",
        "Cross-cutting: identity and access, observability and tracing, an evaluation service, guardrails, cost management, audit, data governance.",
      ], ["Which layer would you build first, and why?"]),
      Q("t-patterns", M, "RAG, fine-tuning, agents or long context: when do you use each?", "", [
        "RAG: knowledge that changes, or must be cited and permission-checked.",
        "Fine-tuning: consistent behaviour or format, narrow tasks, lower cost per call.",
        "Agents: multi-step tasks that need tools and decisions.",
        "Long context with caching: small, stable bodies of knowledge.",
        "Real systems combine them.",
      ]),
      Q("t-patterns", M, "How do you design for multi-tenancy in an AI SaaS product?", "", [
        "Isolate tenants at every layer: data stores (separate indexes or strict filters), caches (tenant in the cache key), prompts and logs, fine-tuned adapters.",
        "Per-tenant quotas, keys and encryption keys (bring-your-own-key if needed), and audit logs.",
        "Test isolation explicitly, including prompt-injection attempts to read another tenant's data.",
      ]),
      Q("t-patterns", P, "Synchronous or asynchronous AI workflows: how do you decide?", "", [
        "Synchronous with streaming for interactive use (chat). Asynchronous (queues, jobs, callbacks) for long tasks such as document processing or agents running for minutes.",
        "Asynchronous designs need job status, retries, idempotency and notifications.",
      ]),
      Q("t-patterns", P, "How do you integrate AI into an event-driven architecture?", "", [
        "AI services consume events (e.g. ticket created → classify → publish the result), using the outbox pattern, idempotent handlers, dead-letter queues, and a schema registry for outputs.",
      ]),
      Q("t-patterns", P, "How do you architect agents that use many internal APIs?", "", [
        "A tool catalogue or registry (e.g. MCP servers), standard auth (user-delegated OAuth), permission scopes per agent, approval policies, and central audit logs.",
      ]),

      // ---------- Security ----------
      Q("t-security", M, "Threat-model an LLM application. What are the main risks?", "Breadth of LLM-specific security knowledge.", [
        "Prompt injection (direct and indirect), data exfiltration through outputs or tools, and \"excessive agency\": tools with too much power.",
        "Sensitive data leaking into prompts, logs or training data; insecure handling of model output (passing it into SQL, HTML or a shell); supply-chain risk (models, plugins, MCP servers); and cost attacks (\"denial of wallet\").",
        "A useful reference is the OWASP Top 10 for LLM Applications.",
        "Controls: least privilege, human approvals, output validation, isolation, monitoring, rate and budget limits, red-teaming.",
      ]),
      Q("t-security", M, "How do you prevent data leakage between users in RAG?", "", [
        "Permission-aware retrieval, with ACLs enforced in the search query.",
        "Sync ACLs from the source systems, never share caches across users without a user or tenant key, and include cross-user access attempts in your tests.",
      ]),
      Q("t-security", M, "What are guardrails, and where do you put them?", "", [
        "Input checks (injection detection, PII, policy), output checks (toxicity, PII leakage, groundedness, format), and tool-level policies (approvals, scopes).",
        "Put them in the gateway and the orchestration layer. Combine rules, classifiers and LLM-based checks, and log and measure false positives.",
      ]),
      Q("t-security", P, "How do you secure third-party models, tools and MCP servers?", "", [
        "Vendor review, pinned versions, scanning model files (prefer safetensors over pickle), sandboxed execution, egress controls, least-privilege credentials, allow-lists.",
      ]),

      // ---------- Governance ----------
      Q("t-gov", M, "What does AI governance look like in practice?", "", [
        "Intake and risk classification for each use case, a list of approved models and providers, and data-usage policies.",
        "Required evals and documentation (system cards), and human oversight for high-risk decisions.",
        "Monitoring, an incident process, audit trails, and a cross-functional review board (legal, security, privacy, business).",
        "Make the paved road easy: platform defaults should implement the policies, so following them is the easiest option.",
      ]),
      Q("t-gov", M, "How does the EU AI Act affect system design?", "Awareness and design impact, not legal advice.", [
        "It's risk-based: some practices are banned, and high-risk systems (e.g. in hiring, credit or critical services) carry obligations: risk management, data governance, logging, transparency, human oversight, accuracy and robustness.",
        "Some systems have transparency duties, e.g. telling users they're interacting with AI, and labelling AI-generated content.",
        "Providers of general-purpose AI models have their own obligations.",
        "Design impact: classify use cases early, keep logs and documentation, and build human oversight and evaluation into the system. Confirm the specifics with legal.",
      ]),
      Q("t-gov", M, "How do you handle PII and GDPR in LLM systems?", "", [
        "Data minimization and purpose limitation; redact or pseudonymize before sending; data-processing agreements with providers; data residency.",
        "Retention limits on prompts and logs, and support for deletion requests, including from vector indexes and caches.",
        "A lawful basis and transparency, plus a data protection impact assessment (DPIA) for high-risk processing.",
      ]),
      Q("t-gov", P, "How do you make AI decisions auditable?", "", [
        "Log inputs, retrieved context, model and prompt versions, outputs, tool calls and approvals under a trace ID.",
        "Store logs immutably under a retention policy, with the ability to replay a decision.",
      ]),
      Q("t-gov", P, "How do you address bias and fairness?", "", [
        "Identify the affected groups, evaluate outputs across them with test sets, mitigate through data, prompts, constraints and human review, then monitor and document.",
      ]),

      // ---------- Cost & ops ----------
      Q("t-cost", M, "How do you build a cost model for an AI initiative?", "", [
        "Volume × tokens per interaction × price per model tier, plus retrieval and infrastructure, plus team costs.",
        "Low, expected and high adoption scenarios, and unit economics: cost per conversation versus value per conversation.",
        "Levers: routing, caching, prompt size, batching, self-hosting at scale, and showback or chargeback per team.",
      ]),
      Q("t-cost", M, "How do you set evaluation and quality standards across many teams?", "", [
        "A shared evaluation service or framework, a required eval set for each use case before launch, and standard metrics (faithfulness, task success, safety).",
        "CI gates for prompt and model changes, production monitoring with sampled judging and user feedback, and periodic quality reviews.",
      ]),
      Q("t-cost", P, "How do you manage model upgrades and deprecations across the company?", "", [
        "Gateway aliases decouple applications from model versions. Run each use case's evals on the new model, roll out in stages, and publish a deprecation calendar.",
      ]),
      Q("t-cost", P, "How do you design for resilience against provider outages?", "", [
        "Multiple providers behind the gateway, fallbacks with prompts already tested on them, graceful degradation, queues for asynchronous work, and SLAs with capacity reservations.",
      ]),
      Q("t-cost", P, "How do you plan capacity for AI across the organization?", "", [
        "Forecast token demand per use case, track provider rate limits and committed capacity, plan GPUs for self-hosted models, and set quotas per team.",
      ]),

      // ---------- Case studies ----------
      Q("t-case", M, "A global insurer wants AI to speed up claims processing. Design the solution.", "", [
        "Discover: claim types, volumes, current cycle time, regulations, existing systems (claims platform, document store).",
        "Phase 1, document intake: OCR and extraction into structured fields, with validation and a human review queue.",
        "Phase 2, adjuster assistant: RAG over policies and similar past claims, summaries, and suggested next steps with citations.",
        "Payout decisions stay with humans (high risk, regulated).",
        "Architecture: an event-driven pipeline, the gateway, permission-aware retrieval, audit logging, evals on labelled claims, regional data residency.",
        "Metrics: cycle time, extraction accuracy, adjuster productivity.",
      ], ["What would you deliver in the first 90 days?"]),
      Q("t-case", M, "15 teams are each building their own chatbot with different vendors. What do you do?", "", [
        "Inventory and assess them: cost, risk, quality. Find the common needs.",
        "Define a paved road: shared gateway, approved providers, a RAG service, an eval framework and guardrails, made easier to use than going alone.",
        "Migrate the high-risk or high-cost ones first, put governance on new projects, and measure adoption and savings.",
      ]),
      Q("t-case", P, "Design an internal knowledge assistant across Confluence, SharePoint, Slack and Jira.", "", [
        "Connectors with incremental sync and ACL mapping, a unified hybrid index, identity mapping across systems, freshness targets, citations, evals per source, and usage analytics.",
      ]),

      // ---------- Leadership ----------
      Q("t-lead", M, "Walk me through an architecture you designed. What would you do differently?", "", [
        "Context, constraints, the options you considered, the decision and why, the results with numbers, and what you'd change (this shows reflection).",
        "Have a diagram and an ADR-style summary ready.",
      ]),
      Q("t-lead", M, "How do you get teams to adopt your architecture when you don't manage them?", "", [
        "Involve them early, solve their actual pain, and provide reference implementations and templates.",
        "Use data from pilots, run office hours, get leadership sponsorship, and measure adoption.",
      ]),
      Q("t-lead", M, "How do you explain an AI trade-off to executives?", "", [
        "Start with the business outcome, show 2–3 options with cost, risk, time and expected impact, give your recommendation, and state what you need from them.",
      ]),
      Q("t-lead", P, "Tell me about a time you pushed back on an AI initiative.", "", [
        "STAR. Show the business reasoning, the alternative you offered, and the outcome.",
      ]),
      Q("t-lead", P, "How do you write an architecture decision record (ADR)?", "", [
        "Title, context, decision, options considered, consequences (pros and cons), and status. Keep it short, and link it to evals or data.",
      ], [], `# ADR-012: Route all LLM traffic through a central gateway
Status: Accepted
Context: 15 teams call 4 providers directly; no cost visibility or policy control.
Decision: All LLM calls go through the platform gateway (model aliases, limits, logging).
Options: (a) status quo (b) shared SDK only (c) gateway  -> chose (c)
Consequences: + central cost/policy/upgrades  - added hop (~20 ms), platform team owns uptime`),
    ],
  };
})();

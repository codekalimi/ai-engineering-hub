// Forward Deployed AI Engineer interview question bank. Q(topic, level, question, whatTheyTest, answerPoints[], followUps[], code)
(function () {
  const M = "must", P = "maybe";
  const Q = (t, lvl, q, test, a, fu, code) => ({ t, lvl, q, test: test || "", a, fu: fu || [], code: code || "" });

  window.QBANK = {
    id: "fde",
    rounds: [
      { name: "Recruiter / hiring manager screen", len: "30 min", what: "Why customer-facing work, appetite for travel and ambiguity, and your background.", tip: "Give a real reason you like working with customers, plus one example." },
      { name: "Technical screen", len: "45–60 min", what: "Practical coding in Python or TypeScript: APIs, data wrangling, working through a messy dataset.", tip: "Ship a working solution first, then clean it up." },
      { name: "Customer case / decomposition", len: "60 min", what: "A vague customer problem that you turn into a scoped, phased plan.", tip: "Ask questions before proposing anything. Define the success metric early." },
      { name: "Live build / prototype", len: "2–4 h", what: "Build something working fast on unfamiliar data or an unfamiliar API.", tip: "Get a thin version working end to end early, then improve it. Narrate your decisions." },
      { name: "Presentation / demo", len: "30–45 min", what: "Present your build to \"customer stakeholders\", often a mix of technical and executive people.", tip: "Lead with the business outcome, then the demo, then risks and next steps." },
      { name: "Behavioral", len: "45 min", what: "Ambiguity, ownership, difficult customers, saying no.", tip: "Stories should show judgment and calm under pressure, not heroics." },
    ],
    topics: [
      { id: "t-case", title: "Customer scenarios", blurb: "The heart of FDE interviews: turning a vague ask into a plan that ships." },
      { id: "t-build", title: "Prototyping & integration", blurb: "Building fast inside someone else's environment, safely." },
      { id: "t-ai", title: "Applied AI in customer settings", blurb: "AI knowledge, framed for customers' constraints." },
      { id: "t-comm", title: "Communication & stakeholders", blurb: "Half of the job. Interviewers listen to how you say it." },
      { id: "t-behav", title: "Behavioral", blurb: "Prepare a STAR story for each." },
    ],
    questions: [
      // ---------- Customer scenarios ----------
      Q("t-case", M, "A bank wants to \"use AI on their support tickets\". What do you do in your first week?", "Discovery before solutioning, and turning ambiguity into a plan.", [
        "Discovery: meet the stakeholders (support lead, IT, security, compliance). Learn the current process, volumes, pain points and the metrics they already track.",
        "Pick 1–2 concrete use cases with measurable value, e.g. ticket triage and routing, or draft replies that agents approve.",
        "Get sample data early (anonymized tickets), and check data access and security constraints.",
        "Agree on success metrics and a baseline (handling time, routing accuracy).",
        "Propose a phased plan: prototype on historical data → pilot with a few agents → production.",
      ], ["What if they can't share any data in week 1?", "Which use case would you pick first, and why?"]),
      Q("t-case", M, "The customer wants a fully autonomous agent that approves refunds. How do you respond?", "Judgment, and saying no constructively.", [
        "Understand the goal behind it (cost, speed) and their risk tolerance. Don't just say no.",
        "Propose a path: first the AI drafts decisions for a human to approve; measure accuracy; then automate low-risk cases (e.g. under an amount threshold) with an audit trail and rollback.",
        "Make the risks concrete: error rate, fraud, prompt injection, regulatory exposure.",
      ]),
      Q("t-case", M, "How do you scope an AI project so it ships in 6–8 weeks?", "", [
        "Narrow it to one workflow, one user group and one data source, and define \"done\" and the metric.",
        "Find blockers early: data access, security review, integrations.",
        "Build the thinnest end-to-end slice first, then improve quality with evals.",
        "Write down what's out of scope and get agreement on it.",
      ]),
      Q("t-case", M, "The pilot looks great in demos, but users don't trust the answers. What do you do?", "", [
        "Talk to the users and collect the cases they don't trust, then measure accuracy on them.",
        "Add citations and confidence signals so users can check the sources.",
        "Fix the main failure types (often retrieval or missing data), and show the improvement with numbers.",
        "Design for verification: easy feedback, and a clear escalation path.",
      ]),
      Q("t-case", M, "A customer executive wants a demo in 3 days. What do you build?", "", [
        "Something that shows value on their own sample data, for one high-impact workflow.",
        "A reliable happy path, a couple of realistic edge cases, and visible sources.",
        "Prepare the story: problem → demo → early measured results → plan and risks. Be honest about what's still a prototype.",
      ]),
      Q("t-case", M, "Their IT team has blocked you for weeks on a security review. What do you do?", "", [
        "Prepare the security package up front: data-flow diagram, data retention, encryption, certifications (SOC 2), and the model provider's terms.",
        "Meet security early and ask for their specific concerns. Offer options, e.g. deploying in their cloud account or redacting PII.",
        "Keep progressing on parts that don't need production data (synthetic data, architecture).",
        "If it's still stuck, escalate through your sponsor with the business impact of the delay.",
      ]),
      Q("t-case", P, "The customer's data is messy and spread across 5 systems. How do you approach it?", "", [
        "Map the sources, owners, access methods and data quality, then pick the 1–2 sources that matter most for the use case.",
        "Build connectors incrementally and normalize everything into one common record or document format with metadata.",
        "Don't wait for perfect data. Use evals to measure how much the quality issues actually hurt.",
      ]),
      Q("t-case", P, "How do you decide whether a problem should use an LLM at all?", "", [
        "LLMs suit language-heavy, fuzzy tasks. Rules, SQL or classic ML suit deterministic, numeric or high-precision tasks.",
        "Consider volume, latency, cost and error tolerance. A hybrid is often best.",
      ]),
      Q("t-case", P, "Two departments at the customer want conflicting things. How do you handle it?", "", [
        "Clarify each side's goal and the shared business objective, and bring data on effort and impact.",
        "Go to the sponsor with a recommendation, not just a description of the conflict.",
      ]),

      // ---------- Build & integration ----------
      Q("t-build", M, "How would you connect an LLM app to a customer's internal systems securely?", "", [
        "Use their identity provider for single sign-on (SAML or OIDC), and act on behalf of the user with scoped OAuth tokens.",
        "Call systems through their existing APIs with least-privilege service accounts, and keep secrets in a vault.",
        "On the network side, deploy in their VPC or connect over a private link if required.",
        "Log and audit every action, and get the security review started early.",
      ]),
      Q("t-build", M, "The customer can't send data to an external LLM API. What are your options?", "", [
        "Enterprise agreements with zero data retention and regional endpoints, or the provider's models through their own cloud account (Bedrock, Vertex AI, Azure).",
        "A self-hosted open-weight model in their environment (e.g. vLLM on their GPUs), with trade-offs in quality and operations.",
        "Redact PII before calls, or go hybrid: sensitive steps stay local, the rest go to the API.",
      ]),
      Q("t-build", M, "Walk me through building a prototype RAG on a customer's documents in two days.", "", [
        "Day 1: get a representative sample of documents; parse, chunk and embed them into a simple vector store; build a basic chain with citations; collect 20–30 real questions from users.",
        "Day 2: run those questions, fix the biggest failures (parsing, chunking, hybrid search), add a simple UI or API, and write a short summary of results and next steps.",
      ]),
      Q("t-build", M, "Build a prototype to extract fields from 10,000 invoice PDFs.", "", [
        "Parse with layout-aware PDF tools, and OCR the scanned ones.",
        "Extract with structured output into a Pydantic schema (vendor, date, line items, totals).",
        "Validate with business rules (line items add up to the total, valid dates), and send low-confidence or failing cases to a human review queue.",
        "Measure accuracy per field on a labeled sample of 100–200 invoices, and process in batches with retries.",
      ], [], `class Invoice(BaseModel):
    vendor: str
    invoice_date: date
    total: Decimal
    lines: list[LineItem]

inv = model.with_structured_output(Invoice).invoke(page_text)
if abs(sum(l.amount for l in inv.lines) - inv.total) > Decimal("0.01"):
    review_queue.put(doc_id)`),
      Q("t-build", M, "How do you hand over a solution to the customer's team?", "", [
        "Documentation: architecture, runbooks, the eval set and how to run it, known limitations.",
        "Pair with their engineers before handover, make sure monitoring and alerts are in place, and agree on clear ownership.",
        "Train the users, and agree on the support path back to your company.",
      ]),
      Q("t-build", P, "How do you handle webhooks and event-driven integrations reliably?", "", [
        "Verify signatures, respond quickly, and do the processing asynchronously from a queue.",
        "Use idempotency keys to handle retries and duplicates, plus a dead-letter queue and replay.",
      ]),
      Q("t-build", P, "How do you get productive in an unfamiliar codebase or environment quickly?", "", [
        "Find the entry points and the data flow, run it locally, read the tests, and ask one knowledgeable person targeted questions.",
        "Make small, safe changes first, and add logging or tracing.",
      ]),
      Q("t-build", P, "What deployment options would you consider for a customer: cloud, on-prem, air-gapped?", "", [
        "SaaS is fastest. The customer's cloud account or VPC gives them data control. On-prem or air-gapped needs self-hosted models, offline packages and their operations team.",
        "Package with Docker and Helm, use infrastructure as code, and plan how updates and model upgrades will work.",
      ]),

      // ---------- Applied AI ----------
      Q("t-ai", M, "How do you prove to a customer that the AI solution works?", "", [
        "Before building, agree on an eval set from their real data and their definition of \"correct\".",
        "Report accuracy, coverage and failure examples, compared with the current baseline (human time or error rate).",
        "Pilot with real users, and track the business metrics.",
      ]),
      Q("t-ai", M, "How do you choose between prompt engineering, RAG and fine-tuning for a customer?", "", [
        "Prompting first. RAG when answers depend on their documents. Fine-tuning for a consistent style or format, or a narrow, high-volume task.",
        "The customer's constraints matter: data volume, capacity to label data, deployment options, budget.",
      ]),
      Q("t-ai", M, "Explain prompt-injection risk to a non-technical customer.", "", [
        "If the AI reads text we don't control, such as emails or web pages, that text can contain hidden instructions. It's like a phishing email aimed at the AI.",
        "So we limit what the AI is allowed to do, require approval for important actions, and check its actions in code.",
      ]),
      Q("t-ai", P, "How do you estimate and control cost for a customer deployment?", "", [
        "Estimate tokens per request × requests per day × price, including retrieval and retries. A small spreadsheet calculator works well.",
        "Controls: model routing, caching, per-user limits, dashboards and alerts.",
      ]),
      Q("t-ai", P, "How do you handle multilingual customer data?", "", [
        "Check model quality per language with evals, use a multilingual embedding model, store the language in metadata, and translate only when needed.",
      ]),

      // ---------- Communication ----------
      Q("t-comm", M, "Explain RAG to a business executive in one minute.", "", [
        "The AI first looks up the relevant pages in your own documents, then answers using only those, and shows its sources. It's like a new employee who checks the handbook before answering.",
        "So answers stay current and can be checked, and no model needs retraining.",
      ]),
      Q("t-comm", M, "How do you say no to a customer request?", "", [
        "Acknowledge the goal, explain the concrete risk or cost, and offer an alternative that delivers most of the value.",
        "Put it in writing with the trade-offs, and involve the account owner if needed.",
      ]),
      Q("t-comm", M, "You have 5 minutes to demo. How do you structure it?", "", [
        "1 minute: the problem and why it matters, using their numbers.",
        "3 minutes: a live demo of the key workflow on their data.",
        "1 minute: results so far, risks, next steps and what you need from them.",
        "Have a recorded backup in case the live demo fails.",
      ]),
      Q("t-comm", P, "How do you write a project update for a customer sponsor?", "", [
        "Keep it short: status (on track or at risk), progress against the metrics, decisions needed, risks, next steps.",
        "Share bad news early, together with a plan.",
      ]),
      Q("t-comm", P, "How do you feed customer learnings back to the product team?", "", [
        "Look for patterns across customers, write short problem statements with evidence and impact, and propose features that generalize instead of one-off hacks.",
      ]),

      // ---------- Behavioral ----------
      Q("t-behav", M, "Tell me about a time you worked with a difficult customer or stakeholder.", "", [
        "STAR. Show empathy and clarity: what they really needed, how you reset expectations, when and how you escalated, and the outcome.",
      ]),
      Q("t-behav", M, "Tell me about a time you delivered under a tight deadline with unclear requirements.", "", [
        "STAR. Show how you clarified the goal, cut the scope, shipped something useful, and what you'd do differently.",
      ]),
      Q("t-behav", M, "Why a forward-deployed role rather than a pure engineering role?", "", [
        "Give an honest motivation: direct impact, variety, solving real problems end to end, working with people.",
        "Show you know the downsides (context switching, travel, ambiguity) and why they suit you.",
      ]),
      Q("t-behav", P, "Tell me about a time you found the customer was solving the wrong problem.", "", [
        "STAR. Show discovery questions, data you gathered, how you reframed the problem, and the better outcome.",
      ]),
      Q("t-behav", P, "How do you handle context switching across several customers?", "", [
        "Have a system: notes per account, weekly priorities, clear status documents, and protected focus time.",
      ]),
    ],
  };
})();

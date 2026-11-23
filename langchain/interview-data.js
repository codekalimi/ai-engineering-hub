// Content for interview.html.
// Each class: say = 30-second answer, methods = [signature, returns, note], qa = [question, answer, top?]
window.INTERVIEW = {
  groups: [
    { id: "core", title: "Core: Runnable & chat models" },
    { id: "prompts-output", title: "Messages, prompts & output" },
    { id: "rag-classes", title: "RAG classes" },
    { id: "tools-agents", title: "Tools & agents" },
    { id: "graph-classes", title: "LangGraph" },
    { id: "observability", title: "Callbacks, tracing & evaluation" },
  ],
  classes: [
    // ================= CORE =================
    {
      id: "runnable", group: "core", name: "Runnable", imp: `from langchain_core.runnables import Runnable`,
      say: `Runnable is the base interface of LangChain Core. Prompts, models, parsers, retrievers, tools and compiled graphs all implement it, so they share invoke, batch and stream, plus async versions. The pipe operator joins Runnables into a RunnableSequence, and every Runnable gets retries, fallbacks, config and tracing for free.`,
      methods: [
        [`invoke(input, config=None)`, `output`, `Run once and wait for the full result.`],
        [`batch(inputs, config=None)`, `list[output]`, `Many inputs in parallel (thread pool). Order kept. config={"max_concurrency": n}.`],
        [`stream(input)`, `iterator of chunks`, `Output piece by piece; lowers time to first token.`],
        [`ainvoke / abatch / astream`, `same, awaitable`, `Non-blocking versions for async servers (FastAPI).`],
        [`astream_events(input, version="v2")`, `async iterator of events`, `Every event from every inner step: tokens, tool starts, retriever results.`],
        [`a | b`, `RunnableSequence`, `Output of a becomes input of b. Dicts become RunnableParallel, functions become RunnableLambda.`],
        [`with_retry(stop_after_attempt=3)`, `Runnable`, `Retry on errors.`],
        [`with_fallbacks([backup])`, `Runnable`, `Try backups if the main one fails.`],
        [`with_config(run_name=, tags=, callbacks=)`, `Runnable`, `Attach config used for tracing and callbacks.`],
        [`bind(**kwargs)`, `Runnable`, `Fix arguments, e.g. model.bind(stop=["\\n"]).`],
        [`get_graph().draw_mermaid()`, `str`, `Visualize a chain or graph.`],
      ],
      qa: [
        [`What is a Runnable and why does LangChain use it?`, `It's one standard interface for every component. Because a prompt, a model and a retriever all have invoke, batch and stream, they can be composed with the pipe operator, and features like streaming, async, retries, fallbacks and tracing work on any of them, including whole chains.`, true],
        [`What's the difference between invoke, batch and stream?`, `invoke runs one input and returns the full output. batch runs a list of inputs in parallel with a thread pool, keeps the order, and lets you cap concurrency. stream yields chunks as they're generated, which lowers time to first token. Total time is similar; perceived latency is better with streaming.`, true],
        [`When do you use the async versions?`, `Inside async code such as a FastAPI server. An LLM call is mostly network wait, so ainvoke frees the event loop to serve other requests meanwhile. If a component has no native async, LangChain runs the sync version in a thread executor.`],
        [`What is LCEL?`, `LangChain Expression Language: composing Runnables declaratively with the pipe operator. a | b builds a RunnableSequence; a dict builds a RunnableParallel. The result is itself a Runnable, so it supports streaming, batching, async and tracing automatically.`, true],
        [`How do you make a chain resilient in production?`, `with_retry for transient errors like rate limits, with_fallbacks to switch to a backup model or provider, timeouts and max_retries on the model itself, and max_concurrency on batch to stay under rate limits.`],
      ],
      gotcha: `A plain Python function in a chain blocks streaming at that step, because it needs the whole input. Write it as a generator if you need streaming to continue.`,
    },
    {
      id: "chat-model", group: "core", name: "BaseChatModel", alias: "ChatAnthropic · ChatOpenAI · init_chat_model",
      imp: `from langchain_anthropic import ChatAnthropic\nfrom langchain.chat_models import init_chat_model`,
      say: `A chat model is a Runnable that takes a list of messages and returns an AIMessage. Each provider class converts LangChain messages to that provider's API format and back, so switching providers is a one-line change. The two methods that matter most beyond invoke are bind_tools and with_structured_output.`,
      methods: [
        [`ChatAnthropic(model=, temperature=, max_tokens=, timeout=, max_retries=)`, `chat model`, `Common constructor params across providers.`],
        [`invoke(messages | str)`, `AIMessage`, `A string is treated as one HumanMessage.`],
        [`stream(messages)`, `AIMessageChunk iterator`, `Chunks can be added together: full = c1 + c2 + ...`],
        [`bind_tools([tools], tool_choice=)`, `Runnable`, `Model may now reply with tool_calls. tool_choice can force a tool.`],
        [`with_structured_output(Schema, include_raw=False)`, `Runnable → Schema instance`, `Pydantic class, TypedDict or JSON schema. Uses tool calling or native JSON mode.`],
        [`init_chat_model("provider:model")`, `chat model`, `Provider-agnostic factory, handy for config-driven apps.`],
      ],
      qa: [
        [`How do you get reliable structured output from an LLM?`, `Use model.with_structured_output(MyPydanticModel). It sends the schema to the model, usually through tool calling or native JSON mode, then validates and returns an instance of the class. It's more reliable than asking for JSON in the prompt and parsing text. Field descriptions act as instructions, and include_raw=True lets you handle parsing failures.`, true],
        [`Chat model vs LLM class: what's the difference?`, `LLM classes are the older text-in, text-out completion interface. Chat models take a list of role-tagged messages and return an AIMessage, which is what modern models are built for and what tool calling and structured output rely on. Use chat models for new work.`],
        [`How would you switch from one provider to another?`, `Swap the class, e.g. ChatOpenAI for ChatAnthropic, or use init_chat_model with a different provider string. Prompts, parsers and chains don't change, because they only depend on the Runnable interface and the standard message types.`],
        [`What does temperature do?`, `It controls randomness when picking the next token. Near 0 gives focused, repeatable answers, good for extraction and classification. Higher values give more varied output for creative tasks.`],
      ],
      gotcha: `Keep API keys in environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY). Never hard-code them or commit them.`,
    },

    // ================= PROMPTS & OUTPUT =================
    {
      id: "messages", group: "prompts-output", name: "Messages", alias: "SystemMessage · HumanMessage · AIMessage · ToolMessage",
      imp: `from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, trim_messages`,
      say: `Messages are LangChain's provider-neutral chat format. SystemMessage sets behaviour, HumanMessage is the user, AIMessage is the model's reply including any tool calls and token usage, and ToolMessage carries a tool's result linked by tool_call_id. Memory is just keeping and re-sending this list.`,
      methods: [
        [`msg.content`, `str or list of blocks`, `The text (or multimodal blocks).`],
        [`ai_msg.tool_calls`, `list[dict]`, `[{"name", "args", "id"}] when the model wants a tool.`],
        [`ai_msg.usage_metadata`, `dict`, `input_tokens, output_tokens, total_tokens: track cost.`],
        [`ai_msg.response_metadata`, `dict`, `Provider details: model name, stop reason.`],
        [`ToolMessage(content, tool_call_id=)`, `ToolMessage`, `Tool result; the id must match the AIMessage's tool call.`],
        [`trim_messages(msgs, max_tokens=, strategy="last", token_counter=)`, `list[messages]`, `Keep history under the context limit.`],
      ],
      qa: [
        [`How is conversation memory implemented?`, `LLMs are stateless, so memory means re-sending previous messages on each call. In LangChain you keep a message list; in LangGraph a checkpointer saves the messages per thread_id automatically. For long chats you trim old messages or summarize them to control token cost.`, true],
        [`Why does a ToolMessage need tool_call_id?`, `The model can request several tool calls in one turn. The id links each result to the request that asked for it, so the model knows which answer belongs to which call. Most providers reject a tool result without a matching id.`],
        [`What's in an AIMessage besides the text?`, `tool_calls for requested tool executions, usage_metadata with token counts for cost tracking, response_metadata with provider details like the stop reason, and an id.`],
      ],
      gotcha: `Long histories cost more on every call, because the whole list is sent each time. Trim or summarize.`,
    },
    {
      id: "chat-prompt", group: "prompts-output", name: "ChatPromptTemplate", alias: "with MessagesPlaceholder",
      imp: `from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder`,
      say: `ChatPromptTemplate is a reusable list of role-tagged messages with {variables}. It's a Runnable that takes a dict and returns a PromptValue, so it's the first step of most chains. MessagesPlaceholder inserts a whole list of messages, usually the chat history.`,
      methods: [
        [`ChatPromptTemplate.from_messages([("system", "..."), ("user", "{q}")])`, `template`, `The usual constructor.`],
        [`ChatPromptTemplate.from_template("... {q}")`, `template`, `Single user message shortcut.`],
        [`prompt.invoke({"q": ...})`, `ChatPromptValue`, `Fill variables; feeds straight into a model.`],
        [`prompt.format_messages(q=...)`, `list[messages]`, `Same, returning plain messages.`],
        [`prompt.partial(role="tutor")`, `template`, `Pre-fill some variables now, the rest later.`],
        [`prompt.input_variables`, `list[str]`, `Which variables are required.`],
        [`MessagesPlaceholder("history")`, `placeholder`, `Slot for a list of messages.`],
      ],
      qa: [
        [`Why use prompt templates instead of f-strings?`, `Templates keep message roles, validate that required variables are present, are reusable and versionable, support partials, and are Runnables, so they plug into chains and show up in traces. An f-string gives you just a string with none of that.`, true],
        [`How do you add chat history to a prompt?`, `Put MessagesPlaceholder("history") between the system message and the new user message, then pass the list of previous messages under that key when invoking.`],
      ],
      gotcha: `Literal braces in a template, such as a JSON example, must be doubled: {{ and }}. Otherwise they're read as variables.`,
    },
    {
      id: "output-parsers", group: "prompts-output", name: "Output parsers", alias: "StrOutputParser · JsonOutputParser · PydanticOutputParser",
      imp: `from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser`,
      say: `Output parsers turn the model's AIMessage into something your code can use: a string, a dict or a Pydantic object. StrOutputParser is the most common final step. For structured data, with_structured_output on the model is usually preferred today, and parsers are the fallback for models without tool calling.`,
      methods: [
        [`StrOutputParser()`, `str`, `Returns .content; streams token by token.`],
        [`JsonOutputParser(pydantic_object=None)`, `dict`, `Parses JSON from text; streams partial JSON.`],
        [`PydanticOutputParser(pydantic_object=X)`, `X instance`, `Validates against your class.`],
        [`parser.get_format_instructions()`, `str`, `Text to put in the prompt describing the expected format.`],
        [`parser.parse(text)`, `parsed value`, `Parse a string directly.`],
      ],
      qa: [
        [`with_structured_output vs an output parser: which and why?`, `with_structured_output uses the model's native tool calling or JSON mode, so the schema is enforced at generation time and is more reliable. Output parsers work on plain text after generation, which is useful for models without those features or when streaming partial JSON, but they fail more often on malformed output.`, true],
      ],
      gotcha: `If a parser fails on malformed output, retry with the error message included in the prompt, or switch to with_structured_output.`,
    },
    {
      id: "runnable-helpers", group: "prompts-output", name: "RunnablePassthrough · RunnableParallel · RunnableLambda",
      imp: `from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda`,
      say: `These are the glue classes of LCEL. RunnableParallel runs several Runnables on the same input and returns a dict. RunnablePassthrough forwards the input unchanged, or adds keys with assign. RunnableLambda wraps any Python function as a Runnable.`,
      methods: [
        [`RunnableParallel(a=r1, b=r2)` , `dict`, `Same input to all, run concurrently. A dict literal in a chain does the same.`],
        [`RunnablePassthrough()`, `input unchanged`, `Forward the question alongside retrieved context.`],
        [`RunnablePassthrough.assign(key=fn)`, `input + new key`, `Add a computed field, keep the rest.`],
        [`RunnableLambda(fn)`, `fn(input)`, `Custom step; a bare function in a chain is wrapped automatically.`],
        [`RunnableBranch((cond, r1), default)`, `output of chosen branch`, `Simple if/else routing in LCEL. For anything complex, use LangGraph.`],
      ],
      qa: [
        [`In a RAG chain, how does the question reach both the retriever and the prompt?`, `With a dict at the start of the chain: {"context": retriever, "question": RunnablePassthrough()}. LangChain turns it into a RunnableParallel, so the same input goes to the retriever, which returns documents, and to the passthrough, which returns the question unchanged. The resulting dict fills the prompt.`, true],
      ],
      gotcha: `Format retrieved documents into a string (e.g. retriever | format_docs) before they reach the prompt, or the prompt receives a list of Document objects.`,
    },

    // ================= RAG =================
    {
      id: "document-loaders", group: "rag-classes", name: "Document · BaseLoader", alias: "PyPDFLoader · WebBaseLoader · DirectoryLoader",
      imp: `from langchain_core.documents import Document\nfrom langchain_community.document_loaders import PyPDFLoader`,
      say: `A Document is page_content plus a metadata dict. Loaders read sources such as PDFs, web pages or databases into Documents. Metadata like the source and page is what you later use for citations and filtering.`,
      methods: [
        [`Document(page_content=, metadata={})`, `Document`, `The unit of text in RAG.`],
        [`loader.load()`, `list[Document]`, `Load everything into memory.`],
        [`loader.lazy_load()`, `iterator[Document]`, `One at a time; use for large datasets.`],
      ],
      qa: [
        [`How would you index 100,000 documents without running out of memory?`, `Use lazy_load to stream documents, split and embed them in batches, and add each batch with vector_store.add_documents. Store stable IDs in metadata so you can update or delete documents later instead of re-indexing everything.`],
      ],
      gotcha: `Scanned PDFs have no text layer. You need OCR before loading, otherwise page_content is empty.`,
    },
    {
      id: "text-splitter", group: "rag-classes", name: "RecursiveCharacterTextSplitter",
      imp: `from langchain_text_splitters import RecursiveCharacterTextSplitter`,
      say: `It splits text into chunks of a target size, trying separators in order: paragraphs, then lines, then spaces, then characters. That keeps related text together. Overlap repeats a little text between chunks so an idea that crosses a boundary isn't lost.`,
      methods: [
        [`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`, `splitter`, `Sizes are in characters by default.`],
        [`split_documents(docs)`, `list[Document]`, `Keeps metadata on each chunk.`],
        [`split_text(text)`, `list[str]`, `Plain strings.`],
        [`from_tiktoken_encoder(chunk_size=, chunk_overlap=)`, `splitter`, `Measure size in tokens instead of characters.`],
        [`separators=["\\n\\n", "\\n", " ", ""]`, `(the default)`, `Override for code or markdown; language-specific splitters exist too.`],
      ],
      qa: [
        [`How do you choose chunk size and overlap?`, `Small chunks give precise matches but lose context; large chunks keep context but dilute the embedding and use more prompt tokens. A common start is 500 to 1000 characters with 10 to 20 percent overlap. Then tune it by checking what the retriever returns on real questions, and match the structure of the content, such as splitting markdown by headings.`, true],
        [`Why split documents at all?`, `Embedding a whole document blurs many topics into one vector, so search gets less precise, and whole documents may not fit in the context window. Chunks give focused vectors and let you send only the relevant parts.`],
      ],
      gotcha: `chunk_size counts characters unless you use from_tiktoken_encoder. Models have token limits, not character limits.`,
    },
    {
      id: "embeddings", group: "rag-classes", name: "Embeddings", alias: "OpenAIEmbeddings · HuggingFaceEmbeddings",
      imp: `from langchain_openai import OpenAIEmbeddings`,
      say: `An embeddings class turns text into vectors so that similar meanings are close together. It has two methods: embed_documents for indexing many texts and embed_query for the user's question.`,
      methods: [
        [`embed_documents(list[str])`, `list[list[float]]`, `Batch-embed chunks at indexing time.`],
        [`embed_query(str)`, `list[float]`, `Embed the question at search time.`],
        [`aembed_documents / aembed_query`, `same, awaitable`, `Async versions.`],
      ],
      qa: [
        [`Why are there separate embed_query and embed_documents methods?`, `Some embedding models are trained to encode questions and passages slightly differently, and documents are usually embedded in large batches while a query is a single call. Separate methods allow both.`],
        [`What is semantic search, and how is it different from keyword search?`, `Semantic search compares embedding vectors, usually by cosine similarity, so it matches meaning: "money back" finds "refund policy". Keyword search like BM25 matches exact words, which is better for names, codes and IDs. Hybrid search combines both.`, true],
      ],
      gotcha: `You must use the same embedding model for indexing and querying. Vectors from different models aren't comparable.`,
    },
    {
      id: "vector-store", group: "rag-classes", name: "VectorStore", alias: "Chroma · FAISS · PGVector · Pinecone",
      imp: `from langchain_chroma import Chroma`,
      say: `A vector store saves embeddings together with their text and metadata, and finds the nearest vectors to a query quickly. Every vector store shares one interface: add documents, run similarity search, and turn the store into a retriever.`,
      methods: [
        [`Chroma.from_documents(docs, embeddings)`, `store`, `Embed and store in one step.`],
        [`add_documents(docs, ids=)`, `list[id]`, `Add more later; pass ids to update.`],
        [`similarity_search(query, k=4, filter=)`, `list[Document]`, `Top-k closest chunks; optional metadata filter.`],
        [`similarity_search_with_score(query, k=)`, `list[(Document, score)]`, `Inspect how close matches are.`],
        [`max_marginal_relevance_search(query, k=, fetch_k=)`, `list[Document]`, `Relevant but diverse results.`],
        [`delete(ids=[...])`, `—`, `Remove outdated chunks.`],
        [`as_retriever(search_type=, search_kwargs=)`, `VectorStoreRetriever`, `Make it a Runnable for chains.`],
      ],
      qa: [
        [`What's the difference between similarity search and MMR?`, `Similarity search returns the k closest chunks, which are often near-duplicates of each other. MMR, maximal marginal relevance, fetches more candidates (fetch_k) and then picks ones that are relevant and also different from each other, so the context covers more ground.`, true],
        [`Which vector store would you choose?`, `Chroma or FAISS for prototypes and local work. In production, pgvector if you already run Postgres and want one database, or a managed service like Pinecone, Qdrant or Weaviate for scale, filtering and hybrid search. The LangChain interface is the same, so switching is cheap.`],
      ],
      gotcha: `Scores mean different things in different stores: some return distance (lower is better), others similarity (higher is better). Check before setting a threshold.`,
    },
    {
      id: "retriever", group: "rag-classes", name: "BaseRetriever", alias: "VectorStoreRetriever and advanced retrievers",
      imp: `retriever = vector_store.as_retriever(search_kwargs={"k": 4})`,
      say: `A retriever is any Runnable that takes a query string and returns a list of Documents. A vector store retriever is the usual one, but retrievers can also wrap keyword search, web search or several retrievers combined. That's why it, not the vector store, is what goes in the chain.`,
      methods: [
        [`retriever.invoke(query)`, `list[Document]`, `Standard Runnable call.`],
        [`as_retriever(search_type="similarity")`, `retriever`, `Plain top-k.`],
        [`as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})`, `retriever`, `Diverse results.`],
        [`as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})`, `retriever`, `Only return good-enough matches.`],
        [`search_kwargs={"filter": {"source": "hr.pdf"}}`, `—`, `Metadata filtering.`],
      ],
      qa: [
        [`Your RAG app gives wrong answers. How do you debug it?`, `First separate retrieval from generation: log the retrieved chunks for the failing question. If the right chunk isn't there, fix retrieval with better chunking, more k, hybrid keyword plus vector search, metadata filters, query rewriting or a reranker. If the right chunk is there but the answer is wrong, fix the prompt, for example by telling the model to answer only from the context and to cite sources. Use LangSmith traces and an evaluation set to measure each change.`, true],
        [`How can you improve retrieval quality?`, `Tune chunking; use hybrid search to combine BM25 and vectors; rewrite or expand the query (multi-query); rerank the top results with a cross-encoder; filter by metadata; and use parent-document retrieval to search small chunks but return larger surrounding context.`, true],
      ],
      gotcha: `More chunks isn't always better: irrelevant context can distract the model and costs tokens.`,
    },

    // ================= TOOLS & AGENTS =================
    {
      id: "tool", group: "tools-agents", name: "BaseTool · @tool", alias: "StructuredTool",
      imp: `from langchain_core.tools import tool`,
      say: `A tool is a function plus a name, a description and an argument schema that the model can read. The @tool decorator builds all three from the function name, docstring and type hints. The model never runs the tool. It returns a tool call, and your code or an agent executes it.`,
      methods: [
        [`@tool`, `BaseTool`, `Name from the function, description from the docstring, schema from type hints.`],
        [`@tool("name", args_schema=MyModel)`, `BaseTool`, `Custom name and Pydantic schema with field descriptions.`],
        [`tool.name / tool.description / tool.args`, `str / str / dict`, `What the model sees.`],
        [`tool.invoke({"city": "Paris"})`, `result`, `Run it directly, e.g. in tests.`],
        [`StructuredTool.from_function(fn)`, `BaseTool`, `Build a tool without the decorator.`],
      ],
      qa: [
        [`How does an LLM decide which tool to call?`, `The tool schemas (name, description, arguments) are sent with the request. The model reads them like part of the prompt and, if a tool fits, returns a structured tool call with arguments matching the schema. So the quality of the name and description directly decides whether the right tool is picked.`, true],
        [`What's the difference between a tool and a utility?`, `A utility, such as WikipediaAPIWrapper, is the low-level client that talks to a service. A tool wraps a capability with a name, description and schema so an LLM can call it. Many built-in tools hold a utility as api_wrapper.`],
        [`How do you keep tool use safe?`, `Validate arguments, give tools the smallest permissions they need, require human approval for side effects like sending email or payments (interrupt in LangGraph), set timeouts, and never pass model output straight into shell or SQL without checks.`],
      ],
      gotcha: `Return errors as text from a tool instead of crashing, so the model can read the error and try again.`,
    },
    {
      id: "create-agent", group: "tools-agents", name: "create_agent", alias: "replaces the older AgentExecutor",
      imp: `from langchain.agents import create_agent`,
      say: `create_agent builds a ready-made tool-calling agent on top of LangGraph: the model is called, any requested tools run, the results go back, and this repeats until the model answers without a tool call. It returns a compiled graph, so it supports streaming, memory through a checkpointer, and human-in-the-loop.`,
      methods: [
        [`create_agent(model, tools=[...], system_prompt="...")`, `compiled graph`, `The minimum.`],
        [`create_agent(..., checkpointer=InMemorySaver())`, `compiled graph`, `Memory per thread_id.`],
        [`create_agent(..., response_format=MySchema)`, `compiled graph`, `Structured final answer.`],
        [`agent.invoke({"messages": [...]}, config)`, `{"messages": [...]}`, `Final state; the answer is the last message.`],
        [`agent.stream(..., stream_mode="updates")`, `iterator`, `See each step as it happens.`],
        [`config={"recursion_limit": 25}`, `—`, `Cap the number of steps.`],
      ],
      qa: [
        [`What is an agent, and how is it different from a chain?`, `A chain follows a fixed path the developer wrote. An agent uses the LLM to decide the path at run time: which tool to call, with what arguments, and when to stop. It's a loop of reason, act and observe until the task is done.`, true],
        [`How do you stop an agent from looping forever or getting expensive?`, `Set recursion_limit to cap steps, write clear tool descriptions and a focused system prompt, return useful errors from tools, limit the number of tools, add timeouts, and monitor steps and tokens in LangSmith.`, true],
        [`What replaced AgentExecutor?`, `In LangChain v1 the recommended agent is create_agent, which is built on LangGraph. It gives you persistence, streaming, human-in-the-loop and middleware hooks that AgentExecutor didn't have. AgentExecutor lives on in the legacy package.`],
      ],
      gotcha: `More tools means more confusion. If an agent needs many tools, group them or split the work across specialized agents.`,
    },

    // ================= LANGGRAPH =================
    {
      id: "state-graph", group: "graph-classes", name: "StateGraph", alias: "START · END · add_messages",
      imp: `from langgraph.graph import StateGraph, START, END, MessagesState`,
      say: `StateGraph is the LangGraph builder. You define a state schema, add nodes (functions that return state updates) and edges (fixed or conditional), then compile. Conditional edges are what make loops and branching possible, which a linear chain can't do.`,
      methods: [
        [`StateGraph(StateSchema)`, `builder`, `State is usually a TypedDict or MessagesState.`],
        [`add_node("name", fn)`, `builder`, `fn(state) returns a dict of updates.`],
        [`add_edge("a", "b")`, `builder`, `Always go from a to b. Use START and END.`],
        [`add_conditional_edges("a", route_fn)`, `builder`, `route_fn(state) returns the next node name.`],
        [`compile(checkpointer=, store=, interrupt_before=[...])`, `CompiledStateGraph`, `A Runnable app.`],
        [`Annotated[list, add_messages]`, `reducer`, `How updates merge: append instead of overwrite.`],
      ],
      qa: [
        [`LangChain vs LangGraph: what's the difference?`, `LangChain provides the building blocks: model integrations, prompts, tools, retrievers, and LCEL for linear chains. LangGraph is a lower-level orchestration framework for stateful, cyclical workflows, built from state, nodes and edges. It adds loops, persistence through checkpointers, human-in-the-loop and durable execution. They work together: nodes usually use LangChain components, and create_agent is built on LangGraph.`, true],
        [`What is a reducer in LangGraph?`, `A function that decides how a node's update is merged into the existing state for a key. By default the new value overwrites the old one. With add_messages, new messages are appended (and matched by id), which is what you want for chat history.`, true],
        [`When would you use LangGraph instead of a chain?`, `When the flow isn't a straight line: loops like retry or agent tool loops, branching on intermediate results, multiple agents, long-running tasks that must survive restarts, or steps that pause for human approval.`],
      ],
      gotcha: `A node returns only the keys it changes. Returning the whole state can duplicate messages when a reducer appends.`,
    },
    {
      id: "compiled-graph", group: "graph-classes", name: "CompiledStateGraph", alias: "the app returned by compile()",
      imp: `app = graph.compile(checkpointer=InMemorySaver())`,
      say: `The compiled graph is a Runnable, so it has invoke, stream and batch. It also has methods to inspect and change saved state: get_state, get_state_history and update_state. Streaming supports several modes, depending on whether you want full state, updates per node or LLM tokens.`,
      methods: [
        [`invoke(input, config)`, `final state`, `config={"configurable": {"thread_id": "1"}}`],
        [`stream(input, config, stream_mode="values")`, `full state after each step`, `For showing the state as it changes.`],
        [`stream(..., stream_mode="updates")`, `{node: update}`, `What each node changed.`],
        [`stream(..., stream_mode="messages")`, `(token chunk, metadata)`, `LLM tokens for chat UIs.`],
        [`get_state(config)`, `StateSnapshot`, `.values, .next (what runs next), .tasks.`],
        [`get_state_history(config)`, `iterator[StateSnapshot]`, `Every checkpoint, newest first: time travel.`],
        [`update_state(config, values, as_node=)`, `config`, `Edit state by hand, e.g. after human review.`],
        [`get_graph().draw_mermaid_png()`, `bytes`, `Picture of the graph.`],
      ],
      qa: [
        [`How do you stream from a LangGraph app?`, `Use stream or astream with a stream_mode. "values" gives the full state after every step, "updates" gives only what each node changed, and "messages" gives LLM tokens as they're generated, for chat UIs. You can pass several modes at once.`, true],
        [`What is time travel in LangGraph?`, `Because every step is checkpointed, get_state_history lists past states. You can resume from an older checkpoint, optionally after changing it with update_state, to replay or fork a run. It's useful for debugging and "undo".`],
      ],
      gotcha: `Without a checkpointer, get_state and interrupts don't work, because there's nothing saved.`,
    },
    {
      id: "checkpointer", group: "graph-classes", name: "Checkpointer · Store", alias: "InMemorySaver · SqliteSaver · PostgresSaver · InMemoryStore",
      imp: `from langgraph.checkpoint.memory import InMemorySaver\nfrom langgraph.store.memory import InMemoryStore`,
      say: `A checkpointer saves the graph state after every step, keyed by thread_id. That gives short-term memory within a conversation, crash recovery and human-in-the-loop. A Store is separate: it holds long-term memory shared across threads, like user preferences.`,
      methods: [
        [`InMemorySaver()`, `checkpointer`, `For learning and tests; lost on restart.`],
        [`SqliteSaver / PostgresSaver`, `checkpointer`, `Durable; separate packages (langgraph-checkpoint-sqlite / -postgres).`],
        [`{"configurable": {"thread_id": "user-42"}}`, `config`, `One thread = one conversation.`],
        [`InMemoryStore()` , `store`, `Long-term key-value memory; compile(store=store).`],
        [`store.put(("users", uid), "prefs", {...})`, `—`, `Save under a namespace.`],
        [`store.search(("users", uid), query="...")`, `list[Item]`, `Find memories, optionally by meaning.`],
      ],
      qa: [
        [`Short-term vs long-term memory in LangGraph?`, `Short-term memory is the conversation state of one thread, saved by the checkpointer and keyed by thread_id. Long-term memory is information that should persist across conversations, like a user's preferences or facts, saved in a Store under namespaces and available from any thread.`, true],
        [`What happens if the server crashes halfway through a graph run?`, `With a durable checkpointer like Postgres, the state after the last completed step is saved. Invoking again with the same thread_id resumes from there instead of starting over.`],
      ],
      gotcha: `Reusing the same thread_id for different users mixes their conversations. Generate one per user session.`,
    },
    {
      id: "interrupt", group: "graph-classes", name: "interrupt · Command", alias: "human in the loop",
      imp: `from langgraph.types import interrupt, Command`,
      say: `interrupt pauses a graph inside a node and returns a payload to the caller, such as a draft to approve. The state is saved by the checkpointer, so the run can wait for any length of time. You resume by invoking with Command(resume=value), and that value becomes interrupt's return value.`,
      methods: [
        [`interrupt(payload)`, `the resume value`, `Call inside a node; pauses the run.`],
        [`result["__interrupt__"]`, `list[Interrupt]`, `What the caller receives while paused.`],
        [`Command(resume=value)`, `command`, `Pass to invoke to continue.`],
        [`Command(goto="node", update={...})`, `command`, `Returned from a node: update state and jump.`],
        [`compile(interrupt_before=["tools"])`, `—`, `Static breakpoints, mainly for debugging.`],
      ],
      qa: [
        [`How do you add human approval to an agent?`, `Call interrupt inside the node before the risky action, for example before sending an email, passing the proposed action. The graph pauses and the state is checkpointed. The app shows it to a person, then calls invoke with Command(resume=decision) on the same thread_id, and the node continues with that decision: approve, edit or reject.`, true],
      ],
      gotcha: `When resumed, the node runs again from its start. Keep side effects after the interrupt call, or make them safe to repeat.`,
    },

    // ================= OBSERVABILITY =================
    {
      id: "callbacks", group: "observability", name: "Callbacks · LangSmith", alias: "BaseCallbackHandler · @traceable · evaluate",
      imp: `from langchain_core.callbacks import BaseCallbackHandler\nfrom langsmith import traceable`,
      say: `Every Runnable emits callback events, such as start, new token, end and error, each with a run id and parent id. LangSmith uses them to record a full trace tree with inputs, outputs, latency and token cost. You turn it on with environment variables, and you can use LangSmith datasets and evaluators to test quality over time.`,
      methods: [
        [`LANGSMITH_TRACING=true, LANGSMITH_API_KEY=...`, `—`, `Turns on tracing with no code changes.`],
        [`@traceable`, `decorator`, `Trace your own non-LangChain functions too.`],
        [`BaseCallbackHandler.on_llm_start / on_llm_new_token / on_llm_end`, `hooks`, `Custom logging, metrics, streaming.`],
        [`on_tool_start / on_tool_end / on_chain_error`, `hooks`, `Track tools and failures.`],
        [`chain.invoke(x, config={"callbacks": [handler]})`, `—`, `Attach a handler for one run.`],
        [`langsmith.evaluate(target, data=, evaluators=)`, `results`, `Run a dataset through your app and score it.`],
      ],
      qa: [
        [`How do you monitor and debug an LLM app in production?`, `Trace every run with LangSmith or a similar tool to see each step's input, output, latency and tokens. Log user feedback, track cost and error rates, and set alerts. When something goes wrong, open the trace for that request to find the step that failed.`, true],
        [`How do you evaluate an LLM or RAG application?`, `Build a dataset of real questions with reference answers, run the app over it, and score outputs with evaluators: exact checks where possible, LLM-as-judge for correctness and faithfulness, and retrieval metrics like whether the right chunk was found. Re-run it after every prompt, model or chunking change to catch regressions.`, true],
      ],
      gotcha: `Traces contain user data. Mask sensitive fields or turn tracing off for private inputs.`,
    },
  ],
};

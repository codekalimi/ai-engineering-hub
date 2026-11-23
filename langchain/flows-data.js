// Data for the step-through flows on under-the-hood.html
const N = (id, x, y, label, sub, o = {}) => ({ id, x, y, label, sub, w: o.w || 140, h: o.h || 60, io: !!o.io });
const E = (id, from, to, o = {}) => ({ id, from, to, ...o });

const FLOWS = [
{
  id: "lcel", tab: "The pipe", title: "1. Runnables and the | pipe",
  intro: "Every piece in LangChain (prompt, model, parser, retriever, tool) is a Runnable: a block with the same plug. invoke() runs one input, batch() runs many, stream() returns pieces as they arrive. The | pipe snaps blocks together.",
  vb: [800, 150],
  nodes: [
    N("input", 10, 45, "Input", "a dict", { w: 120, io: true }),
    N("prompt", 165, 45, "Prompt", "PromptTemplate"),
    N("model", 340, 45, "Chat model", "ChatAnthropic"),
    N("parser", 515, 45, "Parser", "StrOutputParser"),
    N("output", 690, 45, "Output", "a string", { w: 100, io: true }),
  ],
  edges: [
    E("e1", ["input", "r"], ["prompt", "l"]), E("e2", ["prompt", "r"], ["model", "l"]),
    E("e3", ["model", "r"], ["parser", "l"]), E("e4", ["parser", "r"], ["output", "l"]),
  ],
  steps: [
    { on: ["prompt", "model", "parser"], edges: [], title: "You build the chain",
      text: "Writing prompt | model | parser runs nothing yet. The | operator only builds a RunnableSequence object that remembers the order of the blocks.",
      code: `from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model  = ChatAnthropic(model="claude-sonnet-5-5")
parser = StrOutputParser()

chain = prompt | model | parser
# chain is RunnableSequence(prompt, model, parser)` },
    { on: ["input"], edges: [], title: "You call invoke()",
      text: "chain.invoke() hands your input to the first block. Each block then gets the previous block's output, one after another.",
      code: `chain.invoke({"topic": "cats"})

# in the pipe now:
{"topic": "cats"}` },
    { on: ["prompt"], edges: ["e1"], title: "The prompt fills in the template",
      text: "The template replaces {topic} with \"cats\". Its output is not a string. It is a list of chat messages, because chat models take messages.",
      code: `ChatPromptValue(messages=[
  HumanMessage(content="Tell me a joke about cats")
])` },
    { on: ["model"], edges: ["e2"], title: "The model calls the API",
      text: "The chat model converts the messages into the provider's JSON, sends an HTTP request, waits, and wraps the reply in an AIMessage. The reply also carries token usage and metadata.",
      code: `AIMessage(
  content="Why did the cat sit on the computer? To keep an eye on the mouse.",
  usage_metadata={"input_tokens": 14, "output_tokens": 18, ...},
  response_metadata={"model": "claude-sonnet-5-5", "stop_reason": "end_turn"}
)` },
    { on: ["parser"], edges: ["e3"], title: "The parser cleans up",
      text: "StrOutputParser takes the AIMessage and keeps only .content. Other parsers turn the text into JSON or a Pydantic object instead.",
      code: `"Why did the cat sit on the computer? To keep an eye on the mouse."` },
    { on: ["output"], edges: ["e4"], title: "You get the result",
      text: "That's the whole trick: the sequence loops over its blocks and feeds each output into the next. The finished chain is also a Runnable, so .stream() and .batch() work on it, and you can pipe it into a bigger chain.",
      code: `for chunk in chain.stream({"topic": "cats"}):
    print(chunk, end="")     # "Why" " did" " the" ...

chain.batch([{"topic": "cats"}, {"topic": "dogs"}])   # runs in parallel` },
  ],
  oneLiner: "Every component is a Runnable with invoke, batch and stream. The | operator builds a RunnableSequence that feeds each block's output into the next, and the result is itself a Runnable."
},
{
  id: "messages", tab: "Messages & memory", title: "2. Messages and memory",
  intro: "Chat models don't take plain strings. They take a list of messages, and the model itself remembers nothing between calls.",
  vb: [800, 160],
  nodes: [
    N("history", 10, 40, "Message list", "kept by your app", { w: 200, h: 80 }),
    N("adapter", 300, 40, "LangChain chat model", "translates the format", { w: 200, h: 80 }),
    N("provider", 590, 40, "Provider API", "Claude / OpenAI / Gemini", { w: 200, h: 80 }),
  ],
  edges: [
    E("h2a", ["history", "r", -16], ["adapter", "l", -16]), E("a2p", ["adapter", "r", -16], ["provider", "l", -16]),
    E("p2a", ["provider", "l", 16], ["adapter", "r", 16]), E("a2h", ["adapter", "l", 16], ["history", "r", 16]),
  ],
  labels: [{ x: 255, y: 30, t: "request" }, { x: 545, y: 30, t: "request" }, { x: 255, y: 140, t: "reply" }, { x: 545, y: 140, t: "reply" }],
  steps: [
    { on: ["history"], edges: [], title: "Your app builds a message list",
      text: "Each message has a role. SystemMessage sets behaviour, HumanMessage is the user, AIMessage is the model, and ToolMessage is a tool result.",
      code: `messages = [
  SystemMessage("You are a helpful math tutor."),
  HumanMessage("What's 2+2?"),
]` },
    { on: ["adapter"], edges: ["h2a"], title: "LangChain translates the format",
      text: "Every provider wants a slightly different JSON shape. The chat model class (ChatAnthropic, ChatOpenAI and so on) converts LangChain messages into that shape. This is why you can swap providers by changing one line.",
      code: `{
  "model": "claude-sonnet-5-5",
  "system": "You are a helpful math tutor.",
  "messages": [
    {"role": "user", "content": "What's 2+2?"}
  ]
}` },
    { on: ["provider"], edges: ["a2p"], title: "The provider runs the LLM",
      text: "The LLM reads the whole list and predicts the next message. When the request ends, the provider keeps nothing. Every call starts from zero.",
      code: `{
  "role": "assistant",
  "content": [{"type": "text", "text": "2 + 2 = 4"}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 21, "output_tokens": 9}
}` },
    { on: ["adapter"], edges: ["p2a"], title: "The reply becomes an AIMessage",
      text: "LangChain converts the provider's JSON back into a standard AIMessage, so your code looks the same whichever model answered.",
      code: `AIMessage(content="2 + 2 = 4", usage_metadata={...})` },
    { on: ["history"], edges: ["a2h"], title: "Memory is your list getting longer",
      text: "You append the AIMessage to the list. On the next question you send the whole list again. That is all memory is: re-sending the history. Long chats cost more because the full history goes out every time.",
      code: `messages = [
  SystemMessage("You are a helpful math tutor."),
  HumanMessage("What's 2+2?"),
  AIMessage("2 + 2 = 4"),
  HumanMessage("Times 3?"),   # the model only knows "4" because we re-send it
]` },
  ],
  oneLiner: "LLMs are stateless. LangChain standardizes messages across providers, and memory is just re-sending the conversation history on every call, sometimes trimmed or summarized to save tokens."
},
{
  id: "tools", tab: "Tool calling", title: "3. Tool calling",
  intro: "The LLM never runs your code. It can only ask for a tool call. LangChain runs the function and sends the result back.",
  vb: [800, 260],
  nodes: [
    N("app", 10, 30, "Your app + LangChain", "runs the code", { w: 230, h: 80 }),
    N("llm", 560, 30, "LLM", "only writes text and requests", { w: 230, h: 80 }),
    N("tool", 10, 185, "get_weather()", "your Python function", { w: 230, h: 64 }),
  ],
  edges: [
    E("a2l", ["app", "r", -16], ["llm", "l", -16]), E("l2a", ["llm", "l", 16], ["app", "r", 16]),
    E("a2t", ["app", "b", -45], ["tool", "t", -45]), E("t2a", ["tool", "t", 45], ["app", "b", 45]),
  ],
  labels: [{ x: 400, y: 42, t: "messages + tool schemas" }, { x: 400, y: 112, t: "AIMessage" }, { x: 140, y: 150, t: "args → result" }],
  steps: [
    { on: ["tool", "app"], edges: [], title: "Turn a function into a tool",
      text: "The @tool decorator reads the function name, docstring and type hints and builds a JSON schema. The docstring becomes the description the LLM reads to decide when to use the tool.",
      code: `@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return weather_api.lookup(city)

llm_with_tools = llm.bind_tools([get_weather])

# what the model will see:
{"name": "get_weather",
 "description": "Get the current weather for a city.",
 "input_schema": {"type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]}}` },
    { on: ["llm"], edges: ["a2l"], title: "Send the question plus the tool list",
      text: "The request carries the user's question and the schema of every bound tool.",
      code: `llm_with_tools.invoke("What's the weather in Paris?")

# request = your messages + all tool schemas` },
    { on: ["app"], edges: ["l2a"], title: "The LLM asks for a call",
      text: "The model decides a tool would help. It doesn't answer yet. It returns an AIMessage with tool_calls: the tool name and arguments, written to match your schema.",
      code: `AIMessage(
  content="",
  tool_calls=[{"name": "get_weather",
               "args": {"city": "Paris"},
               "id": "toolu_01A"}]
)` },
    { on: ["tool"], edges: ["a2t", "t2a"], title: "Your code runs the tool",
      text: "LangChain (or your own loop) looks up the tool by name and calls it with those arguments. This is the only place real code runs.",
      code: `result = get_weather.invoke({"city": "Paris"})
# "18°C, sunny"` },
    { on: ["llm"], edges: ["a2l"], title: "Send the result back",
      text: "The result is wrapped in a ToolMessage with the same id, so the model can match it to its request. The whole history goes out again.",
      code: `[
  HumanMessage("What's the weather in Paris?"),
  AIMessage(tool_calls=[get_weather(city="Paris")]),
  ToolMessage("18°C, sunny", tool_call_id="toolu_01A"),
]` },
    { on: ["app"], edges: ["l2a"], title: "The LLM writes the final answer",
      text: "Now the model has the fact it needed and answers in plain text. If it needed more, it would ask for another tool call. Repeat that and you have an agent (flow 4).",
      code: `AIMessage(content="It's 18°C and sunny in Paris right now.")` },
  ],
  oneLiner: "The model only returns a structured tool-call request. The framework runs the function and sends the result back as a ToolMessage. The tool's name and description decide whether the model picks it."
},
{
  id: "agent", tab: "Agent loop", title: "4. The agent loop (LangGraph)",
  intro: "An agent is the tool-calling loop repeated until the LLM stops asking for tools. LangGraph draws that loop as a graph: nodes do the work, edges decide where to go next, and a shared state (mostly the message list) moves between them.",
  vb: [800, 260],
  nodes: [
    N("start", 10, 100, "START", "", { w: 100, h: 56, io: true }),
    N("agent", 190, 90, "agent node", "calls the LLM", { w: 200, h: 76 }),
    N("tools", 540, 20, "tools node", "runs requested tools", { w: 200, h: 76 }),
    N("end", 580, 185, "END", "", { w: 120, h: 56, io: true }),
  ],
  edges: [
    E("s2a", ["start", "r"], ["agent", "l"]),
    E("a2t", ["agent", "t", 30], ["tools", "l", -10]),
    E("t2a", ["tools", "b", 0], ["agent", "r", -12]),
    E("a2e", ["agent", "r", 22], ["end", "l", 0]),
  ],
  labels: [{ x: 360, y: 30, t: "has tool_calls" }, { x: 470, y: 236, t: "no tool_calls" }, { x: 600, y: 140, t: "always back" }],
  steps: [
    { on: ["start"], edges: ["s2a"], title: "The question enters the graph",
      text: "The graph starts with a state object. Every node reads the state and returns updates to it.",
      code: `state = {"messages": [
  HumanMessage("Which is warmer right now, Paris or Tokyo?")
]}` },
    { on: ["agent"], edges: [], title: "The agent node calls the LLM",
      text: "The LLM sees the question and the tools. It needs two facts, so it asks for two tool calls at once.",
      code: `AIMessage(tool_calls=[
  {"name": "get_weather", "args": {"city": "Paris"}},
  {"name": "get_weather", "args": {"city": "Tokyo"}},
])` },
    { on: ["tools"], edges: ["a2t"], title: "A conditional edge picks the path",
      text: "After the agent node, a small function checks the last message. It has tool_calls, so the graph goes to the tools node.",
      code: `def should_continue(state):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END` },
    { on: ["agent"], edges: ["t2a"], title: "Tools run, results go back",
      text: "The tools node runs both calls and adds two ToolMessages to the state. A normal edge always sends it back to the agent.",
      code: `state["messages"] += [
  ToolMessage("Paris: 18°C, sunny"),
  ToolMessage("Tokyo: 24°C, cloudy"),
]` },
    { on: ["agent"], edges: [], title: "The agent thinks again",
      text: "With both results in the message list, the LLM can answer. This time it returns plain text with no tool_calls.",
      code: `AIMessage(content="Tokyo is warmer: 24°C versus 18°C in Paris.")` },
    { on: ["end"], edges: ["a2e"], title: "No tool calls means stop",
      text: "should_continue sees no tool_calls and goes to END. A recursion limit stops a loop that never finishes. Here is how the graph is wired:",
      code: `graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode([get_weather]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
app = graph.compile()

# shortcut that builds the same loop:
# create_agent(model, tools=[get_weather])` },
  ],
  oneLiner: "An agent is a loop: the LLM picks a tool, the tool runs, the result goes back, and it repeats until the LLM answers without a tool call. LangGraph models it as nodes, edges and shared state, which makes it controllable and resumable."
},
{
  id: "rag", tab: "RAG", title: "5. RAG: answering from your documents",
  intro: "RAG gives the LLM your own documents. The top row prepares them once. The bottom row runs on every question.",
  vb: [800, 290],
  nodes: [
    N("docs", 10, 30, "Documents", "PDF, web, DB", { w: 135 }),
    N("loader", 172, 30, "Loader", "→ Document objects", { w: 135 }),
    N("splitter", 334, 30, "Splitter", "→ chunks", { w: 135 }),
    N("embed", 496, 30, "Embeddings", "text → numbers", { w: 135 }),
    N("store", 658, 30, "Vector store", "Chroma, FAISS…", { w: 132 }),
    N("question", 10, 210, "Question", "", { w: 135, io: true }),
    N("retriever", 172, 210, "Retriever", "find similar", { w: 135 }),
    N("prompt", 334, 210, "Prompt", "context + question", { w: 135 }),
    N("llm", 496, 210, "LLM", "writes answer", { w: 135 }),
    N("answer", 658, 210, "Answer", "", { w: 132, io: true }),
  ],
  edges: [
    E("d2l", ["docs", "r"], ["loader", "l"]), E("l2s", ["loader", "r"], ["splitter", "l"]),
    E("s2e", ["splitter", "r"], ["embed", "l"]), E("e2v", ["embed", "r"], ["store", "l"]),
    E("q2r", ["question", "r"], ["retriever", "l"]), E("r2v", ["retriever", "t"], ["store", "b", -20]),
    E("r2p", ["retriever", "r"], ["prompt", "l"]), E("p2l", ["prompt", "r"], ["llm", "l"]),
    E("l2a", ["llm", "r"], ["answer", "l"]),
  ],
  labels: [{ x: 10, y: 18, t: "INDEXING · once", zone: true }, { x: 10, y: 198, t: "QUERY · every question", zone: true }, { x: 400, y: 150, t: "similarity search" }],
  steps: [
    { on: ["docs", "loader"], edges: ["d2l"], title: "Load the documents",
      text: "A document loader reads PDFs, web pages or database rows into Document objects: the text plus metadata such as the source and page.",
      code: `Document(
  page_content="Refunds are available within 30 days of purchase...",
  metadata={"source": "policy.pdf", "page": 3}
)` },
    { on: ["splitter"], edges: ["l2s"], title: "Split into chunks",
      text: "Whole documents are too big to send and too vague to search. A text splitter cuts them into overlapping chunks so a sentence isn't lost at a boundary.",
      code: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 1 PDF of 40 pages  ->  about 120 chunks` },
    { on: ["embed"], edges: ["s2e"], title: "Turn text into numbers",
      text: "An embedding model turns each chunk into a vector: a long list of numbers. Chunks with similar meaning get vectors that sit close together.",
      code: `"Refunds are available within 30 days..."
   -> [0.021, -0.113, 0.087, ..., 0.004]   # e.g. 1536 numbers` },
    { on: ["store"], edges: ["e2v"], title: "Store the vectors",
      text: "The vector store saves each vector with its chunk, so it can find the nearest neighbours quickly later.",
      code: `vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})` },
    { on: ["question", "retriever", "store"], edges: ["q2r", "r2v"], title: "A question arrives",
      text: "The retriever embeds the question with the same embedding model and asks the vector store for the 4 closest chunks. It matches by meaning, not by exact words.",
      code: `retriever.invoke("Can I get my money back?")

# finds "Refunds are available within 30 days..."
# even though the two share no keywords` },
    { on: ["prompt"], edges: ["r2p"], title: "Put the chunks into the prompt",
      text: "The retrieved chunks are pasted into the prompt as context. The LLM never searches anything. It only reads what we hand it.",
      code: `Answer using only this context:
[1] Refunds are available within 30 days of purchase...
[2] Refunds go back to the original payment method...

Question: Can I get my money back?` },
    { on: ["llm", "answer"], edges: ["p2l", "l2a"], title: "The LLM answers from your data",
      text: "The model answers from the context, and you can show the source metadata as citations. In code, the whole bottom row is one LCEL chain.",
      code: `chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
chain.invoke("Can I get my money back?")` },
  ],
  oneLiner: "RAG splits and embeds documents into a vector store once. At question time a retriever finds the most similar chunks, and the chain puts them into the prompt as context, so the model answers from your data."
},
{
  id: "callbacks", tab: "Callbacks & tracing", title: "6. Callbacks: the hidden observer",
  intro: "Every Runnable reports what it does as events. Streaming, logging and LangSmith tracing all listen to these events.",
  vb: [800, 250],
  nodes: [
    N("prompt", 60, 30, "Prompt", "", { w: 140 }),
    N("model", 330, 30, "Chat model", "", { w: 140 }),
    N("parser", 600, 30, "Parser", "", { w: 140 }),
    N("handler", 200, 175, "Callback handlers", "stream · logs · LangSmith tracer", { w: 400, h: 62 }),
  ],
  edges: [
    E("pm", ["prompt", "r"], ["model", "l"]), E("mp", ["model", "r"], ["parser", "l"]),
    E("cp", ["prompt", "b"], ["handler", "t", -140], { dashed: true }),
    E("cm", ["model", "b"], ["handler", "t", 0], { dashed: true }),
    E("cpa", ["parser", "b"], ["handler", "t", 140], { dashed: true }),
  ],
  labels: [{ x: 412, y: 140, t: "events" }],
  steps: [
    { on: ["prompt", "handler"], edges: ["cp"], title: "Each block announces itself",
      text: "When a block starts, it fires a start event with a run_id and the id of its parent run. That parent link is how a trace becomes a tree.",
      code: `on_chain_start  RunnableSequence     run_id=a1
on_chain_start  ChatPromptTemplate   run_id=b2  parent=a1
on_chain_end    ChatPromptTemplate   run_id=b2` },
    { on: ["model", "handler"], edges: ["pm", "cm"], title: "The model streams tokens",
      text: "While the model writes, it fires a new-token event for each small piece. stream() and astream_events() are built on these events.",
      code: `on_chat_model_start   run_id=c3  parent=a1
on_llm_new_token      "Why"
on_llm_new_token      " did"
on_llm_new_token      " the cat"
...
on_chat_model_end     14 tokens in / 18 out` },
    { on: ["parser", "handler"], edges: ["mp", "cpa"], title: "The chain finishes",
      text: "The parser reports its end event, and finally the outer sequence reports its own end.",
      code: `on_chain_start  StrOutputParser    run_id=d4  parent=a1
on_chain_end    StrOutputParser    run_id=d4
on_chain_end    RunnableSequence   run_id=a1` },
    { on: ["handler"], edges: [], title: "LangSmith shows the tree",
      text: "Set LANGSMITH_TRACING=true and a tracer callback sends these events to LangSmith. You see each step's input, output, time and token cost, which is how you debug a chain that gives a wrong answer.",
      code: `export LANGSMITH_TRACING=true

RunnableSequence          1.21s
├─ ChatPromptTemplate     0.00s
├─ ChatAnthropic          1.20s   14 → 18 tokens
└─ StrOutputParser        0.00s` },
  ],
  oneLiner: "Runnables emit callback events with run and parent IDs. Streaming, logging and LangSmith tracing are all built on those events, which gives you a full trace tree for debugging."
},
{
  id: "langgraph", tab: "LangGraph", title: "7. LangGraph: state, loops and human approval",
  intro: "LangChain gives you the parts. LangGraph controls how they run. This flow follows an email assistant that drafts a reply, pauses for a person to approve it, loops back if they say no, and saves every step in a checkpointer.",
  vb: [800, 290],
  nodes: [
    N("start", 10, 60, "START", "", { w: 100, h: 56, io: true }),
    N("draft", 140, 50, "draft node", "LLM writes the email", { w: 160, h: 76 }),
    N("human", 390, 50, "approval node", "interrupt() waits", { w: 170, h: 76 }),
    N("send", 620, 50, "send node", "sends the email", { w: 170, h: 76 }),
    N("end", 655, 200, "END", "", { w: 100, h: 56, io: true }),
    N("ckpt", 150, 210, "Checkpointer", "saves state per thread_id", { w: 400, h: 62 }),
  ],
  edges: [
    E("s2d", ["start", "r"], ["draft", "l"]),
    E("d2h", ["draft", "r", -14], ["human", "l", -14]),
    E("h2d", ["human", "l", 14], ["draft", "r", 14]),
    E("h2s", ["human", "r"], ["send", "l"]),
    E("s2e", ["send", "b"], ["end", "t"]),
    E("cd", ["draft", "b"], ["ckpt", "t", -130], { dashed: true }),
    E("ch", ["human", "b"], ["ckpt", "t", 125], { dashed: true }),
  ],
  labels: [{ x: 345, y: 44, t: "draft" }, { x: 345, y: 150, t: "rejected" }, { x: 590, y: 74, t: "ok" }, { x: 350, y: 182, t: "snapshot after each node" }],
  steps: [
    { on: ["start"], edges: ["s2d"], title: "Define the state",
      text: "Everything the graph knows lives in one state object. Each node returns only the keys it changes. A reducer such as add_messages says how to merge an update: append new messages instead of replacing the list.",
      code: `class EmailState(TypedDict):
    messages: Annotated[list, add_messages]   # reducer: append, don't overwrite
    draft: str
    approved: bool

config = {"configurable": {"thread_id": "email-123"}}
app.invoke({"messages": [("user", "Reply to Sam about the late invoice")]}, config)` },
    { on: ["draft", "ckpt"], edges: ["cd"], title: "A node does work and updates the state",
      text: "The draft node calls a LangChain model and returns only the draft key. After every node finishes, the checkpointer saves a snapshot of the whole state under the thread_id.",
      code: `def draft_email(state: EmailState):
    reply = model.invoke(state["messages"])
    return {"draft": reply.content}

# checkpoint saved -> {"messages": [...], "draft": "Hi Sam, sorry about..."}` },
    { on: ["human"], edges: ["d2h"], title: "interrupt() pauses the graph",
      text: "interrupt() stops the run right here and hands the draft to your app to show a person. The run is not waiting in memory. It lives in the checkpointer, so the person can answer minutes or days later.",
      code: `def approval(state: EmailState):
    decision = interrupt({"draft": state["draft"],
                          "question": "Send this email?"})
    if decision == "yes":
        return {"approved": True}
    return {"approved": False, "messages": [("user", decision)]}` },
    { on: ["ckpt"], edges: ["ch"], title: "The paused run is saved",
      text: "invoke() returns early with the interrupt payload. get_state shows the saved state and which node runs next. Even a server restart loses nothing, as long as the checkpointer uses a real database such as Postgres.",
      code: `result = app.invoke({...}, config)
result["__interrupt__"]
# [Interrupt(value={"draft": "Hi Sam, sorry about...",
#                   "question": "Send this email?"})]

app.get_state(config).next    # ("approval",)  <- where it will resume` },
    { on: ["draft"], edges: ["h2d"], title: "Resume with \"no\": the graph loops",
      text: "Command(resume=...) wakes the graph at the exact interrupt. The person's feedback is added to the messages, and a conditional edge sends the run back to draft for a new version. This loop is what a plain LCEL chain cannot do.",
      code: `app.invoke(Command(resume="Make it shorter and friendlier"), config)

def route(state: EmailState):
    return "send" if state["approved"] else "draft"

# approved=False -> back to draft -> new draft -> interrupt again` },
    { on: ["send"], edges: ["h2s"], title: "Resume with \"yes\": send it",
      text: "The second draft is approved. The same route function now picks the send node, which calls the email tool.",
      code: `app.invoke(Command(resume="yes"), config)

def send_email(state: EmailState):
    email_api.send(to="sam@example.com", body=state["draft"])
    return {"messages": [("assistant", "Email sent.")]}` },
    { on: ["end"], edges: ["s2e"], title: "Done, and everything is remembered",
      text: "Any later run with the same thread_id continues from the saved state. That one mechanism gives you memory across turns, human approval and crash recovery. get_state_history even lets you rewind to an older checkpoint (time travel). Here is the full wiring:",
      code: `graph = StateGraph(EmailState)
graph.add_node("draft", draft_email)
graph.add_node("approval", approval)
graph.add_node("send", send_email)

graph.add_edge(START, "draft")
graph.add_edge("draft", "approval")
graph.add_conditional_edges("approval", route)   # loop or continue
graph.add_edge("send", END)

app = graph.compile(checkpointer=PostgresSaver(conn))` },
  ],
  oneLiner: "LangGraph models a workflow as a state graph. Nodes update a shared state, conditional edges allow loops and branches, a checkpointer saves the state per thread_id after every step, and interrupt() pauses for a human and resumes from the exact same point."
},
];
window.FLOWS = FLOWS;

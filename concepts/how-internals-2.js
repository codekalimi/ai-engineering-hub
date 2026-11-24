// "Behind the scenes" part 2: Tools & agents, LangChain & LangGraph. Format: see how-internals-1.js.
(function (root) {
  const P = String.raw;
  const B = (root.HOW_INTERNALS = root.HOW_INTERNALS || {});
  const X = (id, see, code) => { B["hw-" + id] = { see, code: code.replace(/^\n/, "") }; };

  // ===================================================================== TOOLS & AGENTS
  X("toolcall", [
    'LangChain: `ai = model.bind_tools([get_weather]).invoke("...")`, then print `ai.tool_calls`: a list of `{"name", "args", "id"}` dicts. That is the model’s whole contribution.',
    'Print `convert_to_openai_tool(get_weather)` (from `langchain_core.utils.function_calling`) to see the JSON schema your Python function becomes.',
    'Raw SDKs: set `ANTHROPIC_LOG=debug` or `OPENAI_LOG=debug` to log each HTTP request the SDK makes; or print the response object and find the tool-call block.',
  ], P`
# The JSON that actually goes back and forth. Field names vary a little by provider; the shape doesn't.
import json

def get_weather(city):
    return {"Pune": "Rain, 24C", "Delhi": "Sunny, 31C"}.get(city, "unknown")
TOOLS = {"get_weather": get_weather}

# Step 1: your app sends the conversation plus the tool schemas
request = {
    "messages": [{"role": "user", "content": "Do I need an umbrella in Pune today?"}],
    "tools": [{"name": "get_weather", "description": "Current weather for a city",
               "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                                "required": ["city"]}}],
}
print("REQUEST ->", json.dumps(request)[:110], "...")

def fake_model(req):
    last = req["messages"][-1]
    if last["role"] == "user":
        # Step 2: the model replies with a structured tool call instead of text
        return {"role": "assistant", "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Pune"}}]}
    # Step 5: the model uses the result to write its answer
    return {"role": "assistant", "content": f"The forecast says {last['content']}, so yes, take an umbrella."}

reply = fake_model(request)
print("MODEL   ->", json.dumps(reply))
for call in reply["tool_calls"]:
    # Step 3: YOUR code runs the function. The model never executes anything.
    result = TOOLS[call["name"]](**call["arguments"])
    # Step 4: the result goes back as a tool message, linked by the tool call id
    request["messages"] += [reply, {"role": "tool", "tool_call_id": call["id"], "content": result}]
    print("TOOL    ->", json.dumps(request["messages"][-1]))
print("MODEL   ->", fake_model(request)["content"])
`);

  X("agentloop", [
    'LangChain/LangGraph agents: `result = agent.invoke({"messages": [("user", "...")]})`, then `for m in result["messages"]: m.pretty_print()` shows every turn: AI tool calls, tool results, final answer.',
    'Watch it live: `for step in agent.stream(inputs, stream_mode="updates"): print(step)` prints one update per loop iteration.',
    'LangSmith (set `LANGSMITH_TRACING=true`) shows each loop turn as a child run with the exact prompt, tool arguments, timings and tokens.',
  ], P`
import json

def search_flights(src, dst):
    return [{"flight": "AI-101", "price": 5400}, {"flight": "6E-22", "price": 4100}]
def book(flight):
    return f"booked {flight}, ref XK42"
TOOLS = {"search_flights": search_flights, "book": book}

def llm(messages):
    # a scripted stand-in for the model: it reads the conversation and decides the next move
    results = [m for m in messages if m["role"] == "tool"]
    if not results:
        return {"tool_calls": [{"id": "c1", "name": "search_flights", "args": {"src": "PNQ", "dst": "DEL"}}]}
    if len(results) == 1:
        cheapest = min(json.loads(results[0]["content"]), key=lambda f: f["price"])
        return {"tool_calls": [{"id": "c2", "name": "book", "args": {"flight": cheapest["flight"]}}]}
    return {"content": "Booked the cheapest flight, 6E-22 (ref XK42)."}

# Step 1: start with the task (plus, in real life, a system prompt and the tool schemas)
messages = [{"role": "user", "content": "Book me the cheapest flight from Pune to Delhi"}]
MAX_STEPS = 5                                    # always cap the loop
for step in range(1, MAX_STEPS + 1):
    # Step 2: each turn the model either requests tools or answers
    reply = llm(messages)
    messages.append({"role": "assistant", **reply})
    if "tool_calls" not in reply:
        # Step 5: no tool call -> this reply is the final answer
        print(f"step {step}: FINAL -> {reply['content']}")
        break
    for call in reply["tool_calls"]:
        # Step 3: run the tool and append its result (errors go back as text, not exceptions)
        try:
            result = TOOLS[call["name"]](**call["args"])
        except Exception as err:
            result = f"error: {err}"
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
        print(f"step {step}: {call['name']}({call['args']}) -> {result}")
    # Step 4: loop - the model is called again with the longer conversation
else:
    print("stopped: step limit reached")
print(f"{len(messages)} messages in the conversation after {step} LLM calls")
`);

  X("mcp", [
    'MCP Inspector: `npx @modelcontextprotocol/inspector <command that starts your server>` opens a UI where you can call initialize, tools/list and tools/call yourself and read the raw JSON-RPC messages.',
    'A stdio server is just a process reading JSON lines on stdin and writing them on stdout; log them to stderr to watch the conversation.',
    'In your host app’s logs (Claude Desktop, IDE assistants and others write MCP logs), look for the tools/list response to see exactly which tool descriptions the model receives.',
  ], P`
# MCP = JSON-RPC 2.0 messages between a client (inside the host app) and a server (a separate
# program, over stdio or HTTP). Here the server is a function so you can read every message.
import json

TOOLS = [{"name": "get_ticket", "description": "Look up a support ticket by id",
          "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}]
TICKETS = {"T-7": "Printer on floor 3 is offline (open)"}

def server(raw):
    msg = json.loads(raw)
    if msg["method"] == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "helpdesk"}}
    elif msg["method"] == "tools/list":
        result = {"tools": TOOLS}
    elif msg["method"] == "tools/call":
        # Step 5: the server does the real work (API, database, files)
        ticket = TICKETS.get(msg["params"]["arguments"]["id"], "not found")
        result = {"content": [{"type": "text", "text": ticket}]}
    return json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result})

next_id = 0
def send(method, params=None):
    global next_id
    next_id += 1
    raw = json.dumps({"jsonrpc": "2.0", "id": next_id, "method": method, "params": params or {}})
    print("client ->", raw[:105])
    result = json.loads(server(raw))["result"]
    print("server <-", json.dumps(result)[:105])
    return result

# Step 1-2: the host's client connects, exchanges capabilities, lists the tools
send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "my-agent"}})
tools = send("tools/list")["tools"]
# Step 3: these schemas go to the LLM like any other tools; say it picks get_ticket
# Step 4: the client sends tools/call
result = send("tools/call", {"name": "get_ticket", "arguments": {"id": "T-7"}})
print("text handed back to the LLM:", result["content"][0]["text"])
`);

  X("supervisor", [
    'LangGraph: `graph.stream(inputs, stream_mode="updates", subgraphs=True)` prints which agent (node or subgraph) ran at each step and what it returned.',
    'Draw the graph: `print(graph.get_graph().draw_mermaid())`, then paste it into any Mermaid viewer.',
    'In LangSmith, each specialist appears as its own branch of the trace, so you can see every routing decision and its cost.',
  ], P`
# A supervisor is an LLM whose "tools" are other agents. Everything is stubbed so the routing is visible.
def research_agent(task):
    return "Q3 revenue was 4.2M, up 12% on Q2."
def writer_agent(task):
    return "Draft: 'Team - Q3 revenue reached 4.2M (+12%). Great work!'"
AGENTS = {"researcher": research_agent, "writer": writer_agent}

def supervisor_llm(task, results):
    # Step 1-2: decide which specialist should act next (a real supervisor asks an LLM this,
    # usually with structured output like {"next": "writer", "instruction": "..."})
    if "researcher" not in results:
        return "researcher", "find the Q3 revenue numbers"
    if "writer" not in results:
        return "writer", "write a short team email using: " + results["researcher"]
    return "FINISH", results["writer"]

task = "Email the team a summary of Q3 revenue"
results = {}
for turn in range(1, 6):
    choice, instruction = supervisor_llm(task, results)
    if choice == "FINISH":
        # Step 4: the supervisor is satisfied and returns the final answer
        print(f"turn {turn}: supervisor -> FINISH\nfinal: {instruction}")
        break
    print(f"turn {turn}: supervisor -> {choice}: {instruction!r}")
    # Step 3: the specialist runs its own loop (own prompt, own tools) and reports back
    results[choice] = AGENTS[choice](instruction)
    print(f"        {choice} reports: {results[choice]}")
`);

  X("agent-memory", [
    'LangGraph short-term memory: `graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]` shows exactly what the next turn will see.',
    'LangGraph long-term memory: `store.search(("memories", user_id), query="food preferences")` returns stored items with similarity scores; print them before they go into the prompt.',
    'Print the final prompt of a new conversation to check that the right memories (and only those) were injected.',
  ], P`
import json, re
words = lambda t: set(re.findall(r"[a-z]+", t.lower()))

threads = {}   # short-term: each conversation's messages, saved per thread (a checkpointer does this)
store = {}     # long-term: durable facts per user, searchable across conversations (a store does this)

def save_memory(user, fact):
    # Step 3: written by the agent through a "save memory" tool (or by a background job)
    store.setdefault(user, []).append(fact)

def search_memory(user, query):
    # Step 4: in a later conversation, find relevant memories (real stores use embeddings)
    return [f for f in store.get(user, []) if words(f) & words(query)]

def chat(user, thread, message):
    history = threads.setdefault(thread, [])        # Step 1: load this thread's history
    memories = search_memory(user, message)         # Step 2/4: plus relevant long-term facts
    print(f"[{thread}] {message!r}")
    print(f"   prompt = {len(history)} earlier messages + memories {memories}")
    history += [("user", message), ("assistant", "...")]
    if "I'm vegetarian" in message:
        save_memory(user, "User is vegetarian")
    if "aisle" in message:
        save_memory(user, "User prefers an aisle seat on flights")

chat("user-42", "thread-1", "I'm vegetarian, plan a dinner for Friday")
chat("user-42", "thread-1", "Also I like an aisle seat when I fly")
# A brand-new conversation: no history, but the long-term memory comes back
chat("user-42", "thread-2", "Find a vegetarian place near my hotel")
print("long-term store:", json.dumps(store))
`);

  X("hitl", [
    'LangGraph: after the call that pauses, `state = graph.get_state(config)`; `state.next` shows the node that will run on resume and `state.tasks[0].interrupts` holds the payload.',
    'The returned value also carries the pause: look for the `__interrupt__` key in what `invoke` returns.',
    'Use a persistent checkpointer (SQLite or Postgres), stop your Python process after the interrupt, start it again and resume with `Command(resume=...)`: it still works, which proves nothing was held in memory.',
  ], P`
# A tiny version of LangGraph's interrupt() and Command(resume=...).
import json

checkpoints = {}                                   # the checkpointer: thread_id -> saved state

class Interrupt(Exception):
    def __init__(self, payload):
        self.payload = payload

def draft_node(state, resume):
    return {"draft": f"Hi {state['to']}, the invoice is attached."}

def approve_node(state, resume):
    # Step 2: interrupt() pauses with the thing that needs approval
    if resume is None:
        raise Interrupt({"action": "send_email", "draft": state["draft"]})
    # Step 6: on resume, the node runs again from its start and interrupt() returns the decision
    return {"approved": resume["approved"], "draft": resume.get("edited", state["draft"])}

def send_node(state, resume):
    return {"sent": state["approved"]}

NODES = [("draft", draft_node), ("approve", approve_node), ("send", send_node)]

def invoke(inp, thread_id, resume=None):
    saved = checkpoints.get(thread_id)
    state, start = (saved["state"], saved["next"]) if saved else (dict(inp), 0)
    for i in range(start, len(NODES)):
        name, fn = NODES[i]
        try:
            update = fn(state, resume)
        except Interrupt as it:
            # Step 3: save the state and the node to re-run, then return the payload
            checkpoints[thread_id] = {"state": state, "next": i}
            print(f"  paused at '{name}', state saved for {thread_id}")
            return {"__interrupt__": it.payload}
        state.update(update)
        print(f"  ran {name}: {update}")
    checkpoints[thread_id] = {"state": state, "next": len(NODES)}
    return state

# Step 1: invoke with a thread_id
print("app got:", json.dumps(invoke({"to": "Ravi"}, "thread-9")))
# Step 4: a person reviews and edits it in your UI - maybe hours later
decision = {"approved": True, "edited": "Hi Ravi, invoice #881 is attached. Thanks!"}
# Step 5: invoke again with the decision and the SAME thread_id
print("final state:", json.dumps(invoke(None, "thread-9", resume=decision)))
`);

  X("delegated-auth", [
    'Decode a token locally to read its claims (`sub`, `aud`, `scope`, `exp`): the payload is the middle part, base64url-encoded. Don’t paste real tokens into websites.',
    'Call the target API with a token that lacks a scope and confirm you get a 403: that is the API enforcing the user’s permissions, not your agent.',
    'Log which user and which scopes every agent tool call ran with; that audit trail is what security reviewers ask for.',
  ], P`
# Delegated access, simplified: real tokens are JWTs signed with RSA/EC keys by an identity provider.
import base64, hashlib, hmac, json, time

IDP_KEY = b"identity-provider-secret"              # shared by the identity provider and the API

def b64(data): return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
def unb64(s): return json.loads(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)))
def sign(body): return hmac.new(IDP_KEY, body.encode(), hashlib.sha256).hexdigest()[:16]

def issue_token(user, scopes, ttl=600):
    # Step 1-2: the user signs in and approves scopes; the IdP issues a short-lived token
    body = b64({"sub": user, "scope": " ".join(scopes), "aud": "calendar-api", "exp": int(time.time()) + ttl})
    return body + "." + sign(body)

def calendar_api(token, action, owner):
    # Step 4: the API checks signature, audience, expiry, scope AND the user's own permissions
    body, sig = token.split(".")
    if not hmac.compare_digest(sig, sign(body)):
        return "401 bad signature"
    claims = unb64(body)
    if claims["aud"] != "calendar-api" or claims["exp"] < time.time():
        return "401 wrong audience or expired"
    need = "calendar.read" if action == "read" else "calendar.write"
    if need not in claims["scope"].split():
        return f"403 token lacks scope {need}"
    if owner != claims["sub"]:
        return f"403 {claims['sub']} cannot access {owner}'s calendar"
    return f"200 {action} {owner}'s calendar"

token = issue_token("meera", ["calendar.read"])
claims = unb64(token.split(".")[0])
print("token says:", {k: claims[k] for k in ("sub", "aud", "scope")}, "+ an expiry 10 minutes away")
# Step 3: the agent acts with the USER's token, never with a powerful service account
for action, owner in [("read", "meera"), ("write", "meera"), ("read", "ceo")]:
    print(f"agent tries to {action} {owner}'s calendar: {calendar_api(token, action, owner)}")
`);

  // ===================================================================== LANGCHAIN & LANGGRAPH
  X("lcel", [
    'Print the structure: `print(type(chain).__name__, chain.steps)` (a RunnableSequence), or `chain.get_graph().print_ascii()` (needs the `grandalf` package).',
    '`from langchain_core.globals import set_debug; set_debug(True)` prints every step’s input and output as the chain runs.',
    'Read the real thing: open `langchain_core/runnables/base.py` and find `__or__` and `RunnableSequence.invoke`; with the mini version below in mind it reads easily.',
  ], P`
# A from-scratch version of LangChain's Runnable and the | operator.
from concurrent.futures import ThreadPoolExecutor
VERBOSE = True

class Runnable:
    # Step 1: one interface for everything: invoke, batch, stream
    def invoke(self, x):
        raise NotImplementedError
    def __or__(self, other):                       # Step 2: a | b -> RunnableSequence
        return RunnableSequence(self, coerce(other))
    def batch(self, xs):                           # Step 5: many inputs in parallel (thread pool)
        with ThreadPoolExecutor() as pool:
            return list(pool.map(self.invoke, xs))
    def stream(self, x):                           # default: one chunk = the whole result
        yield self.invoke(x)

class RunnableLambda(Runnable):
    def __init__(self, fn, name):
        self.fn, self.name = fn, name
    def invoke(self, x):
        out = self.fn(x)
        if VERBOSE:
            print(f"   {self.name}: {x!r} -> {out!r}")
        return out

class RunnableSequence(Runnable):
    def __init__(self, *steps):
        self.steps = steps
    def __or__(self, other):                       # (a | b) | c flattens into one sequence
        return RunnableSequence(*self.steps, coerce(other))
    def invoke(self, x):                           # Step 3: feed each output into the next step
        for step in self.steps:
            x = step.invoke(x)
        return x
    def stream(self, x):                           # Step 4: run the earlier steps, stream the last
        for step in self.steps[:-1]:
            x = step.invoke(x)
        yield from self.steps[-1].stream(x)

def coerce(obj):                                   # plain functions become runnables
    return obj if isinstance(obj, Runnable) else RunnableLambda(obj, obj.__name__)

class FakeChatModel(Runnable):
    def stream(self, prompt):                      # a chat model streams token by token
        yield from ["Paris", " is", " the", " capital", "."]
    def invoke(self, prompt):
        out = "".join(self.stream(prompt))
        if VERBOSE:
            print(f"   model: {prompt!r} -> {out!r}")
        return out

prompt = RunnableLambda(lambda d: f"What is the capital of {d['country']}?", "prompt")
def parser(text): return text.strip().upper()

chain = prompt | FakeChatModel() | parser
print("chain is a", type(chain).__name__, "with", len(chain.steps), "steps")
print("invoke:")
print(" =>", chain.invoke({"country": "France"}))
print("stream:", list((prompt | FakeChatModel()).stream({"country": "France"})))
VERBOSE = False                                    # threads would interleave the prints
print("batch:", chain.batch([{"country": "France"}, {"country": "Japan"}]))
`);

  X("rag-chain", [
    'Run just the first step to see the dict the prompt receives: `setup = RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough()); print(setup.invoke("..."))`.',
    'Print the filled prompt: `(setup | prompt).invoke("...").to_string()` shows the exact text before the model is called.',
    '`chain.get_graph().print_ascii()` draws the two parallel branches joining at the prompt.',
  ], P`
# What {"context": retriever | format_docs, "question": RunnablePassthrough()} does underneath.
from concurrent.futures import ThreadPoolExecutor

DOCS = ["LangGraph adds state and loops.", "LCEL composes runnables with pipes.", "Paris is in France."]
def retriever(question):                     # returns documents sharing a word with the question
    q = set(question.lower().strip("?").split())
    return [d for d in DOCS if q & set(d.lower().strip(".").split())]
def format_docs(docs): return "\n".join(f"- {d}" for d in docs)
def passthrough(x): return x                 # RunnablePassthrough: the input, unchanged

def pipe(*fns):
    def run(x):
        for fn in fns:
            x = fn(x)
        return x
    return run

def parallel(branches):
    # Step 2: a dict becomes a RunnableParallel - the SAME input goes to every branch, concurrently
    def run(x):
        with ThreadPoolExecutor() as pool:
            futures = {key: pool.submit(fn, x) for key, fn in branches.items()}
            return {key: f.result() for key, f in futures.items()}
    return run

# Step 1 + 3: the context branch turns the question into text; the passthrough keeps the question
setup = parallel({"context": pipe(retriever, format_docs), "question": passthrough})
# Step 4: the dict fills the prompt's {context} and {question}
def prompt(d): return f"Use only this context:\n{d['context']}\n\nQuestion: {d['question']}"
# Step 5: then the usual model -> parser (stubbed)
def model(p): return f"(answer written from {p.count('- ')} context line)"

def show(label):
    def run(x):
        print(f"---- {label} ----\n{x}")
        return x
    return run

chain = pipe(setup, show("after the dict step"), prompt, show("prompt"), model)
print("answer:", chain("What does LCEL compose?"))
`);

  X("langgraph-run", [
    '`for chunk in graph.stream(inputs, stream_mode="debug"): print(chunk)` prints every task start and result, step by step: the engine below, in real life.',
    '`stream_mode="updates"` shows each node’s partial update; `stream_mode="values"` shows the full state after every step.',
    'Hit the limit on purpose with `graph.invoke(inputs, {"recursion_limit": 3})` to see a GraphRecursionError.',
  ], P`
# A mini LangGraph engine: state + nodes + reducers + edges + a step loop.
import operator

REDUCERS = {"messages": operator.add}             # 'messages' appends; other keys overwrite

def merge(state, update):
    # Step 3: reducers merge each node's update into the state
    for k, v in update.items():
        state[k] = REDUCERS[k](state.get(k, []), v) if k in REDUCERS else v
    return state

def agent(state):                                 # pretend LLM: ask for a tool once, then answer
    if not any(m.startswith("tool:") for m in state["messages"]):
        return {"messages": ["ai: call search('langgraph')"]}
    return {"messages": ["ai: LangGraph runs nodes in steps."]}

def tools(state):
    return {"messages": ["tool: 'LangGraph is a graph runtime'"]}

NODES = {"agent": agent, "tools": tools}
EDGES = {"tools": "agent"}                        # fixed edge: tools -> agent
def route_after_agent(state):                     # conditional edge: a function picks the next node
    return "tools" if "call" in state["messages"][-1] else "END"
CONDITIONAL = {"agent": route_after_agent}

checkpoints = []
def invoke(inp, recursion_limit=10):
    state, next_nodes = merge({}, inp), ["agent"]          # Step 1: input -> initial state
    for step in range(1, recursion_limit + 1):
        if not next_nodes:
            return state
        updates = [(n, NODES[n](dict(state))) for n in next_nodes]   # Step 2: run scheduled nodes
        for n, u in updates:
            state = merge(state, u)
        checkpoints.append({"step": step, "state": dict(state)})      # Step 4: checkpoint
        # Step 5: edges decide the next nodes
        nxt = [CONDITIONAL[n](state) if n in CONDITIONAL else EDGES[n] for n, _ in updates]
        print(f"step {step}: ran {[n for n, _ in updates]} -> next {nxt}")
        next_nodes = [n for n in nxt if n != "END"]
    raise RuntimeError("recursion limit reached")           # Step 6

final = invoke({"messages": ["user: what is langgraph?"]})
print("messages:", final["messages"])
print("checkpoints saved:", len(checkpoints))
`);

  X("reducers", [
    'Compare `graph.stream(inputs, stream_mode="updates")` (what each node returned) with `stream_mode="values"` (the state after the reducers ran).',
    'Return two updates to the same plain key from parallel nodes and read the error: LangGraph refuses to guess how to combine them.',
    'Read `add_messages` in `langgraph/graph/message.py`: it is short and does exactly the id-based append-or-replace shown below.',
  ], P`
import itertools, operator

_ids = itertools.count(1)
def msg(role, text, id=None):
    return {"id": id or f"m{next(_ids)}", "role": role, "text": text}

def add_messages(current, new):
    # Step 3: append new messages, but a message with an existing id REPLACES that message
    merged = list(current)
    position = {m["id"]: i for i, m in enumerate(merged)}
    for m in new:
        if m["id"] in position:
            merged[position[m["id"]]] = m
        else:
            position[m["id"]] = len(merged)
            merged.append(m)
    return merged

# Step 1: the schema annotates keys with reducers:
#   messages: Annotated[list, add_messages]   sources: Annotated[list, operator.add]   status: str
REDUCERS = {"messages": add_messages, "sources": operator.add}

def apply(state, update, node):
    # Step 2: for each key in the update, call reducer(current, new)
    # Step 4: keys without a reducer simply take the new value
    for key, value in update.items():
        reducer = REDUCERS.get(key)
        state[key] = reducer(state.get(key, []), value) if reducer else value
    shown = {k: ([m["id"] + ":" + m["text"] for m in v] if k == "messages" else v) for k, v in state.items()}
    print(f"after {node:9} {shown}")

state = {}
apply(state, {"messages": [msg("user", "hi")], "status": "start"}, "input")
apply(state, {"messages": [msg("ai", "draft", id="a1")], "sources": ["doc1"], "status": "drafted"}, "writer")
apply(state, {"sources": ["doc2"]}, "retriever")
apply(state, {"messages": [msg("ai", "final", id="a1")], "status": "done"}, "editor")   # same id -> replaced
`);

  X("checkpoint", [
    '`for snap in graph.get_state_history(config): print(snap.config["configurable"]["checkpoint_id"], snap.next, len(snap.values["messages"]))` lists every saved step.',
    'With `SqliteSaver` (package `langgraph-checkpoint-sqlite`), open the database file in any SQLite browser and look at the checkpoints table.',
    'Time travel: invoke with an older `checkpoint_id` in the config to resume from that point.',
  ], P`
# A checkpointer saves a snapshot of the state after each step, per thread_id.
# Here it is SQLite from the standard library, like langgraph-checkpoint-sqlite without the details.
import json, sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE checkpoints (thread_id TEXT, step INTEGER, state TEXT)")

def save(thread_id, step, state):
    db.execute("INSERT INTO checkpoints VALUES (?, ?, ?)", (thread_id, step, json.dumps(state)))

def latest(thread_id):
    row = db.execute("SELECT step, state FROM checkpoints WHERE thread_id = ? "
                     "ORDER BY step DESC LIMIT 1", (thread_id,)).fetchone()
    return (row[0], json.loads(row[1])) if row else (0, {"messages": []})

def chatbot(state):
    n = sum(m.startswith("user") for m in state["messages"])
    return f"ai: that's message {n} from you"

# Step 1: compile(checkpointer=...) wires this in. Step 2: every call carries a thread_id
def invoke(message, config):
    thread_id = config["configurable"]["thread_id"]
    step, state = latest(thread_id)               # Step 4: load the latest snapshot for this thread
    for update in ("user: " + message, None):
        state["messages"].append(update or chatbot(state))
        step += 1
        save(thread_id, step, state)              # Step 3: snapshot after each step
    return state["messages"][-1]

alice = {"configurable": {"thread_id": "alice"}}
bob = {"configurable": {"thread_id": "bob"}}
print("alice:", invoke("hi", alice))
print("alice:", invoke("how are you?", alice))
print("bob:  ", invoke("hello", bob))
# Step 5: get_state_history - every snapshot, newest first
for step, state in db.execute("SELECT step, state FROM checkpoints WHERE thread_id = 'alice' ORDER BY step DESC"):
    print(f"alice step {step}: {json.loads(state)['messages']}")
`);

  X("streaming", [
    '`curl -N http://localhost:8000/chat/stream` (the `-N` turns off buffering) prints the raw `data:` lines as they arrive.',
    'Browser DevTools, Network tab: click the request and open the EventStream tab to see each event with its arrival time.',
    'LangChain: `async for ev in chain.astream_events(inputs, version="v2"): print(ev["event"], ev["name"])` shows every token event before you forward it.',
  ], P`
# Server-Sent Events: the raw HTTP response body is plain text, one "data:" line per event.
import json

def fake_model_stream(prompt):
    # Step 2: the model is called with streaming on and yields tokens as it produces them
    yield from ["Stream", "ing ", "shows ", "tokens ", "early."]

def sse_endpoint(prompt):
    # Response header: Content-Type: text/event-stream
    full = []
    for chunk in fake_model_stream(prompt):
        full.append(chunk)
        # Step 3: forward each chunk immediately; a blank line ends each event
        yield f"data: {json.dumps({'delta': chunk})}\n\n"
    # Step 5: log the full answer and usage, then send a final 'done' event
    yield f"event: done\ndata: {json.dumps({'chars': len(''.join(full))})}\n\n"

# Step 1: the browser keeps the connection open (EventSource or fetch with a stream reader)
ui = ""
for n, raw in enumerate(sse_endpoint("hi"), 1):
    print(f"event {n} on the wire: {raw!r}")
    # Step 4: the browser parses each event and appends the text to the UI
    for line in raw.splitlines():
        if line.startswith("data:") and "delta" in line:
            ui += json.loads(line[len("data:"):])["delta"]
print("UI now shows:", ui)
`);

  X("callbacks", [
    'Write a handler: subclass `BaseCallbackHandler`, implement `on_chat_model_start`, `on_llm_end` and `on_tool_start`, and pass it with `chain.invoke(x, config={"callbacks": [MyHandler()]})`.',
    'Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to send the same events to LangSmith and browse the run tree.',
    'Each event carries `run_id` and `parent_run_id`: print them to see how the tree below is rebuilt.',
  ], P`
# Every run gets an id and a parent id; handlers receive start/end events.
import itertools
ids = itertools.count(1)

class PrintHandler:                      # like a BaseCallbackHandler with on_*_start / on_*_end
    def on_start(self, run):
        print(f"{'  ' * run['depth']}start {run['type']}:{run['name']} (run {run['id']}, parent {run['parent']})")
    def on_end(self, run, out):
        print(f"{'  ' * run['depth']}end   {run['name']} -> {out!r}")

class Tracer:                            # like LangSmith: collect the runs, rebuild the tree later
    def __init__(self):
        self.runs = []
    def on_start(self, run):
        self.runs.append(run)
    def on_end(self, run, out):
        run["output"] = out

def traced(kind, name):
    def wrap(fn):
        def inner(x, config, parent=None, depth=0):
            run = {"id": next(ids), "parent": parent, "type": kind, "name": name, "depth": depth}  # Step 1
            for h in config["callbacks"]:
                h.on_start(run)          # Step 2-3: events go to every attached handler
            out = fn(x, config, run["id"], depth + 1)
            for h in config["callbacks"]:
                h.on_end(run, out)
            return out
        return inner
    return wrap

@traced("retriever", "search_docs")
def retrieve(q, config, parent, depth):
    return ["doc about " + q]

@traced("llm", "chat_model")
def llm(docs, config, parent, depth):
    return "answer using " + docs[0]

@traced("chain", "rag_chain")
def rag(q, config, parent, depth):
    return llm(retrieve(q, config, parent, depth), config, parent, depth)

tracer = Tracer()
rag("refunds", {"callbacks": [PrintHandler(), tracer]})

# Step 4-5: a tracer sends the runs to a server; parent ids rebuild the tree
children = {}
for r in tracer.runs:
    children.setdefault(r["parent"], []).append(r)
def show(parent, indent=0):
    for r in children.get(parent, []):
        print("  " * indent + f"- {r['name']} ({r['type']})")
        show(r["id"], indent + 1)
print("trace tree:")
show(None)
`);
})(typeof window !== "undefined" ? window : globalThis);

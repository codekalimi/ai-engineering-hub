// Levels 1 and 2, part 2: Tools & agents, LangChain & LangGraph. Format: see how-levels-1.js.
(function (root) {
  const P = String.raw;
  const B = (root.HOW_LEVELS = root.HOW_LEVELS || {});
  const L = (id, use, under) => { B["hw-" + id] = { use, under }; };
  const FALLBACK = `betas=["server-side-fallback-2026-07-01"], fallbacks="default"`;

  // ===================================================================== TOOLS & AGENTS
  L("toolcall", {
    note: "Decorate a function, bind it, and the model can ask for it. You run it; the result goes back as a ToolMessage.",
    script: [{ tool_calls: [{ name: "get_weather", args: { city: "Pune" } }] }, "Yes - it's raining in Pune (24C), so take an umbrella."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return {"Pune": "Rain, 24C"}.get(city, "unknown")

model = init_chat_model("anthropic:claude-opus-5-5").bind_tools([get_weather])
messages = [HumanMessage("Do I need an umbrella in Pune today?")]

ai = model.invoke(messages)
print("tool calls:", ai.tool_calls)
messages += [ai] + [get_weather.invoke(call) for call in ai.tool_calls]   # invoking with the call -> a ToolMessage
print("tool result:", messages[-1].content, "| linked to", messages[-1].tool_call_id)
print("answer:", model.invoke(messages).content)
`,
  }, {
    note: "What LangChain translates that into: the raw Anthropic Messages API. Tools are JSON schemas; the model answers with a tool_use block; you reply with a tool_result block carrying the same id.",
    source: ["langchain_anthropic/chat_models.py (_format_messages)", "Anthropic docs: Tool use"],
    anthropic: [
      { content: [{ type: "tool_use", id: "toolu_01", name: "get_weather", input: { city: "Pune" } }], stop_reason: "tool_use" },
      { content: [{ type: "text", text: "Yes - it's raining in Pune (24C), so take an umbrella." }] },
    ],
    code: P`
import json
import anthropic

client = anthropic.Anthropic()
tools = [{"name": "get_weather", "description": "Current weather for a city.",           # Step 1: schemas
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
messages = [{"role": "user", "content": "Do I need an umbrella in Pune today?"}]

def ask():
    return client.beta.messages.create(model="claude-opus-5-5", max_tokens=16000, tools=tools,
                                       messages=messages, ${FALLBACK})

response = ask()
print("stop_reason:", response.stop_reason)                    # Step 2: "tool_use", not text
tool_use = next(b for b in response.content if b.type == "tool_use")
print("model asked for:", tool_use.name, tool_use.input, "id", tool_use.id)

result = {"Pune": "Rain, 24C"}[tool_use.input["city"]]           # Step 3: YOUR code runs it
messages += [{"role": "assistant", "content": response.content},  # Step 4: echo the call, then the result
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": result}]}]
print("sent back:", json.dumps(messages[-1]))
print("answer:", ask().content[0].text)                          # Step 5
`,
  });

  L("agentloop", {
    note: "create_agent builds the loop for you (on LangGraph). You give it a model, tools and a prompt.",
    script: [
      { tool_calls: [{ name: "search_flights", args: { src: "PNQ", dst: "DEL" } }] },
      { tool_calls: [{ name: "book", args: { flight: "6E-22" } }] },
      "Booked the cheapest flight, 6E-22 for 4100 (ref XK42).",
    ],
    code: P`
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

@tool
def search_flights(src: str, dst: str) -> list:
    """Find flights between two airport codes."""
    return [{"flight": "AI-101", "price": 5400}, {"flight": "6E-22", "price": 4100}]

@tool
def book(flight: str) -> str:
    """Book a flight by its number."""
    return f"booked {flight}, ref XK42"

agent = create_agent(init_chat_model("anthropic:claude-opus-5-5"), [search_flights, book],
                     system_prompt="You book travel. Always pick the cheapest option.")
result = agent.invoke({"messages": [("user", "Book me the cheapest flight from Pune to Delhi")]},
                      {"recursion_limit": 10})                     # the step limit
for m in result["messages"]:
    print(f"{m.type:5}", m.tool_calls if getattr(m, "tool_calls", None) else m.content)
`,
  }, {
    note: "The loop create_agent runs, written out with the real message classes. Each comment names the part of create_agent it replaces.",
    source: ["langchain/agents/factory.py (create_agent)", "langgraph/prebuilt/tool_node.py (ToolNode)"],
    script: [
      { tool_calls: [{ name: "search_flights", args: { src: "PNQ", dst: "DEL" } }] },
      { tool_calls: [{ name: "book", args: { flight: "6E-22" } }] },
      "Booked the cheapest flight, 6E-22 for 4100 (ref XK42).",
    ],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool

@tool
def search_flights(src: str, dst: str) -> list:
    """Find flights between two airport codes."""
    return [{"flight": "AI-101", "price": 5400}, {"flight": "6E-22", "price": 4100}]

@tool
def book(flight: str) -> str:
    """Book a flight by its number."""
    return f"booked {flight}, ref XK42"

tools = {t.name: t for t in [search_flights, book]}
llm_with_tools = init_chat_model("anthropic:claude-opus-5-5").bind_tools(list(tools.values()))

def run_agent(question, max_steps=10):            # <- what create_agent() builds for you
    messages = [SystemMessage("You book travel. Always pick the cheapest option."),   # Step 1
                HumanMessage(question)]
    for step in range(1, max_steps + 1):          # <- recursion_limit
        ai = llm_with_tools.invoke(messages)      # Step 2: one model call ("model" node)
        messages.append(ai)
        if not ai.tool_calls:                     # Step 5: no tool call -> done (edge to END)
            return ai.content
        for call in ai.tool_calls:                # Step 3: the "tools" node (ToolNode)
            try:
                result = tools[call["name"]].invoke(call["args"])
            except Exception as err:
                result = f"Error: {err}"          # errors go back as text, the model can recover
            messages.append(ToolMessage(str(result), tool_call_id=call["id"]))
            print(f"step {step}: {call['name']}({call['args']}) -> {result}")
        # Step 4: loop - call the model again with the longer conversation
    raise RuntimeError("step limit reached")

print("final:", run_agent("Book me the cheapest flight from Pune to Delhi"))
`,
  });

  L("mcp", {
    note: "Connect to an MCP server, get its tools as LangChain tools, hand them to an agent. The server is the 10-line FastMCP program shown in Level 2.",
    script: [{ tool_calls: [{ name: "get_ticket", args: { id: "T-7" } }] }, "Ticket T-7: the printer on floor 3 is offline, and the ticket is still open."],
    files: { "server.py": P`
from mcp.server.fastmcp import FastMCP

server = FastMCP("helpdesk")

@server.tool()
def get_ticket(id: str) -> str:
    """Look up a support ticket by id."""
    return {"T-7": "Printer on floor 3 is offline (open)"}.get(id, "not found")

if __name__ == "__main__":
    server.run()
` },
    code: P`
import asyncio
import sys
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient({
        "helpdesk": {"transport": "stdio", "command": sys.executable, "args": ["server.py"]},
    })
    tools = await client.get_tools()                      # MCP tools -> LangChain tools
    print("tools from the server:", [t.name for t in tools])
    agent = create_agent(init_chat_model("anthropic:claude-opus-5-5"), tools)
    result = await agent.ainvoke({"messages": [("user", "What's the status of ticket T-7?")]})
    print(result["messages"][-1].content)

asyncio.run(main())
`,
  }, {
    note: "Both halves with the official MCP Python SDK (1.x, the version langchain-mcp-adapters supports): a FastMCP server, and a client session doing the handshake, tools/list and tools/call from the diagram. In SDK 2.x, FastMCP is renamed MCPServer (from mcp.server.mcpserver import MCPServer).",
    source: ["mcp/client/session.py (ClientSession)", "mcp/server/fastmcp/server.py"],
    code: P`
import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = '''
from mcp.server.fastmcp import FastMCP
server = FastMCP("helpdesk")

@server.tool()
def get_ticket(id: str) -> str:
    """Look up a support ticket by id."""
    return {"T-7": "Printer on floor 3 is offline (open)"}.get(id, "not found")

server.run()   # stdio transport: JSON-RPC messages on stdin/stdout
'''
open("server.py", "w").write(SERVER)

async def main():
    params = StdioServerParameters(command=sys.executable, args=["server.py"])   # Step 1: start the server
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()                                     # Step 2: handshake
            print("connected to:", info.serverInfo.name, "| protocol", info.protocolVersion)
            listed = await session.list_tools()                                   # tools/list
            for t in listed.tools:
                print("tool:", t.name, "-", t.description)                        # Step 3: goes to the LLM
            result = await session.call_tool("get_ticket", {"id": "T-7"})         # Step 4: tools/call
            print("result:", result.content[0].text)                              # Step 5

asyncio.run(main())
`,
  });

  L("supervisor", {
    note: "The simplest multi-agent setup: specialists are agents, wrapped as tools of a supervisor agent.",
    script: [
      { tool_calls: [{ name: "ask_researcher", args: { question: "What was Q3 revenue and growth?" } }] },
      { tool_calls: [{ name: "get_financials", args: { quarter: "Q3" } }] },
      "Q3 revenue was 4.2M, up 12% on Q2.",
      { tool_calls: [{ name: "ask_writer", args: { brief: "Short team email: Q3 revenue 4.2M, up 12%." } }] },
      "Team - Q3 revenue reached 4.2M, up 12% on Q2. Great work!",
      "Here's the email: 'Team - Q3 revenue reached 4.2M, up 12% on Q2. Great work!'",
    ],
    code: P`
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

model = init_chat_model("anthropic:claude-opus-5-5")

@tool
def get_financials(quarter: str) -> str:
    """Revenue figures for a quarter."""
    return "Q3 revenue 4.2M (Q2: 3.75M)"

researcher = create_agent(model, [get_financials], system_prompt="You find numbers. Be precise.")
writer = create_agent(model, [], system_prompt="You write short, friendly team emails.")

@tool
def ask_researcher(question: str) -> str:
    """Ask the research agent for company numbers."""
    return researcher.invoke({"messages": [("user", question)]})["messages"][-1].content

@tool
def ask_writer(brief: str) -> str:
    """Ask the writing agent to draft text from a brief."""
    return writer.invoke({"messages": [("user", brief)]})["messages"][-1].content

supervisor = create_agent(model, [ask_researcher, ask_writer],
                          system_prompt="Coordinate the specialists to complete the task, then reply.")
result = supervisor.invoke({"messages": [("user", "Email the team a summary of Q3 revenue")]})
for m in result["messages"]:
    print(f"{m.type:5}", m.tool_calls[0]["name"] if getattr(m, "tool_calls", None) else m.content)
`,
  }, {
    note: "A supervisor as an explicit LangGraph graph: a router node picks the next specialist with structured output and hands over control with Command(goto=...).",
    source: ["langgraph/types.py (Command)", "langgraph/graph/state.py (StateGraph)"],
    script: [
      { tool_calls: [{ name: "Route", args: { next: "researcher", instruction: "Find Q3 revenue and growth." } }] },
      "Q3 revenue was 4.2M, up 12% on Q2.",
      { tool_calls: [{ name: "Route", args: { next: "writer", instruction: "Write a two-line team email with those numbers." } }] },
      "Team - Q3 revenue reached 4.2M, up 12%. Great work!",
      { tool_calls: [{ name: "Route", args: { next: "FINISH", instruction: "" } }] },
    ],
    code: P`
from typing import Literal

from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

model = init_chat_model("anthropic:claude-opus-5-5")

class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"]
    instruction: str

router = model.with_structured_output(Route)

def supervisor(state: MessagesState) -> Command[Literal["researcher", "writer", "__end__"]]:
    # Step 2: decide which specialist acts next
    decision = router.invoke([SystemMessage("Route the task to researcher or writer, or FINISH.")] + state["messages"])
    print(f"supervisor -> {decision.next}" + (f": {decision.instruction}" if decision.instruction else ""))
    if decision.next == "FINISH":
        return Command(goto=END)                                      # Step 4
    return Command(goto=decision.next, update={"messages": [HumanMessage(decision.instruction, name="supervisor")]})

def specialist(name, prompt):
    def node(state: MessagesState) -> Command[Literal["supervisor"]]:
        reply = model.invoke([SystemMessage(prompt)] + state["messages"])   # Step 3: its own prompt (and tools)
        print(f"  {name}: {reply.content}")
        return Command(goto="supervisor", update={"messages": [AIMessage(reply.content, name=name)]})
    return node

graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", specialist("researcher", "You find numbers."))
graph.add_node("writer", specialist("writer", "You write short emails."))
graph.add_edge(START, "supervisor")
app = graph.compile()
app.invoke({"messages": [("user", "Email the team a summary of Q3 revenue")]})
`,
  });

  L("agent-memory", {
    note: "Short-term memory = a checkpointer (per thread). Long-term memory = a store (per user), reached from tools through the runtime.",
    script: [
      { tool_calls: [{ name: "save_memory", args: { fact: "User is vegetarian" } }] },
      "Noted - I'll keep dinner suggestions vegetarian.",
      { tool_calls: [{ name: "recall", args: { query: "food preferences" } }] },
      "Since you're vegetarian, try Green Leaf near your hotel.",
    ],
    code: P`
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

@tool
def save_memory(fact: str, runtime: ToolRuntime) -> str:
    """Save a lasting fact about the user."""
    runtime.store.put(("memories", "user-42"), fact, {"fact": fact})
    return "saved"

@tool
def recall(query: str, runtime: ToolRuntime) -> str:
    """Look up what we know about the user."""
    return str([item.value["fact"] for item in runtime.store.search(("memories", "user-42"))])

agent = create_agent(init_chat_model("anthropic:claude-opus-5-5"), [save_memory, recall],
                     checkpointer=InMemorySaver(),        # short-term: per thread
                     store=InMemoryStore())                # long-term: across threads

def chat(thread, text):
    out = agent.invoke({"messages": [("user", text)]}, {"configurable": {"thread_id": thread}})
    print(f"[{thread}] {text!r} -> {out['messages'][-1].content}  ({len(out['messages'])} messages in thread)")

chat("thread-1", "I'm vegetarian. Remember that.")
chat("thread-2", "Suggest a dinner place near my hotel.")    # new thread: no history, but the memory is there
`,
  }, {
    note: "The two memories as objects you can query: the checkpointer's per-thread state, and a store with semantic search over memories.",
    source: ["langgraph/store/memory/__init__.py (InMemoryStore)", "langgraph/checkpoint/memory/__init__.py (InMemorySaver)"],
    script: ["Hello!", "Hi again!"],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore

# Short-term (Step 1): a checkpointer keeps each thread's messages
model = init_chat_model("anthropic:claude-opus-5-5")
graph = StateGraph(MessagesState)
graph.add_node("chat", lambda s: {"messages": [model.invoke(s["messages"])]})
graph.add_edge(START, "chat")
app = graph.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "thread-1"}}
app.invoke({"messages": [("user", "hi")]}, cfg)
app.invoke({"messages": [("user", "hi again")]}, cfg)
print("thread-1 holds", len(app.get_state(cfg).values["messages"]), "messages")
print("thread-2 holds", len(app.get_state({"configurable": {"thread_id": "thread-2"}}).values.get("messages", [])))

# Long-term (Steps 2-4): a store, namespaced per user, with semantic search
store = InMemoryStore(index={"embed": init_embeddings("openai:text-embedding-3-small"), "dims": 384})
ns = ("memories", "user-42")
for key, fact in [("diet", "User is vegetarian"), ("seat", "Prefers an aisle seat on flights"),
                  ("lang", "Writes in British English")]:
    store.put(ns, key, {"text": fact})
for item in store.search(ns, query="what food should I suggest?", limit=2):
    print(f"{item.score:.2f}  {item.value['text']}")
`,
  });

  L("hitl", {
    note: "One middleware line pauses the agent before chosen tools run. A person approves, edits or rejects; the same thread_id resumes it.",
    script: [
      { tool_calls: [{ name: "send_email", args: { to: "ravi@example.com", body: "Hi Ravi, the invoice is attached." } }] },
      "Done - the email was sent with your edits.",
    ],
    code: P`
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}: {body}"

agent = create_agent(init_chat_model("anthropic:claude-opus-5-5"), [send_email],
                     middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})],
                     checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "invoice-42"}}

result = agent.invoke({"messages": [("user", "Email Ravi that the invoice is attached")]}, config)
request = result["__interrupt__"][0].value                      # paused: nothing was sent
print("needs approval:", request["action_requests"][0]["args"])

edited = {"type": "edit", "edited_action": {"name": "send_email",
          "args": {"to": "ravi@example.com", "body": "Hi Ravi, invoice #881 is attached. Thanks!"}}}
result = agent.invoke(Command(resume={"decisions": [edited]}), config)   # same thread_id
print("tool ran:", result["messages"][-2].content.splitlines()[-1])   # the edited call ran
print("agent:", result["messages"][-1].content)
`,
  }, {
    note: "The mechanism the middleware uses: interrupt() inside a node, a checkpointer, and Command(resume=...).",
    source: ["langgraph/types.py (interrupt, Command)"],
    code: P`
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

class State(TypedDict):
    to: str
    draft: str
    sent: bool

def draft(state: State):
    return {"draft": f"Hi {state['to']}, the invoice is attached."}

def approve(state: State):
    decision = interrupt({"action": "send_email", "draft": state["draft"]})   # Step 2: pause here
    print("  resumed with:", decision)                                        # Step 6: the person's answer
    return {"draft": decision.get("edited", state["draft"]), "sent": decision["approved"]}

graph = StateGraph(State)
graph.add_node("draft", draft)
graph.add_node("approve", approve)
graph.add_edge(START, "draft")
graph.add_edge("draft", "approve")
graph.add_edge("approve", END)
app = graph.compile(checkpointer=InMemorySaver())        # Step 3: state saved at the pause

config = {"configurable": {"thread_id": "t-9"}}          # Step 1
out = app.invoke({"to": "Ravi", "draft": "", "sent": False}, config)
print("paused with:", out["__interrupt__"][0].value)
snapshot = app.get_state(config)
print("waiting at node:", snapshot.next)                 # Step 4: your UI shows this to a person

final = app.invoke(Command(resume={"approved": True, "edited": "Hi Ravi, invoice #881 is attached."}), config)  # Step 5
print("final state:", final)
`,
  });

  L("delegated-auth", {
    note: "The user's token travels in the runtime context. The model never sees it (it isn't a tool argument); the tool passes it to the API.",
    script: [
      { tool_calls: [{ name: "read_calendar", args: { day: "today" } }] },
      "You have a design review at 10:00 and lunch with Arjun at 13:00.",
    ],
    code: P`
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool

@dataclass
class UserContext:
    user_id: str
    access_token: str            # issued by your identity provider after the user consented

def calendar_api(token, day):    # stands in for the real API, which checks the token itself
    events = {"meera": ["10:00 design review", "13:00 lunch with Arjun"]}
    return events.get(TOKENS.get(token, ""), "401 unauthorized")
TOKENS = {"tok-meera-read": "meera"}

@tool
def read_calendar(day: str, runtime: ToolRuntime[UserContext]) -> str:
    """Read the signed-in user's calendar for a day."""
    return str(calendar_api(runtime.context.access_token, day))   # the USER's token, not a service account

print("what the model sees of the tool:", read_calendar.args)     # no token in here
agent = create_agent(init_chat_model("anthropic:claude-opus-5-5"), [read_calendar], context_schema=UserContext)
result = agent.invoke({"messages": [("user", "What's on my calendar today?")]},
                      context=UserContext(user_id="meera", access_token="tok-meera-read"))
print(result["messages"][-1].content)
`,
  }, {
    note: "What an auth library and the API do with that token: a token exchange for a narrow, short-lived token, then checks of audience, expiry and scope. The identity provider is mocked; the request shape is the standard OAuth token exchange (RFC 8693).",
    source: ["RFC 8693: OAuth 2.0 Token Exchange"],
    code: P`
import base64
import json
import time
import httpx

def fake_idp(request):                     # stands in for your identity provider's /token endpoint
    form = dict(x.split("=") for x in request.content.decode().split("&"))
    claims = {"sub": "meera", "aud": form["audience"], "scope": form["scope"].replace("+", " "), "exp": int(time.time()) + 300}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return httpx.Response(200, json={"access_token": f"header.{payload}.signature", "expires_in": 300})

idp = httpx.Client(transport=httpx.MockTransport(fake_idp))
# Steps 1-2: swap the user's login token for a narrow token for ONE API
resp = idp.post("https://idp.example.com/oauth/token", data={
    "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
    "subject_token": "<the user's token>",
    "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
    "audience": "calendar-api",
    "scope": "calendar.read",
})
token = resp.json()["access_token"]

# Step 4: what the API checks on every call (after verifying the signature)
part = token.split(".")[1]
claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
print("claims:", {k: claims[k] for k in ("sub", "aud", "scope")}, f"expires in {claims['exp'] - int(time.time())}s")
for needed in ("calendar.read", "calendar.write"):
    ok = claims["aud"] == "calendar-api" and claims["exp"] > time.time() and needed in claims["scope"].split()
    print(f"{needed}: {'allowed' if ok else '403 forbidden'}")
`,
  });

  // ===================================================================== LANGCHAIN & LANGGRAPH
  L("lcel", {
    note: "Pipe components together; the chain gets invoke, stream and batch for free.",
    script: ["Paris.", "Paris is the capital of France.", "Paris.", "Tokyo."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("What is the capital of {country}? One word.")
model = init_chat_model("anthropic:claude-opus-5-5")
chain = prompt | model | StrOutputParser()

print(chain.invoke({"country": "France"}))
print([chunk for chunk in chain.stream({"country": "France"})])          # tokens as they arrive
print(chain.batch([{"country": "France"}, {"country": "Japan"}]))       # in parallel
`,
  }, {
    note: "What | builds and what invoke does with it, using the real classes.",
    source: ["langchain_core/runnables/base.py (Runnable.__or__, RunnableSequence.invoke)"],
    script: ["Paris.", "Paris."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

prompt = ChatPromptTemplate.from_template("What is the capital of {country}? One word.")
model = init_chat_model("anthropic:claude-opus-5-5")
parser = StrOutputParser()

chain = prompt | model | parser                       # Step 2: __or__ -> RunnableSequence
print(type(chain).__name__, "->", [type(s).__name__ for s in chain.steps])
same = RunnableSequence(prompt, model, parser)        # exactly what | built
print(same.get_graph().draw_mermaid().count("-->"), "arrows in the drawn graph")

x = {"country": "France"}                             # Step 3: what invoke does
for step in chain.steps:
    x = step.invoke(x)
    kind = "str" if isinstance(x, str) else type(x).__name__
    print(f"after {type(step).__name__:18} -> {kind}: {str(x)[:55]!r}")
`,
  });

  L("rag-chain", {
    note: "The classic LCEL RAG chain: a dict at the start sends the question to the retriever and keeps it for the prompt.",
    script: ["LCEL composes runnables with the | operator."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

retriever = InMemoryVectorStore.from_texts(
    ["LangGraph adds state and loops.", "LCEL composes runnables with the | operator.", "Paris is in France."],
    init_embeddings("openai:text-embedding-3-small")).as_retriever(search_kwargs={"k": 1})

def format_docs(docs):
    return "\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template("Use only this context:\n{context}\n\nQuestion: {question}")
chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
         | prompt | init_chat_model("anthropic:claude-opus-5-5") | StrOutputParser())
print(chain.invoke("What does LCEL do?"))
`,
  }, {
    note: "The same chain built explicitly: the dict is a RunnableParallel. RunnablePassthrough.assign keeps the documents so you can return citations.",
    source: ["langchain_core/runnables/base.py (RunnableParallel)", "langchain_core/runnables/passthrough.py"],
    script: ["LCEL composes runnables with the | operator."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

retriever = InMemoryVectorStore.from_texts(
    ["LangGraph adds state and loops.", "LCEL composes runnables with the | operator.", "Paris is in France."],
    init_embeddings("openai:text-embedding-3-small")).as_retriever(search_kwargs={"k": 1})
prompt = ChatPromptTemplate.from_template("Use only this context:\n{context}\n\nQuestion: {question}")
model = init_chat_model("anthropic:claude-opus-5-5")

# Steps 1-2: the dict IS a RunnableParallel - same input to every branch
setup = RunnableParallel(docs=retriever, question=RunnablePassthrough())
print("after the parallel step:", setup.invoke("What does LCEL do?"))

# Steps 3-5: format the docs into {context}, keep docs in the output for citations
answer = (lambda d: {"context": "\n".join(x.page_content for x in d["docs"]), "question": d["question"]}) | prompt | model | StrOutputParser()
chain = setup | RunnablePassthrough.assign(answer=answer)
out = chain.invoke("What does LCEL do?")
print("answer:", out["answer"], "| sources:", [d.page_content for d in out["docs"]])
`,
  });

  L("langgraph-run", {
    note: "A graph you build yourself: state, nodes, a conditional edge. This is the agent loop as an explicit graph.",
    script: [{ tool_calls: [{ name: "search", args: { q: "langgraph" } }] }, "LangGraph runs your nodes in steps over shared state."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def search(q: str) -> str:
    """Search the docs."""
    return "LangGraph is a runtime for stateful, multi-step agent graphs."

model = init_chat_model("anthropic:claude-opus-5-5").bind_tools([search])

graph = StateGraph(MessagesState)
graph.add_node("agent", lambda state: {"messages": [model.invoke(state["messages"])]})
graph.add_node("tools", ToolNode([search]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)     # tools if it asked for one, else END
graph.add_edge("tools", "agent")
app = graph.compile()

result = app.invoke({"messages": [("user", "What is LangGraph?")]})
print(result["messages"][-1].content)
`,
  }, {
    note: "Watching the engine run that graph step by step: which nodes run in each step, and what each returned.",
    source: ["langgraph/pregel/main.py (Pregel.stream)"],
    script: [{ tool_calls: [{ name: "search", args: { q: "langgraph" } }] }, "LangGraph runs your nodes in steps over shared state."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def search(q: str) -> str:
    """Search the docs."""
    return "LangGraph is a runtime for stateful, multi-step agent graphs."

model = init_chat_model("anthropic:claude-opus-5-5").bind_tools([search])
graph = StateGraph(MessagesState)
graph.add_node("agent", lambda state: {"messages": [model.invoke(state["messages"])]})
graph.add_node("tools", ToolNode([search]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
app = graph.compile()

for e in app.get_graph().edges:                               # the structure you defined
    print(f"edge {e.source} -> {e.target}" + ("  (conditional)" if e.conditional else ""))
for event in app.stream({"messages": [("user", "What is LangGraph?")]}, stream_mode="debug"):
    if event["type"] == "task_result":                      # Steps 2-5, one line per node run
        update = dict(event["payload"]["result"])
        msg = update["messages"][-1]
        print(f"step {event['step']}: {event['payload']['name']:5} -> {msg.type}: {getattr(msg, 'tool_calls', None) or msg.content}")
`,
  });

  L("reducers", {
    note: "Annotate state keys with a reducer and parallel nodes can safely write to the same key.",
    code: P`
import operator
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    query: str
    results: Annotated[list, operator.add]       # reducer: concatenate updates

def web(state: State):
    return {"results": [f"web hit for {state['query']}"]}

def docs(state: State):
    return {"results": [f"docs hit for {state['query']}"]}

graph = StateGraph(State)
graph.add_node("web", web)
graph.add_node("docs", docs)
graph.add_edge(START, "web")                      # both run in the same step (fan-out)
graph.add_edge(START, "docs")
graph.add_edge("web", END)
graph.add_edge("docs", END)
print(graph.compile().invoke({"query": "refund policy", "results": []}))
`,
  }, {
    note: "What happens without a reducer, and what add_messages does: append, replace by id, delete with RemoveMessage.",
    source: ["langgraph/graph/message.py (add_messages)", "langgraph/errors.py (InvalidUpdateError)"],
    code: P`
from typing import TypedDict
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

class NoReducer(TypedDict):
    results: list                               # Step 4: no reducer -> "last write wins"

g = StateGraph(NoReducer)
g.add_node("web", lambda s: {"results": ["web"]})
g.add_node("docs", lambda s: {"results": ["docs"]})
g.add_edge(START, "web")
g.add_edge(START, "docs")
g.add_edge("web", END)
g.add_edge("docs", END)
try:
    g.compile().invoke({"results": []})
except InvalidUpdateError as err:
    print("two writes, no reducer:", str(err).split(".")[0])

# Steps 2-3: add_messages(current, update)
history = [HumanMessage("hi", id="1"), AIMessage("draft", id="2")]
print([m.content for m in add_messages(history, [AIMessage("hello!", id="3")])])     # new id -> append
print([m.content for m in add_messages(history, [AIMessage("final", id="2")])])      # same id -> replace
print([m.content for m in add_messages(history, [RemoveMessage(id="1")])])           # RemoveMessage -> delete
`,
  });

  L("checkpoint", {
    note: "Compile with a checkpointer and pass a thread_id: each thread remembers its own conversation.",
    script: ["Nice to meet you, Asha!", "Your name is Asha.", "I don't know your name yet."],
    code: P`
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

model = init_chat_model("anthropic:claude-opus-5-5")
graph = StateGraph(MessagesState)
graph.add_node("chat", lambda state: {"messages": [model.invoke(state["messages"])]})
graph.add_edge(START, "chat")
app = graph.compile(checkpointer=InMemorySaver())   # in production: a Postgres or SQLite checkpointer

def say(thread, text):
    out = app.invoke({"messages": [("user", text)]}, {"configurable": {"thread_id": thread}})
    print(f"[{thread}] {text} -> {out['messages'][-1].content}")

say("asha", "Hi, I'm Asha.")
say("asha", "What's my name?")          # same thread: the history is loaded
say("bob", "What's my name?")           # different thread: empty history
`,
  }, {
    note: "The snapshots themselves: every step's saved state, and time travel by resuming from an older checkpoint.",
    source: ["langgraph/checkpoint/memory/__init__.py", "langgraph/pregel/main.py (get_state_history)"],
    script: ["Nice to meet you, Asha!", "Your name is Asha.", "You told me your name is Asha."],
    code: P`
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

model = init_chat_model("anthropic:claude-opus-5-5")
graph = StateGraph(MessagesState)
graph.add_node("chat", lambda state: {"messages": [model.invoke(state["messages"])]})
graph.add_edge(START, "chat")
app = graph.compile(checkpointer=InMemorySaver())            # Step 1
config = {"configurable": {"thread_id": "asha"}}             # Step 2
app.invoke({"messages": [("user", "Hi, I'm Asha.")]}, config)
app.invoke({"messages": [("user", "What's my name?")]}, config)

history = list(app.get_state_history(config))                # Step 5: every snapshot, newest first
for snap in history:                                          # Step 3: one per step
    print(f"step {snap.metadata['step']:2}  next={snap.next}  messages={len(snap.values.get('messages', []))}")

# time travel: go back to the snapshot just before the second answer and run again from there
before = next(s for s in history if s.next == ("chat",))
replay = app.invoke(None, before.config)                      # a new branch from that point
print(f"replayed from step {before.metadata['step']}: new answer -> {replay['messages'][-1].content!r}")
`,
  });

  L("streaming", {
    note: "A FastAPI endpoint that streams model tokens to the browser as Server-Sent Events.",
    script: ["Streaming shows tokens as soon as they exist."],
    code: P`
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from langchain.chat_models import init_chat_model

app = FastAPI()
model = init_chat_model("anthropic:claude-opus-5-5")

@app.get("/chat/stream")
async def chat_stream(q: str):
    async def events():
        async for chunk in model.astream(q):                       # Step 2: stream from the model
            if chunk.content:
                yield f"data: {json.dumps({'delta': chunk.content})}\n\n"   # Step 3: forward each chunk
        yield "event: done\ndata: {}\n\n"                          # Step 5
    return StreamingResponse(events(), media_type="text/event-stream")

# Step 1 + 4: a client reading the stream (in a browser: EventSource or fetch with a reader)
with TestClient(app).stream("GET", "/chat/stream", params={"q": "why stream?"}) as response:
    print(response.headers["content-type"])
    for line in response.iter_lines():
        if line:
            print(line)
`,
  }, {
    note: "What astream builds on: every runnable emits start/stream/end events, and chunks are merged with +.",
    source: ["langchain_core/tracers/event_stream.py", "langchain_core/messages/ai.py (AIMessageChunk.__add__)"],
    script: ["Tokens arrive one by one.", "Tokens arrive one by one."],
    code: P`
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chain = ChatPromptTemplate.from_template("{q}") | init_chat_model("anthropic:claude-opus-5-5") | StrOutputParser()

async def main():
    async for ev in chain.astream_events({"q": "hi"}, version="v2"):
        data = ev["data"].get("chunk")
        print(f"{ev['event']:24} {ev['name']:22} {getattr(data, 'content', data) if data is not None else ''!r}")

asyncio.run(main())

# chunks are AIMessageChunk objects; + merges them into the full message
model = init_chat_model("anthropic:claude-opus-5-5")
full = None
for chunk in model.stream("hi"):
    full = chunk if full is None else full + chunk
print(type(full).__name__, repr(full.content))
`,
  });

  L("callbacks", {
    note: "A callback handler sees every model, tool and chain start and end. LangSmith is the same mechanism, switched on with environment variables.",
    script: ["Refunds take 5 days."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Spy(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kw):
        print("MODEL IN: ", [m.content for m in messages[0]])
    def on_llm_end(self, response, **kw):
        print("MODEL OUT:", response.generations[0][0].text)
    def on_chain_end(self, outputs, *, parent_run_id=None, **kw):
        if parent_run_id is None:                      # only the outermost run, not every step
            print("CHAIN END:", repr(outputs))

chain = ChatPromptTemplate.from_template("Answer briefly: {q}") | init_chat_model("anthropic:claude-opus-5-5") | StrOutputParser()
chain.invoke({"q": "How long do refunds take?"}, config={"callbacks": [Spy()]})
# For LangSmith instead: export LANGSMITH_TRACING=true LANGSMITH_API_KEY=...  (no code changes)
`,
  }, {
    note: "How a tracer rebuilds the run tree: every event carries run_id and parent_run_id.",
    source: ["langchain_core/tracers/base.py", "langchain_core/callbacks/manager.py"],
    script: ["Refunds take 5 days."],
    code: P`
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class TreeTracer(BaseCallbackHandler):                 # what LangSmith's tracer does, minus the upload
    def __init__(self):
        self.runs = {}
    def _start(self, name, run_id, parent_run_id):
        self.runs[run_id] = {"name": name, "parent": parent_run_id}     # Step 1: id + parent
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
        self._start(kw.get("name") or "chain", run_id, parent_run_id)
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        self._start("chat_model", run_id, parent_run_id)

tracer = TreeTracer()
chain = ChatPromptTemplate.from_template("Answer briefly: {q}") | init_chat_model("anthropic:claude-opus-5-5") | StrOutputParser()
chain.invoke({"q": "How long do refunds take?"}, config={"callbacks": [tracer], "run_name": "refund_bot"})

def show(parent=None, depth=0):                        # Step 5: rebuild the tree from parent ids
    for run_id, r in tracer.runs.items():
        if r["parent"] == parent:
            print("  " * depth + "- " + r["name"])
            show(run_id, depth + 1)
show()
`,
  });
})(typeof window !== "undefined" ? window : globalThis);

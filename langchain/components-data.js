// Content for components.html. Add a card by adding an object to ITEMS.
// level: "daily" (use daily) | "often" (use often) | "know" (good to know)
window.COMPONENTS = {
  cats: [
    { id: "models", title: "Models", blurb: "The LLM itself, plus the model that turns text into numbers." },
    { id: "messages", title: "Messages", blurb: "What goes in and out of a chat model." },
    { id: "prompts", title: "Prompts", blurb: "Reusable instructions with {placeholders}." },
    { id: "output", title: "Structured output & parsers", blurb: "Turn the model's reply into a string, a dict or your own class." },
    { id: "compose", title: "Composing chains (LCEL)", blurb: "Glue that connects parts and makes them reliable." },
    { id: "rag", title: "Documents & RAG", blurb: "Get your own data into the prompt." },
    { id: "tools", title: "Tools & agents", blurb: "Let the model call your functions." },
    { id: "graph", title: "LangGraph", blurb: "State, loops, memory and human approval." },
    { id: "observe", title: "Debugging & tracing", blurb: "See what actually happened inside." },
  ],
  items: [
    // ---------- Models ----------
    { cat: "models", name: "Chat model", alias: "ChatAnthropic · ChatOpenAI", level: "daily",
      what: "A connection to an LLM provider. You send messages and get an AIMessage back.",
      when: "Every LangChain app. It's the brain.",
      pkg: "pip install langchain-anthropic",
      code: `from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-5-5",
                      temperature=0)
reply = model.invoke("Explain RAG in one line")
print(reply.content)`,
      tip: "Put the API key in an environment variable (ANTHROPIC_API_KEY), never in code. temperature=0 makes answers more repeatable." },
    { cat: "models", name: "init_chat_model", alias: "any provider from a string", level: "often",
      what: "One function that creates any provider's chat model from a \"provider:model\" string.",
      when: "You want to switch providers from a config file without changing imports.",
      pkg: "pip install langchain",
      code: `from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-5-5")
# another provider = another string,
# as long as its package is installed`,
      tip: "The provider's own package (langchain-anthropic, langchain-openai...) still has to be installed." },
    { cat: "models", name: "Embeddings", alias: "OpenAIEmbeddings · HuggingFaceEmbeddings", level: "daily",
      what: "A model that turns text into a list of numbers (a vector). Texts with similar meaning get vectors that are close together.",
      when: "Semantic search, RAG, finding duplicates, clustering.",
      pkg: "pip install langchain-openai",
      code: `from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vec = emb.embed_query("refund policy")
len(vec)   # 1536 numbers`,
      tip: "Use the same embedding model for your documents and your questions. Anthropic has no embedding model; common picks are OpenAI, Voyage, Cohere or a free Hugging Face model." },

    // ---------- Messages ----------
    { cat: "messages", name: "Message types", alias: "System · Human · AI · Tool", level: "daily",
      what: "The four roles in a chat: SystemMessage sets the rules, HumanMessage is the user, AIMessage is the model, ToolMessage is a tool's result.",
      when: "Multi-turn chats, or whenever you build the history yourself.",
      pkg: "included in langchain-core",
      code: `from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage)

msgs = [SystemMessage("You are concise."),
        HumanMessage("Hi, I'm Sam."),
        AIMessage("Hi Sam!"),
        HumanMessage("What's my name?")]
model.invoke(msgs)   # "Your name is Sam."`,
      tip: "The model only knows what's in this list. Shortcut: tuples like (\"user\", \"Hi\") work almost everywhere." },
    { cat: "messages", name: "AIMessage fields", alias: ".content · .tool_calls · .usage_metadata", level: "often",
      what: "What every chat model returns. The text is in .content; everything else sits beside it.",
      when: "Reading the answer, checking for tool calls, tracking cost.",
      pkg: "included in langchain-core",
      code: `reply = model.invoke("Hi")

reply.content          # the text
reply.tool_calls       # tools it wants to run
reply.usage_metadata   # tokens in / out -> cost`,
      tip: "Log usage_metadata from day one. It's how you'll know what each request costs." },

    // ---------- Prompts ----------
    { cat: "prompts", name: "ChatPromptTemplate", alias: "", level: "daily",
      what: "A reusable list of messages with {placeholders} that get filled in at run time.",
      when: "Any prompt you'll run more than once with different values.",
      pkg: "included in langchain-core",
      code: `from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("user", "{question}"),
])
prompt.invoke({"role": "tutor", "question": "2+2?"})`,
      tip: "Every {name} must be supplied when you invoke, or you get a KeyError. To write a real brace, double it: {{ }}." },
    { cat: "prompts", name: "MessagesPlaceholder", alias: "", level: "often",
      what: "A slot in a prompt where a whole list of messages goes, usually the chat history.",
      when: "Chatbots that send earlier turns along with the new question.",
      pkg: "included in langchain-core",
      code: `from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history"),
    ("user", "{question}"),
])
prompt.invoke({"history": past_msgs, "question": "And then?"})`,
      tip: "\"history\" must be a list of messages, not a string." },

    // ---------- Output ----------
    { cat: "output", name: "StrOutputParser", alias: "", level: "daily",
      what: "Takes the AIMessage and returns only the text, as a plain string.",
      when: "The last step of most text chains.",
      pkg: "included in langchain-core",
      code: `from langchain_core.output_parsers import StrOutputParser

chain = prompt | model | StrOutputParser()
chain.invoke({...})   # "just the text"`,
      tip: "It streams token by token, so it's safe to use in chat UIs." },
    { cat: "output", name: "with_structured_output", alias: "Pydantic in, object out", level: "daily",
      what: "Makes the model reply in your exact shape (a Pydantic class). It uses tool calling under the hood.",
      when: "Extraction, classification, or any time your code reads the answer.",
      pkg: "pip install pydantic",
      code: `from pydantic import BaseModel, Field

class Ticket(BaseModel):
    category: str = Field(description="billing, tech or other")
    urgent: bool

extract = model.with_structured_output(Ticket)
t = extract.invoke("I was charged twice!!")
t.category, t.urgent   # ("billing", True)`,
      tip: "Prefer this to parsing JSON out of text yourself. Field descriptions work as instructions to the model." },
    { cat: "output", name: "JsonOutputParser", alias: "", level: "know",
      what: "Pulls JSON out of the model's text reply. It can also stream partial JSON.",
      when: "Models without tool calling, or a quick dict without defining a class.",
      pkg: "included in langchain-core",
      code: `from langchain_core.output_parsers import JsonOutputParser

chain = prompt | model | JsonOutputParser()
chain.invoke({...})   # {"name": "...", "age": 31}`,
      tip: "Tell the model in the prompt to answer in JSON. with_structured_output is usually more reliable." },

    // ---------- Compose ----------
    { cat: "compose", name: "The pipe |", alias: "RunnableSequence", level: "daily",
      what: "Connects parts so each one's output becomes the next one's input. The result is itself a Runnable.",
      when: "Every chain.",
      pkg: "included in langchain-core",
      code: `chain = prompt | model | StrOutputParser()

chain.invoke({"topic": "cats"})        # one
chain.batch([{...}, {...}])            # many
for piece in chain.stream({...}): ...  # live`,
      tip: "If a chain breaks, check that each step's output type matches what the next step expects." },
    { cat: "compose", name: "RunnablePassthrough", alias: "", level: "daily",
      what: "Passes the input through unchanged. Used to send the user's question along beside other values.",
      when: "RAG chains: the retriever fills context while the question passes through.",
      pkg: "included in langchain-core",
      code: `from langchain_core.runnables import RunnablePassthrough

chain = ({"context": retriever,
          "question": RunnablePassthrough()}
         | prompt | model | StrOutputParser())`,
      tip: "RunnablePassthrough.assign(key=fn) adds a new key and keeps all the others." },
    { cat: "compose", name: "RunnableParallel", alias: "a dict in a chain", level: "often",
      what: "Runs several parts on the same input at the same time and returns a dict of results.",
      when: "You need two or more independent results from one input.",
      pkg: "included in langchain-core",
      code: `from langchain_core.runnables import RunnableParallel

both = RunnableParallel(summary=summary_chain,
                        keywords=keyword_chain)
both.invoke({"text": article})
# {"summary": "...", "keywords": "..."}`,
      tip: "A plain dict inside a chain is turned into a RunnableParallel for you." },
    { cat: "compose", name: "RunnableLambda", alias: "", level: "often",
      what: "Wraps a normal Python function so it can sit in a chain.",
      when: "Small custom steps: cleaning text, reformatting, lookups.",
      pkg: "included in langchain-core",
      code: `from langchain_core.runnables import RunnableLambda

clean = RunnableLambda(lambda s: s.strip().lower())
chain = prompt | model | StrOutputParser() | clean`,
      tip: "A plain function needs the whole input, so streaming stops at that step unless you write it as a generator." },
    { cat: "compose", name: "with_retry / with_fallbacks", alias: "", level: "often",
      what: "Wrappers that work on any Runnable: try again on errors, or switch to a backup.",
      when: "Production: rate limits, timeouts, provider outages.",
      pkg: "included in langchain-core",
      code: `safe = model.with_retry(stop_after_attempt=3)

robust = main_model.with_fallbacks([backup_model])
chain = prompt | robust | StrOutputParser()`,
      tip: "Fallbacks can be a cheaper model or a different provider." },

    // ---------- RAG ----------
    { cat: "rag", name: "Document", alias: "", level: "daily",
      what: "LangChain's unit of text: page_content plus metadata such as the source and page.",
      when: "Everything in RAG passes Documents around.",
      pkg: "included in langchain-core",
      code: `from langchain_core.documents import Document

doc = Document(page_content="Refunds within 30 days",
               metadata={"source": "policy.pdf", "page": 3})`,
      tip: "Keep good metadata. It's what you show as citations and use to filter searches." },
    { cat: "rag", name: "Document loaders", alias: "PyPDFLoader · WebBaseLoader · CSVLoader", level: "daily",
      what: "Read files, websites or databases into a list of Documents.",
      when: "Step 1 of indexing.",
      pkg: "pip install langchain-community pypdf",
      code: `from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("policy.pdf").load()
len(docs)                # one Document per page
docs[0].metadata         # {"source": ..., "page": 0}`,
      tip: "There are hundreds of loaders. Search the docs for your source before writing your own." },
    { cat: "rag", name: "RecursiveCharacterTextSplitter", alias: "text splitter", level: "daily",
      what: "Cuts long documents into overlapping chunks, trying to break at paragraphs, then lines, then words.",
      when: "Step 2 of indexing, before embedding.",
      pkg: "pip install langchain-text-splitters",
      code: `from langchain_text_splitters import (
    RecursiveCharacterTextSplitter)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)`,
      tip: "chunk_size counts characters, not tokens. Start around 500 to 1000 with 10 to 20% overlap, then tune by looking at what gets retrieved." },
    { cat: "rag", name: "Vector store", alias: "Chroma · FAISS · pgvector", level: "daily",
      what: "A database that stores vectors with their chunks and quickly finds the ones closest to a question.",
      when: "Step 3 of indexing, and the search engine behind your retriever.",
      pkg: "pip install langchain-chroma",
      code: `from langchain_chroma import Chroma

store = Chroma.from_documents(chunks, emb,
            persist_directory="./db")
store.similarity_search("refund?", k=4)`,
      tip: "Chroma and FAISS are great for learning. pgvector, Pinecone, Qdrant or Weaviate are common in production. persist_directory keeps the data on disk." },
    { cat: "rag", name: "Retriever", alias: "store.as_retriever()", level: "daily",
      what: "Anything that takes a question and returns relevant Documents. It's a Runnable, so it fits in a chain.",
      when: "The first step of every RAG question.",
      pkg: "included in langchain-core",
      code: `retriever = store.as_retriever(
    search_type="mmr",        # varied, less repetitive
    search_kwargs={"k": 4})
retriever.invoke("Can I get a refund?")`,
      tip: "When answers are wrong, print what the retriever returned first. Most RAG bugs are retrieval bugs." },

    // ---------- Tools ----------
    { cat: "tools", name: "@tool", alias: "", level: "daily",
      what: "Turns a Python function into a tool the model can ask to call.",
      when: "The model needs live data or must take an action.",
      pkg: "included in langchain-core",
      code: `from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return weather_api.lookup(city)`,
      tip: "The name, type hints and docstring become what the model reads. Write the docstring for the model, not for humans." },
    { cat: "tools", name: "bind_tools", alias: "", level: "often",
      what: "Tells the model which tools exist. It may then reply with tool_calls instead of text.",
      when: "Building your own tool loop, or checking what the model would call.",
      pkg: "included in langchain-core",
      code: `llm = model.bind_tools([get_weather])
msg = llm.invoke("Weather in Paris?")
msg.tool_calls
# [{"name": "get_weather",
#   "args": {"city": "Paris"}, "id": ...}]`,
      tip: "bind_tools runs nothing by itself. Something has to execute the call: you, ToolNode, or an agent." },
    { cat: "tools", name: "create_agent", alias: "", level: "daily",
      what: "A ready-made agent: the model plus your tools, looping until it can answer.",
      when: "Most tool-using assistants. Start here before building a custom graph.",
      pkg: "pip install langchain",
      code: `from langchain.agents import create_agent

agent = create_agent(model, tools=[get_weather],
                     system_prompt="Be brief.")
agent.invoke({"messages": [("user", "Paris weather?")]})`,
      tip: "It's built on LangGraph and returns a compiled graph, so you can pass checkpointer= for memory." },

    // ---------- Graph ----------
    { cat: "graph", name: "StateGraph · START · END", alias: "", level: "daily",
      what: "The graph builder. You add nodes (functions) and edges (arrows), then compile it into an app.",
      when: "Any custom workflow: loops, branches, several steps or agents.",
      pkg: "pip install langgraph",
      code: `from langgraph.graph import StateGraph, START, END

g = StateGraph(State)
g.add_node("chatbot", chatbot)
g.add_edge(START, "chatbot")
g.add_edge("chatbot", END)
app = g.compile()`,
      tip: "A node returns only the keys it changes, not the whole state." },
    { cat: "graph", name: "MessagesState · add_messages", alias: "reducers", level: "often",
      what: "A ready-made state with a messages list whose reducer appends new messages instead of replacing the list.",
      when: "Almost every chat or agent graph.",
      pkg: "pip install langgraph",
      code: `from langgraph.graph import MessagesState

class State(MessagesState):
    summary: str      # add your own keys`,
      tip: "Without a reducer, a node that returns messages would overwrite the whole history." },
    { cat: "graph", name: "add_conditional_edges", alias: "", level: "daily",
      what: "An arrow with a decision: a function looks at the state and returns the name of the next node.",
      when: "if/else in a graph, and every loop.",
      pkg: "pip install langgraph",
      code: `def route(state):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

g.add_conditional_edges("agent", route)`,
      tip: "This is the only way to make a loop: route back to an earlier node." },
    { cat: "graph", name: "ToolNode · tools_condition", alias: "prebuilt", level: "often",
      what: "A node that runs whatever tools the model asked for, and the matching ready-made route function.",
      when: "Building your own agent graph without writing the tool loop by hand.",
      pkg: "pip install langgraph",
      code: `from langgraph.prebuilt import ToolNode, tools_condition

g.add_node("tools", ToolNode([get_weather]))
g.add_conditional_edges("agent", tools_condition)
g.add_edge("tools", "agent")`,
      tip: "tools_condition is the route function from the card above, already written for you." },
    { cat: "graph", name: "Checkpointer · thread_id", alias: "InMemorySaver · SqliteSaver · PostgresSaver", level: "daily",
      what: "Saves the state after every step. Each thread_id is its own saved conversation.",
      when: "Chat memory, resuming after a crash, human approval.",
      pkg: "pip install langgraph",
      code: `from langgraph.checkpoint.memory import InMemorySaver

app = g.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "user-42"}}
app.invoke({"messages": [("user", "Hi")]}, cfg)`,
      tip: "InMemorySaver forgets everything on restart. Real apps use SqliteSaver or PostgresSaver (separate packages)." },
    { cat: "graph", name: "interrupt · Command", alias: "human in the loop", level: "know",
      what: "interrupt() pauses the graph and hands a value to your app. Command(resume=...) continues from that exact point.",
      when: "Approvals, asking the user a question halfway, editing a draft.",
      pkg: "pip install langgraph",
      code: `from langgraph.types import interrupt, Command

def approve(state):
    answer = interrupt({"draft": state["draft"]})
    return {"approved": answer == "yes"}

app.invoke(Command(resume="yes"), cfg)`,
      tip: "It needs a checkpointer, because the paused run is stored there." },

    // ---------- Observe ----------
    { cat: "observe", name: "LangSmith tracing", alias: "", level: "daily",
      what: "Records every step of every run: inputs, outputs, time and tokens, shown as a tree.",
      when: "Always while developing, and the first thing to check when an answer is wrong.",
      pkg: "pip install langsmith",
      code: `# in your shell or .env file
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...

# then run your code as usual;
# traces appear in the LangSmith UI`,
      tip: "No code changes needed. It works for chains, agents and graphs." },
    { cat: "observe", name: "astream_events", alias: "", level: "know",
      what: "Streams every event from every step: tokens, tool starts, retriever results.",
      when: "UIs that show progress such as \"Searching...\" or \"Calling a tool...\".",
      pkg: "included in langchain-core",
      code: `async for ev in chain.astream_events(inp, version="v2"):
    if ev["event"] == "on_chat_model_stream":
        print(ev["data"]["chunk"].content, end="")
    elif ev["event"] == "on_tool_start":
        print("\\n[calling", ev["name"], "]")`,
      tip: "Use plain stream() when you only need the final text." },
  ],
};

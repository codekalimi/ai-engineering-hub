"""Test prelude for the Level 1 / Level 2 snippets (run by concepts/run-levels.js).

It swaps the network-bound parts for local fakes, so every snippet runs for real, offline:
  - init_chat_model(...)  -> ScriptedChatModel: replies come from the FAKE_SCRIPT env var (a JSON list).
      "text"                                    -> an AIMessage with that text
      {"tool_calls": [{"name": ..., "args": {...}}], "content": "..."} -> an AIMessage with tool calls
      {"error": "message"}                      -> the call raises TimeoutError (a provider outage)
  - init_embeddings(...)  -> LocalEmbeddings: a real small embedding model (all-MiniLM-L6-v2) run
                             locally instead of a paid embedding API.
Everything else (LangChain, LangGraph, the MCP SDK, PyTorch, tokenizers) is the real library.
"""
import hashlib
import itertools
import json
import math
import os
import re
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_QUEUE = list(json.loads(os.environ.get("FAKE_SCRIPT", "[]")))
_IDS = itertools.count(1)


def _next_message() -> AIMessage:
    item = _QUEUE.pop(0) if _QUEUE else "(scripted model: no more replies)"
    if isinstance(item, dict) and "error" in item:          # simulate a provider failure
        raise TimeoutError(item["error"])
    if isinstance(item, str):
        return AIMessage(content=item, usage_metadata={"input_tokens": 50, "output_tokens": len(item.split()), "total_tokens": 50 + len(item.split())})
    calls = [{"name": c["name"], "args": c.get("args", {}), "id": f"call_{next(_IDS)}", "type": "tool_call"}
             for c in item.get("tool_calls", [])]
    return AIMessage(content=item.get("content", ""), tool_calls=calls,
                     usage_metadata={"input_tokens": 50, "output_tokens": 12, "total_tokens": 62})


class ScriptedChatModel(BaseChatModel):
    """A chat model whose answers are scripted. It supports invoke, stream, bind_tools and
    with_structured_output (through tool calls), which is all the snippets need."""

    model_name: str = "scripted-fake"

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=_next_message())])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = _next_message()
        if msg.tool_calls:
            chunk = ChatGenerationChunk(message=AIMessageChunk(
                content="", tool_call_chunks=[{"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                                              for i, c in enumerate(msg.tool_calls)]))
            yield chunk
            return
        for piece in re.findall(r"\S+\s*", str(msg.content)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

    def bind_tools(self, tools, **kwargs):
        return self


class WordHashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (256 dims). Texts sharing words are similar."""

    dim = 256

    def _embed(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            w = w[:5]                                     # crude stemming: refund/refunds match
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _fake_init_chat_model(model: Optional[str] = None, **kwargs: Any):
    return ScriptedChatModel()


class LocalEmbeddings(Embeddings):
    """A real (small) embedding model running locally: sentence-transformers/all-MiniLM-L6-v2."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def _fake_init_embeddings(model: Optional[str] = None, **kwargs: Any):
    return LocalEmbeddings()


import langchain.chat_models as _cm      # noqa: E402
import langchain.embeddings as _em       # noqa: E402

_cm.init_chat_model = _fake_init_chat_model
_em.init_embeddings = _fake_init_embeddings


# ---- Anthropic SDK: real client, mocked HTTP. The SDK still validates every parameter name;
# the reply comes from FAKE_ANTHROPIC (a JSON list of partial message bodies). ----
import anthropic as _anthropic   # noqa: E402
import httpx2 as _httpx          # noqa: E402  (this SDK release uses httpx2)

_ANTHROPIC_REPLIES = list(json.loads(os.environ.get("FAKE_ANTHROPIC", "[]")))


def _anthropic_handler(request):
    sent = json.loads(request.content or b"{}")
    part = _ANTHROPIC_REPLIES.pop(0) if _ANTHROPIC_REPLIES else {}
    body = {"id": "msg_test", "type": "message", "role": "assistant", "model": sent.get("model", "claude"),
            "content": [{"type": "text", "text": "(mocked reply)"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
    body.update(part)
    return _httpx.Response(200, json=body)


_orig_anthropic_init = _anthropic.Anthropic.__init__


def _mocked_anthropic_init(self, *args, **kwargs):
    kwargs.setdefault("api_key", "test-key")
    kwargs["http_client"] = _httpx.Client(transport=_httpx.MockTransport(_anthropic_handler))
    _orig_anthropic_init(self, *args, **kwargs)


_anthropic.Anthropic.__init__ = _mocked_anthropic_init

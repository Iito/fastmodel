"""GGUF model backend — loads GGUF files via llama-cpp-python.

Requires: pip install llama-cpp-python
    or:   pip install fastmodel[gguf]
"""

import time
from typing import ClassVar

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatInput(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048


class ChatOutput(BaseModel):
    content: str = ""
    done: bool = False
    eval_count: int | None = None
    total_duration: float | None = None


class GGUFModel:
    """Serves a GGUF model using llama-cpp-python.

    Satisfies the ServingModule protocol. Can also be used standalone.
    """

    MODULE_NAME: ClassVar[str] = "gguf"
    MODULE_VERSION: ClassVar[str] = "0.1.0"
    INPUT_TYPE: ClassVar[type[BaseModel]] = ChatInput
    OUTPUT_TYPE: ClassVar[type[BaseModel]] = ChatOutput

    def __init__(self, model_path: str | None = None, n_gpu_layers: int = -1, n_ctx: int = 4096):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._llm = None

    def load(self, path: str | None = None):
        """Load the GGUF model into memory."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF backend.\n"
                "Install it with: pip install llama-cpp-python\n"
                "    or: pip install fastmodel[gguf]"
            )

        path = path or self.model_path
        if not path:
            raise ValueError("No model path provided")

        self._llm = Llama(
            model_path=path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )

    @property
    def _has_chat_template(self) -> bool:
        if not self._llm:
            return False
        return any("chat_template" in k for k in self._llm.metadata)

    def _format_prompt(self, messages: list[ChatMessage]) -> str:
        """Format chat messages for models without an embedded chat template."""
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"System: {m.content}")
            elif m.role == "user":
                parts.append(f"User: {m.content}")
            elif m.role == "assistant":
                parts.append(f"Assistant: {m.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def __call__(self, input: ChatInput) -> ChatOutput:
        if not self._llm:
            raise RuntimeError("Model not loaded — call .load() first")

        t0 = time.perf_counter()

        if self._has_chat_template:
            messages = [{"role": m.role, "content": m.content} for m in input.messages]
            result = self._llm.create_chat_completion(
                messages=messages,
                temperature=input.temperature,
                max_tokens=input.max_tokens,
            )
            content = result["choices"][0]["message"]["content"] if result["choices"] else ""
            usage = result.get("usage", {})
        else:
            prompt = self._format_prompt(input.messages)
            result = self._llm.create_completion(
                prompt=prompt,
                temperature=input.temperature,
                max_tokens=input.max_tokens,
            )
            content = result["choices"][0]["text"] if result["choices"] else ""
            usage = result.get("usage", {})

        return ChatOutput(
            content=content.strip(),
            done=True,
            eval_count=usage.get("completion_tokens"),
            total_duration=time.perf_counter() - t0,
        )

    def stream(self, input: ChatInput):
        """Yield ChatOutput chunks for streaming responses."""
        if not self._llm:
            raise RuntimeError("Model not loaded — call .load() first")

        t0 = time.perf_counter()
        total_tokens = 0

        if self._has_chat_template:
            messages = [{"role": m.role, "content": m.content} for m in input.messages]
            for chunk in self._llm.create_chat_completion(
                messages=messages,
                temperature=input.temperature,
                max_tokens=input.max_tokens,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    total_tokens += 1
                    yield ChatOutput(content=token, done=False)
        else:
            prompt = self._format_prompt(input.messages)
            for chunk in self._llm.create_completion(
                prompt=prompt,
                temperature=input.temperature,
                max_tokens=input.max_tokens,
                stream=True,
            ):
                token = chunk["choices"][0]["text"] if chunk["choices"] else ""
                if token:
                    total_tokens += 1
                    yield ChatOutput(content=token, done=False)

        yield ChatOutput(
            content="",
            done=True,
            eval_count=total_tokens,
            total_duration=time.perf_counter() - t0,
        )

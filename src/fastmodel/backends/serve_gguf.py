"""Serve a GGUF model with an Ollama-compatible /api/chat endpoint."""

import json
import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from uvicorn import run as uvicorn_run

from .gguf import ChatInput, ChatMessage, GGUFModel
from .ollama_resolver import list_models, resolve_gguf

logger = logging.getLogger(__name__)


def create_app(model: GGUFModel, model_name: str = "gguf") -> FastAPI:
    """Build a FastAPI app with Ollama-compatible chat endpoint."""
    app = FastAPI(title=f"fastmodel-gguf ({model_name})")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.get("/api/tags")
    def tags():
        """Ollama-compatible model list."""
        return {"models": [{"name": model_name, "size": 0}]}

    @app.post("/api/chat")
    def chat(request: dict):
        """Ollama-compatible /api/chat endpoint."""
        messages = [ChatMessage(**m) for m in request.get("messages", [])]
        stream = request.get("stream", False)
        chat_input = ChatInput(
            messages=messages,
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens", 2048),
        )

        if stream:
            def generate():
                for chunk in model.stream(chat_input):
                    data = {
                        "message": {"role": "assistant", "content": chunk.content},
                        "done": chunk.done,
                    }
                    if chunk.done:
                        data["eval_count"] = chunk.eval_count
                        data["total_duration"] = int((chunk.total_duration or 0) * 1e9)
                    yield json.dumps(data) + "\n"
            return StreamingResponse(generate(), media_type="application/x-ndjson")

        result = model(chat_input)
        return {
            "message": {"role": "assistant", "content": result.content},
            "done": True,
            "eval_count": result.eval_count,
            "total_duration": int((result.total_duration or 0) * 1e9),
        }

    return app


def run_gguf(
    model_name: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    log_level: str = "info",
):
    """Resolve an Ollama model (or direct GGUF path), load it, and serve it."""
    import os
    if os.path.isfile(model_name) and (model_name.endswith(".gguf") or os.path.getsize(model_name) > 100_000_000):
        gguf_path = os.path.abspath(model_name)
    else:
        gguf_path = resolve_gguf(model_name)
    logger.info(f"Resolved {model_name} -> {gguf_path}")

    model = GGUFModel(model_path=gguf_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    model.load()
    logger.info(f"Model loaded: {model_name}")

    app = create_app(model, model_name)
    uvicorn_run(app, host=host, port=port, log_level=log_level)

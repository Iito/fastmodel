"""Resolve Ollama model names to GGUF blob paths on disk.

Ollama stores models in ~/.ollama/models/:
  manifests/registry.ollama.ai/library/<model>/<tag>  — JSON manifest
  blobs/sha256-<hash>                                  — actual GGUF files
"""

import json
import os

OLLAMA_MODELS_DIR = os.path.expanduser("~/.ollama/models")
MANIFESTS_DIR = os.path.join(OLLAMA_MODELS_DIR, "manifests", "registry.ollama.ai", "library")
BLOBS_DIR = os.path.join(OLLAMA_MODELS_DIR, "blobs")

GGUF_MEDIA_TYPE = "application/vnd.ollama.image.model"


def resolve_gguf(model_name: str) -> str:
    """Resolve 'model:tag' to the absolute path of its GGUF blob.

    Examples:
        resolve_gguf("nemotron-3-nano:4b")
        resolve_gguf("llama3.1")  # defaults to 'latest' tag
    """
    if ":" in model_name:
        name, tag = model_name.rsplit(":", 1)
    else:
        name, tag = model_name, "latest"

    manifest_path = os.path.join(MANIFESTS_DIR, name, tag)
    if not os.path.exists(manifest_path):
        available = list_models()
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {manifest_path}\n"
            f"Available: {', '.join(available) or 'none'}"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == GGUF_MEDIA_TYPE:
            digest = layer["digest"]  # sha256:abc123...
            blob_path = os.path.join(BLOBS_DIR, digest.replace(":", "-"))
            if not os.path.exists(blob_path):
                raise FileNotFoundError(f"Blob missing: {blob_path}")
            return blob_path

    raise ValueError(f"No GGUF layer found in manifest for '{model_name}'")


def list_models() -> list[str]:
    """List all locally available Ollama models."""
    if not os.path.isdir(MANIFESTS_DIR):
        return []
    models = []
    for name in sorted(os.listdir(MANIFESTS_DIR)):
        name_dir = os.path.join(MANIFESTS_DIR, name)
        if not os.path.isdir(name_dir):
            continue
        for tag in sorted(os.listdir(name_dir)):
            if not tag.startswith("."):
                models.append(f"{name}:{tag}")
    return models

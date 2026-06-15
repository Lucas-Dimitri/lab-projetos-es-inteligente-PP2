"""LLM model factory following OCP: extend providers without modifying existing code."""

from typing import Any

# Registry maps provider name → lazy builder callable.
# To add a new provider, insert a new entry here — no other file changes needed.
_PROVIDER_REGISTRY: dict[str, type] = {}


def _register(name: str) -> Any:
    """Decorator that registers a provider builder class under the given name."""

    def decorator(cls: type) -> type:
        _PROVIDER_REGISTRY[name] = cls
        return cls

    return decorator


@_register("ollama")
class _OllamaBuilder:
    """Builds an Agno Ollama model."""

    def __call__(self, model_id: str, host: str = "http://localhost:11434") -> Any:
        """Return a configured Ollama model instance.

        Args:
            model_id: Ollama model tag, e.g. 'llama3.2' or 'qwen2.5'.
            host: Ollama server URL.

        Returns:
            An agno.models.ollama.Ollama instance.
        """
        from agno.models.ollama import Ollama

        return Ollama(id=model_id, host=host)


def build_model(provider: str, model_id: str, **kwargs: Any) -> Any:
    """Instantiate an Agno LLM model for the given provider.

    Args:
        provider: Provider key, e.g. 'ollama'. Must exist in the registry.
        model_id: Model identifier understood by the provider.
        **kwargs: Extra keyword arguments forwarded to the builder (e.g. host).

    Returns:
        A provider-specific Agno model instance.

    Raises:
        ValueError: If the provider key is not registered.
    """
    if provider not in _PROVIDER_REGISTRY:
        supported = list(_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown LLM provider '{provider}'. Supported: {supported}")
    builder = _PROVIDER_REGISTRY[provider]()
    return builder(model_id=model_id, **kwargs)

"""Registry for knowledge editing algorithms."""

from __future__ import annotations
from typing import Dict, Type, List
import logging

from editmind.core.base_editor import BaseKnowledgeEditor

logger = logging.getLogger(__name__)

class EditorRegistry:
    """Central registry for discovering and instantiating knowledge editors."""
    _editors: Dict[str, Type[BaseKnowledgeEditor]] = {}
    _builtins_loaded: bool = False

    @classmethod
    def _ensure_builtins_loaded(cls):
        if not cls._builtins_loaded:
            cls._builtins_loaded = True
            try:
                import editmind.editors  # noqa: F401
            except ImportError:
                pass

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new knowledge editor class."""
        def decorator(subclass: Type[BaseKnowledgeEditor]):
            normalized_name = name.lower().strip()
            if normalized_name in cls._editors:
                logger.warning("Overwriting existing editor registration for: %s", normalized_name)
            cls._editors[normalized_name] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseKnowledgeEditor]:
        """Retrieves an editor class by name."""
        cls._ensure_builtins_loaded()
        normalized_name = name.lower().strip()
        if normalized_name not in cls._editors:
            available = ", ".join(cls.list_available())
            raise KeyError(f"Editor '{name}' not found. Available editors: {available}")
        return cls._editors[normalized_name]

    @classmethod
    def list_available(cls) -> List[str]:
        """Lists all registered editor names."""
        cls._ensure_builtins_loaded()
        return sorted(list(cls._editors.keys()))

    @classmethod
    def create(cls, name: str, model_wrapper: any, config: dict = None) -> BaseKnowledgeEditor:
        """Instantiates an editor with given model wrapper and config."""
        editor_cls = cls.get(name)
        return editor_cls(model_wrapper=model_wrapper, config=config or {})


register_editor = EditorRegistry.register

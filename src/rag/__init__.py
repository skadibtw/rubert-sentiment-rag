from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["ReviewRAG", "build_review_index", "load_review_rag"]

if TYPE_CHECKING:
    from .pipeline import ReviewRAG, build_review_index, load_review_rag


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .pipeline import ReviewRAG, build_review_index, load_review_rag

        exports = {
            "ReviewRAG": ReviewRAG,
            "build_review_index": build_review_index,
            "load_review_rag": load_review_rag,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

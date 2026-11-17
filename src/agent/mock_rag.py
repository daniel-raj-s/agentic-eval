"""Knowledge base loading and lightweight lexical retrieval for the RAG agent."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

DEFAULT_KB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_base.json")
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "will",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation and drop low-signal stop words."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


class KnowledgeRetriever:
    """Retrieves policy and store information from the knowledge base."""

    def __init__(self, kb_path: str = DEFAULT_KB_PATH):
        self.kb_path = kb_path
        self.policies: List[Dict[str, Any]] = []
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        """(Re)load the knowledge base from disk.

        Each call returns a fresh copy, which keeps tests isolated when a tool
        mutates order state (for example cancelling an order).
        """
        if not os.path.exists(self.kb_path):
            self.policies = []
            self.orders = {}
            return

        with open(self.kb_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.policies = list(data.get("policies", []))
        self.orders = {
            str(order_id).upper(): dict(order)
            for order_id, order in data.get("mock_orders", {}).items()
        }

    def search_policies(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Rank policies by lexical overlap with the query.

        Titles are weighted more heavily than body text and exact phrase
        matches receive a bonus. This is deliberately dependency-free: the
        benchmark should run offline and deterministically.
        """
        query_lower = query.lower()
        query_words = _tokenize(query)
        if not query_words:
            return []

        scored: List[tuple[float, Dict[str, Any]]] = []
        for policy in self.policies:
            title = str(policy.get("title", ""))
            content = str(policy.get("content", ""))
            title_words = _tokenize(title)
            content_words = _tokenize(content)

            score = 0.0
            score += 3.0 * len(query_words & title_words)
            score += 1.0 * len(query_words & content_words)

            # Small bonus for contiguous phrase matches such as "return policy".
            for phrase in ("return policy", "shipping", "cancel", "warranty", "refund"):
                if phrase in query_lower and (
                    phrase in title.lower() or phrase in content.lower()
                ):
                    score += 1.5

            if score > 0:
                scored.append((score, policy))

        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("title", ""))))
        return [policy for _, policy in scored[: max(1, top_k)]]

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Look up an order by ID (case-insensitive)."""
        return self.orders.get(order_id.strip().upper())

    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        for policy in self.policies:
            if policy.get("id") == policy_id:
                return policy
        return None

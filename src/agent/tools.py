"""Executable tools for order management, refunds and policy retrieval."""
from __future__ import annotations

from typing import Any, Dict, List

from .mock_rag import KnowledgeRetriever


class AgentTools:
    """Collection of executable tools for the support agent."""

    def __init__(self, retriever: KnowledgeRetriever | None = None):
        self.retriever = retriever or KnowledgeRetriever()

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Look up the status, items, and delivery estimate for an order."""
        normalized_id = str(order_id).strip().upper()
        order = self.retriever.get_order(normalized_id)
        if not order:
            return {"error": f"Order {normalized_id} not found."}

        return {
            "order_id": normalized_id,
            "status": order["status"],
            "delivery_date": order.get("delivery_date"),
            "items": [item["name"] for item in order.get("items", [])],
            "total": order.get("total", 0.0),
        }

    def calculate_refund(self, order_id: str, reason: str = "return_request") -> Dict[str, Any]:
        """Calculate an eligible refund, applying the published return rules."""
        normalized_id = str(order_id).strip().upper()
        order = self.retriever.get_order(normalized_id)
        if not order:
            return {"error": f"Order {normalized_id} not found."}

        normalized_reason = str(reason or "return_request").strip().lower()
        ineligible_reasons: List[str] = []

        for item in order.get("items", []):
            item_name = str(item.get("name", "")).lower()
            if "custom" in item_name or "engraved" in item_name or "personalized" in item_name:
                ineligible_reasons.append(
                    "Personalized, custom, or engraved items are strictly non-refundable."
                )
                break

        if not order.get("eligible_for_return", False):
            ineligible_reasons.append("The order exceeds the 30-day return policy window.")

        if ineligible_reasons:
            return {
                "order_id": normalized_id,
                "eligible": False,
                "refund_amount": 0.0,
                "reason": " ".join(ineligible_reasons),
                "reasons": ineligible_reasons,
            }

        buyer_remorse = any(
            token in normalized_reason
            for token in ("remorse", "mind", "mistake", "unwanted", "changed")
        )
        fee = 5.99 if buyer_remorse else 0.0
        total = float(order.get("total", 0.0))
        refund_amount = max(0.0, round(total - fee, 2))

        return {
            "order_id": normalized_id,
            "eligible": True,
            "original_total": total,
            "deductions": fee,
            "refund_amount": refund_amount,
            "reason": normalized_reason,
            "breakdown": (
                f"Gross: ${total:.2f} - Return Fee: ${fee:.2f} = "
                f"Net: ${refund_amount:.2f}"
            ),
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order if it is in 'Processing' status."""
        normalized_id = str(order_id).strip().upper()
        order = self.retriever.get_order(normalized_id)
        if not order:
            return {"error": f"Order {normalized_id} not found."}

        if order["status"] == "Processing":
            order["status"] = "Cancelled"
            return {
                "order_id": normalized_id,
                "success": True,
                "message": f"Order {normalized_id} has been cancelled successfully.",
            }

        return {
            "order_id": normalized_id,
            "success": False,
            "message": (
                f"Cannot cancel order {normalized_id}. Current status is "
                f"'{order['status']}'. It must be returned after delivery."
            ),
        }

    def search_knowledge_base(self, query: str) -> List[str]:
        """Search store policies, FAQs, and documentation."""
        matched = self.retriever.search_policies(query)
        if not matched:
            return ["No matching store policy found for your query."]
        return [f"[{policy['title']}]: {policy['content']}" for policy in matched]

    # ------------------------------------------------------------------
    # Dispatch and schemas
    # ------------------------------------------------------------------
    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name. Raises ``ValueError`` for unknown tools."""
        if name == "get_order_status":
            return self.get_order_status(**arguments)
        if name == "calculate_refund":
            return self.calculate_refund(**arguments)
        if name == "cancel_order":
            return self.cancel_order(**arguments)
        if name == "search_knowledge_base":
            return self.search_knowledge_base(**arguments)
        raise ValueError(f"Unknown tool: {name}")

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Provider-neutral tool descriptions for function calling."""
        return [
            {
                "name": "get_order_status",
                "description": "Fetch real-time delivery and shipping status for an order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID, e.g. ORD-1001",
                        }
                    },
                    "required": ["order_id"],
                },
            },
            {
                "name": "calculate_refund",
                "description": (
                    "Check return eligibility and calculate the refund amount, including "
                    "policy exclusions and return-shipping deductions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID, e.g. ORD-1001",
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Reason for the return, e.g. buyer_remorse, return_request, "
                                "or damaged"
                            ),
                        },
                    },
                    "required": ["order_id"],
                },
            },
            {
                "name": "cancel_order",
                "description": "Cancel an order if it is still in the Processing state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID, e.g. ORD-1002",
                        }
                    },
                    "required": ["order_id"],
                },
            },
            {
                "name": "search_knowledge_base",
                "description": (
                    "Search store policies covering shipping, delivery, warranties, "
                    "returns, refunds, and cancellations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query terms",
                        }
                    },
                    "required": ["query"],
                },
            },
        ]

    @property
    def openai_tool_schemas(self) -> List[Dict[str, Any]]:
        """Tool schemas in the OpenAI ``tools`` format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            }
            for schema in self.tool_schemas
        ]

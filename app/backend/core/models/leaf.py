from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.backend.core.models.tool_calls import ToolCall

@dataclass
class LlmLeaf:
    description: str
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }

@dataclass
class Leaf:
    id: str
    description: str
    result: str
    parent_leaf: Optional[str]
    child_leaves: List[str]
    tool_calls: List[ToolCall] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"Step: {self.description}"]
        if self.result:
            parts.append(f"Findings: {self.result}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "result": self.result,
            "parent_leaf": self.parent_leaf,
            "child_leaves": self.child_leaves,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }

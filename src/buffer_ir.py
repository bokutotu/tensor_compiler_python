from dataclasses import dataclass, field
from enum import Enum, auto

from loop_ir import Block, Compute, Loop
from tensor_ir import Tensor


class MemoryType(Enum):
    GLOBAL = auto()
    SHARED = auto()
    LOCAL = auto()


@dataclass
class Buffer:
    name: str
    tensor: Tensor
    memory: MemoryType
    body: Block
    attrs: dict[str, object] = field(default_factory=dict)


BlockStmt = Buffer | Loop | Compute

from dataclasses import dataclass, field
from enum import Enum, auto

from tensor_ir import DimExpr, Tensor, ScalarExpr, Symbol, DimRange


class AccessType(Enum):
    READ = auto()
    WRITE = auto()


@dataclass(frozen=True)
class TensorAccess:
    tensor: Tensor
    indices: tuple[DimExpr, ...]
    access_type: AccessType


@dataclass
class Compute:
    name: str
    write: TensorAccess
    expr: ScalarExpr
    attrs: dict[str, object] = field(default_factory=dict)


@dataclass
class Loop:
    iter_var: Symbol
    domain: DimRange
    step: int
    body: list["Stmt"]
    attrs: dict[str, object] = field(default_factory=dict)


@dataclass
class Block:
    stmts: list["Stmt"]


Stmt = Compute | Loop | Block


@dataclass
class Function:
    name: str
    inputs: list[Tensor]
    outputs: list[Tensor]
    body: Block
    symbol_bounds: dict[Symbol, tuple[int, int | None]]

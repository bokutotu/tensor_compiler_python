from dataclasses import dataclass
from enum import Enum, auto

from tensor_ir import DimExpr, Tensor, ScalarExpr, Symbol, DimRange, ReduceOp


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


@dataclass
class Reduce:
    name: str
    write: TensorAccess
    value: ScalarExpr
    reducer: ReduceOp


@dataclass
class Loop:
    iter_var: Symbol
    domain: DimRange
    step: int
    body: list["Stmt"]


@dataclass
class Block:
    stmts: list["Stmt"]


Stmt = Compute | Reduce | Loop | Block


@dataclass
class Function:
    name: str
    inputs: list[Tensor]
    outputs: list[Tensor]
    body: Block
    symbol_bounds: dict[Symbol, tuple[int, int | None]]

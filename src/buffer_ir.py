from dataclasses import dataclass
from enum import Enum, auto

from loop_ir import Block, Compute, Loop, Reduce, TensorAccess
from tensor_ir import Tensor, ScalarExpr, DType, ReduceOp, IndexMap


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
    layout: IndexMap | None = None


class Scope(Enum):
    WARP = auto()
    WARPGROUP = auto()
    BLOCK = auto()
    GRID = auto()


@dataclass
class Barrier:
    scope: Scope


@dataclass
class AsyncCopy:
    dst: TensorAccess
    src: TensorAccess
    bytes: int
    group: int | None = None


@dataclass
class Atomic:
    write: TensorAccess
    value: ScalarExpr
    op: ReduceOp
    scope: Scope


@dataclass
class MMA:
    tile_m: int
    tile_n: int
    tile_k: int
    a: TensorAccess
    b: TensorAccess
    c: TensorAccess
    a_map: IndexMap
    b_map: IndexMap
    c_map: IndexMap
    a_dtype: DType
    b_dtype: DType
    acc_dtype: DType
    scope: Scope
    participants: int


BlockStmt = (
    Buffer
    | Loop
    | Compute
    | Reduce
    | Barrier
    | AsyncCopy
    | Atomic
    | MMA
)

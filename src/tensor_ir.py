from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class DimExpr:
    base: int = 0
    terms: dict[Symbol, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DimRange:
    lower: DimExpr
    upper: DimExpr


def dim(base: int = 0, **coeffs: int) -> DimExpr:
    return DimExpr(base, {Symbol(name): coeff for name, coeff in coeffs.items()})


class OpKind(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()


class DType(Enum):
    FLOAT32 = auto()
    INT64 = auto()
    BOOL = auto()

@dataclass(frozen=True)
class StateVar:
    name: str
    dtype: DType


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[DimExpr, ...]
    dtype: DType


@dataclass(frozen=True)
class IndexMap:
    in_axes: tuple[Symbol, ...]
    out: tuple[DimExpr, ...]


class TensorExpr:
    pass


class ScalarExpr(TensorExpr):
    pass


@dataclass(frozen=True)
class Scan(TensorExpr):
    axis: Symbol
    domain: Mapping[Symbol, DimRange]
    state_vars: tuple[StateVar, ...]
    init: tuple[ScalarExpr, ...]
    step: tuple[ScalarExpr, ...]
    output_index: int


@dataclass(frozen=True)
class StateRead(ScalarExpr):
    index: int


@dataclass(frozen=True)
class TensorInput(ScalarExpr):
    tensor: Tensor
    indices: tuple[DimExpr, ...]


@dataclass(frozen=True)
class ScalarConst(ScalarExpr):
    value: float
    dtype: DType


@dataclass(frozen=True)
class Op(ScalarExpr):
    kind: OpKind
    operands: tuple[ScalarExpr, ...]


class ReduceOp(Enum):
    SUM = auto()
    MAX = auto()
    MIN = auto()
    PROD = auto()


@dataclass(frozen=True)
class Reduction(TensorExpr):
    reducer: ReduceOp
    axes: tuple[Symbol, ...]
    body: TensorExpr
    init: ScalarExpr
    domain: Mapping[Symbol, DimRange]


@dataclass(frozen=True)
class TensorComputeDef:
    name: str
    tensor: Tensor
    axes: tuple[Symbol, ...]
    domain: Mapping[Symbol, DimRange]
    expr: TensorExpr

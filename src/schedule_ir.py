from collections.abc import Sequence
from dataclasses import dataclass

from tensor_ir import Symbol


@dataclass(frozen=True)
class Split:
    loop: Symbol
    factors: tuple[int, ...]


@dataclass(frozen=True)
class Reorder:
    loops: tuple[Symbol, ...]


@dataclass(frozen=True)
class Fuse:
    loops: tuple[Symbol, ...]


@dataclass(frozen=True)
class ComputeAt:
    producer: str
    consumer: str
    loop: Symbol


@dataclass(frozen=True)
class CacheRead:
    tensor: str
    memory: str


@dataclass(frozen=True)
class CacheWrite:
    tensor: str
    memory: str


@dataclass(frozen=True)
class Bind:
    loop: Symbol
    tag: str
    width: int | None = None


@dataclass(frozen=True)
class Tensorize:
    block: str
    intrinsic: str


Schedule = Sequence[
    Split | Reorder | Fuse | ComputeAt | CacheRead | CacheWrite | Bind | Tensorize
]

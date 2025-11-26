from __future__ import annotations

import ctypes
import hashlib
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, cast

import numpy as np

from buffer_codegen import generate_code
from buffer_ir import Buffer
from loop_ir import Block, Compute, If, Loop, Reduce, Stmt
from tensor_ir import Op, ScalarExpr, Tensor, TensorInput


_ALIGNMENT = 128


@dataclass
class CompileConfig:
    cache_dir: Path | None = None
    clang: str = "clang"
    extra_cflags: tuple[str, ...] = ("-O3", "-shared", "-fPIC", "-std=c99")
    keep_c: bool = False
    use_restrict: bool = True
    assume_alignment: int = _ALIGNMENT


@dataclass
class CompiledKernel:
    name: str
    path: Path
    arg_tensors: list[Tensor]
    outputs: tuple[Tensor, ...]
    alignment: int
    use_restrict: bool
    func_ptr: Callable[..., None]
    _cdll: ctypes.CDLL = field(repr=False)

    def __call__(self, *args: ctypes.c_void_p) -> None:
        return self.func_ptr(*args)


class _FuncPtrProtocol(Protocol):
    argtypes: Sequence[object]
    restype: object

    def __call__(self, *args: object) -> None: ...


def compile_buffer(buffer: Buffer, config: CompileConfig | None = None) -> CompiledKernel:
    cfg = config or CompileConfig()
    cache_dir = cfg.cache_dir or Path.home() / ".cache" / "tensor_compiler"
    _ = cache_dir.mkdir(parents=True, exist_ok=True)

    code = generate_code(
        buffer,
        use_restrict=cfg.use_restrict,
        assume_alignment=cfg.assume_alignment,
    )
    cache_hasher = hashlib.sha256()
    cache_hasher.update(code.encode("utf-8"))
    cache_hasher.update(b"\0")
    cache_hasher.update(cfg.clang.encode("utf-8"))
    cache_hasher.update(b"\0")
    cache_hasher.update("\0".join(cfg.extra_cflags).encode("utf-8"))
    cache_hasher.update(b"\0")
    cache_hasher.update(str(cfg.use_restrict).encode("utf-8"))
    cache_hasher.update(b"\0")
    cache_hasher.update(str(cfg.assume_alignment).encode("utf-8"))
    cache_hash = cache_hasher.hexdigest()
    base_name = f"{buffer.name}-{cache_hash}"
    c_path = cache_dir / f"{base_name}.c"
    so_path = cache_dir / f"{base_name}.so"

    if not so_path.exists():
        _ = c_path.write_text(code)
        cmd = [
            cfg.clang,
            *cfg.extra_cflags,
            "-o",
            str(so_path),
            str(c_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Compilation failed (exit {result.returncode}).\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if not cfg.keep_c:
            c_path.unlink(missing_ok=True)

    cdll = ctypes.CDLL(str(so_path))
    arg_tensors = _collect_tensors(buffer.body)
    argtypes = [ctypes.POINTER(ctypes.c_float)] * len(arg_tensors)
    func_ptr_raw = cast(_FuncPtrProtocol, getattr(cdll, buffer.name))
    if not callable(func_ptr_raw):
        raise TypeError(f"Symbol {buffer.name} is not callable in {so_path}")
    func_ptr_raw.argtypes = argtypes
    func_ptr_raw.restype = None
    func_ptr = cast(Callable[..., None], func_ptr_raw)

    return CompiledKernel(
        name=buffer.name,
        path=so_path,
        arg_tensors=arg_tensors,
        outputs=(buffer.tensor,),
        alignment=cfg.assume_alignment,
        use_restrict=cfg.use_restrict,
        func_ptr=func_ptr,
        _cdll=cdll,
    )


def run_kernel(kernel: CompiledKernel, *, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    output_buffers: dict[str, np.ndarray] = outputs.copy() if outputs else {}
    output_names = {t.name for t in kernel.outputs}

    args: list[np.ndarray] = []
    raw_seen: set[int] = set()
    seen_ptrs: set[int] = set()
    processed_by_raw: dict[int, np.ndarray] = {}
    enforce_restrict = kernel.use_restrict
    for tensor in kernel.arg_tensors:
        name = tensor.name
        if name in output_names:
            arr = output_buffers.get(name)
            if arr is None:
                shape = _tensor_shape(tensor)
                if shape is None:
                    raise NotImplementedError("Symbolic dimensions not supported at runtime.")
                arr = _aligned_empty(shape, dtype=np.float32, align=kernel.alignment)
                output_buffers[name] = arr
        elif name in inputs:
            arr = inputs[name]
        else:
            raise ValueError(f"Missing buffer for tensor '{name}'")

        raw_ptr = int(arr.ctypes.data)
        if enforce_restrict:
            if raw_ptr in raw_seen:
                raise ValueError("Restrict/noalias assumption violated: duplicate buffer pointer detected.")
            raw_seen.add(raw_ptr)

        checked = processed_by_raw.get(raw_ptr)
        if checked is None:
            checked = _ensure_c_contiguous_f32(arr)
            checked = _ensure_shape(tensor, checked)
            checked, _ = _ensure_aligned(checked, kernel.alignment)
            if not enforce_restrict:
                processed_by_raw[raw_ptr] = checked
        else:
            checked = _ensure_shape(tensor, checked)

        ptr_val = checked.ctypes.data
        if enforce_restrict:
            if ptr_val in seen_ptrs:
                raise ValueError("Restrict/noalias assumption violated: duplicate buffer pointer detected.")
            seen_ptrs.add(ptr_val)

        args.append(checked)
        if name in output_names:
            output_buffers[name] = checked

    ptr_args = [arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for arr in args]
    kernel.func_ptr(*ptr_args)
    return output_buffers


def _ensure_c_contiguous_f32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32:
        raise TypeError(f"Expected float32 array, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(arr, dtype=np.float32)
    return arr


def _ensure_shape(tensor: Tensor, arr: np.ndarray) -> np.ndarray:
    shape = _tensor_shape(tensor)
    if shape is None:
        raise NotImplementedError("Symbolic dimensions not supported at runtime.")
    if tuple(arr.shape) != shape:
        raise ValueError(f"Shape mismatch for tensor {tensor.name}: expected {shape}, got {tuple(arr.shape)}")
    return arr


def _tensor_shape(tensor: Tensor) -> tuple[int, ...] | None:
    dims: list[int] = []
    for dim in tensor.shape:
        if dim.terms:
            return None
        dims.append(dim.base)
    return tuple(dims)


def _ensure_aligned(arr: np.ndarray, align: int) -> tuple[np.ndarray, bool]:
    if arr.ctypes.data % align == 0:
        return arr, False
    aligned = _aligned_empty(arr.shape, arr.dtype, align)
    aligned[...] = arr
    return aligned, True


def _aligned_empty(shape: Sequence[int], dtype: np.dtype | type | str, align: int) -> np.ndarray:
    np_dtype = np.dtype(dtype)
    size = math.prod(shape)
    nbytes = size * np_dtype.itemsize
    raw = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-raw.ctypes.data) % align
    data = raw[offset : offset + nbytes].view(np_dtype).reshape(tuple(shape))
    return data


def _collect_tensors(block: Block) -> list[Tensor]:
    tensors: dict[str, Tensor] = {}

    def visit(stmt: Stmt) -> None:
        if isinstance(stmt, Loop):
            for s in stmt.body:
                visit(s)
        elif isinstance(stmt, Compute):
            tensors[stmt.write.tensor.name] = stmt.write.tensor
            _visit_expr(stmt.expr)
        elif isinstance(stmt, Reduce):
            tensors[stmt.write.tensor.name] = stmt.write.tensor
            _visit_expr(stmt.value)
        elif isinstance(stmt, If):
            for s in stmt.then_body.stmts:
                visit(s)
            if stmt.else_body is not None:
                for s in stmt.else_body.stmts:
                    visit(s)

    def _visit_expr(expr: ScalarExpr) -> None:
        if isinstance(expr, TensorInput):
            tensors[expr.tensor.name] = expr.tensor
        elif isinstance(expr, Op):
            for op in expr.operands:
                _visit_expr(op)

    for s in block.stmts:
        visit(s)
    return list(tensors.values())

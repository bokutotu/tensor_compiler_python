import ctypes
from pathlib import Path

import numpy as np
import pytest

from buffer_ir import Buffer, MemoryType
from loop_to_buffer import lower_loop_to_buffer
from runtime import CompileConfig, CompiledKernel, compile_buffer, run_kernel
from tensor_ir import (
    DType,
    DimExpr,
    DimRange,
    Op,
    OpKind,
    ReduceOp,
    Reduction,
    ScalarConst,
    Symbol,
    Tensor,
    TensorComputeDef,
    TensorInput,
)
from tensor_to_loop import lower_tensor_to_loop


def _axis_index(symbol: Symbol) -> DimExpr:
    return DimExpr(base=0, terms={symbol: 1})


def _compile(buffer: Buffer, tmp_path: Path) -> CompiledKernel:
    cfg = CompileConfig(cache_dir=tmp_path, keep_c=False)
    kernel = compile_buffer(buffer, cfg)
    assert buffer.memory == MemoryType.GLOBAL
    return kernel


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_runtime_vector_add(tmp_path: Path):
    rng = _rng()
    i = Symbol("i")
    domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=4))}
    shape = (DimExpr(base=4),)

    a = Tensor(name="A", shape=shape, dtype=DType.FLOAT32)
    b = Tensor(name="B", shape=shape, dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=shape, dtype=DType.FLOAT32)

    expr = Op(
        kind=OpKind.ADD,
        operands=(
            TensorInput(tensor=a, indices=(_axis_index(i),)),
            TensorInput(tensor=b, indices=(_axis_index(i),)),
        ),
    )

    compute_def = TensorComputeDef(name="C", tensor=c, axes=(i,), domain=domain, expr=expr)
    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    kernel = _compile(buffer, tmp_path)

    arr_a = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
    arr_b = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
    outputs = run_kernel(kernel, inputs={"A": arr_a, "B": arr_b})
    np.testing.assert_allclose(outputs["C"], arr_a + arr_b)


def test_runtime_reduce_sum(tmp_path: Path):
    rng = _rng()
    i = Symbol("i")
    domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=5))}

    a = Tensor(name="A", shape=(DimExpr(base=5),), dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=(), dtype=DType.FLOAT32)

    reduction = Reduction(
        reducer=ReduceOp.SUM,
        axes=(i,),
        body=TensorInput(tensor=a, indices=(_axis_index(i),)),
        init=ScalarConst(value=0.0, dtype=DType.FLOAT32),
        domain=domain,
    )

    compute_def = TensorComputeDef(name="C", tensor=c, axes=(), domain={}, expr=reduction)
    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    kernel = _compile(buffer, tmp_path)

    arr_a = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
    outputs = run_kernel(kernel, inputs={"A": arr_a})
    np.testing.assert_allclose(outputs["C"], np.array(np.sum(arr_a), dtype=np.float32))


def test_runtime_cache_reuse(tmp_path: Path):
    i = Symbol("i")
    domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=2))}
    shape = (DimExpr(base=2),)

    a = Tensor(name="A", shape=shape, dtype=DType.FLOAT32)
    b = Tensor(name="B", shape=shape, dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=shape, dtype=DType.FLOAT32)

    expr = Op(
        kind=OpKind.ADD,
        operands=(
            TensorInput(tensor=a, indices=(_axis_index(i),)),
            TensorInput(tensor=b, indices=(_axis_index(i),)),
        ),
    )

    compute_def = TensorComputeDef(name="C", tensor=c, axes=(i,), domain=domain, expr=expr)
    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)

    kernel1 = _compile(buffer, tmp_path)
    mtime1 = kernel1.path.stat().st_mtime_ns
    kernel2 = _compile(buffer, tmp_path)
    assert kernel1.path == kernel2.path
    assert kernel2.path.stat().st_mtime_ns == mtime1


def test_runtime_restrict_violation(tmp_path: Path):
    rng = _rng()
    i = Symbol("i")
    domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=3))}
    shape = (DimExpr(base=3),)

    a = Tensor(name="A", shape=shape, dtype=DType.FLOAT32)
    b = Tensor(name="B", shape=shape, dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=shape, dtype=DType.FLOAT32)

    expr = Op(
        kind=OpKind.ADD,
        operands=(
            TensorInput(tensor=a, indices=(_axis_index(i),)),
            TensorInput(tensor=b, indices=(_axis_index(i),)),
        ),
    )

    compute_def = TensorComputeDef(name="C", tensor=c, axes=(i,), domain=domain, expr=expr)
    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    kernel = _compile(buffer, tmp_path)

    shared = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    with pytest.raises(ValueError):
        _ = run_kernel(kernel, inputs={"A": shared, "B": shared})


def test_runtime_alignment_copy(tmp_path: Path):
    rng = _rng()
    i = Symbol("i")
    domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=4))}
    shape = (DimExpr(base=4),)

    a = Tensor(name="A", shape=shape, dtype=DType.FLOAT32)
    b = Tensor(name="B", shape=shape, dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=shape, dtype=DType.FLOAT32)

    expr = Op(
        kind=OpKind.ADD,
        operands=(
            TensorInput(tensor=a, indices=(_axis_index(i),)),
            TensorInput(tensor=b, indices=(_axis_index(i),)),
        ),
    )

    compute_def = TensorComputeDef(name="C", tensor=c, axes=(i,), domain=domain, expr=expr)
    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    kernel = _compile(buffer, tmp_path)

    base_a = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
    base_b = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
    # Misaligned views (offset by one element)
    arr_a = base_a[1:]
    arr_b = base_b[1:]

    outputs = run_kernel(kernel, inputs={"A": arr_a, "B": arr_b})
    np.testing.assert_allclose(outputs["C"], arr_a + arr_b)


def test_runtime_allows_alias_without_restrict(tmp_path: Path):
    rng = _rng()
    i = Symbol("i")
    domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=4))}
    shape = (DimExpr(base=4),)

    a = Tensor(name="A", shape=shape, dtype=DType.FLOAT32)
    b = Tensor(name="B", shape=shape, dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=shape, dtype=DType.FLOAT32)

    expr = Op(
        kind=OpKind.ADD,
        operands=(
            TensorInput(tensor=a, indices=(_axis_index(i),)),
            TensorInput(tensor=b, indices=(_axis_index(i),)),
        ),
    )

    compute_def = TensorComputeDef(name="C", tensor=c, axes=(i,), domain=domain, expr=expr)
    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    cfg = CompileConfig(cache_dir=tmp_path, keep_c=False, use_restrict=False)
    kernel = compile_buffer(buffer, cfg)

    base = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
    shared = base[1:]

    captured_ptrs: list[list[int]] = []
    orig_func = kernel.func_ptr

    def wrapped(*args: ctypes.c_void_p) -> None:
        ptrs: list[int] = []
        for arg in args:
            ptr_val = ctypes.cast(arg, ctypes.c_void_p).value
            assert ptr_val is not None
            ptrs.append(int(ptr_val))
        captured_ptrs.append(ptrs)
        return orig_func(*args)

    kernel.func_ptr = wrapped  # type: ignore[assignment]

    outputs = run_kernel(kernel, inputs={"A": shared, "B": shared})
    np.testing.assert_allclose(outputs["C"], shared * 2)
    assert captured_ptrs, "Kernel was not invoked"
    arg_names = [t.name for t in kernel.arg_tensors]
    captured = captured_ptrs[0]
    assert len(captured) == len(arg_names)
    idx_a = arg_names.index("A")
    idx_b = arg_names.index("B")
    assert captured[idx_a] == captured[idx_b]

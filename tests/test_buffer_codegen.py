from buffer_codegen import generate_code
from buffer_ir import Buffer, MemoryType
from loop_ir import (
    AccessType,
    Block,
    Compare,
    CompareOp,
    Compute,
    If,
    Loop,
    TensorAccess,
)
from loop_to_buffer import lower_loop_to_buffer
from tensor_ir import (
    DType,
    DimExpr,
    DimRange,
    Op,
    OpKind,
    ReduceOp,
    Reduction,
    Scan,
    ScalarConst,
    StateRead,
    StateVar,
    Symbol,
    Tensor,
    TensorComputeDef,
    TensorInput,
)
from tensor_to_loop import lower_tensor_to_loop


def _axis_index(symbol: Symbol) -> DimExpr:
    return DimExpr(base=0, terms={symbol: 1})

def _dim_val(val: int) -> DimExpr:
    return DimExpr(base=val)


def test_codegen_with_restrict_and_alignment():
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

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(i,),
        domain=domain,
        expr=expr,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer, use_restrict=True, assume_alignment=128)

    expected = """void C(float* __restrict C, float* __restrict A, float* __restrict B) {
  float* C_aligned = (float*)__builtin_assume_aligned(C, 128);
  float* A_aligned = (float*)__builtin_assume_aligned(A, 128);
  float* B_aligned = (float*)__builtin_assume_aligned(B, 128);
  for (int i = 0; i < 4; i += 1) {
    C_aligned[i] = (A_aligned[i] + B_aligned[i]);
  }
}"""
    assert code == expected


def test_codegen():
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

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(i,),
        domain=domain,
        expr=expr,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A, float* B) {
  for (int i = 0; i < 4; i += 1) {
    C[i] = (A[i] + B[i]);
  }
}"""
    assert code == expected


def test_codegen_2d():
    i = Symbol("i")
    j = Symbol("j")
    domain = {
        i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=2)),
        j: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=3)),
    }
    shape = (DimExpr(base=2), DimExpr(base=3))

    a = Tensor(name="A", shape=shape, dtype=DType.FLOAT32)
    b = Tensor(name="B", shape=shape, dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=shape, dtype=DType.FLOAT32)

    expr = Op(
        kind=OpKind.ADD,
        operands=(
            TensorInput(tensor=a, indices=(_axis_index(i), _axis_index(j))),
            TensorInput(tensor=b, indices=(_axis_index(i), _axis_index(j))),
        ),
    )

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(i, j),
        domain=domain,
        expr=expr,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A, float* B) {
  for (int i = 0; i < 2; i += 1) {
    for (int j = 0; j < 3; j += 1) {
      C[(i * 3) + j] = (A[(i * 3) + j] + B[(i * 3) + j]);
    }
  }
}"""
    assert code == expected


def test_codegen_reduction_2d_to_1d():
    i = Symbol("i")
    j = Symbol("j")

    output_domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=2))}
    reduction_domain = {j: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=3))}

    a = Tensor(name="A", shape=(DimExpr(base=2), DimExpr(base=3)), dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=(DimExpr(base=2),), dtype=DType.FLOAT32)

    reduction = Reduction(
        reducer=ReduceOp.SUM,
        axes=(j,),
        body=TensorInput(tensor=a, indices=(_axis_index(i), _axis_index(j))),
        init=ScalarConst(value=0.0, dtype=DType.FLOAT32),
        domain=reduction_domain,
    )

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(i,),
        domain=output_domain,
        expr=reduction,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A) {
  for (int i = 0; i < 2; i += 1) {
    C[i] = 0.0;
    for (int j = 0; j < 3; j += 1) {
      C[i] += A[(i * 3) + j];
    }
  }
}"""
    assert code == expected


def test_codegen_reduction_1d_to_scalar():
    i = Symbol("i")

    reduction_domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=4))}

    a = Tensor(name="A", shape=(DimExpr(base=4),), dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=(), dtype=DType.FLOAT32)

    reduction = Reduction(
        reducer=ReduceOp.SUM,
        axes=(i,),
        body=TensorInput(tensor=a, indices=(_axis_index(i),)),
        init=ScalarConst(value=0.0, dtype=DType.FLOAT32),
        domain=reduction_domain,
    )

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(),
        domain={},
        expr=reduction,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A) {
  C[0] = 0.0;
  for (int i = 0; i < 4; i += 1) {
    C[0] += A[i];
  }
}"""
    assert code == expected


def test_codegen_if_guard():
    i = Symbol("i")
    domain = {i: DimRange(lower=_dim_val(0), upper=_dim_val(4))}
    shape = (DimExpr(base=4),)

    a = Tensor(name="A", shape=shape, dtype=DType.FLOAT32)
    b = Tensor(name="B", shape=shape, dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=shape, dtype=DType.FLOAT32)

    then_compute = Compute(
        name="C_then",
        write=TensorAccess(tensor=c, indices=(_axis_index(i),), access_type=AccessType.WRITE),
        expr=TensorInput(tensor=a, indices=(_axis_index(i),)),
    )
    else_compute = Compute(
        name="C_else",
        write=TensorAccess(tensor=c, indices=(_axis_index(i),), access_type=AccessType.WRITE),
        expr=TensorInput(tensor=b, indices=(_axis_index(i),)),
    )

    guard = Compare(lhs=_axis_index(i), rhs=_dim_val(3), op=CompareOp.LT)
    if_stmt = If(cond=guard, then_body=Block([then_compute]), else_body=Block([else_compute]))

    loop = Loop(iter_var=i, domain=domain[i], step=1, body=[if_stmt])
    block = Block(stmts=[loop])
    buffer = Buffer(name="C", tensor=c, memory=MemoryType.GLOBAL, body=block)

    code = generate_code(buffer)

    expected = """void C(float* C, float* A, float* B) {
  for (int i = 0; i < 4; i += 1) {
    if (i < 3) {
      C[i] = A[i];
    } else {
      C[i] = B[i];
    }
  }
}"""
    assert code == expected


def test_codegen_reduction_max():
    i = Symbol("i")

    reduction_domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=5))}

    a = Tensor(name="A", shape=(DimExpr(base=5),), dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=(), dtype=DType.FLOAT32)

    reduction = Reduction(
        reducer=ReduceOp.MAX,
        axes=(i,),
        body=TensorInput(tensor=a, indices=(_axis_index(i),)),
        init=ScalarConst(value=float('-inf'), dtype=DType.FLOAT32),
        domain=reduction_domain,
    )

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(),
        domain={},
        expr=reduction,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A) {
  C[0] = -INFINITY;
  for (int i = 0; i < 5; i += 1) {
    if (A[i] > C[0]) C[0] = A[i];
  }
}"""
    assert code == expected


def test_codegen_reduction_prod():
    i = Symbol("i")

    reduction_domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=3))}

    a = Tensor(name="A", shape=(DimExpr(base=3),), dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=(), dtype=DType.FLOAT32)

    reduction = Reduction(
        reducer=ReduceOp.PROD,
        axes=(i,),
        body=TensorInput(tensor=a, indices=(_axis_index(i),)),
        init=ScalarConst(value=1.0, dtype=DType.FLOAT32),
        domain=reduction_domain,
    )

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(),
        domain={},
        expr=reduction,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A) {
  C[0] = 1.0;
  for (int i = 0; i < 3; i += 1) {
    C[0] *= A[i];
  }
}"""
    assert code == expected


def test_codegen_3d():
    i = Symbol("i")
    j = Symbol("j")
    k = Symbol("k")
    domain = {
        i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=2)),
        j: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=3)),
        k: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=4)),
    }
    shape = (DimExpr(base=2), DimExpr(base=3), DimExpr(base=4))

    a = Tensor(name="A", shape=shape, dtype=DType.FLOAT32)
    b = Tensor(name="B", shape=shape, dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=shape, dtype=DType.FLOAT32)

    expr = Op(
        kind=OpKind.ADD,
        operands=(
            TensorInput(tensor=a, indices=(_axis_index(i), _axis_index(j), _axis_index(k))),
            TensorInput(tensor=b, indices=(_axis_index(i), _axis_index(j), _axis_index(k))),
        ),
    )

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(i, j, k),
        domain=domain,
        expr=expr,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A, float* B) {
  for (int i = 0; i < 2; i += 1) {
    for (int j = 0; j < 3; j += 1) {
      for (int k = 0; k < 4; k += 1) {
        C[(i * 12) + (j * 4) + k] = (A[(i * 12) + (j * 4) + k] + B[(i * 12) + (j * 4) + k]);
      }
    }
  }
}"""
    assert code == expected


def test_codegen_reduction_3d_to_1d():
    i = Symbol("i")
    j = Symbol("j")
    k = Symbol("k")

    output_domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=2))}
    reduction_domain = {
        j: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=3)),
        k: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=4)),
    }

    a = Tensor(name="A", shape=(DimExpr(base=2), DimExpr(base=3), DimExpr(base=4)), dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=(DimExpr(base=2),), dtype=DType.FLOAT32)

    reduction = Reduction(
        reducer=ReduceOp.SUM,
        axes=(j, k),
        body=TensorInput(tensor=a, indices=(_axis_index(i), _axis_index(j), _axis_index(k))),
        init=ScalarConst(value=0.0, dtype=DType.FLOAT32),
        domain=reduction_domain,
    )

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(i,),
        domain=output_domain,
        expr=reduction,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A) {
  for (int i = 0; i < 2; i += 1) {
    C[i] = 0.0;
    for (int j = 0; j < 3; j += 1) {
      for (int k = 0; k < 4; k += 1) {
        C[i] += A[(i * 12) + (j * 4) + k];
      }
    }
  }
}"""
    assert code == expected


def test_codegen_gemm():
    i = Symbol("i")
    j = Symbol("j")
    k = Symbol("k")

    M, N, K = 4, 3, 5

    output_domain = {
        i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=M)),
        j: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=N)),
    }
    reduction_domain = {k: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=K))}

    a = Tensor(name="A", shape=(DimExpr(base=M), DimExpr(base=K)), dtype=DType.FLOAT32)
    b = Tensor(name="B", shape=(DimExpr(base=K), DimExpr(base=N)), dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=(DimExpr(base=M), DimExpr(base=N)), dtype=DType.FLOAT32)

    reduction = Reduction(
        reducer=ReduceOp.SUM,
        axes=(k,),
        body=Op(
            kind=OpKind.MUL,
            operands=(
                TensorInput(tensor=a, indices=(_axis_index(i), _axis_index(k))),
                TensorInput(tensor=b, indices=(_axis_index(k), _axis_index(j))),
            ),
        ),
        init=ScalarConst(value=0.0, dtype=DType.FLOAT32),
        domain=reduction_domain,
    )

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(i, j),
        domain=output_domain,
        expr=reduction,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A, float* B) {
  for (int i = 0; i < 4; i += 1) {
    for (int j = 0; j < 3; j += 1) {
      C[(i * 3) + j] = 0.0;
      for (int k = 0; k < 5; k += 1) {
        C[(i * 3) + j] += (A[(i * 5) + k] * B[(k * 3) + j]);
      }
    }
  }
}"""
    assert code == expected


def test_codegen_scan_sum_to_scalar():
    i = Symbol("i")
    scan_domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=4))}

    a = Tensor(name="A", shape=(DimExpr(base=4),), dtype=DType.FLOAT32)
    c = Tensor(name="C", shape=(), dtype=DType.FLOAT32)

    scan = Scan(
        axis=i,
        domain=scan_domain,
        state_vars=(StateVar(name="s", dtype=DType.FLOAT32),),
        init=(ScalarConst(value=0.0, dtype=DType.FLOAT32),),
        step=(
            Op(
                kind=OpKind.ADD,
                operands=(
                    StateRead(index=0),
                    TensorInput(tensor=a, indices=(_axis_index(i),)),
                ),
            ),
        ),
        output_index=0,
    )

    compute_def = TensorComputeDef(
        name="C",
        tensor=c,
        axes=(),
        domain={},
        expr=scan,
    )

    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A) {
  C[0] = 0.0;
  for (int i = 0; i < 4; i += 1) {
    C[0] = (C[0] + A[i]);
  }
}"""
    assert code == expected

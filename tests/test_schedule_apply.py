from schedule_apply import TileHint, apply_schedule
from loop_ir import Loop, If
from tensor_ir import (
    DType,
    DimExpr,
    DimRange,
    Op,
    OpKind,
    Symbol,
    Tensor,
    TensorComputeDef,
    TensorInput,
)
from tensor_to_loop import lower_tensor_to_loop
from loop_to_buffer import lower_loop_to_buffer
from buffer_codegen import generate_code


def _axis_index(symbol: Symbol) -> DimExpr:
    return DimExpr(base=0, terms={symbol: 1})


def _make_simple_add(n: int):
    i = Symbol("i")
    domain = {i: DimRange(lower=DimExpr(base=0), upper=DimExpr(base=n))}
    shape = (DimExpr(base=n),)

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
    return lower_tensor_to_loop([compute_def]), i


def test_apply_schedule_tile_no_tail():
    loop_func, i = _make_simple_add(4)
    scheduled = apply_schedule(loop_func, (TileHint(loop=i, size=2),))

    i_o = Symbol("i_o")
    i_i = Symbol("i_i")

    outer = scheduled.body.stmts[0]
    assert isinstance(outer, Loop)
    assert outer.iter_var == i_o
    assert outer.domain.lower.base == 0
    assert outer.domain.upper.base == 4
    assert outer.step == 2

    inner = outer.body[0]
    assert isinstance(inner, Loop)
    assert inner.iter_var == i_i
    assert inner.domain.lower.base == 0
    assert inner.domain.upper.base == 2
    assert inner.step == 1

    assert not any(isinstance(stmt, If) for stmt in scheduled.body.stmts[1:])


def test_apply_schedule_tile_with_tail_codegen():
    loop_func, i = _make_simple_add(5)
    scheduled = apply_schedule(loop_func, (TileHint(loop=i, size=2),))
    buffer = lower_loop_to_buffer(scheduled)
    code = generate_code(buffer)

    expected = """void C(float* C, float* A, float* B) {
  for (int i_o = 0; i_o < 4; i_o += 2) {
    for (int i_i = 0; i_i < 2; i_i += 1) {
      C[i_o + i_i] = (A[i_o + i_i] + B[i_o + i_i]);
    }
  }
  if (4 < 5) {
    for (int i = 4; i < 5; i += 1) {
      C[i] = (A[i] + B[i]);
    }
  }
}"""
    assert code == expected

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

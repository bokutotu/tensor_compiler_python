from loop_ir import AccessType, Block, Compute, Loop, TensorAccess
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


def _axis_index(symbol: Symbol) -> DimExpr:
    return DimExpr(base=0, terms={symbol: 1})


def test_lower_tensor_to_loop_add_1d():
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

    lowered = lower_tensor_to_loop([compute_def])

    assert lowered.inputs == [a, b]
    assert lowered.outputs == [c]
    assert lowered.symbol_bounds == {i: (0, 4)}

    expected_compute = Compute(
        name="C",
        write=TensorAccess(
            tensor=c,
            indices=(_axis_index(i),),
            access_type=AccessType.WRITE,
        ),
        expr=expr,
    )

    expected_loop = Loop(
        iter_var=i,
        domain=domain[i],
        step=1,
        body=[expected_compute],
    )

    assert lowered.body == Block(stmts=[expected_loop])

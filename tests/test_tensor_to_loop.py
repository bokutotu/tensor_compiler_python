from loop_ir import AccessType, Block, Compute, Loop, Reduce, TensorAccess
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


def test_lower_tensor_to_loop_add_2d():
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

    lowered = lower_tensor_to_loop([compute_def])

    assert lowered.inputs == [a, b]
    assert lowered.outputs == [c]
    assert lowered.symbol_bounds == {i: (0, 2), j: (0, 3)}

    expected_compute = Compute(
        name="C",
        write=TensorAccess(
            tensor=c,
            indices=(_axis_index(i), _axis_index(j)),
            access_type=AccessType.WRITE,
        ),
        expr=expr,
    )

    expected_inner_loop = Loop(
        iter_var=j,
        domain=domain[j],
        step=1,
        body=[expected_compute],
    )

    expected_outer_loop = Loop(
        iter_var=i,
        domain=domain[i],
        step=1,
        body=[expected_inner_loop],
    )

    assert lowered.body == Block(stmts=[expected_outer_loop])


def test_lower_reduction_sum_2d_to_1d():
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

    lowered = lower_tensor_to_loop([compute_def])

    assert lowered.inputs == [a]
    assert lowered.outputs == [c]
    assert lowered.symbol_bounds == {i: (0, 2), j: (0, 3)}

    expected_init = Compute(
        name="C_init",
        write=TensorAccess(
            tensor=c,
            indices=(_axis_index(i),),
            access_type=AccessType.WRITE,
        ),
        expr=ScalarConst(value=0.0, dtype=DType.FLOAT32),
    )

    expected_reduce = Reduce(
        name="C",
        write=TensorAccess(
            tensor=c,
            indices=(_axis_index(i),),
            access_type=AccessType.WRITE,
        ),
        value=TensorInput(tensor=a, indices=(_axis_index(i), _axis_index(j))),
        reducer=ReduceOp.SUM,
    )

    expected_j_loop = Loop(
        iter_var=j,
        domain=reduction_domain[j],
        step=1,
        body=[expected_reduce],
    )

    expected_i_loop = Loop(
        iter_var=i,
        domain=output_domain[i],
        step=1,
        body=[expected_init, expected_j_loop],
    )

    assert lowered.body == Block(stmts=[expected_i_loop])


def test_lower_reduction_sum_1d_to_scalar():
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

    lowered = lower_tensor_to_loop([compute_def])

    assert lowered.inputs == [a]
    assert lowered.outputs == [c]
    assert lowered.symbol_bounds == {i: (0, 4)}

    expected_init = Compute(
        name="C_init",
        write=TensorAccess(
            tensor=c,
            indices=(),
            access_type=AccessType.WRITE,
        ),
        expr=ScalarConst(value=0.0, dtype=DType.FLOAT32),
    )

    expected_reduce = Reduce(
        name="C",
        write=TensorAccess(
            tensor=c,
            indices=(),
            access_type=AccessType.WRITE,
        ),
        value=TensorInput(tensor=a, indices=(_axis_index(i),)),
        reducer=ReduceOp.SUM,
    )

    expected_i_loop = Loop(
        iter_var=i,
        domain=reduction_domain[i],
        step=1,
        body=[expected_reduce],
    )

    assert lowered.body == Block(stmts=[expected_init, expected_i_loop])


def test_lower_reduction_max():
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

    lowered = lower_tensor_to_loop([compute_def])

    assert lowered.inputs == [a]
    assert lowered.outputs == [c]

    expected_reduce = Reduce(
        name="C",
        write=TensorAccess(
            tensor=c,
            indices=(),
            access_type=AccessType.WRITE,
        ),
        value=TensorInput(tensor=a, indices=(_axis_index(i),)),
        reducer=ReduceOp.MAX,
    )

    assert isinstance(lowered.body.stmts[0], Compute)
    assert isinstance(lowered.body.stmts[1], Loop)
    assert lowered.body.stmts[1].body[0] == expected_reduce

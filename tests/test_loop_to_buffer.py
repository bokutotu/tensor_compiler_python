from loop_ir import AccessType, Block, Compute, Loop, Reduce, TensorAccess
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
from loop_to_buffer import lower_loop_to_buffer
from buffer_ir import MemoryType


def _axis_index(symbol: Symbol) -> DimExpr:
    return DimExpr(base=0, terms={symbol: 1})


def test_lower_loop_to_buffer_add_1d():
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
    assert buffer.name == "C"
    assert buffer.tensor == c
    assert buffer.memory == MemoryType.GLOBAL

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

    assert buffer.body == Block(stmts=[expected_loop])


def test_lower_loop_to_buffer_add_2d():
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
    assert buffer.name == "C"
    assert buffer.tensor == c
    assert buffer.memory == MemoryType.GLOBAL

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

    assert buffer.body == Block(stmts=[expected_outer_loop])


def test_lower_loop_to_buffer_reduction_2d_to_1d():
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

    assert buffer.name == "C"
    assert buffer.tensor == c
    assert buffer.memory == MemoryType.GLOBAL

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

    assert buffer.body == Block(stmts=[expected_i_loop])


def test_lower_loop_to_buffer_reduction_1d_to_scalar():
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

    # Tensor IR -> Loop IR -> Buffer IR
    loop_func = lower_tensor_to_loop([compute_def])
    buffer = lower_loop_to_buffer(loop_func)

    # Verify buffer properties
    assert buffer.name == "C"
    assert buffer.tensor == c
    assert buffer.memory == MemoryType.GLOBAL

    # Verify body structure
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

    assert buffer.body == Block(stmts=[expected_init, expected_i_loop])


def test_lower_loop_to_buffer_scan():
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

    expected_init = Compute(
        name="C_init",
        write=TensorAccess(
            tensor=c,
            indices=(),
            access_type=AccessType.WRITE,
        ),
        expr=ScalarConst(value=0.0, dtype=DType.FLOAT32),
    )

    expected_step = Compute(
        name="C",
        write=TensorAccess(
            tensor=c,
            indices=(),
            access_type=AccessType.WRITE,
        ),
        expr=Op(
            kind=OpKind.ADD,
            operands=(
                TensorInput(tensor=c, indices=()),
                TensorInput(tensor=a, indices=(_axis_index(i),)),
            ),
        ),
    )

    expected_loop = Loop(
        iter_var=i,
        domain=scan_domain[i],
        step=1,
        body=[expected_step],
    )

    assert buffer.name == "C"
    assert buffer.tensor == c
    assert buffer.memory == MemoryType.GLOBAL
    assert buffer.body == Block(stmts=[expected_init, expected_loop])

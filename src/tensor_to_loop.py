from collections.abc import Sequence

from tensor_ir import (
    DimExpr,
    Op,
    ScalarExpr,
    Symbol,
    Tensor,
    TensorComputeDef,
    TensorInput,
    TensorExpr,
)
from loop_ir import AccessType, Block, Compute, Function, Loop, Stmt, TensorAccess


def lower_tensor_to_loop(defs: Sequence[TensorComputeDef]) -> Function:
    if not defs:
        raise ValueError("At least one tensor compute def is required.")
    if len(defs) > 1:
        raise NotImplementedError("Multiple compute definitions are not supported yet.")

    compute_def = defs[0]

    compute = Compute(
        name=compute_def.name,
        write=TensorAccess(
            tensor=compute_def.tensor,
            indices=_indices_for_axes(compute_def.axes),
            access_type=AccessType.WRITE,
        ),
        expr=_extract_scalar_expr(compute_def.expr),
        attrs=dict(compute_def.attrs),
    )

    body: list[Stmt] = [compute]
    for axis in reversed(compute_def.axes):
        domain = compute_def.domain[axis]
        body = [Loop(iter_var=axis, domain=domain, step=1, body=body)]

    symbol_bounds: dict[Symbol, tuple[int, int | None]] = {}
    for axis in compute_def.axes:
        domain = compute_def.domain[axis]
        symbol_bounds[axis] = (
            _dim_expr_to_int(domain.lower),
            _dim_expr_to_int(domain.upper),
        )

    return Function(
        name=compute_def.name,
        inputs=_collect_inputs(compute_def.expr, compute_def.tensor),
        outputs=[compute_def.tensor],
        body=Block(stmts=body),
        symbol_bounds=symbol_bounds,
    )


def _collect_inputs(expr: TensorExpr, output_tensor: Tensor) -> list[Tensor]:
    tensors: list[Tensor] = []
    seen_ids: set[int] = set()

    def visit(node: TensorExpr):
        if isinstance(node, TensorInput):
            tensor = node.tensor
            tensor_id = id(tensor)
            if tensor is not output_tensor and tensor_id not in seen_ids:
                seen_ids.add(tensor_id)
                tensors.append(tensor)
        elif isinstance(node, Op):
            for operand in node.operands:
                visit(operand)

    visit(expr)
    return tensors


def _indices_for_axes(axes: Sequence[Symbol]) -> tuple[DimExpr, ...]:
    return tuple(DimExpr(base=0, terms={axis: 1}) for axis in axes)


def _dim_expr_to_int(expr: DimExpr) -> int:
    if expr.terms:
        raise NotImplementedError("Symbolic bounds are not supported yet.")
    return expr.base


def _extract_scalar_expr(expr: TensorExpr) -> ScalarExpr:
    if isinstance(expr, Op):
        return expr
    raise NotImplementedError("Only direct scalar expressions are supported.")

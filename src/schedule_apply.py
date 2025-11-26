from __future__ import annotations

from dataclasses import dataclass

import islpy as isl

from loop_ir import Block, Compare, CompareOp, Compute, Function, If, Loop, Reduce, Stmt, TensorAccess
from schedule_ir import Split
from tensor_ir import DimExpr, DimRange, Symbol, TensorInput, Op, ScalarConst, ScalarExpr


@dataclass(frozen=True)
class TileHint:
    loop: Symbol
    size: int


def apply_schedule(func: Function, schedule: tuple[Split | TileHint, ...]) -> Function:
    new_func = func
    for step in schedule:
        if isinstance(step, Split):
            size = _split_to_tile_size(step)
            new_func = _apply_tile(new_func, step.loop, size)
        else:
            new_func = _apply_tile(new_func, step.loop, step.size)
    return new_func


def _split_to_tile_size(split: Split) -> int:
    factors = split.factors
    if len(factors) == 0:
        raise ValueError("Split factors must be non-empty.")
    if len(factors) > 1:
        raise NotImplementedError("Multi-level splits are not supported yet.")
    size = factors[0]
    if size <= 0:
        raise ValueError("Tile size must be positive.")
    return size


def _apply_tile(func: Function, target_loop: Symbol, tile: int) -> Function:
    if tile <= 0:
        raise ValueError("Tile size must be positive.")

    new_body, applied = _transform_block(func.body, target_loop, tile)
    if not applied:
        raise ValueError(f"Loop {target_loop.name} not found in function {func.name}.")

    new_symbol_bounds = _collect_symbol_bounds(new_body)
    return Function(
        name=func.name,
        inputs=func.inputs,
        outputs=func.outputs,
        body=new_body,
        symbol_bounds=new_symbol_bounds,
    )


def _transform_block(block: Block, target: Symbol, tile: int) -> tuple[Block, bool]:
    new_stmts: list[Stmt] = []
    applied = False
    for stmt in block.stmts:
        if applied:
            new_stmts.append(stmt)
            continue
        if isinstance(stmt, Loop):
            if stmt.iter_var == target:
                transformed_block = _tile_loop(stmt, tile)
                new_stmts.extend(transformed_block.stmts)
                applied = True
            else:
                new_loop, inner_applied = _transform_loop(stmt, target, tile)
                new_stmts.append(new_loop)
                applied = applied or inner_applied
        elif isinstance(stmt, Block):
            new_block, inner_applied = _transform_block(stmt, target, tile)
            new_stmts.append(new_block)
            applied = applied or inner_applied
        else:
            new_stmts.append(stmt)
    return Block(new_stmts), applied


def _transform_loop(loop: Loop, target: Symbol, tile: int) -> tuple[Loop, bool]:
    new_body, applied = _transform_block(Block(loop.body), target, tile)
    return Loop(iter_var=loop.iter_var, domain=loop.domain, step=loop.step, body=new_body.stmts), applied


def _tile_loop(loop: Loop, tile: int) -> Block:
    if loop.step != 1:
        raise NotImplementedError("Tiling only supports unit step loops.")
    if loop.domain.lower.terms or loop.domain.upper.terms:
        raise NotImplementedError("Symbolic loop bounds are not supported yet.")

    lo = loop.domain.lower.base
    hi = loop.domain.upper.base
    if hi < lo:
        raise ValueError("Loop upper bound must be >= lower bound.")

    hi_full = _compute_full_bound_with_isl(lo, hi, tile)

    outer_sym = Symbol(f"{loop.iter_var.name}_o")
    inner_sym = Symbol(f"{loop.iter_var.name}_i")

    substituted_body = [
        _substitute_stmt(
            stmt,
            {loop.iter_var: DimExpr(base=0, terms={outer_sym: 1, inner_sym: 1})},
        )
        for stmt in loop.body
    ]

    full_outer_domain = DimRange(
        lower=DimExpr(base=lo),
        upper=DimExpr(base=hi_full),
    )
    full_outer_loop = Loop(
        iter_var=outer_sym,
        domain=full_outer_domain,
        step=tile,
        body=[
            Loop(
                iter_var=inner_sym,
                domain=DimRange(lower=DimExpr(base=0), upper=DimExpr(base=tile)),
                step=1,
                body=substituted_body,
            )
        ],
    )

    stmts: list[Stmt] = [full_outer_loop]

    if hi_full < hi:
        tail_loop = Loop(
            iter_var=loop.iter_var,
            domain=DimRange(lower=DimExpr(base=hi_full), upper=loop.domain.upper),
            step=loop.step,
            body=[_substitute_stmt(stmt, {}) for stmt in loop.body],
        )
        cond = Compare(
            lhs=DimExpr(base=hi_full),
            rhs=loop.domain.upper,
            op=CompareOp.LT,
        )
        stmts.append(If(cond=cond, then_body=Block([tail_loop])))

    return Block(stmts=stmts)


def _compute_full_bound_with_isl(lower: int, upper: int, tile: int) -> int:
    if lower >= upper:
        return lower
    ctx = isl.Context()
    set_str = (
        f"{{ [o] : {lower} <= o and o + {tile} <= {upper} and ((o - {lower}) % {tile}) = 0 }}"
    )
    outer_set = isl.Set.read_from_str(ctx, set_str)
    if outer_set.is_empty():
        return lower
    lexmax = outer_set.lexmax()
    point = lexmax.sample_point()
    start_val = point.get_coordinate_val(isl.dim_type.set, 0).to_python()
    return int(start_val) + tile


def _substitute_dim_expr(expr: DimExpr, mapping: dict[Symbol, DimExpr]) -> DimExpr:
    base = expr.base
    terms: dict[Symbol, int] = {}
    for sym, coeff in expr.terms.items():
        replacement = mapping.get(sym)
        if replacement is None:
            terms[sym] = terms.get(sym, 0) + coeff
        else:
            base += coeff * replacement.base
            for rep_sym, rep_coeff in replacement.terms.items():
                terms[rep_sym] = terms.get(rep_sym, 0) + coeff * rep_coeff
    return DimExpr(base=base, terms=terms)


def _substitute_stmt(stmt: Stmt, mapping: dict[Symbol, DimExpr]) -> Stmt:
    if isinstance(stmt, Compute):
        return Compute(
            name=stmt.name,
            write=_substitute_access(stmt.write, mapping),
            expr=_substitute_scalar_expr(stmt.expr, mapping),
        )
    if isinstance(stmt, Reduce):
        return Reduce(
            name=stmt.name,
            write=_substitute_access(stmt.write, mapping),
            value=_substitute_scalar_expr(stmt.value, mapping),
            reducer=stmt.reducer,
        )
    if isinstance(stmt, Loop):
        return Loop(
            iter_var=stmt.iter_var,
            domain=DimRange(
                lower=_substitute_dim_expr(stmt.domain.lower, mapping),
                upper=_substitute_dim_expr(stmt.domain.upper, mapping),
            ),
            step=stmt.step,
            body=[_substitute_stmt(s, mapping) for s in stmt.body],
        )
    if isinstance(stmt, If):
        return If(
            cond=Compare(
                lhs=_substitute_dim_expr(stmt.cond.lhs, mapping),
                rhs=_substitute_dim_expr(stmt.cond.rhs, mapping),
                op=stmt.cond.op,
            ),
            then_body=Block([_substitute_stmt(s, mapping) for s in stmt.then_body.stmts]),
            else_body=Block([_substitute_stmt(s, mapping) for s in stmt.else_body.stmts])
            if stmt.else_body is not None
            else None,
        )
    assert isinstance(stmt, Block)
    return Block([_substitute_stmt(s, mapping) for s in stmt.stmts])


def _substitute_access(access: TensorAccess, mapping: dict[Symbol, DimExpr]) -> TensorAccess:
    return TensorAccess(
        tensor=access.tensor,
        indices=tuple(_substitute_dim_expr(idx, mapping) for idx in access.indices),
        access_type=access.access_type,
    )


def _substitute_scalar_expr(expr: ScalarExpr, mapping: dict[Symbol, DimExpr]) -> ScalarExpr:
    if isinstance(expr, TensorInput):
        return TensorInput(
            tensor=expr.tensor,
            indices=tuple(_substitute_dim_expr(idx, mapping) for idx in expr.indices),
        )
    if isinstance(expr, Op):
        return Op(
            kind=expr.kind,
            operands=tuple(_substitute_scalar_expr(op, mapping) for op in expr.operands),
        )
    if isinstance(expr, ScalarConst):
        return expr
    return expr


def _collect_symbol_bounds(block: Block) -> dict[Symbol, tuple[int, int | None]]:
    bounds: dict[Symbol, tuple[int, int | None]] = {}

    def visit_block(b: Block) -> None:
        for s in b.stmts:
            visit(s)

    def visit(stmt: Stmt) -> None:
        if isinstance(stmt, Loop):
            lo = _dim_expr_to_int(stmt.domain.lower)
            hi = _dim_expr_to_int(stmt.domain.upper)
            bounds[stmt.iter_var] = (lo, hi)
            visit_block(Block(stmt.body))
        elif isinstance(stmt, If):
            visit_block(stmt.then_body)
            if stmt.else_body is not None:
                visit_block(stmt.else_body)

    visit_block(block)
    return bounds


def _dim_expr_to_int(expr: DimExpr) -> int:
    if expr.terms:
        raise NotImplementedError("Symbolic bounds are not supported yet.")
    return expr.base

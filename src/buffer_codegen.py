from collections.abc import Mapping

from buffer_ir import Buffer
from loop_ir import Block, Compute, Loop, Reduce, TensorAccess, Stmt
from tensor_ir import DimExpr, DType, Op, OpKind, ReduceOp, ScalarConst, ScalarExpr, Tensor, TensorInput


def generate_code(buffer: Buffer, *, use_restrict: bool = False, assume_alignment: int | None = None) -> str:
    tensors = _collect_tensors(buffer.body)

    for t in tensors:
        if t.dtype != DType.FLOAT32:
            raise NotImplementedError(f"Only FLOAT32 is supported, got {t.dtype} for tensor {t.name}")

    restrict_kw = " __restrict" if use_restrict else ""
    params = ", ".join(f"float*{restrict_kw} {t.name}" for t in tensors)

    ptr_aliases: dict[str, str] = {}
    alias_lines: list[str] = []
    if assume_alignment is not None:
        for t in tensors:
            alias = f"{t.name}_aligned"
            ptr_aliases[t.name] = alias
            alias_lines.append(
                f"  float* {alias} = (float*)__builtin_assume_aligned({t.name}, {assume_alignment});"
            )

    body_lines: list[str] = []
    body_lines.extend(alias_lines)
    body_lines.append(_gen_block(buffer.body, 1, ptr_aliases))
    body = "\n".join(body_lines)
    return f"void {buffer.name}({params}) {{\n{body}\n}}"


def _gen_block(block: Block, indent: int, ptr_aliases: Mapping[str, str]) -> str:
    return "\n".join(_gen_stmt(s, indent, ptr_aliases) for s in block.stmts)


def _gen_stmt(stmt: Stmt, indent: int, ptr_aliases: Mapping[str, str]) -> str:
    ind = "  " * indent
    if isinstance(stmt, Loop):
        var = stmt.iter_var.name
        lo = _gen_dim(stmt.domain.lower)
        hi = _gen_dim(stmt.domain.upper)
        body = "\n".join(_gen_stmt(s, indent + 1, ptr_aliases) for s in stmt.body)
        return f"{ind}for (int {var} = {lo}; {var} < {hi}; {var} += {stmt.step}) {{\n{body}\n{ind}}}"
    elif isinstance(stmt, Compute):
        lhs = _gen_access(stmt.write, ptr_aliases)
        rhs = _gen_expr(stmt.expr, ptr_aliases)
        return f"{ind}{lhs} = {rhs};"
    elif isinstance(stmt, Reduce):
        lhs = _gen_access(stmt.write, ptr_aliases)
        rhs = _gen_expr(stmt.value, ptr_aliases)
        if stmt.reducer == ReduceOp.SUM:
            return f"{ind}{lhs} += {rhs};"
        elif stmt.reducer == ReduceOp.PROD:
            return f"{ind}{lhs} *= {rhs};"
        elif stmt.reducer == ReduceOp.MAX:
            return f"{ind}if ({rhs} > {lhs}) {lhs} = {rhs};"
        elif stmt.reducer == ReduceOp.MIN:
            return f"{ind}if ({rhs} < {lhs}) {lhs} = {rhs};"
        else:
            raise NotImplementedError(f"Reducer {stmt.reducer} not supported")
    raise NotImplementedError(type(stmt))


def _gen_access(access: TensorAccess, ptr_aliases: Mapping[str, str]) -> str:
    tensor_name = ptr_aliases.get(access.tensor.name, access.tensor.name)
    idx = _gen_index(access.tensor, access.indices)
    return f"{tensor_name}[{idx}]"


def _gen_expr(expr: ScalarExpr, ptr_aliases: Mapping[str, str]) -> str:
    if isinstance(expr, Op):
        l = _gen_expr(expr.operands[0], ptr_aliases)
        r = _gen_expr(expr.operands[1], ptr_aliases)
        op = {OpKind.ADD: "+", OpKind.MUL: "*", OpKind.SUB: "-", OpKind.DIV: "/"}[expr.kind]
        return f"({l} {op} {r})"
    elif isinstance(expr, TensorInput):
        idx = _gen_index(expr.tensor, expr.indices)
        tensor_name = ptr_aliases.get(expr.tensor.name, expr.tensor.name)
        return f"{tensor_name}[{idx}]"
    elif isinstance(expr, ScalarConst):
        return _gen_const(expr)
    raise NotImplementedError(type(expr))


def _gen_const(const: ScalarConst) -> str:
    if const.dtype == DType.FLOAT32:
        if const.value == float('inf'):
            return "INFINITY"
        elif const.value == float('-inf'):
            return "-INFINITY"
        else:
            s = str(const.value)
            return s if '.' in s else f"{s}.0"
    elif const.dtype == DType.INT64:
        return str(int(const.value))
    else:
        raise NotImplementedError(f"Const type {const.dtype} not supported")


def _gen_dim(expr: DimExpr) -> str:
    if expr.base and expr.terms:
        raise NotImplementedError("base + terms")
    if expr.terms:
        sym, coeff = next(iter(expr.terms.items()))
        return sym.name if coeff == 1 else f"{coeff} * {sym.name}"
    return str(expr.base)


def _gen_index(tensor: Tensor, indices: tuple[DimExpr, ...]) -> str:
    if len(indices) == 0:
        return "0"

    if len(indices) == 1:
        return _gen_dim(indices[0])

    parts: list[str] = []
    for i, idx in enumerate(indices):
        if i == len(indices) - 1:
            parts.append(_gen_dim(idx))
        else:
            stride = 1
            for j in range(i + 1, len(tensor.shape)):
                shape_dim = tensor.shape[j]
                if shape_dim.terms:
                    raise NotImplementedError("Symbolic strides not supported")
                stride *= shape_dim.base

            idx_str = _gen_dim(idx)
            parts.append(f"({idx_str} * {stride})")

    return " + ".join(parts)


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
    def _visit_expr(expr: ScalarExpr) -> None:
        if isinstance(expr, TensorInput):
            tensors[expr.tensor.name] = expr.tensor
        elif isinstance(expr, Op):
            for op in expr.operands:
                _visit_expr(op)
    for s in block.stmts:
        visit(s)
    return list(tensors.values())

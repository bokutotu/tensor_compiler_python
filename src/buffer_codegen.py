from buffer_ir import Buffer
from loop_ir import Block, Compute, Loop, TensorAccess, Stmt
from tensor_ir import DimExpr, DType, Op, OpKind, ScalarExpr, Tensor, TensorInput


def generate_code(buffer: Buffer) -> str:
    tensors = _collect_tensors(buffer.body)

    # Verify all tensors are FLOAT32
    for t in tensors:
        if t.dtype != DType.FLOAT32:
            raise NotImplementedError(f"Only FLOAT32 is supported, got {t.dtype} for tensor {t.name}")

    params = ", ".join(f"float* {t.name}" for t in tensors)
    body = _gen_block(buffer.body, 1)
    return f"void {buffer.name}({params}) {{\n{body}\n}}"


def _gen_block(block: Block, indent: int) -> str:
    return "\n".join(_gen_stmt(s, indent) for s in block.stmts)


def _gen_stmt(stmt: Stmt, indent: int) -> str:
    ind = "  " * indent
    if isinstance(stmt, Loop):
        var = stmt.iter_var.name
        lo = _gen_dim(stmt.domain.lower)
        hi = _gen_dim(stmt.domain.upper)
        body = "\n".join(_gen_stmt(s, indent + 1) for s in stmt.body)
        return f"{ind}for (int {var} = {lo}; {var} < {hi}; {var} += {stmt.step}) {{\n{body}\n{ind}}}"
    elif isinstance(stmt, Compute):
        lhs = _gen_access(stmt.write)
        rhs = _gen_expr(stmt.expr)
        return f"{ind}{lhs} = {rhs};"
    raise NotImplementedError(type(stmt))


def _gen_access(access: TensorAccess) -> str:
    idx = _gen_index(access.tensor, access.indices)
    return f"{access.tensor.name}[{idx}]"


def _gen_expr(expr: ScalarExpr) -> str:
    if isinstance(expr, Op):
        l = _gen_expr(expr.operands[0])
        r = _gen_expr(expr.operands[1])
        op = {OpKind.ADD: "+", OpKind.MUL: "*"}[expr.kind]
        return f"({l} {op} {r})"
    elif isinstance(expr, TensorInput):
        idx = _gen_index(expr.tensor, expr.indices)
        return f"{expr.tensor.name}[{idx}]"
    raise NotImplementedError(type(expr))


def _gen_dim(expr: DimExpr) -> str:
    if expr.base and expr.terms:
        raise NotImplementedError("base + terms")
    if expr.terms:
        sym, coeff = next(iter(expr.terms.items()))
        return sym.name if coeff == 1 else f"{coeff} * {sym.name}"
    return str(expr.base)


def _gen_index(tensor: Tensor, indices: tuple[DimExpr, ...]) -> str:
    if len(indices) == 1:
        return _gen_dim(indices[0])

    # Multi-dimensional: compute strides and flatten
    parts: list[str] = []
    for i, idx in enumerate(indices):
        if i == len(indices) - 1:
            # Last dimension: no stride multiplication
            parts.append(_gen_dim(idx))
        else:
            # Compute stride: product of remaining dimensions
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
    def _visit_expr(expr: ScalarExpr) -> None:
        if isinstance(expr, TensorInput):
            tensors[expr.tensor.name] = expr.tensor
        elif isinstance(expr, Op):
            for op in expr.operands:
                _visit_expr(op)
    for s in block.stmts:
        visit(s)
    return list(tensors.values())

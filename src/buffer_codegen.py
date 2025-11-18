from buffer_ir import Buffer
from loop_ir import Block, Compute, Loop, TensorAccess, Stmt
from tensor_ir import DimExpr, Op, OpKind, ScalarExpr, Tensor, TensorInput


def generate_code(buffer: Buffer) -> str:
    tensors = _collect_tensors(buffer.body)
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
    idx = _gen_dim(access.indices[0])
    return f"{access.tensor.name}[{idx}]"


def _gen_expr(expr: ScalarExpr) -> str:
    if isinstance(expr, Op):
        l = _gen_expr(expr.operands[0])
        r = _gen_expr(expr.operands[1])
        op = {OpKind.ADD: "+", OpKind.MUL: "*"}[expr.kind]
        return f"({l} {op} {r})"
    elif isinstance(expr, TensorInput):
        idx = _gen_dim(expr.indices[0])
        return f"{expr.tensor.name}[{idx}]"
    raise NotImplementedError(type(expr))


def _gen_dim(expr: DimExpr) -> str:
    if expr.base and expr.terms:
        raise NotImplementedError("base + terms")
    if expr.terms:
        sym, coeff = next(iter(expr.terms.items()))
        return sym.name if coeff == 1 else f"{coeff} * {sym.name}"
    return str(expr.base)


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

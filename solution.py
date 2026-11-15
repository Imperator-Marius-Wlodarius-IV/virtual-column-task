import ast
import re
import pandas

VALID_LABEL_RE = re.compile(r"^[A-Za-z_]+$")
ALLOWED_OPERATORS = (ast.Add, ast.Sub, ast.Mult)


def _is_valid_label(label: object) -> bool:
    return isinstance(label, str) and bool(VALID_LABEL_RE.fullmatch(label))


def _evaluate_expression(node: ast.AST, df: pandas.DataFrame) -> pandas.Series:
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body, df)

    if isinstance(node, ast.Name):
        if not _is_valid_label(node.id) or node.id not in df.columns:
            raise ValueError(f"Invalid or missing column label: {node.id}")
        return df[node.id]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ALLOWED_OPERATORS):
        left = _evaluate_expression(node.left, df)
        right = _evaluate_expression(node.right, df)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right

    raise ValueError("Unsupported syntax or operator in role expression")


def add_virtual_column(
    df: pandas.DataFrame, role: str, new_column: str
) -> pandas.DataFrame:
    if not isinstance(df, pandas.DataFrame):
        return pandas.DataFrame([])
    if not isinstance(role, str) or not _is_valid_label(new_column):
        return pandas.DataFrame([])
    if not all(_is_valid_label(col) for col in df.columns):
        return pandas.DataFrame([])

    expression = role.strip()
    if not expression:
        return pandas.DataFrame([])

    try:
        parsed_ast = ast.parse(expression, mode="eval")
        virtual_values = _evaluate_expression(parsed_ast, df)
    except (SyntaxError, ValueError, TypeError):
        return pandas.DataFrame([])

    result_df = df.copy()
    result_df[new_column] = virtual_values
    return result_df
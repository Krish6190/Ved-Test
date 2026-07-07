"""2-layer code chunker — AST-aware for Python, paragraph fallback for others.

Layer 1 ("sig"): one chunk per top-level def/class/import/assign. Contains
  the signature line + decorators + docstring (no body). The planner uses
  this layer to discover what symbols exist in a file without loading
  full implementations.

Layer 2 ("body"): one chunk per function/class body. Contains the full
  implementation. The planner uses this when it needs exact code.

Non-Python files fall back to paragraph-based chunking (split on blank
lines, group small ones, split large ones). Non-code text is tagged with
layer="body" so it behaves the same as body chunks for retrieval.

The `layer` field is stored alongside each chunk in LocalVectorDB's
registry. retrieve_rag can filter by layer when the planner wants only
signatures (cheap navigation) or only bodies (exact code).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List


def chunk_python_file(path: str | Path) -> List[Dict]:
    """Parse a Python file and return 2-layer chunks.

    Returns a list of dicts with keys:
      - content: str (the chunk text)
      - layer: "sig" or "body"
      - name: str (symbol name, e.g. function name)
      - lineno: int (1-indexed start line)
      - end_lineno: int (1-indexed end line)

    On SyntaxError, falls back to paragraph chunking.
    """
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_text_paragraphs(source, fallback_tag="py_fallback")

    chunks: List[Dict] = []
    lines = source.splitlines(keepends=True)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(_function_chunks(node, lines))
        elif isinstance(node, ast.ClassDef):
            chunks.extend(_class_chunks(node, lines))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            chunks.append(_node_lines(node, lines, layer="sig", name=_import_name(node)))
        elif isinstance(node, ast.Assign):
            chunks.append(_node_lines(node, lines, layer="sig", name=_assign_name(node)))

    return chunks


def chunk_text_paragraphs(
    text: str,
    max_chunk_size: int = 800,
    fallback_tag: str = "text",
) -> List[Dict]:
    """Paragraph-based chunking for non-Python files.

    Splits on blank lines, groups small paragraphs together, splits large
    paragraphs at max_chunk_size boundaries. Non-code text is tagged with
    layer="body" so retrieval treats it uniformly with code bodies.
    """
    chunks: List[Dict] = []
    if not text.strip():
        return chunks

    paragraphs = text.split("\n\n")
    current_lines: List[str] = []
    current_start_line = 1
    cursor = 1

    def _flush():
        nonlocal current_lines, current_start_line
        if not current_lines:
            return
        content = "\n\n".join(current_lines)
        chunks.append({
            "content": content,
            "layer": "body",
            "name": fallback_tag,
            "lineno": current_start_line,
            "end_lineno": current_start_line + content.count("\n"),
        })
        current_lines = []
        current_start_line = cursor

    for para in paragraphs:
        para = para.strip("\n")
        if not para:
            cursor += 1
            continue
        para_line_count = para.count("\n") + 1

        # If adding this paragraph exceeds the limit and we have content, flush.
        prospective = "\n\n".join(current_lines + [para])
        if len(prospective) > max_chunk_size and current_lines:
            _flush()
            # If a single paragraph is bigger than max_chunk_size, hard-split it.
            if len(para) > max_chunk_size:
                for j in range(0, len(para), max_chunk_size):
                    piece = para[j:j + max_chunk_size]
                    chunks.append({
                        "content": piece,
                        "layer": "body",
                        "name": fallback_tag,
                        "lineno": cursor,
                        "end_lineno": cursor + piece.count("\n"),
                    })
                    cursor += piece.count("\n") + 1
                current_start_line = cursor
            else:
                current_start_line = cursor
                current_lines = [para]
                cursor += para_line_count + 1  # +1 for the blank line
        else:
            current_lines.append(para)
            cursor += para_line_count + 1  # +1 for the blank line

    _flush()
    return chunks


def chunk_file(path: str | Path) -> List[Dict]:
    """Dispatch to the right chunker based on file extension.

    Returns a list of chunk dicts. Each dict has the standard fields
    (content, layer, name, lineno, end_lineno).
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".py":
        return chunk_python_file(p)
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return chunk_text_paragraphs(text, fallback_tag=ext.lstrip(".") or "text")


# ---- Internal helpers ----

def _node_lines(node, lines: List[str], *, layer: str, name: str) -> Dict:
    """Extract the source lines for an AST node, returning a chunk dict."""
    start = max(0, node.lineno - 1)
    end = min(len(lines), node.end_lineno)
    return {
        "content": "".join(lines[start:end]),
        "layer": layer,
        "name": name,
        "lineno": node.lineno,
        "end_lineno": node.end_lineno,
    }


def _function_chunks(node, lines: List[str]) -> List[Dict]:
    """Build sig + body chunks for a function or async function."""
    chunks: List[Dict] = []
    # Determine the signature span: from first decorator (if any) to the
    # line just before the first body statement.
    if node.decorator_list:
        start_line = node.decorator_list[0].lineno
    else:
        start_line = node.lineno
    body_first_line = node.body[0].lineno if node.body else node.end_lineno + 1
    sig_end_line = body_first_line - 1

    sig_lines = lines[start_line - 1:sig_end_line]
    sig_content = "".join(sig_lines)
    docstring = ast.get_docstring(node)
    if docstring:
        # Indent docstring to match the function body indent.
        first_stmt = node.body[0] if node.body else None
        indent = ""
        if first_stmt and isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
            # Compute indent from the docstring line.
            ds_line = first_stmt.lineno - 1
            if ds_line < len(lines):
                line_text = lines[ds_line]
                indent = line_text[: len(line_text) - len(line_text.lstrip())]
        sig_content += f'{indent}"""{docstring}"""\n'

    chunks.append({
        "content": sig_content,
        "layer": "sig",
        "name": node.name,
        "lineno": start_line,
        "end_lineno": sig_end_line,
    })

    # Body chunk: from signature start through the function's full extent.
    body_start = max(0, node.lineno - 1)
    body_end = min(len(lines), node.end_lineno)
    chunks.append({
        "content": "".join(lines[body_start:body_end]),
        "layer": "body",
        "name": node.name,
        "lineno": node.lineno,
        "end_lineno": node.end_lineno,
    })
    return chunks


def _class_chunks(node, lines: List[str]) -> List[Dict]:
    """Build sig + body chunks for a class."""
    chunks: List[Dict] = []
    if node.decorator_list:
        start_line = node.decorator_list[0].lineno
    else:
        start_line = node.lineno
    body_first_line = node.body[0].lineno if node.body else node.end_lineno + 1
    sig_end_line = body_first_line - 1

    sig_lines = lines[start_line - 1:sig_end_line]
    sig_content = "".join(sig_lines)
    docstring = ast.get_docstring(node)
    if docstring:
        first_stmt = node.body[0] if node.body else None
        indent = ""
        if first_stmt and isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
            ds_line = first_stmt.lineno - 1
            if ds_line < len(lines):
                line_text = lines[ds_line]
                indent = line_text[: len(line_text) - len(line_text.lstrip())]
        sig_content += f'{indent}"""{docstring}"""\n'

    chunks.append({
        "content": sig_content,
        "layer": "sig",
        "name": node.name,
        "lineno": start_line,
        "end_lineno": sig_end_line,
    })

    # Body chunk: full class
    body_start = max(0, node.lineno - 1)
    body_end = min(len(lines), node.end_lineno)
    chunks.append({
        "content": "".join(lines[body_start:body_end]),
        "layer": "body",
        "name": node.name,
        "lineno": node.lineno,
        "end_lineno": node.end_lineno,
    })
    return chunks


def _import_name(node) -> str:
    """Human-readable name for an import node."""
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else "import"
    if isinstance(node, ast.ImportFrom):
        first = node.names[0].name if node.names else ""
        return f"{node.module or ''}.{first}".lstrip(".")
    return "import"


def _assign_name(node) -> str:
    """Human-readable name for an assignment target."""
    if node.targets:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return "assignment"


__all__ = ["chunk_python_file", "chunk_text_paragraphs", "chunk_file"]

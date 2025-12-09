"""Script per rimuovere commenti e righe vuote in eccesso dai file Python.

Mantiene:
- Docstring iniziale del modulo (triple quote all'inizio)
- Codice funzionante
- Una sola riga vuota tra le definizioni

Rimuove:
- Commenti inline (# ...)
- Righe solo commento
- Righe vuote multiple (ne lascia max 1)
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import List, Tuple

def extract_module_docstring(lines: List[str]) -> Tuple[List[str], int]:
    """Estrae la docstring iniziale del modulo."""
    if not lines:
        return [], 0
    docstring_lines = []
    i = 0
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    if i >= len(lines):
        return [], 0
    first_line = lines[i].strip()
    if first_line.startswith('"""') or first_line.startswith("'''"):
        quote = first_line[:3]
        if first_line.count(quote) >= 2 and len(first_line) > 6:
            docstring_lines.append(lines[i])
            return docstring_lines, i + 1
        docstring_lines.append(lines[i])
        i += 1
        while i < len(lines):
            docstring_lines.append(lines[i])
            if quote in lines[i]:
                return docstring_lines, i + 1
            i += 1
    return [], 0

def is_string_line(line: str) -> bool:
    """Verifica se la riga è dentro una stringa multilinea."""
    stripped = line.strip()
    return stripped.startswith(('"""', "'''", '"', "'"))

def remove_inline_comment(line: str) -> str:
    """Rimuove commenti inline mantenendo stringhe intatte."""
    result = []
    in_string = False
    string_char = None
    i = 0
    while i < len(line):
        char = line[i]
        if not in_string:
            if char in ('"', "'"):
                if i + 2 < len(line) and line[i:i+3] in ('"""', "'''"):
                    result.append(line[i:i+3])
                    in_string = True
                    string_char = line[i:i+3]
                    i += 3
                    continue
                in_string = True
                string_char = char
                result.append(char)
            elif char == '#':
                break
            else:
                result.append(char)
        else:
            if string_char in ('"""', "'''"):
                if i + 2 < len(line) and line[i:i+3] == string_char:
                    result.append(line[i:i+3])
                    in_string = False
                    string_char = None
                    i += 3
                    continue
            elif char == string_char and (i == 0 or line[i-1] != '\\'):
                in_string = False
                string_char = None
            result.append(char)
        i += 1
    return ''.join(result).rstrip()

def clean_file(filepath: Path) -> bool:
    """Pulisce un singolo file Python."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  Errore lettura {filepath}: {e}")
        return False
    lines = content.splitlines(keepends=True)
    if not lines:
        return True
    docstring_lines, start_idx = extract_module_docstring([l.rstrip('\n\r') for l in lines])
    cleaned_lines = []
    for dl in docstring_lines:
        cleaned_lines.append(dl + '\n')
    in_multiline_string = False
    multiline_quote = None
    prev_was_empty = False
    for i in range(start_idx, len(lines)):
        line = lines[i]
        line_stripped = line.strip()
        line_content = line.rstrip('\n\r')
        if not in_multiline_string:
            for quote in ('"""', "'''"):
                count = line_content.count(quote)
                if count == 1:
                    in_multiline_string = True
                    multiline_quote = quote
                    break
                elif count >= 2:
                    pass
        else:
            if multiline_quote and multiline_quote in line_content:
                in_multiline_string = False
                multiline_quote = None
        if in_multiline_string or (multiline_quote and multiline_quote in line_content):
            cleaned_lines.append(line)
            prev_was_empty = False
            continue
        if line_stripped == '':
            if not prev_was_empty:
                cleaned_lines.append('\n')
                prev_was_empty = True
            continue
        if line_stripped.startswith('#'):
            continue
        cleaned_line = remove_inline_comment(line_content)
        if cleaned_line.strip() == '':
            if not prev_was_empty:
                cleaned_lines.append('\n')
                prev_was_empty = True
            continue
        indent = len(line_content) - len(line_content.lstrip())
        final_line = ' ' * indent + cleaned_line.lstrip() + '\n'
        cleaned_lines.append(final_line)
        prev_was_empty = False
    while cleaned_lines and cleaned_lines[-1].strip() == '':
        cleaned_lines.pop()
    if cleaned_lines:
        cleaned_lines.append('\n')
    try:
        filepath.write_text(''.join(cleaned_lines), encoding='utf-8')
        return True
    except Exception as e:
        print(f"  Errore scrittura {filepath}: {e}")
        return False

def main():
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    app_dir = project_root / "app"
    dirs_to_clean = [src_dir, app_dir]
    files_cleaned = 0
    files_failed = 0
    for dir_path in dirs_to_clean:
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            print(f"Pulizia: {py_file.relative_to(project_root)}")
            if clean_file(py_file):
                files_cleaned += 1
            else:
                files_failed += 1
    print(f"\nCompletato: {files_cleaned} file puliti, {files_failed} falliti")

if __name__ == "__main__":
    main()

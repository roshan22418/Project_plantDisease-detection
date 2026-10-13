# Repository-wide checks; only needs pytest.
import ast
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(p for p in ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
TEXT_FILES = sorted(
    p for p in ROOT.rglob("*")
    if p.suffix in {".py", ".ipynb", ".json", ".md", ".txt", ".yml", ".toml"} and ".git" not in p.parts
)
SECRET_PATTERNS = {
    "W&B key": re.compile(r"wandb\s+login\s+[0-9a-f]{40}"),
    "OpenAI key": re.compile(r"sk-(proj-)?[A-Za-z0-9_-]{32,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
}


def test_notebooks_found():
    assert len(NOTEBOOKS) == 6


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_code_cells_are_valid_python(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        # IPython shell (!) and magic (%) lines are not Python
        lines = [line for line in "".join(cell["source"]).split("\n") if not line.lstrip().startswith(("!", "%"))]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as e:
            pytest.fail(f"cell {i}, line {e.lineno}: {e.msg}: {e.text!r}")


@pytest.mark.parametrize("name, pattern", SECRET_PATTERNS.items(), ids=list(SECRET_PATTERNS))
def test_no_hard_coded_secrets(name, pattern):
    leaks = [str(p.relative_to(ROOT)) for p in TEXT_FILES if pattern.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert leaks == [], f"{name} found in {leaks}"

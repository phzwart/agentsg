"""Sphinx configuration for the agentsg documentation."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "agentsg" / "src"))

project = "agentsg"
author = "P. H. Zwart"
try:
    release = version("agentsg")
except PackageNotFoundError:
    release = "0.5.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "agentsg"
html_theme_options = {
    "source_repository": "https://github.com/phzwart/agentsg",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "class"
napoleon_use_ivar = True
napoleon_numpy_docstring = True

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

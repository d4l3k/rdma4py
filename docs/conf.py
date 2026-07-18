from __future__ import annotations

from importlib.metadata import version as package_version

project = "rdma4py"
author = "Tristan Rice"
copyright = "2026, Tristan Rice"
release = package_version("efa")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "rdma4py"
html_baseurl = "https://d4l3k.github.io/rdma4py/"
html_theme_options = {
    "source_repository": "https://github.com/d4l3k/rdma4py/",
    "source_branch": "main",
    "source_directory": "docs/",
}

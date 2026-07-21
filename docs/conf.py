from __future__ import annotations

from importlib.metadata import version as package_version
from pathlib import Path
from shutil import copytree

import pydata_sphinx_theme

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
html_theme = "pytorch_sphinx_theme2"
html_title = "rdma4py"
html_baseurl = "https://d4l3k.github.io/rdma4py/"
html_theme_options = {
    "article_footer_items": [],
    "article_header_end": [],
    "collapse_navigation": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/d4l3k/rdma4py",
            "icon": "fa-brands fa-github",
        }
    ],
    "logo": {"text": "rdma4py"},
    "navbar_end": ["search-field", "theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_prev_next": True,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "d4l3k",
    "github_repo": "rdma4py",
    "github_version": "main",
    "doc_path": "docs",
}


def _copy_theme_webfonts(app, exception) -> None:
    if exception is not None or app.builder.format != "html":
        return
    # Theme 2's wheel omits the Font Awesome files referenced by its CSS.
    source = (
        Path(pydata_sphinx_theme.__file__).parent
        / "theme"
        / "pydata_sphinx_theme"
        / "static"
        / "vendor"
        / "fontawesome"
        / "6.5.2"
        / "webfonts"
    )
    copytree(source, Path(app.outdir) / "_static" / "webfonts", dirs_exist_ok=True)


def setup(app) -> None:
    app.connect("build-finished", _copy_theme_webfonts)

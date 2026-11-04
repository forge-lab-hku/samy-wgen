"""Sphinx configuration for the samy-wgen user guide."""

from importlib.metadata import version as package_version

project = "samy-wgen"
release = package_version("samy-wgen")
extensions = []
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_title = "samy-wgen documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "fixed_sidebar": True,
    "page_width": "1200px",
    "sidebar_width": "250px",
    "show_relbars": False,
}
html_sidebars = {"**": ["guide-sidebar.html", "searchbox.html"]}

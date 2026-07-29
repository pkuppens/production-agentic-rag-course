"""Shared notebook display configuration."""

import pandas as pd
from IPython.display import HTML, display


def set_wide_output() -> None:
    """Widen pandas tables and the notebook's own display area.

    Pandas: show full column content and all columns instead of truncating
    with "...", which hides useful detail in verbose service/status output.

    Notebook layout: stretch the output area to the full browser/editor
    width (a no-op in editors, like VS Code, that already use full width).
    """
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    display(HTML("<style>.container { width: 100% !important; }</style>"))

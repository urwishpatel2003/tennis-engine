"""Tennis matchup prediction engine."""

import sys

__version__ = "0.1.0"


def _enable_utf8_console() -> None:
    """
    Force UTF-8 on stdout/stderr.

    Windows consoles default to cp1252, which cannot encode the box-drawing and
    arrow characters used throughout this project's CLI output — every script
    would die with a UnicodeEncodeError on its first status line. Done here, at
    package import, so it covers every entry point without each one repeating it.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # already UTF-8, or a stream that does not support reconfigure


_enable_utf8_console()

"""Backwards-compatible entry point.

Usage is unchanged: `python pipeline.py --data-path ...`. The implementation
lives in the evloadcontrol package (see evloadcontrol/cli.py); this file is
kept only so existing commands and the README examples do not need to change.
"""

from evloadcontrol.cli import main

if __name__ == "__main__":
    main()

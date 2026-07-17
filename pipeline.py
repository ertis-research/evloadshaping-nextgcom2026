"""Backwards-compatible entry point.

Usage is unchanged: `python pipeline.py --data-path ...`. The implementation
lives in the evloadshaping package (see evloadshaping/cli.py); this file is
kept only so existing commands, the Dockerfile CMD, and the README examples
do not need to change.
"""

from evloadshaping.cli import main

if __name__ == "__main__":
    main()

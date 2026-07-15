#!/usr/bin/env python
"""Convenience entry point: `python run.py --video home.mp4 --workdir out`.

Equivalent to the installed `video2cad` console script.
"""
import sys

from video2cad.cli import main

if __name__ == "__main__":
    sys.exit(main())

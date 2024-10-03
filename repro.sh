#!/usr/bin/env bash

# requires virtualenv, and `pip install .[dev]`.

PYTHONPATH=. python vyper/cli/vyper_compile.py bug.vy


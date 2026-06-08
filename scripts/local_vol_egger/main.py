"""
Egger & Engl (2005) — entry point.

Set EXAMPLE below, then run from this directory:

    conda run -n TFM_stable python main.py

Or run an example file directly:

    conda run -n TFM_stable python examples/example1.py
"""

# Available examples:
#   "example1"  — §5.4 Example 1, case A: constant vol recovery (no noise)
EXAMPLE = "example1"

if __name__ == "__main__":
    if EXAMPLE == "example1":
        from examples.example1 import run
    else:
        raise ValueError(f"Unknown example '{EXAMPLE}'")

    run()

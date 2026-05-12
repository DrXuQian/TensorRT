#!/usr/bin/env python3
"""End-to-end numerical verification: trt-opt lowered IR vs numpy oracle."""

import subprocess
import numpy as np
import re
import sys
import os
import tempfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRT_OPT = os.path.join(PROJ, "build", "bin", "trt-opt")
SHLO_TRANSLATE = os.path.join(PROJ, "build", "bin", "stablehlo-translate")

from oracle import trt_strict, trt_clamp, trt_fill, trt_wrap, trt_reflect


def make_trt_ir(d, start, size, stride, mode, fill=0, ndim=1):
    """Generate trt dialect MLIR IR for a test case."""
    in_shape = f"{d}xi32"
    out_shape = f"{size}xi32"
    x_vals = ",".join(str(i) for i in range(d))
    fill_operand = ""
    fill_type_in = ""
    fill_const = ""

    if mode == "fill":
        fill_const = f'  %f = stablehlo.constant dense<{fill}> : tensor<i32>\n'
        fill_operand = ", %f"
        fill_type_in = ", tensor<i32>"

    mode_str = {
        "strict_bounds": "kSTRICT_BOUNDS",
        "clamp": "kCLAMP",
        "fill": "kFILL",
        "wrap": "kWRAP",
        "reflect": "kREFLECT",
    }[mode]

    ir = f"""func.func @main() -> tensor<{out_shape}> {{
  %in = stablehlo.constant dense<[{x_vals}]> : tensor<{in_shape}>
{fill_const}  %0 = trt.slice %in{fill_operand} {{
    start = array<i64: {start}>, size = array<i64: {size}>,
    stride = array<i64: {stride}>,
    mode = #trt<sample_mode {mode_str}>
  }} : (tensor<{in_shape}>{fill_type_in}) -> tensor<{out_shape}>
  return %0 : tensor<{out_shape}>
}}
"""
    return ir


def lower_and_interpret(trt_ir):
    """Lower trt IR to stablehlo, then interpret it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(trt_ir)
        f.flush()
        trt_file = f.name

    try:
        # Lower trt -> stablehlo
        r1 = subprocess.run(
            [TRT_OPT, trt_file, "--convert-trt-to-stablehlo"],
            capture_output=True, text=True
        )
        if r1.returncode != 0:
            return None, f"trt-opt error: {r1.stderr}"

        lowered_ir = r1.stdout

        # Write lowered IR
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f2:
            f2.write(lowered_ir)
            f2.flush()
            shlo_file = f2.name

        # Interpret
        r2 = subprocess.run(
            [SHLO_TRANSLATE, "--interpret", shlo_file],
            capture_output=True, text=True
        )
        os.unlink(shlo_file)

        if r2.returncode != 0:
            return lowered_ir, f"stablehlo-translate error: {r2.stderr}"

        # Parse output tensor values - extract from [a, b, c] line
        output = r2.stdout.strip()
        bracket_match = re.search(r"\[([^\]]*)\]", output)
        if bracket_match:
            vals = re.findall(r"[-+]?\d+", bracket_match.group(1))
            result = np.array([int(v) for v in vals], dtype=np.int32)
        else:
            return lowered_ir, f"Could not parse output: {output}"
        return lowered_ir, result

    finally:
        os.unlink(trt_file)


def oracle(x, start, size, stride, mode, fill=0):
    """Get oracle result."""
    if mode == "strict_bounds":
        return trt_strict(x, start, size, stride)
    elif mode == "clamp":
        return trt_clamp(x, start, size, stride)
    elif mode == "fill":
        return trt_fill(x, start, size, stride, fill)
    elif mode == "wrap":
        return trt_wrap(x, start, size, stride)
    elif mode == "reflect":
        return trt_reflect(x, start, size, stride)


# Test cases: (mode, d, start, size, stride, fill)
TEST_CASES = [
    # === kSTRICT_BOUNDS (20 cases) ===
    ("strict_bounds", 10, 2, 3, 2, 0),
    ("strict_bounds", 10, 0, 5, 1, 0),
    ("strict_bounds", 10, 1, 3, 3, 0),
    ("strict_bounds", 8, 7, 3, -2, 0),
    ("strict_bounds", 10, 5, 2, -1, 0),
    ("strict_bounds", 20, 0, 10, 2, 0),
    ("strict_bounds", 15, 3, 4, 3, 0),
    ("strict_bounds", 12, 11, 4, -3, 0),
    ("strict_bounds", 6, 0, 6, 1, 0),
    ("strict_bounds", 10, 9, 5, -2, 0),
    ("strict_bounds", 8, 1, 2, 3, 0),
    ("strict_bounds", 10, 0, 1, 1, 0),
    ("strict_bounds", 10, 9, 1, 1, 0),
    ("strict_bounds", 10, 4, 3, 1, 0),
    ("strict_bounds", 16, 2, 5, 3, 0),
    ("strict_bounds", 10, 8, 3, -1, 0),
    ("strict_bounds", 20, 19, 7, -3, 0),
    ("strict_bounds", 5, 0, 3, 2, 0),
    ("strict_bounds", 7, 6, 2, -3, 0),
    ("strict_bounds", 10, 3, 2, 2, 0),

    # === kFILL (20 cases) ===
    ("fill", 5, -2, 9, 1, 0),
    ("fill", 5, -1, 4, 2, -1),
    ("fill", 5, 3, 5, 1, 99),
    ("fill", 5, 5, 4, -2, 42),
    ("fill", 8, -3, 6, 2, 0),
    ("fill", 10, -5, 20, 1, -7),
    ("fill", 4, -2, 3, 3, 0),
    ("fill", 6, 4, 6, 1, 88),
    ("fill", 3, -1, 5, 1, 0),
    ("fill", 8, 7, 5, -1, 11),
    ("fill", 10, -4, 8, 2, 0),
    ("fill", 5, 0, 5, 1, 0),
    ("fill", 5, -3, 3, 2, -5),
    ("fill", 12, 10, 6, 1, 77),
    ("fill", 7, -2, 4, 3, 0),
    ("fill", 6, 8, 4, -2, 33),
    ("fill", 4, -1, 6, 1, 0),
    ("fill", 10, -1, 12, 1, 0),
    ("fill", 3, 0, 3, 1, 0),
    ("fill", 8, 2, 3, 2, 0),

    # === kCLAMP (20 cases) ===
    ("clamp", 5, -2, 9, 1, 0),
    ("clamp", 5, 3, 5, 1, 0),
    ("clamp", 8, -1, 4, 2, 0),
    ("clamp", 5, 6, 3, -2, 0),
    ("clamp", 10, -3, 8, 2, 0),
    ("clamp", 4, -3, 10, 1, 0),
    ("clamp", 6, 5, 4, 1, 0),
    ("clamp", 10, -5, 6, 3, 0),
    ("clamp", 3, -1, 5, 1, 0),
    ("clamp", 8, 9, 4, -2, 0),
    ("clamp", 12, -2, 8, 2, 0),
    ("clamp", 5, 0, 5, 1, 0),
    ("clamp", 7, -4, 6, 2, 0),
    ("clamp", 6, 3, 6, 1, 0),
    ("clamp", 10, 8, 6, -1, 0),
    ("clamp", 4, -1, 3, 2, 0),
    ("clamp", 15, -2, 10, 2, 0),
    ("clamp", 5, 6, 4, -1, 0),
    ("clamp", 8, -3, 5, 3, 0),
    ("clamp", 3, 0, 3, 1, 0),

    # === kWRAP (20 cases) ===
    ("wrap", 5, -2, 9, 1, 0),
    ("wrap", 5, 3, 5, 1, 0),
    ("wrap", 8, -1, 4, 2, 0),
    ("wrap", 5, 6, 3, -2, 0),
    ("wrap", 4, -2, 6, 1, 0),
    ("wrap", 6, -3, 12, 1, 0),
    ("wrap", 10, 8, 6, 1, 0),
    ("wrap", 3, -1, 7, 1, 0),
    ("wrap", 7, 5, 4, 2, 0),
    ("wrap", 5, 7, 5, -1, 0),
    ("wrap", 4, -4, 8, 1, 0),
    ("wrap", 8, -2, 5, 3, 0),
    ("wrap", 6, 0, 6, 1, 0),
    ("wrap", 10, -1, 3, 3, 0),
    ("wrap", 5, 3, 3, 2, 0),
    ("wrap", 12, -5, 10, 2, 0),
    ("wrap", 4, 5, 6, -1, 0),
    ("wrap", 3, -2, 4, 2, 0),
    ("wrap", 7, 10, 3, -3, 0),
    ("wrap", 8, -1, 10, 1, 0),

    # === kREFLECT (20 cases) ===
    ("reflect", 5, -2, 9, 1, 0),
    ("reflect", 5, 3, 5, 1, 0),
    ("reflect", 8, -1, 4, 2, 0),
    ("reflect", 5, 6, 3, -2, 0),
    ("reflect", 6, -2, 5, 2, 0),
    ("reflect", 10, -3, 8, 2, 0),
    ("reflect", 4, -2, 7, 1, 0),
    ("reflect", 8, 7, 6, -1, 0),
    ("reflect", 6, -1, 3, 2, 0),
    ("reflect", 5, 5, 4, -1, 0),
    ("reflect", 12, -4, 10, 2, 0),
    ("reflect", 3, -1, 5, 1, 0),
    ("reflect", 7, 6, 5, -1, 0),
    ("reflect", 10, -2, 6, 3, 0),
    ("reflect", 5, 0, 5, 1, 0),
    ("reflect", 6, -3, 4, 3, 0),
    ("reflect", 8, 9, 5, -2, 0),
    ("reflect", 4, -1, 6, 1, 0),
    ("reflect", 10, 8, 4, -2, 0),
    ("reflect", 5, -1, 3, 1, 0),
]


def main():
    if not os.path.exists(TRT_OPT):
        print(f"ERROR: trt-opt not found at {TRT_OPT}")
        sys.exit(1)
    if not os.path.exists(SHLO_TRANSLATE):
        print(f"ERROR: stablehlo-translate not found at {SHLO_TRANSLATE}")
        sys.exit(1)

    all_pass = True
    mode_results = {}

    for mode, d, start, size, stride, fill in TEST_CASES:
        x = np.arange(d, dtype=np.int32)
        expected = oracle(x, start, size, stride, mode, fill)

        trt_ir = make_trt_ir(d, start, size, stride, mode, fill)
        lowered_ir, actual = lower_and_interpret(trt_ir)

        if isinstance(actual, str):
            print(f"  FAIL {mode} d={d} s={start} sz={size} st={stride}: {actual}")
            all_pass = False
            continue

        match = np.array_equal(expected, actual)
        status = "ok" if match else "FAIL"
        if not match:
            all_pass = False
            print(f"  FAIL {mode} d={d} s={start} sz={size} st={stride}:")
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")
        else:
            mode_results.setdefault(mode, []).append(True)

    print("\nE2E numerical verification summary:")
    for mode in ["strict_bounds", "fill", "clamp", "wrap", "reflect"]:
        count = len(mode_results.get(mode, []))
        total = sum(1 for m, *_ in TEST_CASES if m == mode)
        mark = "ok" if count == total else "FAIL"
        print(f"  {mode}: {count}/{total} {mark}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

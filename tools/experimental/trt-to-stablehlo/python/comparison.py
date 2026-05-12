#!/usr/bin/env python3
"""Generate side-by-side comparison for each mode."""

import subprocess
import numpy as np
import re
import os
import tempfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRT_OPT = os.path.join(PROJ, "build", "bin", "trt-opt")
SHLO_TRANSLATE = os.path.join(PROJ, "build", "bin", "stablehlo-translate")

from oracle import trt_strict, trt_clamp, trt_fill, trt_wrap, trt_reflect

CASES = [
    ("strict_bounds", 10, 2, 3, 2, 0),
    ("fill", 5, -2, 9, 1, 0),
    ("clamp", 5, -2, 9, 1, 0),
    ("wrap", 5, -2, 9, 1, 0),
    ("reflect", 5, -2, 9, 1, 0),
]

def make_ir(d, start, size, stride, mode, fill=0):
    x_vals = ",".join(str(i) for i in range(d))
    fill_const = ""
    fill_op = ""
    fill_ty = ""
    mode_str = {"strict_bounds":"kSTRICT_BOUNDS","clamp":"kCLAMP",
                "fill":"kFILL","wrap":"kWRAP","reflect":"kREFLECT"}[mode]
    if mode == "fill":
        fill_const = f'  %f = stablehlo.constant dense<{fill}> : tensor<i32>\n'
        fill_op = ", %f"
        fill_ty = ", tensor<i32>"
    return f"""func.func @main() -> tensor<{size}xi32> {{
  %in = stablehlo.constant dense<[{x_vals}]> : tensor<{d}xi32>
{fill_const}  %0 = trt.slice %in{fill_op} {{
    start = array<i64: {start}>, size = array<i64: {size}>,
    stride = array<i64: {stride}>,
    mode = #trt<sample_mode {mode_str}>
  }} : (tensor<{d}xi32>{fill_ty}) -> tensor<{size}xi32>
  return %0 : tensor<{size}xi32>
}}
"""

def run(ir):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(ir); f.flush(); name = f.name
    r1 = subprocess.run([TRT_OPT, name, "--convert-trt-to-stablehlo"],
                        capture_output=True, text=True)
    os.unlink(name)
    lowered = r1.stdout
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f2:
        f2.write(lowered); f2.flush(); name2 = f2.name
    r2 = subprocess.run([SHLO_TRANSLATE, "--interpret", name2],
                        capture_output=True, text=True)
    os.unlink(name2)
    m = re.search(r"\[([^\]]*)\]", r2.stdout)
    vals = np.array([int(v) for v in re.findall(r"[-+]?\d+", m.group(1))], dtype=np.int32) if m else None
    return lowered, vals

for mode, d, start, size, stride, fill in CASES:
    x = np.arange(d, dtype=np.int32)
    fn = {"strict_bounds":trt_strict,"clamp":trt_clamp,"fill":trt_fill,
          "wrap":trt_wrap,"reflect":trt_reflect}[mode]
    expected = fn(x, start, size, stride, fill) if mode == "fill" else fn(x, start, size, stride)

    trt_ir = make_ir(d, start, size, stride, mode, fill)
    lowered_ir, actual = run(trt_ir)

    print(f"\n{'='*70}")
    print(f"MODE={mode} d={d} start={start} size={size} stride={stride} fill={fill}")
    print(f"{'='*70}")
    print(f"\n--- C++ pass output (lowered StableHLO IR) ---")
    print(lowered_ir.strip())
    print(f"\n--- stablehlo-translate numerical result ---")
    print(f"  {actual}")
    print(f"\n--- oracle (numpy) numerical result ---")
    print(f"  {expected}")
    print(f"\n--- MATCH: {np.array_equal(expected, actual)} ---")

#!/usr/bin/env python3
"""E2E verification of dynamic slice gather-based lowering.

Strategy: re-implement the gather index formulas in Python (same as the C++
mapIndex), verify they produce the same output as the oracle. This validates
the ALGORITHM correctness. The MLIR codegen structure is verified by FileCheck.
"""

import numpy as np
import sys

sys.path.insert(0, "python")
from oracle import trt_strict, trt_clamp, trt_fill, trt_wrap, trt_reflect


def gather_eval(x, start, size, stride, mode, fill=0):
    """Python implementation of the gather-based dynamic lowering.
    Mirrors DynSliceConverter::mapIndex exactly."""
    d = x.shape[0]
    raw = np.arange(size, dtype=np.int64) * stride + start

    if mode == "strict_bounds":
        idx = raw
    elif mode == "clamp":
        idx = np.clip(raw, 0, d - 1)
    elif mode == "wrap":
        idx = ((raw % d) + d) % d
    elif mode == "reflect":
        p = 2 * (d - 1)
        c = ((np.abs(raw) % p) + p) % p
        idx = np.where(c >= d, p - c, c)
    elif mode == "fill":
        in_bnd = (raw >= 0) & (raw < d)
        idx = np.clip(raw, 0, d - 1)
        result = x[idx].copy()
        result[~in_bnd] = fill
        return result
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return x[idx]


ORACLE_FN = {
    "strict_bounds": lambda x, s, sz, st, f: trt_strict(x, s, sz, st),
    "clamp": lambda x, s, sz, st, f: trt_clamp(x, s, sz, st),
    "fill": lambda x, s, sz, st, f: trt_fill(x, s, sz, st, f),
    "wrap": lambda x, s, sz, st, f: trt_wrap(x, s, sz, st),
    "reflect": lambda x, s, sz, st, f: trt_reflect(x, s, sz, st),
}


def fuzz_one(rng, mode):
    """Generate a random test case and verify gather_eval == oracle."""
    d = rng.integers(3, 20)
    x = np.arange(d, dtype=np.int64)
    stride = int(rng.integers(1, 4))
    if rng.random() < 0.3:
        stride = -stride

    if mode == "strict_bounds":
        max_size = max(1, (d - 1) // abs(stride) + 1)
        size = int(rng.integers(1, max_size + 1))
        if stride > 0:
            start = int(rng.integers(0, d - (size - 1) * stride))
        else:
            start = int(rng.integers((size - 1) * (-stride), d))
    else:
        size = int(rng.integers(1, d * 2))
        start = int(rng.integers(-d, 2 * d))

    fill = int(rng.integers(-10, 10)) if mode == "fill" else 0

    oracle_result = ORACLE_FN[mode](x, start, size, stride, fill)
    gather_result = gather_eval(x, start, size, stride, mode, fill)

    if not np.array_equal(oracle_result, gather_result):
        return False, (d, start, size, stride, fill, oracle_result, gather_result)
    return True, None


def main():
    rng = np.random.default_rng(42)
    n_trials = 1000
    all_pass = True

    print("Dynamic gather formula verification (1000 trials per mode):")
    for mode in ["strict_bounds", "clamp", "fill", "wrap", "reflect"]:
        passed = 0
        for _ in range(n_trials):
            ok, info = fuzz_one(rng, mode)
            if ok:
                passed += 1
            else:
                d, start, size, stride, fill, expected, actual = info
                print(
                    f"  FAIL {mode}: d={d} start={start} size={size} "
                    f"stride={stride} fill={fill}"
                )
                print(f"    oracle:  {list(expected)}")
                print(f"    gather:  {list(actual)}")
                all_pass = False
        mark = "ok" if passed == n_trials else "FAIL"
        print(f"  {mode:15s}: {passed}/{n_trials} {mark}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

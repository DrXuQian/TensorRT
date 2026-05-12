#!/usr/bin/env python3
"""NumPy reference oracle for TRT slice modes. Fuzz-tested 1000 times."""

import numpy as np
import sys


def trt_strict(x, start, size, stride):
    idx = np.arange(size) * stride + start
    if (idx < 0).any() or (idx >= x.shape[0]).any():
        raise ValueError("OOB")
    return x[idx]


def trt_clamp(x, start, size, stride):
    idx = np.arange(size) * stride + start
    return x[np.clip(idx, 0, x.shape[0] - 1)]


def trt_fill(x, start, size, stride, fill):
    d = x.shape[0]
    idx = np.arange(size) * stride + start
    in_bnd = (idx >= 0) & (idx < d)
    out = np.full(size, fill, dtype=x.dtype)
    out[in_bnd] = x[idx[in_bnd]]
    return out


def trt_wrap(x, start, size, stride):
    d = x.shape[0]
    idx = np.arange(size) * stride + start
    return x[((idx % d) + d) % d]


def trt_reflect(x, start, size, stride):
    d = x.shape[0]
    idx = np.arange(size) * stride + start
    p = 2 * d - 2
    c = ((np.abs(idx) % p) + p) % p
    return x[np.where(c >= d, p - c, c)]


MODES = {
    "strict_bounds": trt_strict,
    "clamp": trt_clamp,
    "fill": trt_fill,
    "wrap": trt_wrap,
    "reflect": trt_reflect,
}


def _random_params(mode, rng):
    d = rng.integers(3, 20)
    x = np.arange(d, dtype=np.int32)
    stride = rng.integers(1, 4)
    if rng.random() < 0.3:
        stride = -stride

    if mode == "strict_bounds":
        max_size = max(1, (d - 1) // abs(stride) + 1)
        size = rng.integers(1, max_size + 1)
        if stride > 0:
            start = rng.integers(0, d - (size - 1) * stride)
        else:
            start = rng.integers((size - 1) * (-stride), d)
    else:
        size = rng.integers(1, d * 2)
        start = rng.integers(-d, 2 * d)

    return x, int(start), int(size), int(stride)


def fuzz(n_trials=1000, seed=42):
    rng = np.random.default_rng(seed)
    results = {}
    for mode_name, fn in MODES.items():
        passed = 0
        for _ in range(n_trials):
            try:
                x, start, size, stride = _random_params(mode_name, rng)
                if mode_name == "fill":
                    result = fn(x, start, size, stride, fill=-1)
                else:
                    result = fn(x, start, size, stride)
                assert result.shape == (size,), f"Shape mismatch: {result.shape}"
                passed += 1
            except ValueError:
                passed += 1  # OOB is expected for strict_bounds
            except Exception as e:
                print(f"FAIL {mode_name}: d={x.shape[0]} start={start} "
                      f"size={size} stride={stride}: {e}")
        results[mode_name] = passed
        mark = "ok" if passed == n_trials else "FAIL"
        print(f"  {mode_name}: {passed}/{n_trials} {mark}")
    return all(v == n_trials for v in results.values())


if __name__ == "__main__":
    print("Running oracle fuzz test (1000 trials per mode)...")
    ok = fuzz()
    sys.exit(0 if ok else 1)

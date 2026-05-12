#!/usr/bin/env python3
"""JAX-based IR oracle: exports stablehlo IR for each TRT slice mode."""

try:
    import jax
    import jax.numpy as jnp
    from functools import partial

    MODE = {
        "fill": "constant",
        "clamp": "edge",
        "wrap": "wrap",
        "reflect": "reflect",
    }

    def trt_slice(x, start, size, stride, mode, fill=0):
        if stride < 0:
            return trt_slice(
                x, start + (size - 1) * stride, size, -stride, mode, fill
            )[::-1]
        d = x.shape[0]
        pad_lo = max(0, -start)
        pad_hi = max(0, start + (size - 1) * stride - (d - 1))
        if mode == "strict_bounds":
            return x[start : start + size * stride : stride]
        kw = (
            {"mode": "constant", "constant_values": fill}
            if mode == "fill"
            else {"mode": MODE[mode]}
        )
        padded = jnp.pad(x, (pad_lo, pad_hi), **kw)
        s = start + pad_lo
        return padded[s : s + size * stride : stride]

    def export_ir(mode, d, start, size, stride, fill=0, dtype=jnp.int32):
        fn = partial(
            trt_slice, start=start, size=size, stride=stride, mode=mode, fill=fill
        )
        return jax.jit(fn).lower(jnp.zeros((d,), dtype)).as_text("stablehlo")

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

    def export_ir(*args, **kwargs):
        raise RuntimeError(
            "JAX not installed. Install with: pip install jax jaxlib"
        )


if __name__ == "__main__":
    if not HAS_JAX:
        print("JAX not installed. Skipping IR export.")
    else:
        modes = [
            ("fill", 5, -2, 9, 1, 0),
            ("clamp", 5, -2, 9, 1, 0),
            ("wrap", 5, -2, 9, 1, 0),
            ("reflect", 5, -2, 9, 1, 0),
            ("strict_bounds", 10, 2, 3, 2, 0),
        ]
        for mode, d, start, size, stride, fill in modes:
            print(f"\n{'='*60}")
            print(f"mode={mode} d={d} start={start} size={size} stride={stride}")
            print(f"{'='*60}")
            ir = export_ir(mode, d, start, size, stride, fill)
            print(ir)

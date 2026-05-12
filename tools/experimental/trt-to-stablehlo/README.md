# trt-to-stablehlo

Minimal standalone MLIR project that lowers TensorRT's `trt.slice` op (with all 5 `SampleMode` variants) to [StableHLO](https://github.com/openxla/stablehlo) ops.

## Quick Start

```bash
# Build LLVM/MLIR (one-time)
git clone --depth 1 https://github.com/llvm/llvm-project /path/to/llvm-project
cmake -S /path/to/llvm-project/llvm -B /path/to/llvm-build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
ninja -C /path/to/llvm-build mlir-opt mlir-tblgen FileCheck count not

# Clone stablehlo
git clone --depth 1 https://github.com/openxla/stablehlo third_party/stablehlo

# Build this project
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm
ninja -C build check-trt-to-stablehlo
```

## The `trt.slice` Op

```mlir
%result = trt.slice %input, %fill_value {
    start  = array<i64: -2>,
    size   = array<i64: 9>,
    stride = array<i64: 1>,
    mode   = #trt<sample_mode kFILL>
} : (tensor<5xi32>, tensor<i32>) -> tensor<9xi32>
```

Parameters mirror TensorRT's `ISliceLayer`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | `DenseI64ArrayAttr` | Start index per axis (can be negative) |
| `size` | `DenseI64ArrayAttr` | Number of elements per axis |
| `stride` | `DenseI64ArrayAttr` | Step per axis (can be negative) |
| `mode` | `SampleModeAttr` | How to handle out-of-bounds access |
| `fill_value` | `Optional<tensor<scalar>>` | Fill value (kFILL mode only) |
| `axes` | `Optional<DenseI64ArrayAttr>` | Subset of axes (optional) |

---

## Lowering: 5 SampleModes → StableHLO

All examples use `tensor<5xi32>` input `[0, 1, 2, 3, 4]` with `start=-2, size=9, stride=1`.

### 1. `kSTRICT_BOUNDS` — Direct slice (no OOB allowed)

The simplest case. Maps 1:1 to `stablehlo.slice`.

**Before:**
```mlir
func.func @kSTRICT_BOUNDS(%in: tensor<10xi32>) -> tensor<3xi32> {
  %0 = trt.slice %in {
    start = array<i64: 2>, size = array<i64: 3>, stride = array<i64: 2>,
    mode = #trt<sample_mode kSTRICT_BOUNDS>
  } : (tensor<10xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}
```

**After:**
```mlir
func.func @kSTRICT_BOUNDS(%arg0: tensor<10xi32>) -> tensor<3xi32> {
  %0 = stablehlo.slice %arg0 [2:7:2] : (tensor<10xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}
```

**Dataflow:**
```
input[10] ──→ stablehlo.slice [2:7:2] ──→ result[3]
```

---

### 2. `kFILL` — Pad with constant fill value

Out-of-bounds positions are filled with a scalar constant using `stablehlo.pad`.

**Before:**
```mlir
func.func @kFILL(%in: tensor<5xi32>, %f: tensor<i32>) -> tensor<9xi32> {
  %0 = trt.slice %in, %f {
    start = array<i64: -2>, size = array<i64: 9>, stride = array<i64: 1>,
    mode = #trt<sample_mode kFILL>
  } : (tensor<5xi32>, tensor<i32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}
```

**After:**
```mlir
func.func @kFILL(%arg0: tensor<5xi32>, %arg1: tensor<i32>) -> tensor<9xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [2], high = [2], interior = [0]
       : (tensor<5xi32>, tensor<i32>) -> tensor<9xi32>
  %1 = stablehlo.slice %0 [0:9] : (tensor<9xi32>) -> tensor<9xi32>
  return %1 : tensor<9xi32>
}
```

**Dataflow:**
```
input[5] ──→ stablehlo.pad (lo=2, hi=2, fill=0) ──→ padded[9] ──→ stablehlo.slice [0:9] ──→ result[9]

Index:        -2  -1 │ 0   1   2   3   4 │  5   6
Value:         0   0 │ 0   1   2   3   4 │  0   0
              fill    │     original       │  fill
```

---

### 3. `kCLAMP` — Edge replication

Out-of-bounds positions clamp to the nearest edge value.

**Before:**
```mlir
func.func @kCLAMP(%in: tensor<5xi32>) -> tensor<9xi32> {
  %0 = trt.slice %in {
    start = array<i64: -2>, size = array<i64: 9>, stride = array<i64: 1>,
    mode = #trt<sample_mode kCLAMP>
  } : (tensor<5xi32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}
```

**After:**
```mlir
func.func @kCLAMP(%arg0: tensor<5xi32>) -> tensor<9xi32> {
  %0 = stablehlo.slice %arg0 [0:1]                                      // low edge  → [0]
  %1 = stablehlo.broadcast_in_dim %0, dims = [0] : ... -> tensor<2xi32> // replicate → [0, 0]
  %2 = stablehlo.slice %arg0 [4:5]                                      // high edge → [4]
  %3 = stablehlo.broadcast_in_dim %2, dims = [0] : ... -> tensor<2xi32> // replicate → [4, 4]
  %4 = stablehlo.concatenate %1, %arg0, %3, dim = 0                     // [0,0,0,1,2,3,4,4,4]
  %5 = stablehlo.slice %4 [0:9]
  return %5 : tensor<9xi32>
}
```

**Dataflow:**
```
                  ┌──── slice [0:1] ──→ broadcast ──→ [0, 0] ───┐
input[5] ────────┤                                                ├──→ concatenate ──→ padded[9] ──→ slice ──→ result[9]
                  └──── slice [4:5] ──→ broadcast ──→ [4, 4] ───┘

Index:    -2  -1 │ 0   1   2   3   4 │  5   6
Value:     0   0 │ 0   1   2   3   4 │  4   4
          clamp  │     original       │  clamp
```

---

### 4. `kWRAP` — Wrap around (circular)

Out-of-bounds positions wrap around to the other side of the tensor.

**Before:**
```mlir
func.func @kWRAP(%in: tensor<5xi32>) -> tensor<9xi32> {
  %0 = trt.slice %in {
    start = array<i64: -2>, size = array<i64: 9>, stride = array<i64: 1>,
    mode = #trt<sample_mode kWRAP>
  } : (tensor<5xi32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}
```

**After:**
```mlir
func.func @kWRAP(%arg0: tensor<5xi32>) -> tensor<9xi32> {
  %0 = stablehlo.slice %arg0 [3:5]       // tail of input → [3, 4]
  %1 = stablehlo.slice %arg0 [0:2]       // head of input → [0, 1]
  %2 = stablehlo.concatenate %0, %arg0, %1, dim = 0   // [3,4,0,1,2,3,4,0,1]
  %3 = stablehlo.slice %2 [0:9]
  return %3 : tensor<9xi32>
}
```

**Dataflow:**
```
             ┌──── slice [3:5] (tail) ──→ [3, 4] ───┐
input[5] ───┤                                         ├──→ concatenate ──→ padded[9] ──→ slice ──→ result[9]
             └──── slice [0:2] (head) ──→ [0, 1] ───┘

Index:    -2  -1 │ 0   1   2   3   4 │  5   6
Value:     3   4 │ 0   1   2   3   4 │  0   1
          wrap   │     original       │  wrap
```

---

### 5. `kREFLECT` — Mirror reflection

Out-of-bounds positions mirror-reflect across the tensor boundary.

**Before:**
```mlir
func.func @kREFLECT(%in: tensor<5xi32>) -> tensor<9xi32> {
  %0 = trt.slice %in {
    start = array<i64: -2>, size = array<i64: 9>, stride = array<i64: 1>,
    mode = #trt<sample_mode kREFLECT>
  } : (tensor<5xi32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}
```

**After:**
```mlir
func.func @kREFLECT(%arg0: tensor<5xi32>) -> tensor<9xi32> {
  %0 = stablehlo.slice %arg0 [1:3]       // [1, 2]
  %1 = stablehlo.reverse %0, dims = [0]  // [2, 1]
  %2 = stablehlo.slice %arg0 [2:4]       // [2, 3]
  %3 = stablehlo.reverse %2, dims = [0]  // [3, 2]
  %4 = stablehlo.concatenate %1, %arg0, %3, dim = 0   // [2,1,0,1,2,3,4,3,2]
  %5 = stablehlo.slice %4 [0:9]
  return %5 : tensor<9xi32>
}
```

**Dataflow:**
```
             ┌──── slice [1:3] ──→ reverse ──→ [2, 1] ───┐
input[5] ───┤                                              ├──→ concatenate ──→ padded[9] ──→ slice ──→ result[9]
             └──── slice [2:4] ──→ reverse ──→ [3, 2] ───┘

Index:    -2  -1 │ 0   1   2   3   4 │  5   6
Value:     2   1 │ 0   1   2   3   4 │  3   2
         reflect │     original       │ reflect
```

---

## Edge Cases: kWRAP and kREFLECT

### kWRAP with stride > 1

`input=[0,1,2,3,4]`, `start=-3, size=6, stride=2` → `[2, 4, 1, 3, 0, 2]`

Every other element from a wrap-around view starting at index -3:

```
Wrapped view:  [2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]
                ↑     ↑     ↑     ↑     ↑     ↑     stride=2
Result:        [2,    4,    1,    3,    0,    2]
```

**Lowered IR:**
```mlir
%0 = stablehlo.slice %arg0 [2:5]          // tail [2,3,4]
%1 = stablehlo.slice %arg0 [0:3]          // head [0,1,2]
%2 = stablehlo.concatenate %0, %arg0, %1  // [2,3,4,0,1,2,3,4,0,1,2]
%3 = stablehlo.slice %2 [0:11:2]          // stride-2 pick → [2,4,1,3,0,2]
```

### kWRAP with pad > d (divmod tiling)

`input=[0,1,2]`, `start=-5, size=11, stride=1` → `[1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]`

When padding (5) exceeds input size (3), we tile full copies via divmod:
`5 = 1*3 + 2` → 1 full copy + 2-element remainder.

**Lowered IR:**
```mlir
%0 = stablehlo.slice %arg0 [1:3]                        // remainder [1,2]
%1 = stablehlo.concatenate %0, %arg0, %arg0, %arg0      // [1,2] ++ [0,1,2] × 3
// = [1,2,0,1,2,0,1,2,0,1,2]  (size=11)
%2 = stablehlo.slice %1 [0:11]
```

### kWRAP with negative stride

`input=[0,1,2,3,4]`, `start=7, size=4, stride=-2` → `[2, 0, 3, 1]`

Accesses indices 7, 5, 3, 1 with wrap: `2, 0, 3, 1`.

**Lowered IR:**
```mlir
%0 = stablehlo.slice %arg0 [0:3]          // head for hi-pad
%1 = stablehlo.concatenate %arg0, %0      // [0,1,2,3,4,0,1,2]
%2 = stablehlo.slice %1 [1:8:2]           // [1,3,0,2] (positive stride)
%3 = stablehlo.reverse %2, dims = [0]     // [2,0,3,1]
```

---

### kREFLECT with stride > 1

`input=[0,1,2,3,4]`, `start=-3, size=5, stride=2` → `[3, 1, 1, 3, 3]`

Every other element from a reflected view starting at index -3:

```
Reflected view: [3, 2, 1, 0, 1, 2, 3, 4, 3]
                 ↑     ↑     ↑     ↑     ↑    stride=2
Result:         [3,    1,    1,    3,    3]
```

**Lowered IR:**
```mlir
%0 = stablehlo.slice %arg0 [1:4]          // [1,2,3]
%1 = stablehlo.reverse %0, dims = [0]     // [3,2,1]
%2 = stablehlo.slice %arg0 [3:4]          // [3]
%3 = stablehlo.reverse %2, dims = [0]     // [3]
%4 = stablehlo.concatenate %1, %arg0, %3  // [3,2,1,0,1,2,3,4,3]
%5 = stablehlo.slice %4 [0:9:2]           // [3,1,1,3,3]
```

### kREFLECT with pad > d-1 (multi-round)

`input=[0,1,2]`, `start=-5, size=11, stride=1` → `[1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1]`

Each reflect round can add at most `d-1=2` elements. Multiple rounds needed:

```
Round 1: [1,0] ++ [0,1,2] ++ [1,0]  →  [1,0,0,1,2,1,0]  (d=7)
Round 2: [0,1,2,1,0,0] ++ [1,0,0,1,2,1,0] ++ [1]  →  [0,1,2,1,0,0,1,0,0,1,2,1,0,1] (not quite)
...
```

The converter incrementally grows the tensor, each round reflecting `min(remaining, curD-1)` elements, until padding is satisfied.

**Lowered IR (2 rounds):**
```mlir
// Round 1: reflect ±2 on d=3
%0 = stablehlo.slice %arg0 [1:3]          // [1,2]
%1 = stablehlo.reverse %0                 // [2,1]
%2 = stablehlo.slice %arg0 [0:2]          // [0,1]
%3 = stablehlo.reverse %2                 // [1,0]
%4 = stablehlo.concatenate %1, %arg0, %3  // [2,1,0,1,2,1,0]  d=7

// Round 2: reflect lo=3, hi=1 on d=7
%5 = stablehlo.slice %4 [1:4]             // [1,0,1]
%6 = stablehlo.reverse %5                 // [1,0,1]
%7 = stablehlo.slice %4 [5:6]             // [1]
%8 = stablehlo.reverse %7                 // [1]
%9 = stablehlo.concatenate %6, %4, %8     // [1,0,1,2,1,0,1,2,1,0,1]  d=11
%10 = stablehlo.slice %9 [0:11]
```

### kREFLECT with negative stride

`input=[0,1,2,3,4]`, `start=6, size=3, stride=-2` → `[2, 4, 2]`

Accesses indices 6, 4, 2 with reflect: `2, 4, 2`.

**Lowered IR:**
```mlir
%0 = stablehlo.slice %arg0 [2:4]          // [2,3]
%1 = stablehlo.reverse %0                 // [3,2]
%2 = stablehlo.concatenate %arg0, %1      // [0,1,2,3,4,3,2]
%3 = stablehlo.slice %2 [2:7:2]           // [2,4,2] (positive stride)
%4 = stablehlo.reverse %3, dims = [0]     // [2,4,2]
```

---



Negative strides are normalized before mode processing:

1. Convert: `new_start = start + (size-1)*stride`, `new_stride = -stride`
2. Apply the mode-specific padding with positive stride
3. Append `stablehlo.reverse` at the end

```mlir
// Before: start=6, size=3, stride=-2 → accesses [6, 4, 2]
// After normalization: start=2, size=3, stride=2 → accesses [2, 4, 6]
// Then reverse the result: [6, 4, 2]

%0 = stablehlo.slice %arg0 [2:7:2] : (tensor<10xi32>) -> tensor<3xi32>
%1 = stablehlo.reverse %0, dims = [0] : tensor<3xi32>
```

---

## Standalone Utility: `SliceModeEmitter.h`

For integration into existing codebases that already have an `mlir::OpBuilder`, use `SliceModeEmitter.h` (header-only, no extra deps beyond stablehlo):

```cpp
#include "TRT/SliceModeEmitter.h"

// In your builder code:
mlir::Value result = SliceModeEmitter::emit(
    op_builder, loc, input,
    start, end, stride,
    SliceModeEmitter::Mode::kCLAMP);
```

---

## Verification

```bash
# Lit tests (FileCheck)
ninja -C build check-trt-to-stablehlo

# Oracle fuzz test (1000 random cases per mode)
python3 python/oracle.py

# End-to-end numerical verification (20 cases per mode)
# Runs: trt-opt → stablehlo-translate --interpret → compare with numpy
python3 python/e2e_verify.py
```

## Visualizing the IR

Use `trt-opt` to see the before/after transformation:

```bash
# Dump lowered StableHLO IR
./build/bin/trt-opt input.mlir --convert-trt-to-stablehlo

# Interpret numerically (requires stablehlo-translate)
./build/bin/trt-opt input.mlir --convert-trt-to-stablehlo | \
  ./build/bin/stablehlo-translate --interpret
```

For graphical visualization, pipe the output MLIR to any MLIR visualizer:
- [mlir-viz](https://github.com/nicholasgasior/mlir-viz)
- [MLIR Explorer](https://mlir.app/) (paste IR into the web UI)
- `--mlir-print-ir-after-all` flag on `trt-opt` to see each pass step

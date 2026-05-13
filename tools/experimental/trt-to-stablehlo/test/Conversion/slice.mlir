// RUN: trt-opt %s --convert-trt-to-stablehlo --split-input-file | FileCheck %s

// ============================================================
// kSTRICT_BOUNDS
// ============================================================

// CHECK-LABEL: @strict_basic
func.func @strict_basic(%in: tensor<10xi32>) -> tensor<3xi32> {
  // CHECK: stablehlo.slice %arg0 [2:7:2]
  %0 = trt.slice %in {
    start = array<i64: 2>, size = array<i64: 3>, stride = array<i64: 2>,
    mode = #trt<sample_mode kSTRICT_BOUNDS>
  } : (tensor<10xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

// ============================================================
// kFILL
// ============================================================

// CHECK-LABEL: @fill_basic
func.func @fill_basic(%in: tensor<5xi32>, %f: tensor<i32>) -> tensor<9xi32> {
  // CHECK: stablehlo.pad %arg0, %arg1, low = [2], high = [2], interior = [0]
  // CHECK: stablehlo.slice %{{.*}} [0:9]
  %0 = trt.slice %in, %f {
    start = array<i64: -2>, size = array<i64: 9>, stride = array<i64: 1>,
    mode = #trt<sample_mode kFILL>
  } : (tensor<5xi32>, tensor<i32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// CHECK-LABEL: @fill_stride2
func.func @fill_stride2(%in: tensor<5xi32>, %f: tensor<i32>) -> tensor<4xi32> {
  // CHECK: stablehlo.pad %arg0, %arg1, low = [1], high = [1], interior = [0]
  // CHECK: stablehlo.slice %{{.*}} [0:7:2]
  %0 = trt.slice %in, %f {
    start = array<i64: -1>, size = array<i64: 4>, stride = array<i64: 2>,
    mode = #trt<sample_mode kFILL>
  } : (tensor<5xi32>, tensor<i32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// ============================================================
// kCLAMP
// ============================================================

// CHECK-LABEL: @clamp_basic
func.func @clamp_basic(%in: tensor<5xi32>) -> tensor<9xi32> {
  // CHECK: %[[E0:.*]] = stablehlo.slice %arg0 [0:1]
  // CHECK: %[[B0:.*]] = stablehlo.broadcast_in_dim %[[E0]], dims = [0]
  // CHECK: %[[E1:.*]] = stablehlo.slice %arg0 [4:5]
  // CHECK: %[[B1:.*]] = stablehlo.broadcast_in_dim %[[E1]], dims = [0]
  // CHECK: stablehlo.concatenate %[[B0]], %arg0, %[[B1]], dim = 0
  // CHECK: stablehlo.slice
  %0 = trt.slice %in {
    start = array<i64: -2>, size = array<i64: 9>, stride = array<i64: 1>,
    mode = #trt<sample_mode kCLAMP>
  } : (tensor<5xi32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// ============================================================
// kWRAP
// ============================================================

// CHECK-LABEL: @wrap_basic
func.func @wrap_basic(%in: tensor<5xi32>) -> tensor<9xi32> {
  // CHECK: %[[T:.*]] = stablehlo.slice %arg0 [3:5]
  // CHECK: %[[H:.*]] = stablehlo.slice %arg0 [0:2]
  // CHECK: stablehlo.concatenate %[[T]], %arg0, %[[H]], dim = 0
  // CHECK: stablehlo.slice
  %0 = trt.slice %in {
    start = array<i64: -2>, size = array<i64: 9>, stride = array<i64: 1>,
    mode = #trt<sample_mode kWRAP>
  } : (tensor<5xi32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// ============================================================
// kREFLECT
// ============================================================

// CHECK-LABEL: @reflect_basic
func.func @reflect_basic(%in: tensor<5xi32>) -> tensor<9xi32> {
  // CHECK-DAG: stablehlo.slice %arg0
  // CHECK-DAG: stablehlo.reverse
  // CHECK: stablehlo.concatenate
  // CHECK: stablehlo.slice
  %0 = trt.slice %in {
    start = array<i64: -2>, size = array<i64: 9>, stride = array<i64: 1>,
    mode = #trt<sample_mode kREFLECT>
  } : (tensor<5xi32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// ============================================================
// Negative stride
// ============================================================

// CHECK-LABEL: @neg_stride_strict
func.func @neg_stride_strict(%in: tensor<10xi32>) -> tensor<3xi32> {
  // CHECK: %[[S:.*]] = stablehlo.slice %arg0 [2:7:2]
  // CHECK: stablehlo.reverse %[[S]], dims = [0]
  %0 = trt.slice %in {
    start = array<i64: 6>, size = array<i64: 3>, stride = array<i64: -2>,
    mode = #trt<sample_mode kSTRICT_BOUNDS>
  } : (tensor<10xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @neg_stride_fill
func.func @neg_stride_fill(%in: tensor<5xi32>, %f: tensor<i32>) -> tensor<4xi32> {
  // CHECK: stablehlo.pad
  // CHECK: stablehlo.slice
  // CHECK: stablehlo.reverse %{{.*}}, dims = [0]
  %0 = trt.slice %in, %f {
    start = array<i64: 5>, size = array<i64: 4>, stride = array<i64: -2>,
    mode = #trt<sample_mode kFILL>
  } : (tensor<5xi32>, tensor<i32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// ============================================================
// axes subset
// ============================================================

// CHECK-LABEL: @axes_subset
func.func @axes_subset(%in: tensor<4x6xi32>) -> tensor<4x3xi32> {
  // CHECK: stablehlo.slice %arg0 [0:4, 1:6:2]
  %0 = trt.slice %in {
    start = array<i64: 1>, size = array<i64: 3>, stride = array<i64: 2>,
    mode = #trt<sample_mode kSTRICT_BOUNDS>,
    axes = array<i64: 1>
  } : (tensor<4x6xi32>) -> tensor<4x3xi32>
  return %0 : tensor<4x3xi32>
}

// -----

// ============================================================
// Multi-dimensional
// ============================================================

// CHECK-LABEL: @fill_2d
func.func @fill_2d(%in: tensor<4x6xi32>, %f: tensor<i32>) -> tensor<6x8xi32> {
  // CHECK: stablehlo.pad %arg0, %arg1, low = [1, 1], high = [1, 1], interior = [0, 0]
  // CHECK: stablehlo.slice
  %0 = trt.slice %in, %f {
    start = array<i64: -1, -1>, size = array<i64: 6, 8>,
    stride = array<i64: 1, 1>,
    mode = #trt<sample_mode kFILL>
  } : (tensor<4x6xi32>, tensor<i32>) -> tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

// -----

// ============================================================
// kREFLECT: negative stride + pad > d-1
// ============================================================

// CHECK-LABEL: @reflect_neg_stride_big_pad
func.func @reflect_neg_stride_big_pad(%in: tensor<3xi32>) -> tensor<12xi32> {
  // CHECK-DAG: stablehlo.slice
  // CHECK-DAG: stablehlo.reverse
  // CHECK: stablehlo.concatenate
  // CHECK: stablehlo.slice
  // CHECK: stablehlo.reverse %{{.*}}, dims = [0]
  %0 = trt.slice %in {
    start = array<i64: 10>, size = array<i64: 12>, stride = array<i64: -1>,
    mode = #trt<sample_mode kREFLECT>
  } : (tensor<3xi32>) -> tensor<12xi32>
  return %0 : tensor<12xi32>
}

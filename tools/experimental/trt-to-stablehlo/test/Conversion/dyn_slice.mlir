// RUN: trt-opt %s --convert-trt-to-stablehlo --split-input-file | FileCheck %s

// ============================================================
// kSTRICT_BOUNDS (dynamic)
// ============================================================

// CHECK-LABEL: @dyn_strict_basic
func.func @dyn_strict_basic(%in: tensor<10xi32>,
    %start: tensor<1xi64>, %size: tensor<1xi64>, %stride: tensor<1xi64>)
    -> tensor<3xi32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = trt.dyn_slice %in, %start, %size, %stride {
    mode = #trt<sample_mode kSTRICT_BOUNDS>
  } : (tensor<10xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @dyn_strict_neg_stride
func.func @dyn_strict_neg_stride(%in: tensor<10xi32>,
    %start: tensor<1xi64>, %size: tensor<1xi64>, %stride: tensor<1xi64>)
    -> tensor<3xi32> {
  // CHECK: stablehlo.compare
  // CHECK: stablehlo.select
  // CHECK: stablehlo.real_dynamic_slice
  %0 = trt.dyn_slice %in, %start, %size, %stride {
    mode = #trt<sample_mode kSTRICT_BOUNDS>
  } : (tensor<10xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

// ============================================================
// kFILL (dynamic)
// ============================================================

// CHECK-LABEL: @dyn_fill_basic
func.func @dyn_fill_basic(%in: tensor<5xi32>,
    %start: tensor<1xi64>, %size: tensor<1xi64>, %stride: tensor<1xi64>,
    %fill: tensor<i32>) -> tensor<9xi32> {
  // CHECK: stablehlo.get_dimension_size
  // CHECK: stablehlo.dynamic_pad
  // CHECK: stablehlo.real_dynamic_slice
  %0 = trt.dyn_slice %in, %start, %size, %stride, %fill {
    mode = #trt<sample_mode kFILL>
  } : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// ============================================================
// E2E numerical: constants as dynamic values
// ============================================================

// CHECK-LABEL: @dyn_e2e_strict
func.func @dyn_e2e_strict() -> tensor<3xi32> {
  %in = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  %start = stablehlo.constant dense<[2]> : tensor<1xi64>
  %size = stablehlo.constant dense<[3]> : tensor<1xi64>
  %stride = stablehlo.constant dense<[2]> : tensor<1xi64>
  // CHECK: stablehlo.real_dynamic_slice
  %0 = trt.dyn_slice %in, %start, %size, %stride {
    mode = #trt<sample_mode kSTRICT_BOUNDS>
  } : (tensor<10xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @dyn_e2e_fill
func.func @dyn_e2e_fill() -> tensor<9xi32> {
  %in = stablehlo.constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %start = stablehlo.constant dense<[-2]> : tensor<1xi64>
  %size = stablehlo.constant dense<[9]> : tensor<1xi64>
  %stride = stablehlo.constant dense<[1]> : tensor<1xi64>
  %fill = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: stablehlo.dynamic_pad
  // CHECK: stablehlo.real_dynamic_slice
  %0 = trt.dyn_slice %in, %start, %size, %stride, %fill {
    mode = #trt<sample_mode kFILL>
  } : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// CHECK-LABEL: @dyn_e2e_strict_neg_stride
func.func @dyn_e2e_strict_neg_stride() -> tensor<3xi32> {
  %in = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  %start = stablehlo.constant dense<[6]> : tensor<1xi64>
  %size = stablehlo.constant dense<[3]> : tensor<1xi64>
  %stride = stablehlo.constant dense<[-2]> : tensor<1xi64>
  // CHECK: stablehlo.real_dynamic_slice
  // CHECK: stablehlo.select
  %0 = trt.dyn_slice %in, %start, %size, %stride {
    mode = #trt<sample_mode kSTRICT_BOUNDS>
  } : (tensor<10xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

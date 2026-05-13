// RUN: trt-opt %s --convert-trt-to-stablehlo --split-input-file | FileCheck %s

// ============================================================
// kSTRICT_BOUNDS — iota + gather
// ============================================================

// CHECK-LABEL: @dyn_strict
func.func @dyn_strict(%in: tensor<10xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>) -> tensor<3xi32> {
  // CHECK: stablehlo.iota
  // CHECK: stablehlo.gather
  %0 = trt.dyn_slice %in, %s, %sz, %st {mode = #trt<sample_mode kSTRICT_BOUNDS>}
    : (tensor<10xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

// ============================================================
// kCLAMP — iota + clamp + gather
// ============================================================

// CHECK-LABEL: @dyn_clamp
func.func @dyn_clamp(%in: tensor<5xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>) -> tensor<9xi32> {
  // CHECK: stablehlo.iota
  // CHECK: stablehlo.clamp
  // CHECK: stablehlo.gather
  %0 = trt.dyn_slice %in, %s, %sz, %st {mode = #trt<sample_mode kCLAMP>}
    : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// ============================================================
// kWRAP — iota + remainder + gather
// ============================================================

// CHECK-LABEL: @dyn_wrap
func.func @dyn_wrap(%in: tensor<5xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>) -> tensor<9xi32> {
  // CHECK: stablehlo.iota
  // CHECK: stablehlo.remainder
  // CHECK: stablehlo.gather
  %0 = trt.dyn_slice %in, %s, %sz, %st {mode = #trt<sample_mode kWRAP>}
    : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// ============================================================
// kREFLECT — iota + abs + remainder + select + gather
// ============================================================

// CHECK-LABEL: @dyn_reflect
func.func @dyn_reflect(%in: tensor<5xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>) -> tensor<9xi32> {
  // CHECK: stablehlo.iota
  // CHECK: stablehlo.abs
  // CHECK: stablehlo.remainder
  // CHECK: stablehlo.select
  // CHECK: stablehlo.gather
  %0 = trt.dyn_slice %in, %s, %sz, %st {mode = #trt<sample_mode kREFLECT>}
    : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// ============================================================
// kFILL — dynamic_pad + real_dynamic_slice (no gather)
// ============================================================

// CHECK-LABEL: @dyn_fill
func.func @dyn_fill(%in: tensor<5xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>,
    %f: tensor<i32>) -> tensor<9xi32> {
  // CHECK: stablehlo.dynamic_pad
  // CHECK: stablehlo.real_dynamic_slice
  %0 = trt.dyn_slice %in, %s, %sz, %st, %f {mode = #trt<sample_mode kFILL>}
    : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

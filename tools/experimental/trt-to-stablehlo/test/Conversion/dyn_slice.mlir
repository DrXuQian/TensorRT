// RUN: trt-opt %s --convert-trt-to-stablehlo --split-input-file | FileCheck %s

// CHECK-LABEL: @dyn_strict
func.func @dyn_strict(%in: tensor<10xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>) -> tensor<3xi32> {
  // CHECK: stablehlo.dynamic_iota
  // CHECK: stablehlo.dynamic_gather
  %0 = trt.dyn_slice %in, %s, %sz, %st {mode = #trt<sample_mode kSTRICT_BOUNDS>}
    : (tensor<10xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @dyn_clamp
func.func @dyn_clamp(%in: tensor<5xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>) -> tensor<9xi32> {
  // CHECK: stablehlo.dynamic_iota
  // CHECK: stablehlo.clamp
  // CHECK: stablehlo.dynamic_gather
  %0 = trt.dyn_slice %in, %s, %sz, %st {mode = #trt<sample_mode kCLAMP>}
    : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// CHECK-LABEL: @dyn_wrap
func.func @dyn_wrap(%in: tensor<5xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>) -> tensor<9xi32> {
  // CHECK: stablehlo.dynamic_iota
  // CHECK: stablehlo.remainder
  // CHECK: stablehlo.dynamic_gather
  %0 = trt.dyn_slice %in, %s, %sz, %st {mode = #trt<sample_mode kWRAP>}
    : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// CHECK-LABEL: @dyn_reflect
func.func @dyn_reflect(%in: tensor<5xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>) -> tensor<9xi32> {
  // CHECK: stablehlo.dynamic_iota
  // CHECK: stablehlo.abs
  // CHECK: stablehlo.select
  // CHECK: stablehlo.dynamic_gather
  %0 = trt.dyn_slice %in, %s, %sz, %st {mode = #trt<sample_mode kREFLECT>}
    : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

// -----

// CHECK-LABEL: @dyn_fill
func.func @dyn_fill(%in: tensor<5xi32>,
    %s: tensor<1xi64>, %sz: tensor<1xi64>, %st: tensor<1xi64>,
    %f: tensor<i32>) -> tensor<9xi32> {
  // CHECK: stablehlo.dynamic_iota
  // CHECK: stablehlo.clamp
  // CHECK: stablehlo.dynamic_gather
  // CHECK: stablehlo.select
  %0 = trt.dyn_slice %in, %s, %sz, %st, %f {mode = #trt<sample_mode kFILL>}
    : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
}

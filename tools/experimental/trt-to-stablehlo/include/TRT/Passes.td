#ifndef TRT_PASSES
#define TRT_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTRTToStablehlo : Pass<"convert-trt-to-stablehlo"> {
  let summary = "Convert TRT dialect ops to StableHLO";
  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect"
  ];
}

#endif // TRT_PASSES

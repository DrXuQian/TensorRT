#ifndef TRT_OPS
#define TRT_OPS

include "TRT/TRTDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SampleMode enum
//===----------------------------------------------------------------------===//

def TRT_SampleMode_STRICT_BOUNDS : I32EnumAttrCase<"kSTRICT_BOUNDS", 0>;
def TRT_SampleMode_WRAP          : I32EnumAttrCase<"kWRAP", 1>;
def TRT_SampleMode_CLAMP         : I32EnumAttrCase<"kCLAMP", 2>;
def TRT_SampleMode_FILL          : I32EnumAttrCase<"kFILL", 3>;
def TRT_SampleMode_REFLECT       : I32EnumAttrCase<"kREFLECT", 4>;

def TRT_SampleMode : I32EnumAttr<"SampleMode", "TRT slice sample mode", [
    TRT_SampleMode_STRICT_BOUNDS, TRT_SampleMode_WRAP, TRT_SampleMode_CLAMP,
    TRT_SampleMode_FILL,          TRT_SampleMode_REFLECT
]> {
  let cppNamespace = "::trt";
  let genSpecializedAttr = 0;
}

def TRT_SampleModeAttr
    : EnumAttr<TRT_Dialect, TRT_SampleMode, "sample_mode">;

//===----------------------------------------------------------------------===//
// trt.slice op
//===----------------------------------------------------------------------===//

def TRT_SliceOp : TRT_Op<"slice", [Pure]> {
  let summary = "TensorRT-style slice with 5 SampleMode variants";
  let arguments = (ins
    AnyRankedTensor:$input,
    DenseI64ArrayAttr:$start,
    DenseI64ArrayAttr:$size,
    DenseI64ArrayAttr:$stride,
    TRT_SampleModeAttr:$mode,
    Optional<AnyRankedTensor>:$fill_value,
    OptionalAttr<DenseI64ArrayAttr>:$axes
  );
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $input (`,` $fill_value^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

#endif // TRT_OPS

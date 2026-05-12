#include "TRT/TRTOps.h"
#include "TRT/TRTDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "TRT/TRTOps.cpp.inc"

//===----------------------------------------------------------------------===//
// SliceOp verifier
//===----------------------------------------------------------------------===//

LogicalResult trt::SliceOp::verify() {
  auto startArr = getStart();
  auto sizeArr = getSize();
  auto strideArr = getStride();

  if (startArr.size() != sizeArr.size() ||
      startArr.size() != strideArr.size())
    return emitOpError("start, size, stride must have the same length");

  auto inTy = cast<RankedTensorType>(getInput().getType());

  if (auto axes = getAxes()) {
    if (axes->size() != startArr.size())
      return emitOpError("axes length must match start/size/stride length");
    for (int64_t a : *axes) {
      if (a < 0 || a >= inTy.getRank())
        return emitOpError("axes value out of range");
    }
  } else {
    if (static_cast<int64_t>(startArr.size()) != inTy.getRank())
      return emitOpError(
          "start/size/stride length must match input rank when axes "
          "not specified");
  }

  for (size_t i = 0; i < sizeArr.size(); ++i) {
    if (sizeArr[i] <= 0)
      return emitOpError("size must be positive");
    if (strideArr[i] == 0)
      return emitOpError("stride must be non-zero");
  }

  if (getMode() == trt::SampleMode::kFILL && !getFillValue())
    return emitOpError("kFILL mode requires fill_value");

  return success();
}

#include "TRT/Passes.h"
#include "TRT/TRTDialect.h"
#include "TRT/TRTOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include <numeric>

using namespace mlir;
namespace sh = mlir::stablehlo;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

struct PadAmount {
  int64_t lo, hi;
};

static SmallVector<PadAmount>
computePads(ArrayRef<int64_t> start, ArrayRef<int64_t> size,
            ArrayRef<int64_t> stride, ArrayRef<int64_t> shape) {
  SmallVector<PadAmount> pads;
  for (size_t i = 0; i < start.size(); ++i) {
    int64_t last = start[i] + (size[i] - 1) * stride[i];
    int64_t lo = std::min(start[i], last);
    int64_t hi = std::max(start[i], last);
    pads.push_back(
        {std::max<int64_t>(0, -lo), std::max<int64_t>(0, hi - (shape[i] - 1))});
  }
  return pads;
}

static SmallVector<int64_t>
normalizeNegativeStrides(SmallVectorImpl<int64_t> &start,
                         ArrayRef<int64_t> size,
                         SmallVectorImpl<int64_t> &stride) {
  SmallVector<int64_t> reverseDims;
  for (size_t i = 0; i < stride.size(); ++i) {
    if (stride[i] < 0) {
      start[i] = start[i] + (size[i] - 1) * stride[i];
      stride[i] = -stride[i];
      reverseDims.push_back(i);
    }
  }
  return reverseDims;
}

static void expandAxes(ArrayRef<int64_t> axes, ArrayRef<int64_t> shape,
                       SmallVectorImpl<int64_t> &start,
                       SmallVectorImpl<int64_t> &size,
                       SmallVectorImpl<int64_t> &stride) {
  int rank = shape.size();
  SmallVector<int64_t> fullStart(rank, 0);
  SmallVector<int64_t> fullSize(shape.begin(), shape.end());
  SmallVector<int64_t> fullStride(rank, 1);
  for (size_t i = 0; i < axes.size(); ++i) {
    fullStart[axes[i]] = start[i];
    fullSize[axes[i]] = size[i];
    fullStride[axes[i]] = stride[i];
  }
  start = std::move(fullStart);
  size = std::move(fullSize);
  stride = std::move(fullStride);
}

/// Slice a single axis [start:end] with stride 1.
static Value sliceAxis(OpBuilder &b, Location loc, Value v, int axis,
                       int64_t s, int64_t e) {
  auto ty = cast<RankedTensorType>(v.getType());
  int rank = ty.getRank();
  SmallVector<int64_t> starts(rank, 0);
  SmallVector<int64_t> limits(ty.getShape().begin(), ty.getShape().end());
  SmallVector<int64_t> strides(rank, 1);
  starts[axis] = s;
  limits[axis] = e;

  SmallVector<int64_t> resultShape(ty.getShape().begin(), ty.getShape().end());
  resultShape[axis] = e - s;
  auto resultTy = RankedTensorType::get(resultShape, ty.getElementType());

  return b.create<sh::SliceOp>(loc, resultTy, v,
                               b.getDenseI64ArrayAttr(starts),
                               b.getDenseI64ArrayAttr(limits),
                               b.getDenseI64ArrayAttr(strides));
}

//===----------------------------------------------------------------------===//
// Per-mode padders
//===----------------------------------------------------------------------===//

/// kFILL: stablehlo.pad with the fill scalar.
static Value padFill(OpBuilder &b, Location loc, Value in, Value fill,
                     ArrayRef<PadAmount> pads) {
  auto inTy = cast<RankedTensorType>(in.getType());
  int rank = inTy.getRank();
  SmallVector<int64_t> lo(rank), hi(rank), interior(rank, 0);
  SmallVector<int64_t> resultShape(inTy.getShape().begin(),
                                   inTy.getShape().end());
  for (int i = 0; i < rank; ++i) {
    lo[i] = pads[i].lo;
    hi[i] = pads[i].hi;
    resultShape[i] += lo[i] + hi[i];
  }
  auto resultTy = RankedTensorType::get(resultShape, inTy.getElementType());
  return b.create<sh::PadOp>(loc, resultTy, in, fill,
                              b.getDenseI64ArrayAttr(lo),
                              b.getDenseI64ArrayAttr(hi),
                              b.getDenseI64ArrayAttr(interior));
}

/// kCLAMP: per-axis edge-replicate via slice + broadcast_in_dim + concatenate.
static Value padClamp(OpBuilder &b, Location loc, Value in,
                      ArrayRef<PadAmount> pads) {
  for (size_t axis = 0; axis < pads.size(); ++axis) {
    if (pads[axis].lo == 0 && pads[axis].hi == 0)
      continue;
    auto inTy = cast<RankedTensorType>(in.getType());
    SmallVector<int64_t> shape(inTy.getShape().begin(), inTy.getShape().end());
    int rank = inTy.getRank();

    SmallVector<int64_t> broadDims(rank);
    std::iota(broadDims.begin(), broadDims.end(), 0);
    SmallVector<Value> parts;

    if (pads[axis].lo > 0) {
      Value edge = sliceAxis(b, loc, in, axis, 0, 1);
      SmallVector<int64_t> broadShape(shape);
      broadShape[axis] = pads[axis].lo;
      auto broadTy = RankedTensorType::get(broadShape, inTy.getElementType());
      Value tile = b.create<sh::BroadcastInDimOp>(
          loc, broadTy, edge, b.getDenseI64ArrayAttr(broadDims));
      parts.push_back(tile);
    }

    parts.push_back(in);

    if (pads[axis].hi > 0) {
      Value edge = sliceAxis(b, loc, in, axis, shape[axis] - 1, shape[axis]);
      SmallVector<int64_t> broadShape(shape);
      broadShape[axis] = pads[axis].hi;
      auto broadTy = RankedTensorType::get(broadShape, inTy.getElementType());
      Value tile = b.create<sh::BroadcastInDimOp>(
          loc, broadTy, edge, b.getDenseI64ArrayAttr(broadDims));
      parts.push_back(tile);
    }

    int64_t newDim = 0;
    for (auto part : parts)
      newDim += cast<RankedTensorType>(part.getType()).getDimSize(axis);
    SmallVector<int64_t> concatShape(shape);
    concatShape[axis] = newDim;
    auto concatTy = RankedTensorType::get(concatShape, inTy.getElementType());
    in = b.create<sh::ConcatenateOp>(loc, concatTy, parts,
                                     b.getI64IntegerAttr(axis));
  }
  return in;
}

/// kWRAP: per-axis wrap-around via slice + concatenate.
static Value padWrap(OpBuilder &b, Location loc, Value in,
                     ArrayRef<PadAmount> pads) {
  for (size_t axis = 0; axis < pads.size(); ++axis) {
    if (pads[axis].lo == 0 && pads[axis].hi == 0)
      continue;
    auto inTy = cast<RankedTensorType>(in.getType());
    int64_t d = inTy.getDimSize(axis);
    int64_t needLo = pads[axis].lo;
    int64_t needHi = pads[axis].hi;

    // divmod: how many full copies of input we need on each side,
    // plus a remainder slice.
    //   needLo = fullLo * d + remLo
    //   needHi = fullHi * d + remHi
    int64_t fullLo = needLo / d, remLo = needLo % d;
    int64_t fullHi = needHi / d, remHi = needHi % d;

    SmallVector<Value> parts;

    // Left remainder (partial wrap)
    if (remLo > 0)
      parts.push_back(sliceAxis(b, loc, in, axis, d - remLo, d));
    // Left full copies
    for (int64_t i = 0; i < fullLo; ++i)
      parts.push_back(in);
    // Original
    parts.push_back(in);
    // Right full copies
    for (int64_t i = 0; i < fullHi; ++i)
      parts.push_back(in);
    // Right remainder (partial wrap)
    if (remHi > 0)
      parts.push_back(sliceAxis(b, loc, in, axis, 0, remHi));

    int64_t newDim = 0;
    for (auto part : parts)
      newDim += cast<RankedTensorType>(part.getType()).getDimSize(axis);
    SmallVector<int64_t> shape(inTy.getShape().begin(),
                               inTy.getShape().end());
    shape[axis] = newDim;
    auto concatTy = RankedTensorType::get(shape, inTy.getElementType());
    in = b.create<sh::ConcatenateOp>(loc, concatTy, parts,
                                     b.getI64IntegerAttr(axis));
  }
  return in;
}

/// kREFLECT: per-axis reflection via slice + reverse + concatenate.
/// Multi-round: each round can reflect at most (d-1) elements per side.
static Value padReflect(OpBuilder &b, Location loc, Value in,
                        ArrayRef<PadAmount> pads) {
  for (size_t axis = 0; axis < pads.size(); ++axis) {
    if (pads[axis].lo == 0 && pads[axis].hi == 0)
      continue;
    int64_t needLo = pads[axis].lo;
    int64_t needHi = pads[axis].hi;

    while (needLo > 0 || needHi > 0) {
      auto curTy = cast<RankedTensorType>(in.getType());
      int64_t curD = curTy.getDimSize(axis);
      SmallVector<int64_t> shape(curTy.getShape().begin(),
                                 curTy.getShape().end());
      // Each round reflects at most curD-1 elements.
      int64_t lo = std::min(needLo, curD - 1);
      int64_t hi = std::min(needHi, curD - 1);

      SmallVector<Value> parts;
      if (lo > 0) {
        Value loSrc = sliceAxis(b, loc, in, axis, 1, 1 + lo);
        Value loRev = b.create<sh::ReverseOp>(
            loc, loSrc.getType(), loSrc,
            b.getDenseI64ArrayAttr(SmallVector<int64_t>{(int64_t)axis}));
        parts.push_back(loRev);
      }
      parts.push_back(in);
      if (hi > 0) {
        Value hiSrc = sliceAxis(b, loc, in, axis, curD - 1 - hi, curD - 1);
        Value hiRev = b.create<sh::ReverseOp>(
            loc, hiSrc.getType(), hiSrc,
            b.getDenseI64ArrayAttr(SmallVector<int64_t>{(int64_t)axis}));
        parts.push_back(hiRev);
      }

      int64_t newDim = 0;
      for (auto part : parts)
        newDim += cast<RankedTensorType>(part.getType()).getDimSize(axis);
      SmallVector<int64_t> concatShape(shape);
      concatShape[axis] = newDim;
      auto concatTy =
          RankedTensorType::get(concatShape, curTy.getElementType());
      in = b.create<sh::ConcatenateOp>(loc, concatTy, parts,
                                       b.getI64IntegerAttr(axis));
      needLo -= lo;
      needHi -= hi;
    }
  }
  return in;
}

//===----------------------------------------------------------------------===//
// SliceConverter pattern
//===----------------------------------------------------------------------===//

struct SliceConverter : OpConversionPattern<trt::SliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(trt::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inTy = cast<RankedTensorType>(adaptor.getInput().getType());
    int rank = inTy.getRank();

    SmallVector<int64_t> start(op.getStart());
    SmallVector<int64_t> size(op.getSize());
    SmallVector<int64_t> stride(op.getStride());

    // Expand axes subset to full rank.
    if (auto axes = op.getAxes())
      expandAxes(*axes, inTy.getShape(), start, size, stride);

    // Normalize negative strides: flip start, negate stride, record dims.
    auto reverseDims = normalizeNegativeStrides(start, size, stride);

    // Compute how much out-of-bounds padding is needed per axis.
    auto pads = computePads(start, size, stride, inTy.getShape());

    Value padded;
    switch (op.getMode()) {
    case trt::SampleMode::kSTRICT_BOUNDS:
      padded = adaptor.getInput();
      for (size_t i = 0; i < pads.size(); ++i) {
        if (pads[i].lo || pads[i].hi)
          return op.emitOpError("kSTRICT_BOUNDS would go out-of-bounds");
      }
      break;
    case trt::SampleMode::kFILL:
      padded = padFill(rewriter, loc, adaptor.getInput(),
                       adaptor.getFillValue(), pads);
      break;
    case trt::SampleMode::kCLAMP:
      padded = padClamp(rewriter, loc, adaptor.getInput(), pads);
      break;
    case trt::SampleMode::kWRAP:
      padded = padWrap(rewriter, loc, adaptor.getInput(), pads);
      break;
    case trt::SampleMode::kREFLECT:
      padded = padReflect(rewriter, loc, adaptor.getInput(), pads);
      break;
    }

    // Final strided slice on the padded tensor.
    SmallVector<int64_t> sliceStart(rank), sliceLimit(rank), sliceStride(rank);
    for (int i = 0; i < rank; ++i) {
      sliceStart[i] = start[i] + pads[i].lo;
      sliceLimit[i] = sliceStart[i] + (size[i] - 1) * stride[i] + 1;
      sliceStride[i] = stride[i];
    }

    SmallVector<int64_t> resultShape(size.begin(), size.end());
    auto elemTy = inTy.getElementType();
    auto sliceResultTy = RankedTensorType::get(resultShape, elemTy);

    Value result = rewriter.create<sh::SliceOp>(
        loc, sliceResultTy, padded, rewriter.getDenseI64ArrayAttr(sliceStart),
        rewriter.getDenseI64ArrayAttr(sliceLimit),
        rewriter.getDenseI64ArrayAttr(sliceStride));

    // Reverse for originally-negative strides.
    if (!reverseDims.empty())
      result = rewriter.create<sh::ReverseOp>(
          loc, result.getType(), result,
          rewriter.getDenseI64ArrayAttr(reverseDims));

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void trt::populateTRTToStablehloPatterns(RewritePatternSet &patterns) {
  patterns.add<SliceConverter>(patterns.getContext());
}

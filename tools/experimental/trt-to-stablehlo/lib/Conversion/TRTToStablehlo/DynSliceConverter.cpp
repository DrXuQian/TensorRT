/// Dynamic trt.dyn_slice → StableHLO lowering via index computation + gather.
///
/// For kCLAMP/kWRAP/kREFLECT/kSTRICT_BOUNDS:
///   Per-axis: iota → raw_idx = start + iota*stride → mode-specific index
///             mapping → gather along axis.
///   No concat, no while, no pad. Negative strides handled naturally.
///
/// For kFILL: dynamic_pad + real_dynamic_slice (gather can't produce fill values).

#include "TRT/Passes.h"
#include "TRT/TRTDialect.h"
#include "TRT/TRTOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace sh = mlir::stablehlo;

namespace {

//===----------------------------------------------------------------------===//
// Tiny helpers
//===----------------------------------------------------------------------===//

/// Scalar i64 constant (0-d tensor).
static Value i64Scalar(OpBuilder &b, Location loc, int64_t v) {
  return b.create<sh::ConstantOp>(
      loc, DenseElementsAttr::get(
               RankedTensorType::get({}, b.getI64Type()),
               b.getI64IntegerAttr(v)));
}

/// Extract tensor<Nxi64>[idx] as tensor<i64> (scalar).
static Value extractIdx(OpBuilder &b, Location loc, Value vec, int idx) {
  auto one = RankedTensorType::get({1}, b.getI64Type());
  Value sl = b.create<sh::SliceOp>(
      loc, one, vec,
      b.getDenseI64ArrayAttr({(int64_t)idx}),
      b.getDenseI64ArrayAttr({(int64_t)(idx + 1)}),
      b.getDenseI64ArrayAttr({1}));
  return b.create<sh::ReshapeOp>(
      loc, RankedTensorType::get({}, b.getI64Type()), sl);
}

/// Broadcast scalar tensor<i64> → tensor<n x i64>.
static Value bcast(OpBuilder &b, Location loc, Value scalar, int64_t n) {
  return b.create<sh::BroadcastInDimOp>(
      loc, RankedTensorType::get({n}, b.getI64Type()), scalar,
      b.getDenseI64ArrayAttr({}));
}

/// Build tensor<n x i64> containing the input dim size at `axis`, broadcasted.
static Value getDimBcast(OpBuilder &b, Location loc, Value input,
                         int axis, int64_t n) {
  Value d32 = b.create<sh::GetDimensionSizeOp>(
      loc, RankedTensorType::get({}, b.getI32Type()), input,
      b.getI64IntegerAttr(axis));
  Value d64 = b.create<sh::ConvertOp>(
      loc, RankedTensorType::get({}, b.getI64Type()), d32);
  return bcast(b, loc, d64, n);
}

//===----------------------------------------------------------------------===//
// Mode-specific index mapping (all operate on tensor<n x i64>)
//===----------------------------------------------------------------------===//

/// Map raw indices to valid [0, d-1] range based on mode.
static Value mapIndex(OpBuilder &b, Location loc, Value raw, Value d,
                      trt::SampleMode mode, int64_t n) {
  auto ty = RankedTensorType::get({n}, b.getI64Type());
  Value zero = b.create<sh::ConstantOp>(loc, DenseElementsAttr::get(ty, b.getI64IntegerAttr(0)));
  Value one  = b.create<sh::ConstantOp>(loc, DenseElementsAttr::get(ty, b.getI64IntegerAttr(1)));
  Value two  = b.create<sh::ConstantOp>(loc, DenseElementsAttr::get(ty, b.getI64IntegerAttr(2)));
  Value dm1  = b.create<sh::SubtractOp>(loc, d, one);

  switch (mode) {
  case trt::SampleMode::kSTRICT_BOUNDS:
    return raw;

  case trt::SampleMode::kCLAMP:
    return b.create<sh::ClampOp>(loc, zero, raw, dm1);

  case trt::SampleMode::kWRAP: {
    // ((raw % d) + d) % d  — handles negatives
    Value r = b.create<sh::RemOp>(loc, raw, d);
    return b.create<sh::RemOp>(loc, b.create<sh::AddOp>(loc, r, d), d);
  }

  case trt::SampleMode::kREFLECT: {
    // p = 2*(d-1); c = ((|raw| % p) + p) % p; select(c>=d, p-c, c)
    Value p = b.create<sh::MulOp>(loc, dm1, two);
    Value absRaw = b.create<sh::AbsOp>(loc, raw);
    Value r = b.create<sh::RemOp>(loc, absRaw, p);
    Value c = b.create<sh::RemOp>(loc, b.create<sh::AddOp>(loc, r, p), p);
    Value cGeD = b.create<sh::CompareOp>(loc, c, d,
                     sh::ComparisonDirection::GE);
    return b.create<sh::SelectOp>(loc, cGeD,
               b.create<sh::SubtractOp>(loc, p, c), c);
  }

  case trt::SampleMode::kFILL:
    // gather with clamped indices; OOB mask applied in caller
    return b.create<sh::ClampOp>(loc, zero, raw, dm1);
  }
}

//===----------------------------------------------------------------------===//
// Gather along a single axis (index_select pattern)
//===----------------------------------------------------------------------===//

static Value gatherAxis(OpBuilder &b, Location loc, Value input,
                        Value indices, int axis) {
  auto inTy = cast<RankedTensorType>(input.getType());
  int rank = inTy.getRank();
  int64_t n = cast<RankedTensorType>(indices.getType()).getDimSize(0);

  // offset_dims = all result positions except `axis`
  SmallVector<int64_t> offsetDims;
  for (int i = 0; i < rank; ++i)
    if (i != axis) offsetDims.push_back(i);

  // slice_sizes = input shape with dim[axis] = 1
  SmallVector<int64_t> sliceSizes(inTy.getShape());
  sliceSizes[axis] = 1;

  // result shape = input shape with dim[axis] = n
  SmallVector<int64_t> resultShape(inTy.getShape());
  resultShape[axis] = n;
  auto resultTy = RankedTensorType::get(resultShape, inTy.getElementType());

  auto dimNumbers = sh::GatherDimensionNumbersAttr::get(
      b.getContext(),
      /*offsetDims=*/offsetDims,
      /*collapsedSliceDims=*/{axis},
      /*operandBatchingDims=*/{},
      /*startIndicesBatchingDims=*/{},
      /*startIndexMap=*/{axis},
      /*indexVectorDim=*/1);

  return b.create<sh::GatherOp>(loc, resultTy, input, indices,
                                dimNumbers,
                                b.getDenseI64ArrayAttr(sliceSizes));
}

//===----------------------------------------------------------------------===//
// DynSliceConverter
//===----------------------------------------------------------------------===//

struct DynSliceConverter : OpConversionPattern<trt::DynSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(trt::DynSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    auto inTy = cast<RankedTensorType>(adaptor.getInput().getType());
    int rank = inTy.getRank();

    Value input = adaptor.getInput();
    bool isFill = (op.getMode() == trt::SampleMode::kFILL);

    // --- All modes: iota + mapIndex + gather (per-axis) ---
    // For kFILL: also track per-axis OOB masks for final select.
    SmallVector<Value> oobMasks; // per-axis: tensor<outDim[axis] x i1>

    for (int axis = 0; axis < rank; ++axis) {
      int64_t outDim = resultTy.getDimSize(axis);

      Value start = bcast(rewriter, loc,
                          extractIdx(rewriter, loc, adaptor.getStart(), axis),
                          outDim);
      Value stride = bcast(rewriter, loc,
                           extractIdx(rewriter, loc, adaptor.getStride(), axis),
                           outDim);
      Value d = getDimBcast(rewriter, loc, adaptor.getInput(), axis, outDim);

      // raw_idx = start + iota * stride
      auto iotaTy = RankedTensorType::get({outDim}, rewriter.getI64Type());
      Value iota = rewriter.create<sh::IotaOp>(loc, iotaTy,
                                                rewriter.getI64IntegerAttr(0));
      Value raw = rewriter.create<sh::AddOp>(loc, start,
                      rewriter.create<sh::MulOp>(loc, iota, stride));

      // For kFILL: record which indices are OOB on this axis
      if (isFill) {
        auto i64Ty = rewriter.getI64Type();
        Value zero = rewriter.create<sh::ConstantOp>(loc,
            DenseElementsAttr::get(iotaTy, rewriter.getI64IntegerAttr(0)));
        Value geLo = rewriter.create<sh::CompareOp>(loc, raw, zero,
                         sh::ComparisonDirection::GE);
        Value ltHi = rewriter.create<sh::CompareOp>(loc, raw, d,
                         sh::ComparisonDirection::LT);
        oobMasks.push_back(rewriter.create<sh::AndOp>(loc, geLo, ltHi));
      }

      Value idx = mapIndex(rewriter, loc, raw, d, op.getMode(), outDim);
      input = gatherAxis(rewriter, loc, input, idx, axis);
    }

    // kFILL: replace OOB positions with fill value
    if (isFill) {
      // Combine per-axis masks: inBounds = mask0[i0] AND mask1[i1] AND ...
      // Each mask is 1-D; broadcast to full output shape, then AND together.
      auto boolTy = RankedTensorType::get(resultTy.getShape(),
                                          rewriter.getI1Type());
      Value combinedMask;
      for (int axis = 0; axis < rank; ++axis) {
        // Broadcast 1-D mask to full output shape along this axis
        SmallVector<int64_t> broadDims = {(int64_t)axis};
        Value mask = rewriter.create<sh::BroadcastInDimOp>(
            loc, boolTy, oobMasks[axis],
            rewriter.getDenseI64ArrayAttr(broadDims));
        combinedMask = combinedMask
            ? rewriter.create<sh::AndOp>(loc, combinedMask, mask).getResult()
            : mask;
      }

      // Broadcast fill scalar to output shape
      Value fill = rewriter.create<sh::BroadcastInDimOp>(
          loc, resultTy, adaptor.getFillValue(),
          rewriter.getDenseI64ArrayAttr({}));

      input = rewriter.create<sh::SelectOp>(loc, combinedMask, input, fill);
    }

    rewriter.replaceOp(op, input);
    return success();
  }

  // No private methods needed — all modes use the unified gather path.
};

} // namespace

void trt::populateTRTDynToStablehloPatterns(RewritePatternSet &patterns) {
  patterns.add<DynSliceConverter>(patterns.getContext());
}

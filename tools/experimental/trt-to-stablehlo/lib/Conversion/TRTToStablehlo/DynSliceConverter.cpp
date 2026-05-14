/// Dynamic trt.dyn_slice → StableHLO lowering via index computation + gather.
///
/// All 5 modes: per-axis iota → raw_idx = start + iota*stride → mode-specific
/// index mapping → gather. Negative strides handled naturally.
/// kFILL: gather with clamped indices + OOB mask + select with fill value.

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

/// Ensure a Value is a 0-d (scalar) tensor; reshape from 1-d if needed.
static Value ensureScalar(OpBuilder &b, Location loc, Value v) {
  auto ty = cast<RankedTensorType>(v.getType());
  if (ty.getRank() == 0) return v;
  auto scalarTy = RankedTensorType::get({}, ty.getElementType());
  auto reshaped = b.create<sh::ReshapeOp>(loc, scalarTy, v);
  assert(cast<RankedTensorType>(reshaped.getType()).getRank() == 0);
  return reshaped;
}

//===----------------------------------------------------------------------===//
// Mode-specific index mapping (all operate on tensor<n x i64>)
//===----------------------------------------------------------------------===//

/// Result of mapIndex: mapped indices + optional in-bounds mask (kFILL only).
struct MappedIndex {
  Value idx;
  Value inBoundsMask; // null for non-FILL modes
};

/// Map raw indices to valid [0, d-1] range based on mode.
/// For kFILL, also returns a per-element in-bounds mask.
static MappedIndex mapIndex(OpBuilder &b, Location loc, Value raw, Value d,
                            trt::SampleMode mode, int64_t n) {
  auto ty = RankedTensorType::get({n}, b.getI64Type());
  Value zero = b.create<sh::ConstantOp>(loc, DenseElementsAttr::get(ty, b.getI64IntegerAttr(0)));
  Value one  = b.create<sh::ConstantOp>(loc, DenseElementsAttr::get(ty, b.getI64IntegerAttr(1)));
  Value dm1  = b.create<sh::SubtractOp>(loc, d, one);

  switch (mode) {
  case trt::SampleMode::kSTRICT_BOUNDS:
    // NOTE: no OOB check — gather with OOB indices is UB.
    return {raw, {}};

  case trt::SampleMode::kCLAMP:
    return {b.create<sh::ClampOp>(loc, zero, raw, dm1), {}};

  case trt::SampleMode::kWRAP: {
    Value r = b.create<sh::RemOp>(loc, raw, d);
    return {b.create<sh::RemOp>(loc, b.create<sh::AddOp>(loc, r, d), d), {}};
  }

  case trt::SampleMode::kREFLECT: {
    Value two = b.create<sh::ConstantOp>(loc, DenseElementsAttr::get(ty, b.getI64IntegerAttr(2)));
    Value p = b.create<sh::MulOp>(loc, dm1, two);
    Value c = b.create<sh::RemOp>(loc, b.create<sh::AbsOp>(loc, raw), p);
    Value cGeD = b.create<sh::CompareOp>(loc, c, d, sh::ComparisonDirection::GE);
    return {b.create<sh::SelectOp>(loc, cGeD, b.create<sh::SubtractOp>(loc, p, c), c), {}};
  }

  case trt::SampleMode::kFILL: {
    Value geLo = b.create<sh::CompareOp>(loc, raw, zero, sh::ComparisonDirection::GE);
    Value ltHi = b.create<sh::CompareOp>(loc, raw, d, sh::ComparisonDirection::LT);
    Value mask = b.create<sh::AndOp>(loc, geLo, ltHi);
    return {b.create<sh::ClampOp>(loc, zero, raw, dm1), mask};
  }
  }
  llvm_unreachable("invalid SampleMode");
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

    // rank=0: scalar, nothing to slice
    if (rank == 0) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Require static output dims (dynamic_iota needed otherwise)
    for (int i = 0; i < rank; ++i) {
      if (resultTy.getDimSize(i) == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(op,
            "dynamic output shape not yet supported");
    }

    Value input = adaptor.getInput();
    SmallVector<Value> masks; // per-axis in-bounds masks (kFILL only)

    // --- Uniform loop: iota + mapIndex + gather, all modes ---
    for (int axis = 0; axis < rank; ++axis) {
      int64_t outDim = resultTy.getDimSize(axis);

      Value start = bcast(rewriter, loc,
                          extractIdx(rewriter, loc, adaptor.getStart(), axis),
                          outDim);
      Value stride = bcast(rewriter, loc,
                           extractIdx(rewriter, loc, adaptor.getStride(), axis),
                           outDim);
      Value d = getDimBcast(rewriter, loc, adaptor.getInput(), axis, outDim);

      auto iotaTy = RankedTensorType::get({outDim}, rewriter.getI64Type());
      Value iota = rewriter.create<sh::IotaOp>(loc, iotaTy,
                                                rewriter.getI64IntegerAttr(0));
      Value raw = rewriter.create<sh::AddOp>(loc, start,
                      rewriter.create<sh::MulOp>(loc, iota, stride));

      auto [idx, mask] = mapIndex(rewriter, loc, raw, d, op.getMode(), outDim);
      if (mask) masks.push_back(mask);
      input = gatherAxis(rewriter, loc, input, idx, axis);
    }

    // Apply OOB mask if any mode produced one (currently kFILL only)
    if (!masks.empty()) {
      auto boolTy = RankedTensorType::get(resultTy.getShape(),
                                          rewriter.getI1Type());
      Value combined;
      for (int i = 0; i < (int)masks.size(); ++i) {
        Value m = rewriter.create<sh::BroadcastInDimOp>(
            loc, boolTy, masks[i],
            rewriter.getDenseI64ArrayAttr({(int64_t)i}));
        combined = combined
            ? rewriter.create<sh::AndOp>(loc, combined, m).getResult() : m;
      }
      Value fv = ensureScalar(rewriter, loc, adaptor.getFillValue());
      Value fill = rewriter.create<sh::BroadcastInDimOp>(
          loc, resultTy, fv, rewriter.getDenseI64ArrayAttr({}));
      input = rewriter.create<sh::SelectOp>(loc, combined, input, fill);
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

} // namespace

void trt::populateTRTDynToStablehloPatterns(RewritePatternSet &patterns) {
  patterns.add<DynSliceConverter>(patterns.getContext());
}

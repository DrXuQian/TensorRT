#include "TRT/Passes.h"
#include "TRT/TRTDialect.h"
#include "TRT/TRTOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace sh = mlir::stablehlo;

namespace {

//===----------------------------------------------------------------------===//
// Helpers for building element-wise tensor<rank x i64> arithmetic
//===----------------------------------------------------------------------===//

/// Create a splat constant tensor<rank x i64> filled with `val`.
static Value makeConst(OpBuilder &b, Location loc, int rank, int64_t val) {
  auto ty = RankedTensorType::get({rank}, b.getI64Type());
  return b.create<sh::ConstantOp>(
      loc, DenseElementsAttr::get(ty, b.getI64IntegerAttr(val)));
}

/// Element-wise add on tensor<rank x i64>.
static Value add(OpBuilder &b, Location loc, Value a, Value c) {
  return b.create<sh::AddOp>(loc, a, c);
}

/// Element-wise subtract on tensor<rank x i64>.
static Value sub(OpBuilder &b, Location loc, Value a, Value c) {
  return b.create<sh::SubtractOp>(loc, a, c);
}

/// Element-wise multiply on tensor<rank x i64>.
static Value mul(OpBuilder &b, Location loc, Value a, Value c) {
  return b.create<sh::MulOp>(loc, a, c);
}

/// Element-wise negate on tensor<rank x i64>.
static Value neg(OpBuilder &b, Location loc, Value a) {
  return b.create<sh::NegOp>(loc, a);
}

/// Element-wise max(a, b) on tensor<rank x i64>.
static Value emax(OpBuilder &b, Location loc, Value a, Value c) {
  return b.create<sh::MaxOp>(loc, a, c);
}

/// Build a tensor<rank x i64> containing the input tensor's dim sizes.
static Value getInputShape(OpBuilder &b, Location loc, Value input, int rank) {
  auto i64Ty = b.getI64Type();
  auto scalar0d = RankedTensorType::get({}, b.getI32Type());
  auto scalar0d_i64 = RankedTensorType::get({}, i64Ty);
  auto elem1d = RankedTensorType::get({1}, i64Ty);

  SmallVector<Value> dims;
  for (int i = 0; i < rank; ++i) {
    // get_dimension_size returns tensor<i32>
    Value d_i32 = b.create<sh::GetDimensionSizeOp>(loc, scalar0d, input,
                                                    b.getI64IntegerAttr(i));
    // convert i32 → i64
    Value d_i64 = b.create<sh::ConvertOp>(loc, scalar0d_i64, d_i32);
    // reshape 0-d → 1-d
    Value d_1d = b.create<sh::ReshapeOp>(loc, elem1d, d_i64);
    dims.push_back(d_1d);
  }
  if (rank == 1)
    return dims[0];
  auto resultTy = RankedTensorType::get({rank}, i64Ty);
  return b.create<sh::ConcatenateOp>(loc, resultTy, dims,
                                     b.getI64IntegerAttr(0));
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
    auto inTy = cast<RankedTensorType>(adaptor.getInput().getType());
    int rank = inTy.getRank();

    Value input = adaptor.getInput();
    Value start = adaptor.getStart();
    Value size = adaptor.getSize();
    Value stride = adaptor.getStride();

    Value zero = makeConst(rewriter, loc, rank, 0);
    Value one = makeConst(rewriter, loc, rank, 1);

    // --- Handle negative strides ---
    // For each dim where stride < 0:
    //   new_start = start + (size - 1) * stride
    //   new_stride = -stride
    //   mark dim for final reverse
    //
    // For dynamic case, we do both paths and select per-element.
    Value absStride = b_abs(rewriter, loc, stride, zero);
    Value isNeg = rewriter.create<sh::CompareOp>(
        loc, stride, zero, sh::ComparisonDirection::LT);

    // new_start when negative: start + (size - 1) * stride
    Value sizeMinus1 = sub(rewriter, loc, size, one);
    Value negStart = add(rewriter, loc, start, mul(rewriter, loc, sizeMinus1, stride));
    // Select: use negStart where stride < 0, else original start
    Value normStart = rewriter.create<sh::SelectOp>(loc, isNeg, negStart, start);
    Value normStride = rewriter.create<sh::SelectOp>(
        loc, isNeg, neg(rewriter, loc, stride), stride);

    // Compute limit = normStart + (size - 1) * normStride + 1
    Value offset = mul(rewriter, loc, sizeMinus1, normStride);
    Value limit = add(rewriter, loc, normStart, add(rewriter, loc, offset, one));

    Value result;
    switch (op.getMode()) {
    case trt::SampleMode::kSTRICT_BOUNDS: {
      result = rewriter.create<sh::RealDynamicSliceOp>(
          loc, op.getResult().getType(), input, normStart, limit, normStride);
      break;
    }

    case trt::SampleMode::kFILL: {
      // pad_lo = max(0, -normStart)
      Value padLo = emax(rewriter, loc, zero, neg(rewriter, loc, normStart));

      // pad_hi = max(0, normStart + (size-1)*normStride - (d - 1))
      Value dimSizes = getInputShape(rewriter, loc, input, rank);
      Value dMinus1 = sub(rewriter, loc, dimSizes, one);
      Value lastIdx = add(rewriter, loc, normStart, offset);
      Value padHi = emax(rewriter, loc, zero, sub(rewriter, loc, lastIdx, dMinus1));

      Value interior = zero;

      // dynamic_pad
      Value fill = adaptor.getFillValue();
      Value padded = rewriter.create<sh::DynamicPadOp>(
          loc,
          /* result type: dynamic */ op.getResult().getType(),
          input, fill, padLo, padHi, interior);

      // Adjust start for padded tensor: newStart = normStart + padLo
      Value newStart = add(rewriter, loc, normStart, padLo);
      Value newLimit = add(rewriter, loc, newStart,
                           add(rewriter, loc, offset, one));

      result = rewriter.create<sh::RealDynamicSliceOp>(
          loc, op.getResult().getType(), padded, newStart, newLimit, normStride);
      break;
    }

    default:
      return op.emitOpError("dynamic lowering not yet supported for this mode");
    }

    // Reverse for negative strides — only if any stride was negative.
    // Check statically if we can tell; otherwise emit conditional reverse.
    // For simplicity: always emit reverse on dims where isNeg, using a
    // loop over dims and static check (the isNeg tensor is element-wise,
    // but reverse needs a list of dims).
    //
    // Since rank is known, we can emit per-dim conditionals.
    // But stablehlo.reverse is unconditional on a fixed dim list.
    // The simplest correct approach: emit reverse on ALL dims, then
    // use select to pick between reversed and non-reversed based on isNeg.
    //
    // However, this creates O(rank) reverse ops. For rank <= 8 this is fine.
    for (int i = 0; i < rank; ++i) {
      Value reversed = rewriter.create<sh::ReverseOp>(
          loc, result.getType(), result,
          rewriter.getDenseI64ArrayAttr({(int64_t)i}));
      // Extract the i-th element of isNeg (i is compile-time known → static slice)
      auto i1Ty = rewriter.getI1Type();
      Value isNegSlice = rewriter.create<sh::SliceOp>(
          loc, RankedTensorType::get({1}, i1Ty), isNeg,
          rewriter.getDenseI64ArrayAttr({(int64_t)i}),
          rewriter.getDenseI64ArrayAttr({(int64_t)(i + 1)}),
          rewriter.getDenseI64ArrayAttr({(int64_t)1}));
      Value isNegScalar = rewriter.create<sh::ReshapeOp>(
          loc, RankedTensorType::get({}, i1Ty), isNegSlice);
      // Broadcast scalar predicate to result shape for select
      Value bcast = rewriter.create<sh::BroadcastInDimOp>(
          loc, RankedTensorType::get(
                   cast<RankedTensorType>(result.getType()).getShape(),
                   rewriter.getI1Type()),
          isNegScalar, rewriter.getDenseI64ArrayAttr({}));
      result = rewriter.create<sh::SelectOp>(loc, bcast, reversed, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  /// abs(x) = select(x < 0, -x, x)
  static Value b_abs(OpBuilder &b, Location loc, Value x, Value zero) {
    Value isNeg = b.create<sh::CompareOp>(
        loc, x, zero, sh::ComparisonDirection::LT);
    return b.create<sh::SelectOp>(loc, isNeg, b.create<sh::NegOp>(loc, x), x);
  }
};

} // namespace

// Register dynamic pattern alongside static ones
void trt::populateTRTDynToStablehloPatterns(RewritePatternSet &patterns) {
  patterns.add<DynSliceConverter>(patterns.getContext());
}

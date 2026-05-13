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

  default:
    llvm_unreachable("kFILL handled separately");
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

    // --- kFILL: use dynamic_pad approach (gather can't synthesize fill) ---
    if (op.getMode() == trt::SampleMode::kFILL)
      return lowerFill(op, adaptor, rewriter);

    // --- kSTRICT/kCLAMP/kWRAP/kREFLECT: iota + mapIndex + gather ---
    for (int axis = 0; axis < rank; ++axis) {
      int64_t outDim = resultTy.getDimSize(axis);

      // Extract start[axis], stride[axis] as scalars, broadcast to outDim
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

      // Map to valid range
      Value idx = mapIndex(rewriter, loc, raw, d, op.getMode(), outDim);

      // Gather along this axis
      input = gatherAxis(rewriter, loc, input, idx, axis);
    }

    rewriter.replaceOp(op, input);
    return success();
  }

private:
  /// kFILL: compute pad amounts dynamically, use dynamic_pad + real_dynamic_slice.
  LogicalResult lowerFill(trt::DynSliceOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto inTy = cast<RankedTensorType>(adaptor.getInput().getType());
    int rank = inTy.getRank();
    auto i64 = rewriter.getI64Type();
    auto vecTy = RankedTensorType::get({rank}, i64);

    Value start = adaptor.getStart();
    Value size = adaptor.getSize();
    Value stride = adaptor.getStride();

    // Constants
    Value zero = rewriter.create<sh::ConstantOp>(loc,
        DenseElementsAttr::get(vecTy, rewriter.getI64IntegerAttr(0)));
    Value one = rewriter.create<sh::ConstantOp>(loc,
        DenseElementsAttr::get(vecTy, rewriter.getI64IntegerAttr(1)));

    // Handle negative strides: normStart = select(stride<0, start+(size-1)*stride, start)
    Value isNeg = rewriter.create<sh::CompareOp>(loc, stride, zero,
                      sh::ComparisonDirection::LT);
    Value sm1 = rewriter.create<sh::SubtractOp>(loc, size, one);
    Value negAdj = rewriter.create<sh::AddOp>(loc, start,
                       rewriter.create<sh::MulOp>(loc, sm1, stride));
    Value normStart = rewriter.create<sh::SelectOp>(loc, isNeg, negAdj, start);
    Value normStride = rewriter.create<sh::SelectOp>(loc, isNeg,
                           rewriter.create<sh::NegOp>(loc, stride), stride);

    // Pad amounts
    Value padLo = rewriter.create<sh::MaxOp>(loc, zero,
                      rewriter.create<sh::NegOp>(loc, normStart));
    // dimSizes tensor
    SmallVector<Value> dims;
    for (int i = 0; i < rank; ++i) {
      Value d32 = rewriter.create<sh::GetDimensionSizeOp>(loc,
          RankedTensorType::get({}, rewriter.getI32Type()),
          adaptor.getInput(), rewriter.getI64IntegerAttr(i));
      Value d64 = rewriter.create<sh::ConvertOp>(loc,
          RankedTensorType::get({}, i64), d32);
      dims.push_back(rewriter.create<sh::ReshapeOp>(loc,
          RankedTensorType::get({1}, i64), d64));
    }
    Value dimSizes = (rank == 1) ? dims[0]
        : rewriter.create<sh::ConcatenateOp>(loc, vecTy, dims,
              rewriter.getI64IntegerAttr(0)).getResult();

    Value dm1 = rewriter.create<sh::SubtractOp>(loc, dimSizes, one);
    Value offset = rewriter.create<sh::MulOp>(loc, sm1, normStride);
    Value lastIdx = rewriter.create<sh::AddOp>(loc, normStart, offset);
    Value padHi = rewriter.create<sh::MaxOp>(loc, zero,
                      rewriter.create<sh::SubtractOp>(loc, lastIdx, dm1));

    // dynamic_pad + real_dynamic_slice
    Value padded = rewriter.create<sh::DynamicPadOp>(loc,
        op.getResult().getType(), adaptor.getInput(), adaptor.getFillValue(),
        padLo, padHi, zero);

    Value newStart = rewriter.create<sh::AddOp>(loc, normStart, padLo);
    Value newLimit = rewriter.create<sh::AddOp>(loc, newStart,
                         rewriter.create<sh::AddOp>(loc, offset, one));
    Value result = rewriter.create<sh::RealDynamicSliceOp>(loc,
        op.getResult().getType(), padded, newStart, newLimit, normStride);

    // Reverse for negative strides (per-dim select)
    for (int i = 0; i < rank; ++i) {
      Value rev = rewriter.create<sh::ReverseOp>(loc, result.getType(), result,
                      rewriter.getDenseI64ArrayAttr({(int64_t)i}));
      auto i1Ty = rewriter.getI1Type();
      Value flag = rewriter.create<sh::SliceOp>(loc,
          RankedTensorType::get({1}, i1Ty), isNeg,
          rewriter.getDenseI64ArrayAttr({(int64_t)i}),
          rewriter.getDenseI64ArrayAttr({(int64_t)(i + 1)}),
          rewriter.getDenseI64ArrayAttr({1}));
      Value flagScalar = rewriter.create<sh::ReshapeOp>(loc,
          RankedTensorType::get({}, i1Ty), flag);
      Value mask = rewriter.create<sh::BroadcastInDimOp>(loc,
          RankedTensorType::get(
              cast<RankedTensorType>(result.getType()).getShape(), i1Ty),
          flagScalar, rewriter.getDenseI64ArrayAttr({}));
      result = rewriter.create<sh::SelectOp>(loc, mask, rev, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void trt::populateTRTDynToStablehloPatterns(RewritePatternSet &patterns) {
  patterns.add<DynSliceConverter>(patterns.getContext());
}

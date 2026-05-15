/// Dynamic trt.dyn_slice → StableHLO lowering via index computation + gather.
///
/// All 5 modes: per-axis dynamic_iota → raw_idx → mode-specific index map
/// → dynamic_gather. Fully dynamic: no static shapes assumed anywhere.

#include "TRT/Passes.h"
#include "TRT/TRTDialect.h"
#include "TRT/TRTOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace sh = mlir::stablehlo;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value i64Const0D(OpBuilder &b, Location loc, int64_t v) {
  return b.create<sh::ConstantOp>(loc,
      DenseElementsAttr::get(RankedTensorType::get({}, b.getI64Type()),
                             b.getI64IntegerAttr(v)));
}

/// Extract tensor<Nxi64>[idx] as 0-D tensor<i64>.
static Value extractScalar(OpBuilder &b, Location loc, Value vec, int idx) {
  auto et = cast<RankedTensorType>(vec.getType()).getElementType();
  Value sl = b.create<sh::SliceOp>(loc,
      RankedTensorType::get({1}, et), vec,
      b.getDenseI64ArrayAttr({(int64_t)idx}),
      b.getDenseI64ArrayAttr({(int64_t)(idx + 1)}),
      b.getDenseI64ArrayAttr({1}));
  Value scalar = b.create<sh::ReshapeOp>(loc,
      RankedTensorType::get({}, et), sl);
  if (!et.isInteger(64))
    scalar = b.create<sh::ConvertOp>(loc,
        RankedTensorType::get({}, b.getI64Type()), scalar);
  return scalar;
}

/// Get input dim at axis as 0-D tensor<i64>.
static Value getInputDim(OpBuilder &b, Location loc, Value input, int axis) {
  Value d32 = b.create<sh::GetDimensionSizeOp>(loc,
      RankedTensorType::get({}, b.getI32Type()),
      input, b.getI64IntegerAttr(axis));
  return b.create<sh::ConvertOp>(loc,
      RankedTensorType::get({}, b.getI64Type()), d32);
}

/// Pack N 0-D scalars → tensor<N x i64>.
static Value pack1D(OpBuilder &b, Location loc,
                    ArrayRef<Value> scalars) {
  auto i64 = b.getI64Type();
  SmallVector<Value> elems;
  for (auto s : scalars)
    elems.push_back(b.create<sh::ReshapeOp>(loc,
        RankedTensorType::get({1}, i64), s));
  if (elems.size() == 1) return elems[0];
  return b.create<sh::ConcatenateOp>(loc,
      RankedTensorType::get({(int64_t)elems.size()}, i64),
      elems, b.getI64IntegerAttr(0));
}

/// Dynamic broadcast 0-D scalar → tensor<? x i64>.
static Value dynBcast(OpBuilder &b, Location loc,
                      Value scalar, Value shape1D) {
  return b.create<sh::DynamicBroadcastInDimOp>(loc,
      RankedTensorType::get({ShapedType::kDynamic}, b.getI64Type()),
      scalar, shape1D, b.getDenseI64ArrayAttr({}));
}

/// Ensure a Value is 0-d; reshape from 1-d if needed.
static Value ensureScalar(OpBuilder &b, Location loc, Value v) {
  auto ty = cast<RankedTensorType>(v.getType());
  if (ty.getRank() == 0) return v;
  auto scalarTy = RankedTensorType::get({}, ty.getElementType());
  auto reshaped = b.create<sh::ReshapeOp>(loc, scalarTy, v);
  assert(cast<RankedTensorType>(reshaped.getType()).getRank() == 0);
  return reshaped;
}

//===----------------------------------------------------------------------===//
// Mode-specific index mapping (all on tensor<? x i64>)
//===----------------------------------------------------------------------===//

struct MappedIndex { Value idx; Value mask; /* null for non-FILL */ };

static MappedIndex mapIndex(OpBuilder &b, Location loc, Value raw,
                            Value dim, Value zero, Value one,
                            trt::SampleMode mode) {
  Value dm1 = b.create<sh::SubtractOp>(loc, dim, one);

  switch (mode) {
  case trt::SampleMode::kSTRICT_BOUNDS:
    return {raw, {}};

  case trt::SampleMode::kCLAMP:
    return {b.create<sh::ClampOp>(loc, zero, raw, dm1), {}};

  case trt::SampleMode::kWRAP: {
    Value r = b.create<sh::RemOp>(loc, raw, dim);
    return {b.create<sh::RemOp>(loc, b.create<sh::AddOp>(loc, r, dim), dim), {}};
  }

  case trt::SampleMode::kREFLECT: {
    Value two = b.create<sh::AddOp>(loc, one, one);
    Value p = b.create<sh::MulOp>(loc, dm1, two);
    Value c = b.create<sh::RemOp>(loc, b.create<sh::AbsOp>(loc, raw), p);
    Value cGeD = b.create<sh::CompareOp>(loc, c, dim,
                     sh::ComparisonDirection::GE);
    return {b.create<sh::SelectOp>(loc, cGeD,
                b.create<sh::SubtractOp>(loc, p, c), c), {}};
  }

  case trt::SampleMode::kFILL: {
    Value geLo = b.create<sh::CompareOp>(loc, raw, zero,
                     sh::ComparisonDirection::GE);
    Value ltHi = b.create<sh::CompareOp>(loc, raw, dim,
                     sh::ComparisonDirection::LT);
    Value mask = b.create<sh::AndOp>(loc, geLo, ltHi);
    return {b.create<sh::ClampOp>(loc, zero, raw, dm1), mask};
  }
  }
  llvm_unreachable("invalid SampleMode");
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
    auto elemTy = inTy.getElementType();
    auto i64 = rewriter.getI64Type();
    auto i1 = rewriter.getI1Type();

    if (rank == 0) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    Value current = adaptor.getInput();
    SmallVector<std::pair<int, Value>> oobMasks; // axis → mask

    auto dynVecTy = RankedTensorType::get({ShapedType::kDynamic}, i64);

    for (int axis = 0; axis < rank; ++axis) {
      Value startS  = extractScalar(rewriter, loc, adaptor.getStart(), axis);
      Value strideS = extractScalar(rewriter, loc, adaptor.getStride(), axis);
      Value sizeS   = extractScalar(rewriter, loc, adaptor.getSize(), axis);
      Value dimS    = getInputDim(rewriter, loc, adaptor.getInput(), axis);

      // shape tensor for dynamic ops: tensor<1xi64> = [sizeS]
      Value shape1D = rewriter.create<sh::ReshapeOp>(loc,
          RankedTensorType::get({1}, i64), sizeS);

      // dynamic_iota
      Value iota = rewriter.create<sh::DynamicIotaOp>(loc,
          dynVecTy, shape1D, rewriter.getI64IntegerAttr(0));

      // broadcast scalars to dynamic 1-D
      Value startB  = dynBcast(rewriter, loc, startS, shape1D);
      Value strideB = dynBcast(rewriter, loc, strideS, shape1D);
      Value dimB    = dynBcast(rewriter, loc, dimS, shape1D);
      Value zero    = dynBcast(rewriter, loc, i64Const0D(rewriter, loc, 0), shape1D);
      Value one     = dynBcast(rewriter, loc, i64Const0D(rewriter, loc, 1), shape1D);

      // raw = start + iota * stride
      Value raw = rewriter.create<sh::AddOp>(loc, startB,
                      rewriter.create<sh::MulOp>(loc, iota, strideB));

      // mapIndex
      auto [idx, mask] = mapIndex(rewriter, loc, raw, dimB, zero, one,
                                  op.getMode());
      if (mask) oobMasks.emplace_back(axis, mask);

      // dynamic_gather: slice_sizes = input shape with dim[axis] = 1
      auto curTy = cast<RankedTensorType>(current.getType());
      SmallVector<Value> ss(rank);
      for (int d = 0; d < rank; ++d)
        ss[d] = (d == axis) ? i64Const0D(rewriter, loc, 1)
                            : getInputDim(rewriter, loc, current, d);
      Value sliceSizesT = pack1D(rewriter, loc, ss);

      // result type: dim[axis] = dynamic, others preserved
      SmallVector<int64_t> outShape(curTy.getShape());
      outShape[axis] = ShapedType::kDynamic;
      auto outTy = RankedTensorType::get(outShape, elemTy);

      SmallVector<int64_t> offDims;
      for (int d = 0; d < rank; ++d)
        if (d != axis) offDims.push_back(d);

      auto dn = sh::GatherDimensionNumbersAttr::get(rewriter.getContext(),
          offDims, {axis}, {}, {}, {axis}, /*indexVectorDim=*/1);

      current = rewriter.create<sh::DynamicGatherOp>(loc, outTy,
          current, idx, sliceSizesT, dn);
    }

    // kFILL: combine per-axis masks + select
    if (!oobMasks.empty()) {
      auto curTy = cast<RankedTensorType>(current.getType());
      auto boolTy = RankedTensorType::get(curTy.getShape(), i1);

      // output shape as runtime tensor<rank x i64>
      SmallVector<Value> shapeScalars(rank);
      for (int d = 0; d < rank; ++d)
        shapeScalars[d] = getInputDim(rewriter, loc, current, d);
      Value curShape = pack1D(rewriter, loc, shapeScalars);

      Value combined;
      for (auto &[axis, m] : oobMasks) {
        Value bm = rewriter.create<sh::DynamicBroadcastInDimOp>(
            loc, boolTy, m, curShape,
            rewriter.getDenseI64ArrayAttr({(int64_t)axis}));
        combined = combined
            ? rewriter.create<sh::AndOp>(loc, combined, bm).getResult() : bm;
      }

      Value fv = ensureScalar(rewriter, loc, adaptor.getFillValue());
      Value fill = rewriter.create<sh::DynamicBroadcastInDimOp>(
          loc, curTy, fv, curShape, rewriter.getDenseI64ArrayAttr({}));

      current = rewriter.create<sh::SelectOp>(loc, combined, current, fill);
    }

    // Refine dynamic→static type if the op result type is more specific
    auto targetTy = cast<RankedTensorType>(op.getResult().getType());
    if (current.getType() != targetTy) {
      SmallVector<Value> shapeDims(rank);
      for (int d = 0; d < rank; ++d) {
        shapeDims[d] = targetTy.isDynamicDim(d)
            ? getInputDim(rewriter, loc, current, d)
            : i64Const0D(rewriter, loc, targetTy.getDimSize(d));
      }
      Value shapeT = pack1D(rewriter, loc, shapeDims);
      current = rewriter.create<sh::DynamicReshapeOp>(loc, targetTy,
                                                       current, shapeT);
    }

    rewriter.replaceOp(op, current);
    return success();
  }
};

} // namespace

void trt::populateTRTDynToStablehloPatterns(RewritePatternSet &patterns) {
  patterns.add<DynSliceConverter>(patterns.getContext());
}

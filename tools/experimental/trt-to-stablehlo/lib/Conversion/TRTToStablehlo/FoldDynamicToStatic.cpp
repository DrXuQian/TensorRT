/// Lightweight pass: fold real_dynamic_slice/dynamic_pad with constant
/// operands back to static slice/pad so stablehlo-translate --interpret works.

#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace sh = mlir::stablehlo;

namespace {

/// Try to extract a constant DenseI64Array from a Value (stablehlo.constant).
static std::optional<SmallVector<int64_t>> getConstI64(Value v) {
  auto defOp = v.getDefiningOp<sh::ConstantOp>();
  if (!defOp) return std::nullopt;
  auto attr = dyn_cast<DenseIntElementsAttr>(defOp.getValue());
  if (!attr) return std::nullopt;
  SmallVector<int64_t> vals;
  for (auto e : attr.getValues<APInt>())
    vals.push_back(e.getSExtValue());
  return vals;
}

/// real_dynamic_slice(X, const_start, const_limit, const_strides) → slice
struct FoldRealDynamicSlice : OpRewritePattern<sh::RealDynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(sh::RealDynamicSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto start = getConstI64(op.getStartIndices());
    auto limit = getConstI64(op.getLimitIndices());
    auto strides = getConstI64(op.getStrides());
    if (!start || !limit || !strides) return failure();

    rewriter.replaceOpWithNewOp<sh::SliceOp>(
        op, op.getType(), op.getOperand(),
        rewriter.getDenseI64ArrayAttr(*start),
        rewriter.getDenseI64ArrayAttr(*limit),
        rewriter.getDenseI64ArrayAttr(*strides));
    return success();
  }
};

/// dynamic_pad(X, pv, const_lo, const_hi, const_int) → pad
struct FoldDynamicPad : OpRewritePattern<sh::DynamicPadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(sh::DynamicPadOp op,
                                PatternRewriter &rewriter) const override {
    auto lo = getConstI64(op.getEdgePaddingLow());
    auto hi = getConstI64(op.getEdgePaddingHigh());
    auto interior = getConstI64(op.getInteriorPadding());
    if (!lo || !hi || !interior) return failure();

    rewriter.replaceOpWithNewOp<sh::PadOp>(
        op, op.getType(), op.getOperand(), op.getPaddingValue(),
        rewriter.getDenseI64ArrayAttr(*lo),
        rewriter.getDenseI64ArrayAttr(*hi),
        rewriter.getDenseI64ArrayAttr(*interior));
    return success();
  }
};

struct FoldDynamicToStaticPass
    : PassWrapper<FoldDynamicToStaticPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldDynamicToStaticPass)
  StringRef getArgument() const override { return "fold-dynamic-to-static"; }
  StringRef getDescription() const override {
    return "Fold stablehlo dynamic ops with constant operands to static";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldRealDynamicSlice, FoldDynamicPad>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace trt {
void registerFoldDynamicToStaticPass() {
  PassRegistration<FoldDynamicToStaticPass>();
}
} // namespace trt

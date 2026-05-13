#include "TRT/Passes.h"
#include "TRT/TRTDialect.h"
#include "TRT/TRTOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace trt {

#define GEN_PASS_DEF_CONVERTTRTTOSTABLEHLO
#include "TRT/Passes.h.inc"

namespace {
struct ConvertTRTToStablehloPass
    : public impl::ConvertTRTToStablehloBase<ConvertTRTToStablehloPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addIllegalDialect<trt::TRTDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    populateTRTToStablehloPatterns(patterns);
    populateTRTDynToStablehloPatterns(patterns);

    if (failed(mlir::applyPartialConversion(getOperation(), target,
                                            std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace trt

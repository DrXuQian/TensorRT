#ifndef TRT_PASSES_H
#define TRT_PASSES_H

#include "mlir/Pass/Pass.h"

namespace trt {

void populateTRTToStablehloPatterns(mlir::RewritePatternSet &patterns);
void populateTRTDynToStablehloPatterns(mlir::RewritePatternSet &patterns);
void registerFoldDynamicToStaticPass();

#define GEN_PASS_DECL
#include "TRT/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "TRT/Passes.h.inc"

} // namespace trt

#endif // TRT_PASSES_H

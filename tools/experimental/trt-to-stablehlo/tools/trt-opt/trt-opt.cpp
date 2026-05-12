#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "TRT/TRTDialect.h"
#include "TRT/TRTOps.h"
#include "TRT/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  trt::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<trt::TRTDialect, mlir::func::FuncDialect,
                  mlir::stablehlo::StablehloDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TRT-to-StableHLO optimizer driver\n",
                        registry));
}

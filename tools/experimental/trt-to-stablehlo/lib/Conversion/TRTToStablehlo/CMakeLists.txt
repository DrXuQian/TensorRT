add_mlir_dialect_library(MLIRTRTToStablehlo
  TRTToStablehlo.cpp
  SliceConverter.cpp

  DEPENDS
  MLIRTRTOpsIncGen
  MLIRTRTPassesIncGen

  LINK_LIBS PUBLIC
  MLIRTRTDialect
  StablehloOps
  MLIRFuncDialect
  MLIRTransformUtils
  MLIRPass
  MLIRIR
)

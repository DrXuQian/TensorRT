add_mlir_dialect_library(MLIRTRTDialect
  TRTDialect.cpp
  TRTOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${TRT_SOURCE_DIR}/include/TRT

  DEPENDS
  MLIRTRTOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

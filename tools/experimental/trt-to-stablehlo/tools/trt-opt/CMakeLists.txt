set(LIBS
  MLIRTRTDialect
  MLIRTRTToStablehlo
  StablehloOps
  MLIRFuncDialect
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

add_llvm_executable(trt-opt trt-opt.cpp)
llvm_update_compile_flags(trt-opt)
target_link_libraries(trt-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(trt-opt)

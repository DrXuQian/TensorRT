configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TRT_TEST_DEPENDS
  FileCheck count not
  trt-opt
)

add_lit_testsuite(check-trt-to-stablehlo
  "Running the trt-to-stablehlo regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TRT_TEST_DEPENDS}
)
set_target_properties(check-trt-to-stablehlo PROPERTIES FOLDER "Tests")

add_lit_testsuites(TRT_TO_STABLEHLO ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${TRT_TEST_DEPENDS}
)

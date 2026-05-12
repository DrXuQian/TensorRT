#include "TRT/TRTDialect.h"
#include "TRT/TRTOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

// Enum implementation
#include "TRT/TRTOpsEnums.cpp.inc"

// AttrDef full implementation (must come before addAttributes)
#define GET_ATTRDEF_CLASSES
#include "TRT/TRTOpsAttrs.cpp.inc"

// Dialect implementation
#include "TRT/TRTOpsDialect.cpp.inc"

void trt::TRTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TRT/TRTOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TRT/TRTOpsAttrs.cpp.inc"
      >();
}

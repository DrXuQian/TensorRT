#ifndef TRT_TRTOPS_H
#define TRT_TRTOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TRT/TRTOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TRT/TRTOpsAttrs.h.inc"

#define GET_OP_CLASSES
#include "TRT/TRTOps.h.inc"

#endif // TRT_TRTOPS_H

#ifndef TRT_DIALECT
#define TRT_DIALECT

include "mlir/IR/OpBase.td"

def TRT_Dialect : Dialect {
  let name = "trt";
  let summary = "Minimal TensorRT dialect for slice lowering";
  let cppNamespace = "::trt";
  let useDefaultAttributePrinterParser = 1;
}

class TRT_Op<string mnemonic, list<Trait> traits = []>
    : Op<TRT_Dialect, mnemonic, traits>;

#endif // TRT_DIALECT

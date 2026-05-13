// SliceModeEmitter.h
// Drop this file into your project. No extra build deps beyond stablehlo.
//
// Usage:
//   #include "SliceModeEmitter.h"
//
//   // In your op_slice or wherever:
//   mlir::Value result = SliceModeEmitter::emit(
//       op_builder, loc, input,
//       start, end, stride,
//       SliceModeEmitter::Mode::kCLAMP);
//
//   // For kFILL mode, pass fill value:
//   mlir::Value result = SliceModeEmitter::emit(
//       op_builder, loc, input,
//       start, end, stride,
//       SliceModeEmitter::Mode::kFILL, fillValue);

#pragma once

#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <numeric>

class SliceModeEmitter {
public:
    enum class Mode { kSTRICT_BOUNDS = 0, kWRAP, kCLAMP, kFILL, kREFLECT };

    /// Main entry point.
    /// start/end/stride follow stablehlo convention (start_indices, limit_indices, strides).
    /// fill is required only for kFILL mode (scalar 0-d tensor).
    static mlir::Value emit(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value input,
                            llvm::ArrayRef<int64_t> start,
                            llvm::ArrayRef<int64_t> end,
                            llvm::ArrayRef<int64_t> stride,
                            Mode mode,
                            mlir::Value fill = {}) {
        auto inTy = mlir::cast<mlir::RankedTensorType>(input.getType());
        int rank = inTy.getRank();

        // (start, end, stride) -> (start, size, stride)
        llvm::SmallVector<int64_t> startVec(start), strideVec(stride);
        llvm::SmallVector<int64_t> size(rank);
        for (int i = 0; i < rank; ++i)
            size[i] = (end[i] - start[i] + stride[i] - 1) / stride[i];

        // Normalize negative strides
        llvm::SmallVector<int64_t> reverseDims;
        for (int i = 0; i < rank; ++i) {
            if (strideVec[i] < 0) {
                startVec[i] += (size[i] - 1) * strideVec[i];
                strideVec[i] = -strideVec[i];
                reverseDims.push_back(i);
            }
        }

        // Compute padding per axis
        llvm::SmallVector<Pad> pads;
        for (int i = 0; i < rank; ++i) {
            int64_t last = startVec[i] + (size[i] - 1) * strideVec[i];
            int64_t lo = std::min(startVec[i], last);
            int64_t hi = std::max(startVec[i], last);
            pads.push_back({std::max<int64_t>(0, -lo),
                            std::max<int64_t>(0, hi - (inTy.getDimSize(i) - 1))});
        }

        // Mode-specific padding
        mlir::Value padded = input;
        switch (mode) {
        case Mode::kSTRICT_BOUNDS:
            for (int i = 0; i < rank; ++i) {
                assert(pads[i].lo == 0 && pads[i].hi == 0 &&
                       "kSTRICT_BOUNDS: slice goes out-of-bounds");
            }
            break;
        case Mode::kFILL:    padded = doPadFill(b, loc, input, fill, pads); break;
        case Mode::kCLAMP:   padded = doPadClamp(b, loc, input, pads); break;
        case Mode::kWRAP:    padded = doPadWrap(b, loc, input, pads); break;
        case Mode::kREFLECT: padded = doPadReflect(b, loc, input, pads); break;
        }

        // Final strided slice
        llvm::SmallVector<int64_t> sS(rank), sL(rank), sT(rank);
        for (int i = 0; i < rank; ++i) {
            sS[i] = startVec[i] + pads[i].lo;
            sL[i] = sS[i] + (size[i] - 1) * strideVec[i] + 1;
            sT[i] = strideVec[i];
        }
        auto resTy = mlir::RankedTensorType::get(size, inTy.getElementType());
        mlir::Value result = b.create<mlir::stablehlo::SliceOp>(
            loc, resTy, padded,
            b.getDenseI64ArrayAttr(sS),
            b.getDenseI64ArrayAttr(sL),
            b.getDenseI64ArrayAttr(sT));

        // Reverse for originally-negative strides
        if (!reverseDims.empty())
            result = b.create<mlir::stablehlo::ReverseOp>(
                loc, result.getType(), result,
                b.getDenseI64ArrayAttr(reverseDims));

        return result;
    }

private:
    struct Pad { int64_t lo, hi; };

    // ---- axis helper ----
    static mlir::Value sliceAxis(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::Value v, int axis, int64_t s, int64_t e) {
        auto ty = mlir::cast<mlir::RankedTensorType>(v.getType());
        int rank = ty.getRank();
        llvm::SmallVector<int64_t> lo(rank, 0), hi(ty.getShape()), st(rank, 1);
        lo[axis] = s; hi[axis] = e;
        llvm::SmallVector<int64_t> rs(ty.getShape());
        rs[axis] = e - s;
        return b.create<mlir::stablehlo::SliceOp>(loc,
            mlir::RankedTensorType::get(rs, ty.getElementType()), v,
            b.getDenseI64ArrayAttr(lo), b.getDenseI64ArrayAttr(hi),
            b.getDenseI64ArrayAttr(st));
    }

    // ---- concat helper ----
    static mlir::Value concat(mlir::OpBuilder &b, mlir::Location loc,
                              llvm::ArrayRef<mlir::Value> parts, int axis) {
        auto ty0 = mlir::cast<mlir::RankedTensorType>(parts[0].getType());
        llvm::SmallVector<int64_t> shape(ty0.getShape());
        shape[axis] = 0;
        for (auto p : parts)
            shape[axis] += mlir::cast<mlir::RankedTensorType>(p.getType()).getDimSize(axis);
        return b.create<mlir::stablehlo::ConcatenateOp>(loc,
            mlir::RankedTensorType::get(shape, ty0.getElementType()),
            parts, b.getI64IntegerAttr(axis));
    }

    // ---- kFILL: stablehlo.pad ----
    static mlir::Value doPadFill(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::Value in, mlir::Value fill,
                                 llvm::ArrayRef<Pad> pads) {
        auto inTy = mlir::cast<mlir::RankedTensorType>(in.getType());
        int rank = inTy.getRank();
        llvm::SmallVector<int64_t> lo(rank), hi(rank), interior(rank, 0), rs(inTy.getShape());
        for (int i = 0; i < rank; ++i) {
            lo[i] = pads[i].lo; hi[i] = pads[i].hi;
            rs[i] += lo[i] + hi[i];
        }
        return b.create<mlir::stablehlo::PadOp>(loc,
            mlir::RankedTensorType::get(rs, inTy.getElementType()),
            in, fill,
            b.getDenseI64ArrayAttr(lo), b.getDenseI64ArrayAttr(hi),
            b.getDenseI64ArrayAttr(interior));
    }

    // ---- kCLAMP: edge replicate ----
    static mlir::Value doPadClamp(mlir::OpBuilder &b, mlir::Location loc,
                                  mlir::Value in, llvm::ArrayRef<Pad> pads) {
        int rank = mlir::cast<mlir::RankedTensorType>(in.getType()).getRank();
        for (int axis = 0; axis < rank; ++axis) {
            if (pads[axis].lo == 0 && pads[axis].hi == 0) continue;
            auto curTy = mlir::cast<mlir::RankedTensorType>(in.getType());
            llvm::SmallVector<int64_t> shape(curTy.getShape());
            llvm::SmallVector<int64_t> dims(rank);
            std::iota(dims.begin(), dims.end(), 0);
            llvm::SmallVector<mlir::Value> parts;
            if (pads[axis].lo > 0) {
                auto edge = sliceAxis(b, loc, in, axis, 0, 1);
                llvm::SmallVector<int64_t> bs(shape); bs[axis] = pads[axis].lo;
                parts.push_back(b.create<mlir::stablehlo::BroadcastInDimOp>(loc,
                    mlir::RankedTensorType::get(bs, curTy.getElementType()),
                    edge, b.getDenseI64ArrayAttr(dims)));
            }
            parts.push_back(in);
            if (pads[axis].hi > 0) {
                auto edge = sliceAxis(b, loc, in, axis, shape[axis]-1, shape[axis]);
                llvm::SmallVector<int64_t> bs(shape); bs[axis] = pads[axis].hi;
                parts.push_back(b.create<mlir::stablehlo::BroadcastInDimOp>(loc,
                    mlir::RankedTensorType::get(bs, curTy.getElementType()),
                    edge, b.getDenseI64ArrayAttr(dims)));
            }
            in = concat(b, loc, parts, axis);
        }
        return in;
    }

    // ---- kWRAP: wrap around (handles pad > d via divmod) ----
    static mlir::Value doPadWrap(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::Value in, llvm::ArrayRef<Pad> pads) {
        int rank = mlir::cast<mlir::RankedTensorType>(in.getType()).getRank();
        for (int axis = 0; axis < rank; ++axis) {
            if (pads[axis].lo == 0 && pads[axis].hi == 0) continue;
            int64_t d = mlir::cast<mlir::RankedTensorType>(in.getType()).getDimSize(axis);
            int64_t fullLo = pads[axis].lo / d, remLo = pads[axis].lo % d;
            int64_t fullHi = pads[axis].hi / d, remHi = pads[axis].hi % d;
            llvm::SmallVector<mlir::Value> parts;
            if (remLo > 0) parts.push_back(sliceAxis(b, loc, in, axis, d - remLo, d));
            for (int64_t i = 0; i < fullLo; ++i) parts.push_back(in);
            parts.push_back(in);
            for (int64_t i = 0; i < fullHi; ++i) parts.push_back(in);
            if (remHi > 0) parts.push_back(sliceAxis(b, loc, in, axis, 0, remHi));
            in = concat(b, loc, parts, axis);
        }
        return in;
    }

    // ---- helper: reverse along one axis ----
    static mlir::Value reverseAxis(mlir::OpBuilder &b, mlir::Location loc,
                                   mlir::Value v, int axis) {
        return b.create<mlir::stablehlo::ReverseOp>(
            loc, v.getType(), v, b.getDenseI64ArrayAttr({(int64_t)axis}));
    }

    // ---- kREFLECT: reflection via half-period chunks ----
    static mlir::Value doPadReflect(mlir::OpBuilder &b, mlir::Location loc,
                                    mlir::Value in, llvm::ArrayRef<Pad> pads) {
        int rank = mlir::cast<mlir::RankedTensorType>(in.getType()).getRank();
        for (int axis = 0; axis < rank; ++axis) {
            if (pads[axis].lo == 0 && pads[axis].hi == 0) continue;
            int64_t d = mlir::cast<mlir::RankedTensorType>(in.getType()).getDimSize(axis);
            int64_t hp = d - 1;

            auto fwdLo = sliceAxis(b, loc, in, axis, 0, hp);
            auto fwdHi = sliceAxis(b, loc, in, axis, 1, d);
            auto revLo = reverseAxis(b, loc, fwdLo, axis);
            auto revHi = reverseAxis(b, loc, fwdHi, axis);

            // Right: revLo, fwdHi, revLo, fwdHi, ...
            llvm::SmallVector<mlir::Value> rightParts;
            { int64_t rem = pads[axis].hi; int tog = 0;
              while (rem > 0) {
                auto chunk = (tog % 2 == 0) ? revLo : fwdHi;
                int64_t take = std::min(rem, hp);
                if (take < hp) chunk = sliceAxis(b, loc, chunk, axis, 0, take);
                rightParts.push_back(chunk);
                rem -= take; tog++;
            }}

            // Left (nearest-to-farthest): revHi, fwdLo, revHi, fwdLo, ...
            llvm::SmallVector<mlir::Value> leftParts;
            { llvm::SmallVector<mlir::Value> chunks;
              int64_t rem = pads[axis].lo; int tog = 0;
              while (rem > 0) {
                auto chunk = (tog % 2 == 0) ? revHi : fwdLo;
                int64_t take = std::min(rem, hp);
                if (take < hp) chunk = sliceAxis(b, loc, chunk, axis, hp - take, hp);
                chunks.push_back(chunk);
                rem -= take; tog++;
              }
              for (auto it = chunks.rbegin(); it != chunks.rend(); ++it)
                leftParts.push_back(*it);
            }

            llvm::SmallVector<mlir::Value> parts;
            parts.append(leftParts.begin(), leftParts.end());
            parts.push_back(in);
            parts.append(rightParts.begin(), rightParts.end());
            in = concat(b, loc, parts, axis);
        }
        return in;
    }
};

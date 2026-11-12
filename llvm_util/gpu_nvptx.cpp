// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// NVPTX: the NVPTX target's LLVM IR (llc -mtriple=nvptx64-nvidia-cuda):
// ptx_kernel kernels, address spaces global 1, shared 3, const 4 and local 5,
// and llvm.nvvm.* intrinsics. PTX itself (what the backend prints) is lifted
// to this form by scripts/ptx-tv, after the PTX ISA's semantics
// (scripts/ptx-tv/ptx.formal), which this target also gives the NVVM
// intrinsics that stand for one PTX instruction (ptx_formal.h).

#include "llvm_util/gpu_target.h"
#include "llvm_util/ptx_formal.h"
#include "util/config.h"
#include "llvm/ADT/StringSwitch.h"

namespace llvm_util {
namespace gpu {
namespace {

struct NVPTXLowering : GPULowering {
  using GPULowering::GPULowering;

  bool lowerCall(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    if (!Name.starts_with("llvm.nvvm."))
      return false;
    return lowerNVVM(B, CI, Name.drop_front(strlen("llvm.nvvm.")), R);
  }

  /// NVPTX's intrinsics: the special registers (PTX ISA, "Special
  /// Registers": %tid, %ntid, %ctaid, %nctaid; a warp is 32 consecutive
  /// threads of the CTA, %laneid the thread's index in it), barriers and
  /// memory fences. Others stay unknown calls.
  bool lowerNVVM(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    Type *Ty = CI.getType();
    if (Name.consume_front("read.ptx.sreg.")) {
      StringRef Role = StringSwitch<StringRef>(Name.drop_back(2))
        .Case("tid", "thread_position_in_threadgroup")
        .Case("ntid", "threads_per_threadgroup")
        .Case("ctaid", "threadgroup_position_in_grid")
        .Case("nctaid", "threadgroups_per_grid")
        .Default("");
      if (!Role.empty() && Name.size() >= 2 && Name[Name.size() - 2] == '.' &&
          Ty->isIntegerTy()) {
        size_t C = StringRef("xyz").find(Name.back());
        if (C != StringRef::npos) {
          R = B.CreateZExtOrTrunc(systemValue(B, Role)[C], Ty);
          return true;
        }
      }
      if (Name == "laneid" && Ty->isIntegerTy()) {
        R = B.CreateZExtOrTrunc(systemValue(B, "thread_index_in_simdgroup")[0],
                                Ty);
        return true;
      }
      if (Name == "warpsize" && Ty->isIntegerTy()) {
        R = ConstantInt::get(Ty, 32);
        return true;
      }
      return false;
    }
    // bar.sync / barrier.sync of the whole CTA (barrier0, barrier.sync(id),
    // barrier.cta.sync[.aligned].all(id)): every thread waits, and memory is
    // ordered across it.
    if (Name == "barrier0" || Name == "barrier.sync" ||
        Name == "bar.sync" ||
        (Name.starts_with("barrier.cta.sync") && Name.ends_with(".all"))) {
      barrier(B, true);
      return true;
    }
    if (Name.starts_with("membar.") || Name.starts_with("fence.sc.") ||
        Name.starts_with("fence.acq_rel.")) {
      fence(B);
      return true;
    }
    return lowerPTXInstruction(B, CI, Name, R);
  }

  /// An intrinsic that stands for one PTX instruction (llvm.nvvm.f2i.rz:
  /// cvt.rzi.s32.f32, llvm.nvvm.prmt: prmt.b32, ...): the instruction's
  /// meaning in the PTX formalization (scripts/ptx-tv/ptx.formal, built in),
  /// the same that ptx-tv lifts PTX by.
  bool lowerPTXInstruction(IRBuilder<> &B, CallInst &CI, StringRef Name,
                           Value *&R) {
    string Op = ptx_instruction_of_nvvm(Name);
    Type *Ty = CI.getType();
    if (Op.empty() || !(Ty->isIntegerTy() || Ty->isFloatingPointTy()))
      return false;
    SmallVector<Value *, 4> Bits;
    for (Value *A : CI.args()) {
      Type *T = A->getType();
      if (T->isFloatingPointTy())
        A = B.CreateBitCast(A, B.getIntNTy(T->getPrimitiveSizeInBits()));
      else if (!T->isIntegerTy())
        return false;
      Bits.push_back(A);
    }
    string Err;
    Value *V = PTXFormal::builtin().lift(B, Op, Bits, Err);
    if (!Err.empty())
      util::config::dbg() << "ERROR: " << Err << '\n';
    if (!V)
      return false;
    unsigned W = V->getType()->getIntegerBitWidth();
    if (Ty->isFloatingPointTy()) {
      if (Ty->getPrimitiveSizeInBits() != W)
        return false;
      R = B.CreateBitCast(V, Ty);
    } else {
      R = B.CreateZExtOrTrunc(V, Ty);
    }
    return true;
  }

};

class NVPTXTarget : public GPUTarget {
public:
  const char *name() const override { return "nvptx"; }

  bool isModule(const Module &M) const override {
    if (StringRef(M.getTargetTriple().str()).starts_with("nvptx"))
      return true;
    for (const Function &F : M)
      if (F.getName().starts_with("llvm.nvvm.") ||
          F.getCallingConv() == CallingConv::PTX_Kernel)
        return true;
    return false;
  }

  bool isKernel(const Function &Fn) const override {
    return Fn.getCallingConv() == CallingConv::PTX_Kernel;
  }

  /// PTX passes a kernel's values in the parameter space, as LLVM IR does.
  bool scalarsInConstantBuffers() const override { return false; }

  bool lowerCall(GPUBuilder &G, IRBuilder<> &B, CallInst &CI, StringRef Name,
                 Value *&R) const override {
    return NVPTXLowering(G).lowerCall(B, CI, Name, R);
  }
};

} // namespace

const GPUTarget &nvptx_target() {
  static const NVPTXTarget T;
  return T;
}

} // namespace gpu
} // namespace llvm_util

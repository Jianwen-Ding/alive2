// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// Upstream GPU IR: what a front end hands a GPU backend, LLVM IR with the
// OpenCL / SPIR-V work-item builtins (get_global_id, barrier, ...), CUDA's
// high multiplications and kernels marked as such.

#include "llvm_util/gpu_target.h"
#include "llvm/ADT/StringSwitch.h"

namespace llvm_util {
namespace gpu {

// OpenCL / SPIR-V work-item builtins, as upstream GPU IR calls them.
StringRef opencl_builtin(StringRef Name) {
  return StringSwitch<StringRef>(Name)
    .Case("_Z12get_local_idj", "get_local_id")
    .Case("_Z12get_group_idj", "get_group_id")
    .Case("_Z13get_global_idj", "get_global_id")
    .Case("_Z14get_local_sizej", "get_local_size")
    .Case("_Z14get_num_groupsj", "get_num_groups")
    .Case("_Z15get_global_sizej", "get_global_size")
    .Case("_Z16get_sub_group_idv", "get_sub_group_id")
    .Case("_Z18get_num_sub_groupsv", "get_num_sub_groups")
    .Case("_Z18get_sub_group_sizev", "get_sub_group_size")
    .Case("_Z22get_sub_group_local_idv", "get_sub_group_local_id")
    .Case("_Z7barrierj", "barrier")
    .Default("");
}


namespace {

struct UpstreamLowering : GPULowering {
  using GPULowering::GPULowering;

  bool lowerCall(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    auto Arg = [&](unsigned I) { return CI.getArgOperand(I); };
    unsigned N = CI.arg_size();
    Type *Ty = CI.getType();

    // Upstream: OpenCL work-item builtins and barriers, CUDA's high
    // multiplications.
    if (StringRef CL = opencl_builtin(Name); !CL.empty()) {
      if (CL == "barrier") {
        auto *Flags = N ? dyn_cast<ConstantInt>(Arg(0)) : nullptr;
        barrier(B, !Flags || !Flags->isZero());
        return true;
      }
      if (CL == "get_sub_group_id")
        R = systemValue(B, "simdgroup_index_in_threadgroup", Ty);
      else if (CL == "get_num_sub_groups")
        R = systemValue(B, "simdgroups_per_threadgroup", Ty);
      else if (CL == "get_sub_group_local_id")
        R = systemValue(B, "thread_index_in_simdgroup", Ty);
      else if (CL == "get_sub_group_size") {
        // The threads of this SIMD-group: 32, or fewer in the last one.
        Value *Left = B.CreateSub(threadsPerThreadgroup(B),
                                  B.CreateAnd(threadIndex(B), ~31u));
        R = B.CreateZExtOrTrunc(
            B.CreateIntrinsic(Intrinsic::umin, {I32}, {Left, B.getInt32(32)}),
            Ty);
      } else if (N == 1 && Ty->isIntegerTy()) {
        Value *V = workItem(B, CL, Arg(0));
        R = V ? B.CreateZExtOrTrunc(V, Ty) : nullptr;
      }
      return R != nullptr;
    }
    if ((Name == "__mulhi" || Name == "__umulhi") && N == 2 &&
        Ty->isIntegerTy()) {
      bool Signed = Name == "__mulhi";
      unsigned Bits = Ty->getIntegerBitWidth();
      Type *Wide = B.getIntNTy(2 * Bits);
      Value *P = B.CreateMul(B.CreateIntCast(Arg(0), Wide, Signed),
                             B.CreateIntCast(Arg(1), Wide, Signed));
      R = B.CreateTrunc(B.CreateLShr(P, Bits), Ty);
      return true;
    }

    return false;
  }
};

class UpstreamTarget : public GPUTarget {
public:
  const char *name() const override { return "upstream"; }

  bool isModule(const Module &M) const override {
    for (const Function &F : M)
      if (!opencl_builtin(F.getName()).empty() ||
          F.getCallingConv() == CallingConv::SPIR_KERNEL)
        return true;
    return false;
  }

  bool isKernel(const Function &Fn) const override {
    return Fn.getCallingConv() == CallingConv::SPIR_KERNEL ||
           Fn.hasFnAttribute("kernel") || Fn.hasFnAttribute("gpu.kernel");
  }

  bool lowerCall(GPUBuilder &G, IRBuilder<> &B, CallInst &CI, StringRef Name,
                 Value *&R) const override {
    return UpstreamLowering(G).lowerCall(B, CI, Name, R);
  }
};

} // namespace

const GPUTarget &upstream_target() {
  static const UpstreamTarget T;
  return T;
}

} // namespace gpu
} // namespace llvm_util

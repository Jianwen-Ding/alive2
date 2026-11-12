// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// AGX IR: the module after the AGX backend's agx-lower-air
// (opt -mtriple=agx -passes=agx-lower-air): agx_kernels, AGX's address
// spaces and llvm.agx.* intrinsics.

#include "llvm_util/gpu_target.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Attributes.h"

namespace llvm_util {
namespace gpu {
namespace {

// AGX's kernel calling convention (CallingConv::AGX_KERNEL), by number, so
// that this file builds against an LLVM without the AGX target.
constexpr unsigned AGXKernelCC = 128;

struct AGXLowering : GPULowering {
  using GPULowering::GPULowering;

  bool lowerCall(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    if (!Name.starts_with("llvm.agx."))
      return false;
    return lowerAGX(B, CI, Name.drop_front(strlen("llvm.agx.")), R);
  }

  bool lowerAGX(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    // System values: the special registers.
    SmallVector<StringRef, 2> Parts;
    Name.split(Parts, '.');
    string Joined = Name.str();
    replace(Joined, '.', '_');
    StringRef J(Joined);
    for (StringRef Role : {"thread_position_in_grid",
                           "threadgroup_position_in_grid",
                           "thread_position_in_threadgroup",
                           "threads_per_threadgroup", "threadgroups_per_grid"})
      for (unsigned C = 0; C < 3; ++C)
        if (J == (Role + Twine("_") + Twine("xyz"[C])).str()) {
          R = B.CreateZExtOrTrunc(systemValue(B, Role)[C], CI.getType());
          return true;
        }
    for (StringRef Role : {"thread_index_in_threadgroup",
                           "thread_index_in_simdgroup",
                           "simdgroup_index_in_threadgroup"})
      if (J == Role) {
        R = B.CreateZExtOrTrunc(systemValue(B, Role)[0], CI.getType());
        return true;
      }
    if (Name == "threadgroup.barrier") {
      convergent(B, "threadgroup.barrier", B.getVoidTy(), {});
      return true;
    }
    // AIR's fast_rsqrt, as AGX keeps it.
    if (Name.starts_with("rsqrt.") && CI.arg_size() == 1) {
      R = B.CreateCall(decl(("air.fast_" + Name).str(), CI.getFunctionType(),
                            MemoryEffects::none(), true),
                       {CI.getArgOperand(0)});
      return true;
    }
    // The SIMD-group functions.
    if (Name.starts_with("simd.") || Name.starts_with("quad.")) {
      StringRef Base = Name;
      if (CI.getCalledFunction()->isIntrinsic()) {
        StringRef IntrBase =
            Intrinsic::getBaseName(CI.getCalledFunction()->getIntrinsicID());
        Base = IntrBase.drop_front(strlen("llvm.agx."));
      }
      bool Quad = Base.starts_with("quad.");
      StringRef Mode = Base.drop_front(5);
      if (Mode.consume_front("shuffle") && CI.arg_size() == 2 &&
          (Mode.empty() || Mode == ".xor" || Mode == ".up" ||
           Mode == ".down")) {
        Mode.consume_front(".");
        R = shuffle(B, Quad, Mode, CI.getArgOperand(0), CI.getArgOperand(1));
        return true;
      }
      // Reductions and scans: AGX::SIMDOp names the operation.
      if ((Base.ends_with(".reduce") || Base.ends_with(".prefix")) &&
          CI.arg_size() == 2)
        if (auto *Op = dyn_cast<ConstantInt>(CI.getArgOperand(1))) {
          StringRef Red;
          switch (Op->getZExtValue()) {
          case 0: Red = "and"; break;
          case 1: Red = "fadd"; break;
          case 2: Red = "or"; break;
          case 3: Red = "fmul"; break;
          case 4: Red = "xor"; break;
          case 5: Red = "fmin"; break;
          case 7: Red = "fmax"; break;
          case 16: Red = "iadd"; break;
          case 20: Red = "smin"; break;
          case 22: Red = "smax"; break;
          case 28: Red = "umin"; break;
          case 30: Red = "umax"; break;
          }
          if (!Red.empty()) {
            R = convergent(B, (Base + "." + Red).str(), CI.getType(),
                           {CI.getArgOperand(0)});
            return true;
          }
        }
      SmallVector<Value *, 3> Args(CI.args());
      R = convergent(B, Base, CI.getType(), Args);
      if (CI.getType()->isVoidTy())
        R = nullptr;
      return true;
    }
    return false;
  }

};

class AGXTarget : public GPUTarget {
public:
  const char *name() const override { return "agx"; }

  bool isModule(const Module &M) const override {
    if (StringRef(M.getTargetTriple().str()).starts_with("agx"))
      return true;
    for (const Function &F : M)
      if (F.getName().starts_with("llvm.agx.") ||
          F.getCallingConv() == AGXKernelCC)
        return true;
    return false;
  }

  bool isKernel(const Function &Fn) const override {
    return Fn.getCallingConv() == AGXKernelCC;
  }

  optional<unsigned> privateAddressSpace() const override { return 5; }

  /// AGX's dynamic threadgroup memory: a zero-sized threadgroup variable
  /// without a definition.
  static bool isDynamicThreadgroupMemory(const GlobalVariable &G) {
    auto *AT = dyn_cast<ArrayType>(G.getValueType());
    return G.getAddressSpace() == ThreadgroupAS && G.isDeclaration() &&
           AT && AT->getNumElements() == 0;
  }

  /// The dynamic threadgroup memory Fn uses.
  void dynamicThreadgroupMemory(
      Function &Fn, SmallVectorImpl<GlobalVariable *> &Out) const override {
    GlobalVariable *DynTG = nullptr;
    for (GlobalVariable &G : Fn.getParent()->globals())
      if (isDynamicThreadgroupMemory(G) &&
          any_of(G.users(), [&](User *U) {
            auto *I = dyn_cast<Instruction>(U);
            return I && I->getFunction() == &Fn;
          }))
        DynTG = &G;
    if (DynTG)
      Out.push_back(DynTG);
  }

  /// What AGX says about its kernels' buffers comes from the ABI.
  void dropABIAttributes(const Function &F, Function &NewF) const override {
    if (F.getCallingConv() != AGXKernelCC)
      return;
    for (Argument &A : NewF.args())
      for (auto K : {Attribute::NoAlias, Attribute::NonNull,
                     Attribute::Dereferenceable,
                     Attribute::DereferenceableOrNull, Attribute::Alignment,
                     Attribute::NoUndef, Attribute::ReadOnly})
        A.removeAttr(K);
  }

  bool lowerCall(GPUBuilder &G, IRBuilder<> &B, CallInst &CI, StringRef Name,
                 Value *&R) const override {
    return AGXLowering(G).lowerCall(B, CI, Name, R);
  }
};

} // namespace

const GPUTarget &agx_target() {
  static const AGXTarget T;
  return T;
}

} // namespace gpu
} // namespace llvm_util

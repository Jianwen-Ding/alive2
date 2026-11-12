// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.
#pragma once

// The parts of alive2's GPU semantics (gpu.h) that are common to every GPU
// target, and the interface a target implements.
//
// The canonical form (gpu.h) is the same for all targets: one address space,
// system values computed from twelve primitive ones, barriers and SIMD-group
// (warp) functions as convergent calls, a sequential memory model. What
// differs between targets is how their code says these things, which a
// GPUTarget translates:
//
//  * which modules and functions are its code, and which are kernels;
//  * how its kernels take their parameters (system values among them, scalars
//    in constant buffers, dynamic threadgroup memory);
//  * what its intrinsics and library functions mean, in terms of the
//    builder's primitives (GPUBuilder).
//
// The targets: upstream GPU IR (OpenCL / SPIR-V builtins, gpu_upstream.cpp),
// AIR (gpu_air.cpp), AGX (gpu_agx.cpp) and NVPTX (gpu_nvptx.cpp).
//
// This header is internal to llvm_util's GPU files.

#include "llvm_util/gpu.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include <optional>
#include <string>
#include <utility>

namespace llvm_util {
namespace gpu {

using namespace llvm;
using namespace std;

/// Threadgroup memory (CUDA's shared memory) is address space 3 in AIR, AGX
/// and NVPTX alike.
enum : unsigned { ThreadgroupAS = 3 };

/// The primitives of the canonical form, which every target's code is
/// translated into: declarations, system values, barriers, fences and
/// SIMD-group functions.
struct GPUBuilder {
  GPUCanonicalFunction &Out;
  Module &M;
  LLVMContext &Ctx;
  // The twelve primitive system values, by name ("threads_per_threadgroup.x").
  StringMap<Value *> SR;
  IntegerType *I32;

  GPUBuilder(GPUCanonicalFunction &Out, Module &M)
      : Out(Out), M(M), Ctx(M.getContext()), I32(Type::getInt32Ty(Ctx)) {}


  /// A function of the canonical form, made the first time it is asked for.
  Function *decl(StringRef Name, FunctionType *FT,
                 optional<MemoryEffects> ME, bool Speculatable = false) {
    if (Function *Fn = M.getFunction(Name)) {
      if (Fn->getFunctionType() == FT)
        return Fn;
      // The same name with another type (an AIR function called at two
      // types): a function of its own.
      return decl((Name + ".").str() + to_string((uintptr_t)FT), FT, ME,
                  Speculatable);
    }
    Function *Fn = Function::Create(FT, GlobalValue::ExternalLinkage, Name, M);
    Fn->setDoesNotThrow();
    Fn->setWillReturn();
    Fn->addFnAttr(Attribute::NoFree);
    Fn->addFnAttr(Attribute::NoSync);
    if (ME)
      Fn->setMemoryEffects(*ME);
    if (Speculatable)
      Fn->addFnAttr(Attribute::Speculatable);
    Out.created.push_back(Fn);
    return Fn;
  }

  /// Another thread's view: may change any memory others can see.
  void fence(IRBuilder<> &B) {
    B.CreateCall(decl("alive.gpu.fence", FunctionType::get(B.getVoidTy(), false),
                      nullopt));
  }

  /// A SIMD-group function or barrier: keeps its number and order.
  Value *convergent(IRBuilder<> &B, StringRef Name, Type *Ret,
                    ArrayRef<Value *> Args) {
    SmallVector<Type *, 4> Tys;
    for (Value *A : Args)
      Tys.push_back(A->getType());
    string Full = ("alive.gpu." + Name).str();
    if (!Ret->isVoidTy() || !Args.empty()) {
      // One function per type, as an overloaded intrinsic.
      raw_string_ostream OS(Full);
      if (!Ret->isVoidTy())
        OS << '.' << *Ret;
    }
    return B.CreateCall(
        decl(Full, FunctionType::get(Ret, Tys, false),
             MemoryEffects::inaccessibleMemOnly()),
        Args);
  }

  /// A shuffle of the SIMD-group (quad-group): mode "" reads lane Arg, "xor"
  /// lane ^ Arg, "up" lane - Arg and "down" lane + Arg, as MSL defines them:
  /// a thread whose source lane would be outside of the group (the lowest
  /// Arg lanes for up, the highest for down) keeps its value. All are reads
  /// of a lane, one primitive.
  Value *shuffle(IRBuilder<> &B, bool Quad, StringRef Mode, Value *X,
                 Value *Arg) {
    Value *Self = systemValue(B, "thread_index_in_simdgroup")[0];
    unsigned N = Quad ? 4 : 32;
    if (Quad)
      Self = B.CreateAnd(Self, 3);
    Arg = B.CreateZExtOrTrunc(Arg, I32);
    Value *Src = Arg;
    if (Mode == "xor")
      Src = B.CreateXor(Self, Arg);
    else if (Mode == "up")
      Src = B.CreateSelect(B.CreateICmpUGE(Self, Arg), B.CreateSub(Self, Arg),
                           Self);
    else if (Mode == "down")
      Src = B.CreateSelect(
          B.CreateICmpULT(Arg, B.getInt32(N)),
          B.CreateSelect(B.CreateICmpULT(B.CreateAdd(Self, Arg),
                                         B.getInt32(N)),
                         B.CreateAdd(Self, Arg), Self),
          Self);
    return convergent(B, Quad ? "quad.shuffle" : "simd.shuffle", X->getType(),
                      {X, Src});
  }

  void barrier(IRBuilder<> &B, bool Memory) {
    if (Memory)
      fence(B);
    convergent(B, "threadgroup.barrier", B.getVoidTy(), {});
    if (Memory)
      fence(B);
  }

  // -------------------------------------------------------------------------
  // System values.

  Value *sr(StringRef Name) { return SR.lookup(Name); }

  /// The twelve primitive values at the function's entry, and what every
  /// launch guarantees about them that is linear (so that it costs the
  /// solver little): 1 to 1024 threads in each dimension of a threadgroup,
  /// and threadgroup and grid positions within their bounds. (That a
  /// threadgroup has at most 1024 threads in all, and a grid fewer than 2^32
  /// in each dimension, are left out: products.)
  void makePrimitives(IRBuilder<> &B) {
    static const char *const Names[] = {
        "thread_position_in_threadgroup", "threadgroup_position_in_grid",
        "threads_per_threadgroup", "threadgroups_per_grid"};
    for (const char *N : Names)
      for (char C : {'x', 'y', 'z'}) {
        string Name = string(N) + "." + C;
        SR[Name] = B.CreateCall(
            decl("alive.gpu.sr." + Name, FunctionType::get(I32, false),
                 MemoryEffects::none(), true),
            {}, Name);
      }
    Function *Assume = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::assume);
    auto assume = [&](Value *C) { B.CreateCall(Assume, {C}); };
    for (char C : {'x', 'y', 'z'}) {
      string S(1, C);
      Value *Tid = sr("thread_position_in_threadgroup." + S);
      Value *Tpt = sr("threads_per_threadgroup." + S);
      Value *Pos = sr("threadgroup_position_in_grid." + S);
      Value *Num = sr("threadgroups_per_grid." + S);
      assume(B.CreateICmpNE(Tpt, B.getInt32(0)));
      assume(B.CreateICmpULE(Tpt, B.getInt32(1024)));
      assume(B.CreateICmpULT(Tid, Tpt));
      assume(B.CreateICmpULT(Pos, Num));
    }
  }

  Value *threadIndex(IRBuilder<> &B) {
    // Linear: x + X * (y + Y * z).
    Value *Y = B.CreateAdd(sr("thread_position_in_threadgroup.y"),
                           B.CreateMul(sr("threads_per_threadgroup.y"),
                                       sr("thread_position_in_threadgroup.z")));
    return B.CreateAdd(sr("thread_position_in_threadgroup.x"),
                       B.CreateMul(sr("threads_per_threadgroup.x"), Y));
  }

  Value *threadsPerThreadgroup(IRBuilder<> &B) {
    return B.CreateMul(B.CreateMul(sr("threads_per_threadgroup.x"),
                                   sr("threads_per_threadgroup.y")),
                       sr("threads_per_threadgroup.z"));
  }

  /// The components of the system value Role, as i32 (one for a scalar one),
  /// or none for a role AIR does not have.
  SmallVector<Value *, 3> systemValue(IRBuilder<> &B, StringRef Role) {
    SmallVector<Value *, 3> V;
    auto xyz = [&](StringRef N) {
      for (char C : {'x', 'y', 'z'})
        V.push_back(sr((N + "." + Twine(C)).str()));
    };
    if (Role == "thread_position_in_threadgroup" ||
        Role == "threadgroup_position_in_grid" ||
        Role == "threads_per_threadgroup" || Role == "threadgroups_per_grid")
      xyz(Role);
    else if (Role == "thread_position_in_grid")
      for (char C : {'x', 'y', 'z'}) {
        string S(1, C);
        V.push_back(B.CreateAdd(
            B.CreateMul(sr("threadgroup_position_in_grid." + S),
                        sr("threads_per_threadgroup." + S)),
            sr("thread_position_in_threadgroup." + S)));
      }
    else if (Role == "threads_per_grid")
      for (char C : {'x', 'y', 'z'}) {
        string S(1, C);
        V.push_back(B.CreateMul(sr("threadgroups_per_grid." + S),
                                sr("threads_per_threadgroup." + S)));
      }
    else if (Role == "thread_index_in_threadgroup")
      V.push_back(threadIndex(B));
    // SIMD-groups are 32 consecutive threads of the threadgroup.
    else if (Role == "thread_index_in_simdgroup")
      V.push_back(B.CreateAnd(threadIndex(B), 31));
    else if (Role == "simdgroup_index_in_threadgroup")
      V.push_back(B.CreateLShr(threadIndex(B), 5));
    else if (Role == "threads_per_simdgroup")
      V.push_back(B.getInt32(32));
    else if (Role == "simdgroups_per_threadgroup")
      V.push_back(B.CreateLShr(B.CreateAdd(threadsPerThreadgroup(B),
                                           B.getInt32(31)), 5));
    return V;
  }

  /// The system value Role as a value of type Ty: a vector, array or
  /// structure of its components, or its first one; integers of any width.
  Value *systemValue(IRBuilder<> &B, StringRef Role, Type *Ty) {
    SmallVector<Value *, 3> C = systemValue(B, Role);
    if (C.empty())
      return nullptr;
    auto component = [&](unsigned I, Type *T) -> Value * {
      if (!T->isIntegerTy())
        return nullptr;
      return I < C.size() ? B.CreateZExtOrTrunc(C[I], T)
                          : ConstantInt::get(T, 0);
    };
    if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
      Value *V = PoisonValue::get(VT);
      for (unsigned I = 0; I < VT->getNumElements(); ++I) {
        Value *E = component(I, VT->getElementType());
        if (!E)
          return nullptr;
        V = B.CreateInsertElement(V, E, I);
      }
      return V;
    }
    if (isa<ArrayType>(Ty) || isa<StructType>(Ty)) {
      Value *V = PoisonValue::get(Ty);
      unsigned N = isa<ArrayType>(Ty) ? Ty->getArrayNumElements()
                                      : Ty->getStructNumElements();
      for (unsigned I = 0; I < N; ++I) {
        Type *ET = isa<ArrayType>(Ty) ? Ty->getArrayElementType()
                                      : Ty->getStructElementType(I);
        Value *E = component(I, ET);
        if (!E)
          return nullptr;
        V = B.CreateInsertValue(V, E, I);
      }
      return V;
    }
    return component(0, Ty);
  }

  /// OpenCL's work-item function Fn for dimension Dim: 0 (ids) or 1 (sizes)
  /// beyond the three dimensions.
  Value *workItem(IRBuilder<> &B, StringRef Fn, Value *Dim) {
    StringRef Role = StringSwitch<StringRef>(Fn)
      .Case("get_local_id", "thread_position_in_threadgroup")
      .Case("get_group_id", "threadgroup_position_in_grid")
      .Case("get_global_id", "thread_position_in_grid")
      .Case("get_local_size", "threads_per_threadgroup")
      .Case("get_num_groups", "threadgroups_per_grid")
      .Case("get_global_size", "threads_per_grid")
      .Default("");
    if (Role.empty())
      return nullptr;
    SmallVector<Value *, 3> C = systemValue(B, Role);
    Value *Outside = B.getInt32(Fn.ends_with("_id") ? 0 : 1);
    Value *D = B.CreateZExtOrTrunc(Dim, I32);
    Value *R = Outside;
    for (int I = 2; I >= 0; --I)
      R = B.CreateSelect(B.CreateICmpEQ(D, B.getInt32(I)), C[I], R);
    return R;
  }

};

/// The base of a target's call lowering: the builder's primitives, by the
/// same names, so that a target's code reads as if it were the builder's.
struct GPULowering {
  GPUBuilder &G;
  Module &M;
  LLVMContext &Ctx;
  IntegerType *I32;

  explicit GPULowering(GPUBuilder &G)
      : G(G), M(G.M), Ctx(G.Ctx), I32(G.I32) {}

  Function *decl(StringRef Name, FunctionType *FT, optional<MemoryEffects> ME,
                 bool Speculatable = false) {
    return G.decl(Name, FT, ME, Speculatable);
  }
  void fence(IRBuilder<> &B) { G.fence(B); }
  Value *convergent(IRBuilder<> &B, StringRef Name, Type *Ret,
                    ArrayRef<Value *> Args) {
    return G.convergent(B, Name, Ret, Args);
  }
  Value *shuffle(IRBuilder<> &B, bool Quad, StringRef Mode, Value *X,
                 Value *Arg) {
    return G.shuffle(B, Quad, Mode, X, Arg);
  }
  void barrier(IRBuilder<> &B, bool Memory) { G.barrier(B, Memory); }
  Value *sr(StringRef Name) { return G.sr(Name); }
  SmallVector<Value *, 3> systemValue(IRBuilder<> &B, StringRef Role) {
    return G.systemValue(B, Role);
  }
  Value *systemValue(IRBuilder<> &B, StringRef Role, Type *Ty) {
    return G.systemValue(B, Role, Ty);
  }
  Value *threadIndex(IRBuilder<> &B) { return G.threadIndex(B); }
  Value *threadsPerThreadgroup(IRBuilder<> &B) {
    return G.threadsPerThreadgroup(B);
  }
  Value *workItem(IRBuilder<> &B, StringRef Fn, Value *Dim) {
    return G.workItem(B, Fn, Dim);
  }
};

/// A GPU target: how its code says what the canonical form says.
class GPUTarget {
public:
  virtual ~GPUTarget() = default;
  virtual const char *name() const = 0;

  /// Whether M is (or has) code of this target: its triple, or functions it
  /// declares or defines (kernels, the target's intrinsics).
  virtual bool isModule(const Module &M) const = 0;

  /// Whether Fn is one of this target's kernels.
  virtual bool isKernel(const Function &Fn) const { return false; }

  /// The parameters of kernel Fn that are system values: Roles[i] is the
  /// role of parameter i ("thread_position_in_grid"), left empty for others.
  virtual void kernelRoles(const Function &Fn,
                           SmallVectorImpl<string> &Roles) const {}

  /// Whether the canonical form of this target's kernels takes a scalar
  /// parameter in a constant buffer (a pointer to it), as AIR passes it.
  virtual bool scalarsInConstantBuffers() const { return true; }

  /// The variable that stands for the launch's dynamic threadgroup memory,
  /// if Fn uses one (it becomes a trailing parameter).
  virtual GlobalVariable *dynamicThreadgroupMemory(Function &Fn) const {
    return nullptr;
  }

  /// Drop from the canonical function NewF of F the parameter attributes
  /// that come from the target's ABI, which the canonical form does not
  /// have.
  virtual void dropABIAttributes(const Function &F, Function &NewF) const {}

  /// Lower a call of one of the target's functions (Name, the callee's
  /// name): true if it did, with R the call's value (none: poison).
  virtual bool lowerCall(GPUBuilder &B, IRBuilder<> &IRB, CallInst &CI,
                         StringRef Name, Value *&R) const = 0;
};

const GPUTarget &upstream_target();
const GPUTarget &air_target();
const GPUTarget &agx_target();
const GPUTarget &nvptx_target();

/// Every target, in the order in which they lower calls.
ArrayRef<const GPUTarget *> targets();

/// The target whose conventions M's kernels follow.
const GPUTarget &module_target(const Module &M);

} // namespace gpu
} // namespace llvm_util

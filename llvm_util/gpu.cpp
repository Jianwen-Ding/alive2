// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "llvm_util/gpu.h"
#include "util/config.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <optional>

using namespace llvm;
using namespace std;
using namespace util;

namespace {

// AIR's address spaces (llvm/lib/Target/AIR/AIRAddressSpaces.h).
// Threadgroup memory is 3 in AIR's and AGX's.
enum : unsigned { AIRThreadgroup = 3 };
// AGX's kernel calling convention (CallingConv::AGX_KERNEL), by number, so
// that this file builds against an LLVM without the AGX target.
constexpr unsigned AGXKernelCC = 128;

bool is_agx_module(const Module &M) {
  return StringRef(M.getTargetTriple().str()).starts_with("agx");
}

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

// The system values of AIR (!air.kernel roles and air.<role> functions).
bool is_system_value(StringRef Role) {
  return StringSwitch<bool>(Role)
    .Cases({"thread_position_in_grid", "threadgroup_position_in_grid",
            "thread_position_in_threadgroup", "threads_per_threadgroup",
            "threadgroups_per_grid", "threads_per_grid",
            "thread_index_in_threadgroup", "thread_index_in_simdgroup",
            "simdgroup_index_in_threadgroup", "threads_per_simdgroup",
            "simdgroups_per_threadgroup"}, true)
    .Default(false);
}

// The math functions of AIR, the inverse of the AIR target's
// AIRIntrinsicMappings.td: "air." ["fast_"] name "." ["s."|"u."] <type>.
struct MathFn {
  const char *AIR;
  unsigned Signedness; // 0: none, 1: ".s", 2: ".u"
  const char *LLVM;
  bool HasFast;
};
const MathFn math_fns[] = {
  {"fabs", 0, "llvm.fabs", true},   {"floor", 0, "llvm.floor", true},
  {"ceil", 0, "llvm.ceil", true},   {"trunc", 0, "llvm.trunc", true},
  {"rint", 0, "llvm.rint", true},   {"round", 0, "llvm.round", true},
  {"sqrt", 0, "llvm.sqrt", true},   {"exp", 0, "llvm.exp", true},
  {"exp2", 0, "llvm.exp2", true},   {"exp10", 0, "llvm.exp10", true},
  {"log", 0, "llvm.log", true},     {"log2", 0, "llvm.log2", true},
  {"log10", 0, "llvm.log10", true}, {"sin", 0, "llvm.sin", true},
  {"cos", 0, "llvm.cos", true},     {"tan", 0, "llvm.tan", true},
  {"asin", 0, "llvm.asin", true},   {"acos", 0, "llvm.acos", true},
  {"atan", 0, "llvm.atan", true},   {"atan2", 0, "llvm.atan2", true},
  {"sinh", 0, "llvm.sinh", true},   {"cosh", 0, "llvm.cosh", true},
  {"tanh", 0, "llvm.tanh", true},   {"asinh", 0, "llvm.asinh", true},
  {"acosh", 0, "llvm.acosh", true}, {"atanh", 0, "llvm.atanh", true},
  {"pow", 0, "llvm.pow", true},
  // Metal's fmin and fmax return the other operand of a NaN: minnum, maxnum.
  {"fmin", 0, "llvm.minnum", true}, {"fmax", 0, "llvm.maxnum", true},
  {"fma", 0, "llvm.fma", false},
  {"min", 1, "llvm.smin", false},   {"max", 1, "llvm.smax", false},
  {"min", 2, "llvm.umin", false},   {"max", 2, "llvm.umax", false},
  {"abs", 1, "llvm.abs", false},    {"popcount", 0, "llvm.ctpop", false},
  {"clz", 0, "llvm.ctlz", false},   {"ctz", 0, "llvm.cttz", false},
  {"reverse_bits", 0, "llvm.bitreverse", false},
};

// MSL's memory_order, as AIR's atomic functions take it.
AtomicOrdering msl_ordering(Value *Order) {
  auto *C = dyn_cast<ConstantInt>(Order);
  switch (C ? C->getZExtValue() : 5) {
  case 0: return AtomicOrdering::Monotonic;
  case 1:
  case 2: return AtomicOrdering::Acquire;
  case 3: return AtomicOrdering::Release;
  case 4: return AtomicOrdering::AcquireRelease;
  default: return AtomicOrdering::SequentiallyConsistent;
  }
}

/// Pointers of every address space as alive2's (0), in every type. alive2
/// has one address space: the canonical form keeps no address spaces, so
/// that memories of different address spaces may alias there (which only
/// keeps alive2 from proving what depends on their being disjoint).
class FlatTypes : public ValueMapTypeRemapper {
  DenseMap<Type *, Type *> Cache;

public:
  static unsigned addrSpace(unsigned) { return 0; }

  Type *remapType(Type *T) override {
    if (auto It = Cache.find(T); It != Cache.end())
      return It->second;
    Type *R = T;
    if (auto *PT = dyn_cast<PointerType>(T)) {
      R = PointerType::get(T->getContext(), addrSpace(PT->getAddressSpace()));
    } else if (auto *VT = dyn_cast<VectorType>(T)) {
      R = VectorType::get(remapType(VT->getElementType()),
                          VT->getElementCount());
    } else if (auto *AT = dyn_cast<ArrayType>(T)) {
      R = ArrayType::get(remapType(AT->getElementType()),
                         AT->getNumElements());
    } else if (auto *FT = dyn_cast<FunctionType>(T)) {
      SmallVector<Type *, 8> Params;
      for (Type *P : FT->params())
        Params.push_back(remapType(P));
      R = FunctionType::get(remapType(FT->getReturnType()), Params,
                            FT->isVarArg());
    } else if (auto *ST = dyn_cast<StructType>(T); ST && !ST->isOpaque()) {
      SmallVector<Type *, 8> Elts;
      for (Type *E : ST->elements())
        Elts.push_back(remapType(E));
      if (!equal(Elts, ST->elements()))
        R = ST->isLiteral()
                ? StructType::get(T->getContext(), Elts, ST->isPacked())
                : StructType::create(T->getContext(), Elts,
                                     ST->getName().str() + ".air",
                                     ST->isPacked());
    }
    Cache[T] = R;
    return R;
  }
};

} // namespace

namespace llvm_util {

struct GPUCanonicalizer : ValueMaterializer {
  GPUCanonicalFunction &Out;
  Function &F;
  Module &M;
  LLVMContext &Ctx;
  bool AGX;
  FlatTypes Types;
  // The twelve primitive system values, by name ("threads_per_threadgroup.x").
  StringMap<Value *> SR;
  IntegerType *I32;

  GPUCanonicalizer(GPUCanonicalFunction &Out, Function &F)
      : Out(Out), F(F), M(*F.getParent()), Ctx(F.getContext()),
        AGX(is_agx_module(M)), I32(Type::getInt32Ty(Ctx)) {}

  // -------------------------------------------------------------------------
  // Declarations of the canonical form.

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
  /// of a lane, one primitive: a quad-group's lane j is the SIMD-group's
  /// lane (lane & ~3) + j, so its shuffles are the SIMD-group's reads too.
  Value *shuffle(IRBuilder<> &B, bool Quad, StringRef Mode, Value *X,
                 Value *Arg) {
    Value *Lane = systemValue(B, "thread_index_in_simdgroup")[0];
    Value *Self = Lane;
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
    // (a quad lane past the quad keeps its value, as the masked hardware
    // recordings of quad_shuffle and quad_shuffle_xor by 4 and 255 show)
    if (Quad)
      Src = B.CreateOr(B.CreateAnd(Lane, ~3u),
                       B.CreateSelect(B.CreateICmpULT(Src, B.getInt32(4)), Src,
                                      Self));
    return convergent(B, "simd.shuffle", X->getType(), {X, Src});
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

  // -------------------------------------------------------------------------
  // The canonical function.

  /// Parameter roles from !air.kernel: system values (their role), or none.
  SmallVector<string, 8> kernelRoles() {
    SmallVector<string, 8> R(F.arg_size());
    NamedMDNode *MD = M.getNamedMetadata("air.kernel");
    if (!MD)
      return R;
    for (MDNode *N : MD->operands()) {
      if (N->getNumOperands() < 3 ||
          mdconst::dyn_extract_or_null<Function>(N->getOperand(0)) != &F)
        continue;
      auto *Ps = dyn_cast<MDNode>(N->getOperand(2));
      if (!Ps)
        continue;
      for (const MDOperand &PO : Ps->operands()) {
        auto *P = dyn_cast<MDNode>(PO);
        if (!P || P->getNumOperands() < 2)
          continue;
        auto *Idx = mdconst::dyn_extract_or_null<ConstantInt>(P->getOperand(0));
        auto *Role = dyn_cast<MDString>(P->getOperand(1));
        if (!Idx || !Role || Idx->getZExtValue() >= F.arg_size())
          continue;
        StringRef S = Role->getString();
        if (S.consume_front("air.") && is_system_value(S))
          R[Idx->getZExtValue()] = S.str();
      }
    }
    return R;
  }

  /// AGX's dynamic threadgroup memory: a zero-sized threadgroup variable
  /// without a definition.
  static bool isDynamicThreadgroupMemory(const GlobalVariable &G) {
    auto *AT = dyn_cast<ArrayType>(G.getValueType());
    return G.getAddressSpace() == AIRThreadgroup && G.isDeclaration() &&
           AT && AT->getNumElements() == 0;
  }

  /// Values of F's module the canonical function cannot use as they are: a
  /// function or variable whose type has pointers of other address spaces.
  Value *materialize(Value *V) override {
    if (auto *Fn = dyn_cast<Function>(V)) {
      auto *FT = cast<FunctionType>(Types.remapType(Fn->getFunctionType()));
      if (FT == Fn->getFunctionType())
        return nullptr;
      if (Fn->isIntrinsic()) {
        SmallVector<Type *, 4> Overloads;
        if (Intrinsic::isSignatureValid(Fn->getIntrinsicID(), FT, Overloads))
          return Intrinsic::getOrInsertDeclaration(&M, Fn->getIntrinsicID(),
                                                   Overloads);
      }
      string Name = Fn->getName().str();
      Out.renamed.push_back({Fn, Name});
      Fn->setName(Name + ".alive2.as");
      Function *New = Function::Create(FT, GlobalValue::ExternalLinkage, Name, M);
      New->copyAttributesFrom(Fn);
      Out.created.push_back(New);
      return New;
    }
    if (auto *G = dyn_cast<GlobalVariable>(V)) {
      unsigned AS = FlatTypes::addrSpace(G->getAddressSpace());
      Type *VT = Types.remapType(G->getValueType());
      if (AS == G->getAddressSpace() && VT == G->getValueType())
        return nullptr;
      string Name = G->getName().str();
      Out.renamed.push_back({G, Name});
      G->setName(Name + ".alive2.as");
      // An initializer of the same type is kept; others are unknown.
      Constant *Init = G->hasInitializer() && VT == G->getValueType()
                           ? G->getInitializer() : nullptr;
      auto *New = new GlobalVariable(M, VT, G->isConstant(),
                                     Init ? G->getLinkage()
                                          : GlobalValue::ExternalLinkage,
                                     Init, Name, nullptr,
                                     G->getThreadLocalMode(), AS);
      New->setAlignment(G->getAlign());
      Out.created.push_back(New);
      return New;
    }
    return nullptr;
  }

  /// The parameter attributes alive2 translates.
  static bool alive2Supports(Attribute::AttrKind K) {
    switch (K) {
    case Attribute::InReg: case Attribute::SExt: case Attribute::ZExt:
    case Attribute::ByVal: case Attribute::NonNull: case Attribute::Captures:
    case Attribute::ReadOnly: case Attribute::WriteOnly:
    case Attribute::ReadNone: case Attribute::Dereferenceable:
    case Attribute::DereferenceableOrNull: case Attribute::Writable:
    case Attribute::Alignment: case Attribute::NoUndef: case Attribute::Range:
    case Attribute::NoFPClass: case Attribute::Returned:
    case Attribute::AllocatedPointer: case Attribute::AllocAlign:
    case Attribute::DeadOnUnwind: case Attribute::DeadOnReturn:
    case Attribute::Initializes: case Attribute::NoFree:
    case Attribute::NoFreeObj:
      return true;
    default:
      return false;
    }
  }

  /// Whether F is a kernel: in !air.kernel, or marked as one (upstream), or
  /// an agx_kernel.
  bool isKernel(Function &Fn) {
    if (Fn.getCallingConv() == AGXKernelCC ||
        Fn.getCallingConv() == CallingConv::SPIR_KERNEL ||
        Fn.hasFnAttribute("air-kernel") || Fn.hasFnAttribute("kernel") ||
        Fn.hasFnAttribute("gpu.kernel"))
      return true;
    if (NamedMDNode *MD = M.getNamedMetadata("air.kernel"))
      for (MDNode *N : MD->operands())
        if (N->getNumOperands() &&
            mdconst::dyn_extract_or_null<Function>(N->getOperand(0)) == &Fn)
          return true;
    return false;
  }

  /// F with the functions it calls inlined, as far as they are defined (a
  /// call of a function is its body; AGX inlines them all), in a copy.
  Function *inlinedCopy() {
    ValueToValueMapTy VMap;
    Function *Copy = CloneFunction(&F, VMap);
    Copy->setName(F.getName() + ".alive2.inlined");
    for (unsigned Round = 0; Round < 16; ++Round) {
      SmallVector<CallBase *, 8> Calls;
      for (Instruction &I : instructions(*Copy))
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (Function *Callee = CB->getCalledFunction();
              Callee && !Callee->isDeclaration() && Callee != &F &&
              Callee != Copy && !Callee->isVarArg() &&
              CB->getFunctionType() == Callee->getFunctionType())
            Calls.push_back(CB);
      if (Calls.empty())
        break;
      for (CallBase *CB : Calls) {
        InlineFunctionInfo IFI;
        (void)InlineFunction(*CB, IFI);
      }
    }
    return Copy;
  }

  Function *run() {
    SmallVector<string, 8> Roles = kernelRoles();
    bool Kernel = isKernel(F);
    // The body to translate: F's, its calls inlined.
    Function *Src = inlinedCopy();
    Out.created.push_back(Src);

    // The dynamic threadgroup memory F uses (AGX).
    GlobalVariable *DynTG = nullptr;
    if (AGX)
      for (GlobalVariable &G : M.globals())
        if (isDynamicThreadgroupMemory(G) &&
            any_of(G.users(), [&](User *U) {
              auto *I = dyn_cast<Instruction>(U);
              return I && I->getFunction() == Src;
            }))
          DynTG = &G;

    // Parameters: the others in order, then the threadgroup buffers. A
    // kernel takes a value as AIR passes it, in a constant buffer.
    SmallVector<Argument *, 8> Kept, Trailing;
    SmallPtrSet<Argument *, 4> ByValue;
    for (Argument &A : Src->args()) {
      if (!Roles[A.getArgNo()].empty())
        continue;
      auto *PT = dyn_cast<PointerType>(A.getType());
      (PT && PT->getAddressSpace() == AIRThreadgroup ? Trailing : Kept)
          .push_back(&A);
      if (Kernel && !PT && A.getType()->isFirstClassType() &&
          !A.getType()->isStructTy() && !A.hasByValAttr())
        ByValue.insert(&A);
    }
    SmallVector<Type *, 8> Params;
    for (Argument *A : Kept)
      Params.push_back(ByValue.count(A)
                           ? PointerType::get(Ctx, 0)
                           : Types.remapType(A->getType()));
    for (Argument *A : Trailing)
      Params.push_back(Types.remapType(A->getType()));
    if (DynTG)
      Params.push_back(Types.remapType(DynTG->getType()));
    auto *FT = FunctionType::get(Types.remapType(F.getReturnType()), Params,
                                 F.isVarArg());
    // The kernel parameters of !air.kernel are F's.

    Out.orig_name = F.getName().str();
    F.setName(Out.orig_name + ".alive2.orig");
    Function *NewF = Function::Create(FT, F.getLinkage(), F.getAddressSpace(),
                                      Out.orig_name, &M);
    Out.canon = NewF;

    ValueToValueMapTy VMap;
    unsigned I = 0;
    for (Argument *A : Kept)
      VMap[A] = NewF->getArg(I++);
    for (Argument *A : Trailing)
      VMap[A] = NewF->getArg(I++);
    if (DynTG)
      VMap[DynTG] = NewF->getArg(I++);
    // The entry: the system values, then F's body.
    BasicBlock *Pre = BasicBlock::Create(Ctx, "entry.gpu", NewF);
    IRBuilder<> B(Pre);
    makePrimitives(B);
    // A value parameter: its constant buffer's bytes.
    const DataLayout &DL = M.getDataLayout();
    for (Argument *A : Kept)
      if (ByValue.count(A)) {
        Argument *Buf = cast<Argument>(VMap[A]);
        VMap[A] = B.CreateAlignedLoad(A->getType(), Buf,
                                      DL.getABITypeAlign(A->getType()),
                                      A->getName() + ".val");
      }
    for (Argument &A : Src->args())
      if (!Roles[A.getArgNo()].empty()) {
        Value *V = systemValue(B, Roles[A.getArgNo()], A.getType());
        VMap[&A] = V ? V : PoisonValue::get(A.getType());
      }

    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewF, Src, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns, "", nullptr, &Types, this);
    // Casts between address spaces are no casts now.
    for (Instruction &I : make_early_inc_range(instructions(*NewF)))
      if (auto *C = dyn_cast<AddrSpaceCastInst>(&I);
          C && C->getSrcTy() == C->getDestTy()) {
        C->replaceAllUsesWith(C->getOperand(0));
        C->eraseFromParent();
      }
    for (unsigned J = 0; J < Kept.size(); ++J)
      NewF->getArg(J)->setName(Kept[J]->getName());
    for (Argument *A : Trailing)
      cast<Argument>(VMap[A])->setName(A->getName());
    if (DynTG)
      NewF->getArg(NewF->arg_size() - 1)->setName(DynTG->getName());

    NewF->setCallingConv(CallingConv::C);
    // byref is how AGX passes a constant buffer, not a copy; and what AGX
    // says about its kernels' buffers comes from the ABI, which this form
    // does not have.
    bool AGXKernel = F.getCallingConv() == AGXKernelCC;
    for (Argument &A : NewF->args()) {
      A.removeAttr(Attribute::ByRef);
      if (AGXKernel)
        for (auto K : {Attribute::NoAlias, Attribute::NonNull,
                       Attribute::Dereferenceable,
                       Attribute::DereferenceableOrNull, Attribute::Alignment,
                       Attribute::NoUndef, Attribute::ReadOnly})
          A.removeAttr(K);
      A.removeAttrs(AttributeFuncs::typeIncompatible(A.getType(),
                                                     A.getAttributes()));
      // What alive2 has no meaning for (noalias): fewer assumptions.
      AttributeMask Unsupported;
      for (Attribute At : A.getAttributes())
        if (!At.isStringAttribute() && !alive2Supports(At.getKindAsEnum()))
          Unsupported.addAttribute(At.getKindAsEnum());
      A.removeAttrs(Unsupported);
    }
    // A constant buffer of a value, as the AIR target describes it.
    for (unsigned J = 0; J < Kept.size(); ++J)
      if (ByValue.count(Kept[J])) {
        Argument *A = NewF->getArg(J);
        Type *T = Kept[J]->getType();
        A->removeAttrs(A->getAttributes());
        A->addAttr(Attribute::NoUndef);
        A->addAttr(Attribute::NonNull);
        A->addAttr(Attribute::ReadOnly);
        A->addAttr(Attribute::getWithAlignment(Ctx, DL.getABITypeAlign(T)));
        A->addAttr(Attribute::getWithDereferenceableBytes(
            Ctx, DL.getTypeStoreSize(T)));
      }

    auto *Body = cast<BasicBlock>(VMap[&Src->getEntryBlock()]);
    B.CreateBr(Body);
    MergeBlockIntoPredecessor(Body);

    lowerCalls(*NewF);
    defaultFPEnvironment(*NewF);
    sequentialMemory(*NewF);
    if (config::gpu_print_canonical) {
      config::dbg() << "\n----------------------------------------\n"
                               "GPU canonical form of " << Out.orig_name
                            << ":\n";
      // (a string: the opt plugin has no raw_os_ostream)
      string Text;
      raw_string_ostream OS(Text);
      NewF->print(OS);
      config::dbg() << Text;
    }
    return NewF;
  }

  // -------------------------------------------------------------------------
  // Calls.

  void lowerCalls(Function &Fn) {
    SmallVector<CallInst *, 16> Calls;
    for (Instruction &I : instructions(Fn))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (CI->getCalledFunction())
          Calls.push_back(CI);
    for (CallInst *CI : Calls) {
      StringRef Name = CI->getCalledFunction()->getName();
      IRBuilder<> B(CI);
      Value *Result = nullptr;
      if (!lowerCall(B, *CI, Name, Result))
        continue;
      if (!CI->getType()->isVoidTy()) {
        if (Result)
          Result->takeName(CI);
        CI->replaceAllUsesWith(Result ? Result
                                      : PoisonValue::get(CI->getType()));
      }
      CI->eraseFromParent();
    }
  }

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

    // AGX.
    if (Name.starts_with("llvm.agx."))
      return lowerAGX(B, CI, Name.drop_front(strlen("llvm.agx.")), R);

    if (!Name.consume_front("air."))
      return false;

    // System values, as functions (before the AIR target makes them
    // parameters).
    StringRef Role = Name.take_until([](char C) { return C == '.'; });
    if (N == 0 && is_system_value(Role)) {
      R = systemValue(B, Role, Ty);
      return R != nullptr;
    }
    if (Name.starts_with("atomic."))
      return lowerAtomic(B, CI, Name, R);
    if (Name == "wg.barrier" || Name == "threadgroup.barrier" ||
        Name == "simdgroup.barrier") {
      // mem_flags: device 1, threadgroup 2, texture 4. A SIMD-group runs in
      // lockstep: its barrier only orders memory.
      auto *Flags = N ? dyn_cast<ConstantInt>(Arg(0)) : nullptr;
      bool Memory = !Flags || !Flags->isZero();
      if (Name == "simdgroup.barrier") {
        if (Memory)
          fence(B);
      } else {
        barrier(B, Memory);
      }
      return true;
    }
    if (Name.starts_with("simd_") || Name.starts_with("quad_"))
      return lowerSIMD(B, CI, Name, R);
    return lowerOther(B, CI, Name, R) || lowerMath(B, CI, Name, R);
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

  // air.atomic.fence(flags, order, scope), the atomic operations
  //   air.atomic.<space>.<op>.[s.|u.]<type>(ptr, value, order, scope, volatile)
  // and air.atomic.<space>.cmpxchg.weak.i32(ptr, ptr expected, desired,
  // success order, failure order, scope, [flags,] volatile), which returns
  // the value it found and, as MSL's compare-exchange does, leaves it in
  // *expected (whether or not it succeeds). It returns no success flag, so
  // that a spurious failure would be undetectable: it succeeds when it finds
  // the expected value, as the AIR target (which lowers LLVM's strong
  // cmpxchg to it) and the AGX backend (which lowers it to one) take it, and
  // as an Apple M2 Max does.
  bool lowerAtomic(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    auto Arg = [&](unsigned I) { return CI.getArgOperand(I); };
    if (Name == "atomic.fence") {
      if (CI.arg_size() != 3)
        return false;
      if (msl_ordering(Arg(1)) != AtomicOrdering::Monotonic)
        fence(B);
      return true;
    }
    StringRef Rest = Name.drop_front(strlen("atomic."));
    if (!Rest.consume_front("global.") && !Rest.consume_front("local."))
      return false;
    if (Rest == "cmpxchg.weak.i32") {
      if (CI.arg_size() != 7 && CI.arg_size() != 8)
        return false;
      Value *Expected = B.CreateLoad(B.getInt32Ty(), Arg(1));
      AtomicOrdering Success = msl_ordering(Arg(3));
      AtomicOrdering Failure = msl_ordering(Arg(4));
      if (Failure == AtomicOrdering::Release)
        Failure = AtomicOrdering::Monotonic;
      else if (Failure == AtomicOrdering::AcquireRelease)
        Failure = AtomicOrdering::Acquire;
      if (isStrongerThan(Failure, Success))
        Success = Failure;
      auto *X = B.CreateAtomicCmpXchg(Arg(0), Expected, Arg(2), MaybeAlign(),
                                      Success, Failure);
      R = B.CreateExtractValue(X, 0);
      B.CreateStore(R, Arg(1));
      return true;
    }
    StringRef Op = Rest.take_until([](char C) { return C == '.'; });
    Rest = Rest.drop_front(Op.size());
    Rest.consume_front(".");
    bool Unsigned = Rest.consume_front("u.");
    Rest.consume_front("s.");
    bool Float = Rest.starts_with("f");
    optional<AtomicRMWInst::BinOp> BinOp =
        StringSwitch<optional<AtomicRMWInst::BinOp>>(Op)
            .Case("xchg", AtomicRMWInst::Xchg)
            .Case("add", Float ? AtomicRMWInst::FAdd : AtomicRMWInst::Add)
            .Case("sub", Float ? AtomicRMWInst::FSub : AtomicRMWInst::Sub)
            .Case("max", Float      ? AtomicRMWInst::FMax
                         : Unsigned ? AtomicRMWInst::UMax
                                    : AtomicRMWInst::Max)
            .Case("min", Float      ? AtomicRMWInst::FMin
                         : Unsigned ? AtomicRMWInst::UMin
                                    : AtomicRMWInst::Min)
            .Case("and", AtomicRMWInst::And)
            .Case("or", AtomicRMWInst::Or)
            .Case("xor", AtomicRMWInst::Xor)
            .Default(nullopt);
    if (!BinOp || CI.arg_size() != 5)
      return false;
    R = B.CreateAtomicRMW(*BinOp, Arg(0), Arg(1), MaybeAlign(),
                          msl_ordering(Arg(2)));
    return true;
  }

  // air.<name>.[s.|u.]<type> and air.fast_<name>.<type>: LLVM's intrinsics,
  // the fast ones with afn.
  bool lowerMath(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    StringRef Rest = Name;
    bool Fast = Rest.consume_front("fast_");
    StringRef Base = Rest.take_until([](char C) { return C == '.'; });
    Rest = Rest.drop_front(Base.size());
    Rest.consume_front(".");
    unsigned Signedness = Rest.consume_front("s.") ? 1
                          : Rest.consume_front("u.") ? 2 : 0;
    const MathFn *Fn = nullptr;
    for (const MathFn &M : math_fns)
      if (Base == M.AIR && Signedness == M.Signedness && (!Fast || M.HasFast))
        Fn = &M;
    if (!Fn)
      return false;
    Intrinsic::ID ID = Intrinsic::lookupIntrinsicID(Fn->LLVM);
    if (ID == Intrinsic::not_intrinsic || CI.getType()->isVoidTy())
      return false;
    Function *Decl = Intrinsic::getOrInsertDeclaration(&M, ID, {CI.getType()});
    SmallVector<Value *, 4> Args(CI.args());
    // The operands AIR leaves out (llvm.abs's is_int_min_poison) are false.
    while (Args.size() < Decl->arg_size())
      Args.push_back(B.getFalse());
    if (Args.size() != Decl->arg_size())
      return false;
    for (unsigned I = 0; I < Args.size(); ++I)
      if (Args[I]->getType() != Decl->getFunctionType()->getParamType(I))
        return false;
    CallInst *New = B.CreateCall(Decl, Args);
    if (isa<FPMathOperator>(New)) {
      FastMathFlags FMF = CI.getFastMathFlags();
      FMF.setApproxFunc(FMF.approxFunc() || Fast);
      New->setFastMathFlags(FMF);
    }
    R = New;
    return true;
  }

  // The SIMD-group and quad-group functions: shuffles, reductions, prefix
  // scans, ballots and votes, on AGX's operations (AGX::SIMDOp numbers the
  // reductions).
  bool lowerSIMD(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    bool Quad = Name.starts_with("quad_");
    StringRef Group = Quad ? "quad." : "simd.";
    StringRef Rest = Name.drop_front(5);
    StringRef Op = Rest.take_until([](char C) { return C == '.'; });
    Rest = Rest.drop_front(Op.size());
    Rest.consume_front(".");
    bool Unsigned = Rest.starts_with("u.");
    Type *Ty = CI.getType();
    auto Arg = [&](unsigned I) { return CI.getArgOperand(I); };
    auto Lane = [&](unsigned I) { return B.CreateZExtOrTrunc(Arg(I), I32); };
    auto Fn = [&](StringRef Op, Type *Ret, ArrayRef<Value *> Args) {
      return convergent(B, (Group + Op).str(), Ret, Args);
    };
    auto Active = [&] { return Fn("ballot", I32, {B.getTrue()}); };

    optional<StringRef> Shuffle = StringSwitch<optional<StringRef>>(Op)
      .Cases({"shuffle", "broadcast"}, StringRef())
      .Case("shuffle_xor", StringRef("xor"))
      .Case("shuffle_up", StringRef("up"))
      .Case("shuffle_down", StringRef("down"))
      .Default(nullopt);
    if (Shuffle && CI.arg_size() == 2) {
      R = shuffle(B, Quad, *Shuffle, Arg(0), Arg(1));
      return true;
    }
    if (Op == "broadcast_first" && CI.arg_size() == 1) {
      Value *First = B.CreateIntrinsic(Intrinsic::cttz, {I32},
                                       {Active(), B.getFalse()});
      R = Fn("shuffle", Ty, {Arg(0), First});
      return true;
    }
    // shuffle_and_fill_{down,up}(data, fill, delta, modulo): in segments of
    // modulo threads, data delta threads down (up), or fill from the start
    // (end) of the segment where that leaves it.
    if ((Op == "shuffle_and_fill_down" || Op == "shuffle_and_fill_up") &&
        CI.arg_size() == 4) {
      bool Down = Op == "shuffle_and_fill_down";
      Value *Self = systemValue(B, "thread_index_in_simdgroup")[0];
      if (Quad)
        Self = B.CreateAnd(Self, 3);
      Value *Delta = Lane(2), *Mod = Lane(3);
      Value *Pos = B.CreateURem(Self, Mod);
      Value *Base = B.CreateSub(Self, Pos);
      Value *Inside, *Src;
      if (Down) {
        Value *To = B.CreateAdd(Pos, Delta);
        Inside = B.CreateICmpULT(To, Mod);
        Src = B.CreateAdd(Base,
                          B.CreateSelect(Inside, To, B.CreateSub(To, Mod)));
      } else {
        Inside = B.CreateICmpUGE(Pos, Delta);
        Value *From = B.CreateSub(Pos, Delta);
        Src = B.CreateAdd(Base,
                          B.CreateSelect(Inside, From, B.CreateAdd(From, Mod)));
      }
      Value *Data = Fn("shuffle", Ty, {Arg(0), Src});
      Value *Fill = Fn("shuffle", Ty, {Arg(1), Src});
      R = B.CreateSelect(Inside, Data, Fill);
      return true;
    }
    if (Op == "ballot" && CI.arg_size() == 1 && Ty->isIntegerTy()) {
      R = B.CreateZExtOrTrunc(Fn("ballot", I32, {Arg(0)}), Ty);
      return true;
    }
    if ((Op == "vote_any" || Op == "vote_all") && CI.arg_size() == 1 &&
        Arg(0)->getType()->isIntegerTy()) {
      Value *Mask = Arg(0);
      R = Op == "vote_any"
              ? B.CreateIsNotNull(Mask)
              : B.CreateICmpEQ(Mask,
                               B.CreateZExtOrTrunc(Active(), Mask->getType()));
      R = B.CreateZExtOrTrunc(R, Ty);
      return true;
    }

    // Reductions and (exclusive) scans, named by their operation.
    bool Float = Ty->getScalarType()->isFloatingPointTy();
    bool Prefix = Op.consume_front("prefix_");
    bool Inclusive = Prefix && Op.consume_front("inclusive_");
    if (Prefix && !Inclusive && !Op.consume_front("exclusive_"))
      return false;
    StringRef Red = StringSwitch<StringRef>(Op)
      .Case("sum", Float ? "fadd" : "iadd")
      .Case("product", Float ? "fmul" : "")
      .Case("min", Float ? "fmin" : Unsigned ? "umin" : "smin")
      .Case("max", Float ? "fmax" : Unsigned ? "umax" : "smax")
      .Case("and", Float ? "" : "and")
      .Case("or", Float ? "" : "or")
      .Case("xor", Float ? "" : "xor")
      .Default("");
    if (Red.empty() || CI.arg_size() != 1 ||
        (Prefix && Op != "sum" && Op != "product"))
      return false;
    Value *X = Arg(0);
    R = Fn(((Prefix ? "prefix." : "reduce.") + Red).str(), Ty, {X});
    // The hardware's scans are exclusive.
    if (Inclusive)
      R = Op == "sum" ? (Float ? B.CreateFAdd(R, X) : B.CreateAdd(R, X))
                      : B.CreateFMul(R, X);
    return true;
  }

  // AIR functions that are neither LLVM intrinsics nor SIMD-group functions.
  bool lowerOther(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    SmallVector<StringRef, 6> Parts;
    Name.split(Parts, '.');
    StringRef Fn = Parts[0];
    Type *Ty = CI.getType();
    unsigned N = CI.arg_size();
    auto Arg = [&](unsigned I) { return CI.getArgOperand(I); };

    // OpenCL-style work-item functions, as OpenCL defines them.
    if (Fn.starts_with("get_") && N == 1 && Ty->isIntegerTy()) {
      Value *V = workItem(B, Fn, Arg(0));
      if (!V)
        return false;
      R = B.CreateZExtOrTrunc(V, Ty);
      return true;
    }
    // air.convert.<to>.<from>, each a kind (f, s or u) and a type; floats
    // become integers saturating.
    if (Fn == "convert" && Parts.size() == 5 && N == 1) {
      Value *X = Arg(0);
      char To = Parts[1][0], From = Parts[3][0];
      if (To == 'f')
        R = From == 'f'   ? B.CreateFPCast(X, Ty)
            : From == 's' ? B.CreateSIToFP(X, Ty)
                          : B.CreateUIToFP(X, Ty);
      else if (From == 'f')
        R = B.CreateIntrinsic(To == 's' ? Intrinsic::fptosi_sat
                                        : Intrinsic::fptoui_sat,
                              {Ty, X->getType()}, {X});
      else
        R = B.CreateIntCast(X, Ty, From == 's');
      return true;
    }
    if (Fn == "dot" && N == 2) {
      auto *VT = dyn_cast<FixedVectorType>(Arg(0)->getType());
      if (!VT)
        return false;
      Value *Sum = nullptr;
      for (unsigned I = 0; I < VT->getNumElements(); ++I) {
        Value *P = B.CreateFMul(B.CreateExtractElement(Arg(0), I),
                                B.CreateExtractElement(Arg(1), I));
        Sum = Sum ? B.CreateFAdd(Sum, P) : P;
      }
      R = Sum;
      return true;
    }
    // air.pack.{unorm,snorm}{4x8,2x16}.<vector> and air.unpack: normalized
    // integers, rounded to nearest even, the first element lowest.
    if ((Fn == "pack" || Fn == "unpack") && Parts.size() >= 2 && N == 1) {
      StringRef Format = Parts[1];
      bool Signed = Format.starts_with("snorm");
      unsigned Count, Bits;
      if (Format.ends_with("4x8"))
        Count = 4, Bits = 8;
      else if (Format.ends_with("2x16"))
        Count = 2, Bits = 16;
      else
        return false;
      if (!Signed && !Format.starts_with("unorm"))
        return false;
      double Scale = Signed ? (1 << (Bits - 1)) - 1 : (1 << Bits) - 1;
      Type *F32 = B.getFloatTy();
      if (Fn == "pack") {
        Value *Word = B.getInt32(0);
        for (unsigned I = 0; I < Count; ++I) {
          Value *X = B.CreateFPCast(B.CreateExtractElement(Arg(0), I), F32);
          X = B.CreateMaxNum(X, ConstantFP::get(F32, Signed ? -1.0 : 0.0));
          X = B.CreateMinNum(X, ConstantFP::get(F32, 1.0));
          X = B.CreateIntrinsic(Intrinsic::rint, {F32},
                                {B.CreateFMul(X, ConstantFP::get(F32, Scale))});
          Value *I32V = B.CreateFPToSI(X, I32);
          I32V = B.CreateAnd(I32V, B.getInt32((1u << Bits) - 1));
          Word = B.CreateOr(Word, B.CreateShl(I32V, I * Bits));
        }
        R = B.CreateZExtOrTrunc(Word, Ty);
        return true;
      }
      auto *VT = dyn_cast<FixedVectorType>(Ty);
      if (!VT || VT->getNumElements() != Count)
        return false;
      Value *Word = B.CreateZExtOrTrunc(Arg(0), I32);
      Value *V = PoisonValue::get(VT);
      for (unsigned I = 0; I < Count; ++I) {
        Value *E = B.CreateTrunc(B.CreateLShr(Word, I * Bits),
                                 B.getIntNTy(Bits));
        Value *Fv = Signed ? B.CreateSIToFP(E, F32) : B.CreateUIToFP(E, F32);
        Fv = B.CreateFDiv(Fv, ConstantFP::get(F32, Scale));
        if (Signed)
          Fv = B.CreateMaxNum(Fv, ConstantFP::get(F32, -1.0));
        V = B.CreateInsertElement(V, B.CreateFPCast(Fv, VT->getElementType()),
                                  I);
      }
      R = V;
      return true;
    }
    if ((Fn == "sincos" || Fn == "fast_sincos") && N == 2) {
      FastMathFlags FMF = CI.getFastMathFlags();
      FMF.setApproxFunc(FMF.approxFunc() || Fn == "fast_sincos");
      IRBuilder<>::FastMathFlagGuard Guard(B);
      B.setFastMathFlags(FMF);
      R = B.CreateUnaryIntrinsic(Intrinsic::sin, Arg(0));
      B.CreateStore(B.CreateUnaryIntrinsic(Intrinsic::cos, Arg(0)), Arg(1));
      return true;
    }
    if (Fn == "fract" && N == 1 && Ty->isFPOrFPVectorTy()) {
      // x - floor(x), below 1.
      Value *X = Arg(0);
      Value *Fr = B.CreateFSub(X, B.CreateUnaryIntrinsic(Intrinsic::floor, X));
      APFloat BelowOne =
          APFloat::getOne(Ty->getScalarType()->getFltSemantics());
      BelowOne.next(/*nextDown=*/true);
      R = B.CreateMinNum(Fr, ConstantFP::get(Ty, BelowOne));
      return true;
    }
    if (Fn == "rsqrt" && N == 1 && Ty->isFPOrFPVectorTy()) {
      R = B.CreateFDiv(ConstantFP::get(Ty, 1.0),
                       B.CreateUnaryIntrinsic(Intrinsic::sqrt, Arg(0)));
      return true;
    }
    if (Fn == "fmod" && N == 2) {
      R = B.CreateFRem(Arg(0), Arg(1));
      return true;
    }
    if (Fn == "mul_hi" && Parts.size() == 3 && N == 2 && Ty->isIntOrIntVectorTy()) {
      bool Signed = Parts[1] == "s";
      unsigned Bits = Ty->getScalarSizeInBits();
      Type *Wide = Ty->getWithNewBitWidth(Bits * 2);
      Value *P = B.CreateMul(B.CreateIntCast(Arg(0), Wide, Signed),
                             B.CreateIntCast(Arg(1), Wide, Signed));
      R = B.CreateTrunc(B.CreateLShr(P, Bits), Ty);
      return true;
    }
    // Device-coherent accesses are accesses (the memory model is sequential).
    if (Fn == "load" && Parts.size() >= 2 && Parts[1] == "device_coherent" &&
        N == 1) {
      R = B.CreateLoad(Ty, Arg(0));
      return true;
    }
    if (Fn == "store" && Parts.size() >= 2 && Parts[1] == "device_coherent" &&
        N == 2) {
      B.CreateStore(Arg(0), Arg(1));
      return true;
    }
    return false;
  }

  // -------------------------------------------------------------------------
  // The floating-point environment.

  /// AIR and AGX have no floating-point environment to change: outside
  /// strictfp functions it rounds to nearest even and raises nothing, so a
  /// constrained operation that rounds to nearest (or dynamically) and
  /// ignores exceptions is the plain operation (agx-lower-air restores AIR's
  /// software arithmetic as such operations).
  void defaultFPEnvironment(Function &Fn) {
    if (Fn.hasFnAttribute(Attribute::StrictFP))
      return;
    for (Instruction &I : make_early_inc_range(instructions(Fn))) {
      auto *C = dyn_cast<ConstrainedFPIntrinsic>(&I);
      if (!C)
        continue;
      std::optional<RoundingMode> RM = C->getRoundingMode();
      std::optional<fp::ExceptionBehavior> EB = C->getExceptionBehavior();
      if ((RM && *RM != RoundingMode::NearestTiesToEven &&
           *RM != RoundingMode::Dynamic) ||
          (EB && *EB != fp::ebIgnore))
        continue;
      IRBuilder<> B(C);
      Value *A0 = C->getArgOperand(0);
      Value *R = nullptr;
      switch (C->getIntrinsicID()) {
      case Intrinsic::experimental_constrained_fadd:
        R = B.CreateFAdd(A0, C->getArgOperand(1)); break;
      case Intrinsic::experimental_constrained_fsub:
        R = B.CreateFSub(A0, C->getArgOperand(1)); break;
      case Intrinsic::experimental_constrained_fmul:
        R = B.CreateFMul(A0, C->getArgOperand(1)); break;
      case Intrinsic::experimental_constrained_fdiv:
        R = B.CreateFDiv(A0, C->getArgOperand(1)); break;
      case Intrinsic::experimental_constrained_frem:
        R = B.CreateFRem(A0, C->getArgOperand(1)); break;
      case Intrinsic::experimental_constrained_fma:
        R = B.CreateIntrinsic(Intrinsic::fma, {C->getType()},
                              {A0, C->getArgOperand(1), C->getArgOperand(2)});
        break;
      case Intrinsic::experimental_constrained_sqrt:
        R = B.CreateUnaryIntrinsic(Intrinsic::sqrt, A0); break;
      case Intrinsic::experimental_constrained_fptrunc:
        R = B.CreateFPTrunc(A0, C->getType()); break;
      case Intrinsic::experimental_constrained_fpext:
        R = B.CreateFPExt(A0, C->getType()); break;
      case Intrinsic::experimental_constrained_sitofp:
        R = B.CreateSIToFP(A0, C->getType()); break;
      case Intrinsic::experimental_constrained_uitofp:
        R = B.CreateUIToFP(A0, C->getType()); break;
      default:
        break;
      }
      if (!R)
        continue;
      if (auto *RI = dyn_cast<Instruction>(R); RI && isa<FPMathOperator>(RI))
        RI->setFastMathFlags(C->getFastMathFlags());
      R->takeName(C);
      C->replaceAllUsesWith(R);
      C->eraseFromParent();
    }
  }

  // -------------------------------------------------------------------------
  // The sequential memory model.

  void sequentialMemory(Function &Fn) {
    SmallVector<Instruction *, 16> Work;
    for (Instruction &I : instructions(Fn))
      if (isa<FenceInst>(I) || isa<AtomicRMWInst>(I) ||
          isa<AtomicCmpXchgInst>(I) ||
          (isa<LoadInst>(I) && !cast<LoadInst>(I).isSimple()) ||
          (isa<StoreInst>(I) && !cast<StoreInst>(I).isSimple()))
        Work.push_back(&I);
    for (Instruction *I : Work) {
      IRBuilder<> B(I);
      // Atomics of the thread alone are its accesses.
      if (auto *RMW = dyn_cast<AtomicRMWInst>(I);
          RMW && RMW->getSyncScopeID() == SyncScope::SingleThread)
        RMW->setOrdering(AtomicOrdering::Monotonic);
      auto ordered = [](AtomicOrdering O) {
        return isStrongerThanMonotonic(O);
      };
      auto after = [&] {
        IRBuilder<> A(I->getNextNode());
        fence(A);
      };
      if (auto *Fe = dyn_cast<FenceInst>(I)) {
        // A fence of the thread alone orders nothing for other threads.
        if (Fe->getSyncScopeID() != SyncScope::SingleThread)
          fence(B);
        I->eraseFromParent();
      } else if (auto *L = dyn_cast<LoadInst>(I)) {
        if (ordered(L->getOrdering()))
          after();
        L->setAtomic(AtomicOrdering::NotAtomic);
        L->setVolatile(false);
      } else if (auto *S = dyn_cast<StoreInst>(I)) {
        if (ordered(S->getOrdering()))
          fence(B);
        S->setAtomic(AtomicOrdering::NotAtomic);
        S->setVolatile(false);
      } else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
        Value *Old = sequentialRMW(B, *RMW);
        if (!Old)
          continue;
        Old->takeName(RMW);
        RMW->replaceAllUsesWith(Old);
        RMW->eraseFromParent();
      } else if (auto *X = dyn_cast<AtomicCmpXchgInst>(I)) {
        sequentialCmpXchg(*X);
      }
    }
  }

  /// atomicrmw as load, operation, store; an ordered one between fences.
  Value *sequentialRMW(IRBuilder<> &B, AtomicRMWInst &RMW) {
    bool Ordered = isStrongerThanMonotonic(RMW.getOrdering());
    Type *Ty = RMW.getType();
    Value *Ptr = RMW.getPointerOperand(), *V = RMW.getValOperand();
    // Check the operation first, so that nothing is emitted for one this
    // form does not have.
    switch (RMW.getOperation()) {
    case AtomicRMWInst::Xchg: case AtomicRMWInst::Add: case AtomicRMWInst::Sub:
    case AtomicRMWInst::And: case AtomicRMWInst::Nand: case AtomicRMWInst::Or:
    case AtomicRMWInst::Xor: case AtomicRMWInst::Max: case AtomicRMWInst::Min:
    case AtomicRMWInst::UMax: case AtomicRMWInst::UMin:
    case AtomicRMWInst::FAdd: case AtomicRMWInst::FSub:
    case AtomicRMWInst::FMax: case AtomicRMWInst::FMin:
    case AtomicRMWInst::FMaximum: case AtomicRMWInst::FMinimum:
    case AtomicRMWInst::UIncWrap: case AtomicRMWInst::UDecWrap:
      break;
    default:
      return nullptr;
    }
    if (Ordered)
      fence(B);
    LoadInst *Old = B.CreateAlignedLoad(Ty, Ptr, RMW.getAlign());
    Value *New = nullptr;
    auto intr = [&](Intrinsic::ID ID) {
      return B.CreateIntrinsic(ID, {Ty}, {Old, V});
    };
    switch (RMW.getOperation()) {
    case AtomicRMWInst::Xchg: New = V; break;
    case AtomicRMWInst::Add: New = B.CreateAdd(Old, V); break;
    case AtomicRMWInst::Sub: New = B.CreateSub(Old, V); break;
    case AtomicRMWInst::And: New = B.CreateAnd(Old, V); break;
    case AtomicRMWInst::Nand: New = B.CreateNot(B.CreateAnd(Old, V)); break;
    case AtomicRMWInst::Or: New = B.CreateOr(Old, V); break;
    case AtomicRMWInst::Xor: New = B.CreateXor(Old, V); break;
    case AtomicRMWInst::Max: New = intr(Intrinsic::smax); break;
    case AtomicRMWInst::Min: New = intr(Intrinsic::smin); break;
    case AtomicRMWInst::UMax: New = intr(Intrinsic::umax); break;
    case AtomicRMWInst::UMin: New = intr(Intrinsic::umin); break;
    case AtomicRMWInst::FAdd: New = B.CreateFAdd(Old, V); break;
    case AtomicRMWInst::FSub: New = B.CreateFSub(Old, V); break;
    case AtomicRMWInst::FMax: New = intr(Intrinsic::maxnum); break;
    case AtomicRMWInst::FMin: New = intr(Intrinsic::minnum); break;
    case AtomicRMWInst::FMaximum: New = intr(Intrinsic::maximum); break;
    case AtomicRMWInst::FMinimum: New = intr(Intrinsic::minimum); break;
    case AtomicRMWInst::UIncWrap:
      New = B.CreateSelect(B.CreateICmpUGE(Old, V), ConstantInt::get(Ty, 0),
                           B.CreateAdd(Old, ConstantInt::get(Ty, 1)));
      break;
    case AtomicRMWInst::UDecWrap:
      New = B.CreateSelect(
          B.CreateOr(B.CreateIsNull(Old), B.CreateICmpUGT(Old, V)), V,
          B.CreateSub(Old, ConstantInt::get(Ty, 1)));
      break;
    default:
      llvm_unreachable("checked above");
    }
    B.CreateAlignedStore(New, Ptr, RMW.getAlign());
    if (Ordered)
      fence(B);
    return Old;
  }

  /// cmpxchg as a load and a store when it succeeds; a weak one may fail
  /// anyway.
  void sequentialCmpXchg(AtomicCmpXchgInst &X) {
    IRBuilder<> B(&X);
    bool Ordered = isStrongerThanMonotonic(X.getSuccessOrdering()) ||
                   isStrongerThanMonotonic(X.getFailureOrdering());
    if (Ordered)
      fence(B);
    Type *Ty = X.getCompareOperand()->getType();
    Value *Ptr = X.getPointerOperand();
    LoadInst *Old = B.CreateAlignedLoad(Ty, Ptr, X.getAlign());
    Value *Eq = B.CreateICmpEQ(Old, X.getCompareOperand());
    if (X.isWeak())
      Eq = B.CreateAnd(Eq, B.CreateFreeze(PoisonValue::get(B.getInt1Ty())));
    Instruction *Then = SplitBlockAndInsertIfThen(Eq, &X, false);
    IRBuilder<>(Then).CreateAlignedStore(X.getNewValOperand(), Ptr,
                                         X.getAlign());
    B.SetInsertPoint(&X);
    Value *R = B.CreateInsertValue(PoisonValue::get(X.getType()), Old, 0);
    R = B.CreateInsertValue(R, Eq, 1);
    if (Ordered)
      fence(B);
    R->takeName(&X);
    X.replaceAllUsesWith(R);
    X.eraseFromParent();
  }
};

bool is_gpu_module(const Module &M) {
  StringRef T(M.getTargetTriple().str());
  if (T.starts_with("agx") || T.starts_with("air") ||
      M.getNamedMetadata("air.kernel"))
    return true;
  for (const Function &F : M) {
    StringRef N = F.getName();
    if (N.starts_with("air.") || N.starts_with("llvm.agx.") ||
        !opencl_builtin(N).empty() ||
        F.getCallingConv() == CallingConv::SPIR_KERNEL ||
        F.getCallingConv() == AGXKernelCC ||
        F.hasFnAttribute("air-kernel"))
      return true;
  }
  return false;
}

GPUCanonicalFunction::GPUCanonicalFunction(Function &F) : orig(&F) {
  GPUCanonicalizer(*this, F).run();
}

GPUCanonicalFunction::~GPUCanonicalFunction() {
  if (canon) {
    canon->dropAllReferences();
    canon->eraseFromParent();
  }
  // The functions and variables made for the canonical form, then the names
  // they borrowed.
  for (auto I = created.rbegin(), E = created.rend(); I != E; ++I) {
    (*I)->removeDeadConstantUsers();
    if ((*I)->use_empty())
      (*I)->eraseFromParent();
    else
      (*I)->setName((*I)->getName() + ".alive2.canon");
  }
  for (auto I = renamed.rbegin(), E = renamed.rend(); I != E; ++I)
    I->first->setName(I->second);
  orig->setName(orig_name);
}

}

// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// The canonical form of GPU code (gpu.h): what is common to every target.
// The targets' parts are gpu_upstream.cpp, gpu_air.cpp, gpu_agx.cpp and
// gpu_nvptx.cpp (gpu_target.h).

#include "llvm_util/gpu.h"
#include "llvm_util/gpu_target.h"
#include "util/config.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;
using namespace std;
using namespace util;
using namespace llvm_util::gpu;

namespace {

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

namespace gpu {

ArrayRef<const GPUTarget *> targets() {
  // Upstream first: its builtins are plain functions any form may call.
  static const GPUTarget *const All[] = {&upstream_target(), &agx_target(),
                                         &nvptx_target(), &air_target()};
  return All;
}

const GPUTarget &module_target(const Module &M) {
  // A module of a backend's form may still call upstream builtins: the most
  // specific form first.
  for (const GPUTarget *T : {&nvptx_target(), &agx_target(), &air_target()})
    if (T->isModule(M))
      return *T;
  return upstream_target();
}

} // namespace gpu

struct GPUCanonicalizer : GPUBuilder, ValueMaterializer {
  Function &F;
  const GPUTarget &Target;
  FlatTypes Types;

  GPUCanonicalizer(GPUCanonicalFunction &Out, Function &F)
      : GPUBuilder(Out, *F.getParent()), F(F),
        Target(module_target(*F.getParent())) {}

  // -------------------------------------------------------------------------
  // The canonical function.

  /// Parameter roles: system values (their role), or none.
  SmallVector<string, 8> kernelRoles() {
    SmallVector<string, 8> R(F.arg_size());
    for (const GPUTarget *T : targets())
      T->kernelRoles(F, R);
    return R;
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

  /// Whether Fn is a kernel of any of the targets.
  bool isKernel(Function &Fn) {
    for (const GPUTarget *T : targets())
      if (T->isKernel(Fn))
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
    // Casts between address spaces in constants (NVPTX's generic pointers
    // to shared variables) as instructions, which become no casts below.
    {
      SmallVector<Constant *, 4> Casts;
      SmallPtrSet<Constant *, 8> Seen;
      SmallVector<Constant *, 8> Work;
      for (Instruction &I : instructions(*Src))
        for (Value *Op : I.operands())
          if (auto *CE = dyn_cast<ConstantExpr>(Op))
            Work.push_back(CE);
      while (!Work.empty()) {
        Constant *C = Work.pop_back_val();
        if (!Seen.insert(C).second)
          continue;
        if (auto *CE = dyn_cast<ConstantExpr>(C)) {
          if (CE->getOpcode() == Instruction::AddrSpaceCast)
            Casts.push_back(CE);
          for (Value *Op : CE->operands())
            if (auto *OC = dyn_cast<ConstantExpr>(Op))
              Work.push_back(OC);
        }
      }
      if (!Casts.empty())
        convertUsersOfConstantsToInstructions(Casts, Src, false, true);
    }

    // Casts between address spaces that have no meaning.
    for (Instruction &I : make_early_inc_range(instructions(*Src)))
      if (auto *C = dyn_cast<AddrSpaceCastInst>(&I);
          C && Target.castIsPoison(C->getSrcAddressSpace(),
                                   C->getDestAddressSpace())) {
        C->replaceAllUsesWith(PoisonValue::get(C->getType()));
        C->eraseFromParent();
      }

    // Atomics on memory private to the thread are of the thread alone (the
    // address spaces are gone below).
    if (optional<unsigned> Private = Target.privateAddressSpace())
      for (Instruction &I : instructions(*Src)) {
        Value *P = getLoadStorePointerOperand(&I);
        if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
          P = RMW->getPointerOperand();
        if (auto *X = dyn_cast<AtomicCmpXchgInst>(&I))
          P = X->getPointerOperand();
        if (!P || P->getType()->getPointerAddressSpace() != *Private)
          continue;
        if (auto *L = dyn_cast<LoadInst>(&I); L && L->isAtomic())
          L->setSyncScopeID(SyncScope::SingleThread);
        if (auto *S = dyn_cast<StoreInst>(&I); S && S->isAtomic())
          S->setSyncScopeID(SyncScope::SingleThread);
        if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
          RMW->setSyncScopeID(SyncScope::SingleThread);
        if (auto *X = dyn_cast<AtomicCmpXchgInst>(&I))
          X->setSyncScopeID(SyncScope::SingleThread);
      }

    // The dynamic threadgroup memory F uses (AGX's).
    SmallVector<GlobalVariable *, 2> DynTGs;
    Target.dynamicThreadgroupMemory(*Src, DynTGs);
    GlobalVariable *DynTG = DynTGs.empty() ? nullptr : DynTGs[0];

    // Parameters: the others in order, then the threadgroup buffers. A
    // kernel of a target that says so takes a value as AIR passes it, in a
    // constant buffer (GPUTarget::scalarsInConstantBuffers).
    SmallVector<Argument *, 8> Kept, Trailing;
    SmallPtrSet<Argument *, 4> ByValue;
    for (Argument &A : Src->args()) {
      if (!Roles[A.getArgNo()].empty())
        continue;
      auto *PT = dyn_cast<PointerType>(A.getType());
      (PT && PT->getAddressSpace() == ThreadgroupAS ? Trailing : Kept)
          .push_back(&A);
      if (Kernel && Target.scalarsInConstantBuffers() && !PT &&
          A.getType()->isFirstClassType() && !A.getType()->isStructTy() &&
          !A.hasByValAttr())
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
    if (DynTG) {
      // Each variable that stands for it is the one object.
      for (GlobalVariable *G : DynTGs)
        VMap[G] = NewF->getArg(I);
      ++I;
    }
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
    // byref is how AGX passes a constant buffer, not a copy; and what a
    // target says about its kernels' buffers comes from the ABI, which this
    // form does not have.
    Target.dropABIAttributes(F, *NewF);
    for (Argument &A : NewF->args()) {
      A.removeAttr(Attribute::ByRef);
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
    mergeFences(*NewF);
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
    for (const GPUTarget *T : targets())
      if (T->lowerCall(*this, B, CI, Name, R))
        return true;
    return false;
  }

  // -------------------------------------------------------------------------
  // The floating-point environment.

  /// The GPUs (AIR, AGX, PTX) have no floating-point environment to change:
  /// outside strictfp functions it rounds to nearest even and raises
  /// nothing, so a constrained operation that rounds to nearest (or
  /// dynamically) and ignores exceptions is the plain operation
  /// (agx-lower-air restores AIR's software arithmetic as such operations).
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
      } else if (auto *L = dyn_cast<LoadInst>(I);
                 L && L->isAtomic() &&
                 L->getSyncScopeID() == SyncScope::SingleThread) {
        L->setAtomic(AtomicOrdering::NotAtomic);
        L->setVolatile(false);
      } else if (auto *S = dyn_cast<StoreInst>(I);
                 S && S->isAtomic() &&
                 S->getSyncScopeID() == SyncScope::SingleThread) {
        S->setAtomic(AtomicOrdering::NotAtomic);
        S->setVolatile(false);
      } else if (auto *L = dyn_cast<LoadInst>(I)) {
        // An acquire load is the load, then what other threads did; a
        // sequentially consistent one is also after what they did.
        if (L->getOrdering() == AtomicOrdering::SequentiallyConsistent)
          fence(B);
        if (ordered(L->getOrdering()))
          after();
        L->setAtomic(AtomicOrdering::NotAtomic);
        L->setVolatile(false);
      } else if (auto *S = dyn_cast<StoreInst>(I)) {
        // A release or sequentially consistent store is what other threads
        // did, then the store (PTX: fence.sc, then the store).
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

  /// Fences with no access of memory between them are one: what other
  /// threads may do at the first, they may at the second.
  void mergeFences(Function &Fn) {
    Function *Fence = M.getFunction("alive.gpu.fence");
    if (!Fence)
      return;
    for (BasicBlock &BB : Fn) {
      CallInst *Last = nullptr;
      for (Instruction &I : make_early_inc_range(BB)) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (CI && CI->getCalledFunction() == Fence) {
          if (Last) {
            CI->eraseFromParent();
            continue;
          }
          Last = CI;
        } else if (I.mayReadOrWriteMemory() || I.isTerminator()) {
          Last = nullptr;
        }
      }
    }
  }

  /// atomicrmw as load, operation, store; an ordered one between fences.
  Value *sequentialRMW(IRBuilder<> &B, AtomicRMWInst &RMW) {
    // Release: what this thread did is done first; acquire: what others did
    // is seen after.
    bool Release = isReleaseOrStronger(RMW.getOrdering());
    bool Acquire = isAcquireOrStronger(RMW.getOrdering());
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
    if (Release)
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
    if (Acquire)
      fence(B);
    return Old;
  }

  /// cmpxchg as a load and a store when it succeeds; a weak one may fail
  /// anyway.
  void sequentialCmpXchg(AtomicCmpXchgInst &X) {
    IRBuilder<> B(&X);
    bool Release = isReleaseOrStronger(X.getSuccessOrdering());
    bool Acquire = isAcquireOrStronger(X.getSuccessOrdering()) ||
                   isAcquireOrStronger(X.getFailureOrdering());
    if (Release)
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
    if (Acquire)
      fence(B);
    R->takeName(&X);
    X.replaceAllUsesWith(R);
    X.eraseFromParent();
  }
};

bool is_gpu_module(const Module &M) {
  for (const GPUTarget *T : targets())
    if (T->isModule(M))
      return true;
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

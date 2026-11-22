// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// AIR: the module after the AIR target's portable passes
// (llc -mtriple=air -air-portable): calls of AIR's functions (air.*) and
// kernels whose system values are parameters described by !air.kernel.

#include "llvm_util/gpu_target.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Metadata.h"

namespace llvm_util {
namespace gpu {

StringRef opencl_builtin(StringRef Name);

namespace {

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

struct AIRLowering : GPULowering {
  using GPULowering::GPULowering;

  bool lowerCall(IRBuilder<> &B, CallInst &CI, StringRef Name, Value *&R) {
    auto Arg = [&](unsigned I) { return CI.getArgOperand(I); };
    unsigned N = CI.arg_size();
    Type *Ty = CI.getType();

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

};

class AIRTarget : public GPUTarget {
public:
  const char *name() const override { return "air"; }

  bool isModule(const Module &M) const override {
    if (StringRef(M.getTargetTriple().str()).starts_with("air") ||
        M.getNamedMetadata("air.kernel"))
      return true;
    for (const Function &F : M)
      if (F.getName().starts_with("air.") || F.hasFnAttribute("air-kernel"))
        return true;
    return false;
  }

  bool isKernel(const Function &Fn) const override {
    if (Fn.hasFnAttribute("air-kernel"))
      return true;
    if (NamedMDNode *MD = Fn.getParent()->getNamedMetadata("air.kernel"))
      for (MDNode *N : MD->operands())
        if (N->getNumOperands() &&
            mdconst::dyn_extract_or_null<Function>(N->getOperand(0)) == &Fn)
          return true;
    return false;
  }

  /// Parameter roles from !air.kernel.
  void kernelRoles(const Function &F,
                   SmallVectorImpl<string> &R) const override {
    const Module &M = *F.getParent();
    NamedMDNode *MD = M.getNamedMetadata("air.kernel");
    if (!MD)
      return;
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
  }


  bool lowerCall(GPUBuilder &G, IRBuilder<> &B, CallInst &CI, StringRef Name,
                 Value *&R) const override {
    return AIRLowering(G).lowerCall(B, CI, Name, R);
  }
};

} // namespace

const GPUTarget &air_target() {
  static const AIRTarget T;
  return T;
}

} // namespace gpu
} // namespace llvm_util

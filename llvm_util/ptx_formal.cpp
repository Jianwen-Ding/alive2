// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// The reader and interpreter of the PTX formalization (ptx_formal.h): the
// same as scripts/ptx-tv/ptx_formal.py, the format's other interpreter.
// Which instructions there are, and what they compute, is all ptx.formal's.

#include "llvm_util/ptx_formal.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include <cctype>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace llvm;
using namespace std;

namespace llvm_util::gpu {

namespace {

// The formalization's text, as the build found it.
const char *const BuiltinText =
#include "ptx_formal_text.inc"
    ;

// ---------------------------------------------------------------------------
// S-expressions.

struct SExpr {
  enum Kind { Int, Float, Str, Sym, List } K = List;
  int64_t I = 0;
  double F = 0;
  string S;
  vector<SExpr> L;

  bool isSym(StringRef N) const { return K == Sym && S == N; }
  bool isList() const { return K == List; }
  StringRef head() const {
    return K == List && !L.empty() && L[0].K == Sym ? StringRef(L[0].S)
                                                    : StringRef();
  }
};

struct FormalError : runtime_error {
  using runtime_error::runtime_error;
};

vector<string> tokenize(StringRef T) {
  vector<string> Out;
  size_t I = 0, N = T.size();
  while (I < N) {
    char C = T[I];
    if (C == ';') {
      while (I < N && T[I] != '\n')
        ++I;
    } else if (isspace((unsigned char)C)) {
      ++I;
    } else if (C == '(' || C == ')') {
      Out.push_back(string(1, C));
      ++I;
    } else if (C == '"') {
      size_t J = T.find('"', I + 1);
      if (J == StringRef::npos)
        throw FormalError("unterminated string");
      Out.push_back(T.substr(I, J - I + 1).str());
      I = J + 1;
    } else {
      size_t J = I;
      while (J < N && !isspace((unsigned char)T[J]) && T[J] != '(' &&
             T[J] != ')' && T[J] != ';')
        ++J;
      Out.push_back(T.substr(I, J - I).str());
      I = J;
    }
  }
  return Out;
}

SExpr atom(const string &T) {
  SExpr E;
  if (T[0] == '"') {
    E.K = SExpr::Str;
    E.S = T.substr(1, T.size() - 2);
    return E;
  }
  StringRef R(T);
  int64_t V;
  if (!R.getAsInteger(10, V)) {
    E.K = SExpr::Int;
    E.I = V;
    return E;
  }
  if (R.contains('.') && (isdigit((unsigned char)R[0]) ||
                          (R[0] == '-' && R.size() > 1))) {
    double D;
    if (!R.getAsDouble(D)) {
      E.K = SExpr::Float;
      E.F = D;
      return E;
    }
  }
  E.K = SExpr::Sym;
  E.S = T;
  return E;
}

vector<SExpr> parseAll(StringRef Text) {
  vector<string> Toks = tokenize(Text);
  vector<SExpr> Top;
  vector<SExpr> Stack;
  for (const string &T : Toks) {
    if (T == "(") {
      Stack.emplace_back();
    } else if (T == ")") {
      if (Stack.empty())
        throw FormalError("unbalanced )");
      SExpr E = std::move(Stack.back());
      Stack.pop_back();
      (Stack.empty() ? Top : Stack.back().L).push_back(std::move(E));
    } else {
      (Stack.empty() ? Top : Stack.back().L).push_back(atom(T));
    }
  }
  if (!Stack.empty())
    throw FormalError("unbalanced (");
  return Top;
}

SExpr subst(const SExpr &E, const map<string, SExpr> &M) {
  if (E.K == SExpr::Sym) {
    auto It = M.find(E.S);
    return It == M.end() ? E : It->second;
  }
  if (E.K != SExpr::List)
    return E;
  SExpr R;
  for (const SExpr &C : E.L)
    R.L.push_back(subst(C, M));
  return R;
}

// ---------------------------------------------------------------------------
// Types.

const vector<string> &typeClass(StringRef C) {
  static const vector<string> Int = {"s8", "s16", "s32", "s64",
                                     "u8", "u16", "u32", "u64"};
  static const map<string, vector<string>> Classes = [] {
    map<string, vector<string>> M;
    vector<string> S, U, Bits = {"b8", "b16", "b32", "b64"};
    for (const string &T : Int)
      (T[0] == 's' ? S : U).push_back(T);
    M["int"] = Int;
    M["sint"] = S;
    M["uint"] = U;
    M["int16-32"] = {"s16", "u16", "s32", "u32"};
    M["int32-64"] = {"s32", "u32", "s64", "u64"};
    M["bits"] = Bits;
    M["b32"] = {"b32"};
    M["bits32-64"] = {"b32", "b64"};
    vector<string> BP = Bits;
    BP.push_back("pred");
    M["bitsp"] = BP;
    vector<string> UB = U;
    UB.insert(UB.end(), Bits.begin(), Bits.end());
    M["ubits"] = UB;
    vector<string> CI = Int;
    CI.insert(CI.end(), Bits.begin(), Bits.end());
    M["cmpint"] = CI;
    M["float"] = {"f16", "bf16", "f32", "f64"};
    M["f32"] = {"f32"};
    M["f64"] = {"f64"};
    vector<string> Any = CI;
    Any.insert(Any.end(), {"f16", "bf16", "f32", "f64", "pred"});
    M["any"] = Any;
    return M;
  }();
  auto It = Classes.find(C.str());
  if (It == Classes.end())
    throw FormalError("type class " + C.str());
  return It->second;
}

unsigned typeWidth(StringRef T) {
  if (T == "pred")
    return 1;
  unsigned W = 0;
  for (char C : T)
    if (isdigit((unsigned char)C))
      W = W * 10 + (C - '0');
  return W;
}

// ---------------------------------------------------------------------------
// Entries.

struct Entry {
  vector<SExpr> Pattern;
  vector<string> Operands;
  SExpr Body;
};

struct Bindings {
  map<string, string> Vals; // type variables, (one N ...) names, has:m
  vector<string> Vars;      // the type variables, in order
};

bool matchFrom(const vector<SExpr> &P, size_t I, const vector<string> &Parts,
               size_t J, Bindings &B) {
  if (I == P.size())
    return J == Parts.size();
  const SExpr &E = P[I];
  if (E.head() == "opt") {
    if (J < Parts.size() && Parts[J] == E.L[1].S) {
      Bindings B2 = B;
      B2.Vals["has:" + E.L[1].S] = "1";
      if (matchFrom(P, I + 1, Parts, J + 1, B2)) {
        B = std::move(B2);
        return true;
      }
    }
    return matchFrom(P, I + 1, Parts, J, B);
  }
  if (J >= Parts.size())
    return false;
  const string &T = Parts[J];
  if (E.head() == "one") {
    bool Found = false;
    for (size_t K = 2; K < E.L.size(); ++K)
      Found |= E.L[K].S == T;
    if (!Found)
      return false;
    Bindings B2 = B;
    B2.Vals[E.L[1].S] = T;
    if (!matchFrom(P, I + 1, Parts, J + 1, B2))
      return false;
    B = std::move(B2);
    return true;
  }
  if (E.K == SExpr::Sym && !E.S.empty() && E.S[0] == '$') {
    StringRef Spec = StringRef(E.S).drop_front();
    auto [Var, Cls] = Spec.split(':');
    const vector<string> &C = typeClass(Cls);
    if (find(C.begin(), C.end(), T) == C.end())
      return false;
    Bindings B2 = B;
    auto It = B.Vals.find(Var.str());
    if (It != B.Vals.end()) {
      if (It->second != T)
        return false;
    } else {
      B2.Vals[Var.str()] = T;
      B2.Vars.push_back(Var.str());
    }
    if (!matchFrom(P, I + 1, Parts, J + 1, B2))
      return false;
    B = std::move(B2);
    return true;
  }
  if (E.K == SExpr::Sym && E.S == T)
    return matchFrom(P, I + 1, Parts, J + 1, B);
  return false;
}

// ---------------------------------------------------------------------------
// The interpreter.

struct NeedWidth {};

struct Val {
  Value *V;
  unsigned W;
};

struct Interp;

struct Thunk {
  SExpr Expr;
  map<string, shared_ptr<Thunk>> Env;
  map<unsigned, Val> Cache; // by wanted width (0: none)
};

using Env = map<string, shared_ptr<Thunk>>;

Type *fpType(LLVMContext &C, unsigned W) {
  switch (W) {
  case 16: return Type::getHalfTy(C);
  case 32: return Type::getFloatTy(C);
  case 64: return Type::getDoubleTy(C);
  }
  throw FormalError("no float type of " + to_string(W) + " bits");
}

struct Interp {
  const map<string, pair<vector<string>, SExpr>> &Defines;
  IRBuilder<> &B;
  const Entry &E;
  const Bindings &Bd;
  ArrayRef<Value *> Args;
  string First;
  map<string, pair<unsigned, string>> Ops; // name -> (index, annotation)
  map<string, Val> OpCache;

  Interp(const map<string, pair<vector<string>, SExpr>> &D, IRBuilder<> &B,
         const Entry &E, const Bindings &Bd, ArrayRef<Value *> Args)
      : Defines(D), B(B), E(E), Bd(Bd), Args(Args) {
    if (!Bd.Vars.empty())
      First = Bd.Vars[0];
    for (unsigned I = 0; I < E.Operands.size(); ++I) {
      auto [Name, Ann] = StringRef(E.Operands[I]).split(':');
      Ops[Name.str()] = {I, Ann.str()};
    }
  }

  unsigned varWidth(const string &V) const {
    auto It = Bd.Vals.find(V);
    if (It == Bd.Vals.end())
      throw FormalError("type variable " + V + " unbound");
    return typeWidth(It->second);
  }

  bool isSigned(const string &V) const {
    auto It = Bd.Vals.find(V.empty() ? First : V);
    return It != Bd.Vals.end() && !It->second.empty() &&
           It->second[0] == 's';
  }

  optional<int64_t> stat(const SExpr &X) const {
    if (X.K == SExpr::Int)
      return X.I;
    if (X.K == SExpr::Sym) {
      if (X.S == "W")
        return varWidth(First);
      if (X.S == "WD" || X.S == "WS")
        return varWidth(X.S.substr(1));
      return nullopt;
    }
    StringRef H = X.head();
    if (H == "+" || H == "-" || H == "*") {
      optional<int64_t> R;
      for (size_t I = 1; I < X.L.size(); ++I) {
        optional<int64_t> V = stat(X.L[I]);
        if (!V)
          return nullopt;
        if (!R)
          R = *V;
        else
          R = H == "+" ? *R + *V : H == "-" ? *R - *V : *R * *V;
      }
      return R;
    }
    return nullopt;
  }

  int64_t statOrFail(const SExpr &X) const {
    optional<int64_t> V = stat(X);
    if (!V)
      throw FormalError("not a static value");
    return *V;
  }

  pair<unsigned, bool> annWidth(StringRef Ann) const {
    if (Ann.empty())
      return {varWidth(First), isSigned(First)};
    if (Ann == "pred")
      return {1, false};
    if (Bd.Vals.count(Ann.str()))
      return {varWidth(Ann.str()), isSigned(Ann.str())};
    // kW, kWD, kWS.
    size_t P = Ann.find('W');
    if (P != StringRef::npos) {
      unsigned K = 1;
      if (P)
        Ann.substr(0, P).getAsInteger(10, K);
      string V = Ann.substr(P + 1).empty() ? First : Ann.substr(P + 1).str();
      return {K * varWidth(V), isSigned(V)};
    }
    unsigned N;
    if (!Ann.getAsInteger(10, N))
      return {N, false};
    throw FormalError("operand annotation " + Ann.str());
  }

  Val lit(int64_t V, unsigned W) {
    APInt A(64, (uint64_t)V, true);
    A = W <= 64 ? A.trunc(W) : A.sext(W);
    return {ConstantInt::get(B.getContext(), A), W};
  }

  Value *resize(Value *V, unsigned From, unsigned To, bool Signed) {
    if (From == To)
      return V;
    Type *T = B.getIntNTy(To);
    if (From > To)
      return B.CreateTrunc(V, T);
    return Signed ? B.CreateSExt(V, T) : B.CreateZExt(V, T);
  }

  Val operand(const string &Name) {
    auto [I, Ann] = Ops.at(Name);
    auto [W, Sg] = annWidth(Ann);
    if (I == 0)
      throw FormalError("the destination " + Name + " read");
    if (I - 1 >= Args.size())
      throw FormalError("operand " + Name + " missing");
    Value *A = Args[I - 1];
    unsigned AW = A->getType()->getIntegerBitWidth();
    return {resize(A, AW, W, Sg), W};
  }

  /// The float type of W bits: bf16 for 16 bits where the instruction's
  /// types are bf16 (and not f16).
  Type *ftype(unsigned W) {
    if (W == 16) {
      bool BF = false, F16 = false;
      for (const string &V : Bd.Vars) {
        BF |= Bd.Vals.at(V) == "bf16";
        F16 |= Bd.Vals.at(V) == "f16";
      }
      if (BF && !F16)
        return Type::getBFloatTy(B.getContext());
    }
    return fpType(B.getContext(), W);
  }
  Value *fp(Value *V, unsigned W) { return B.CreateBitCast(V, ftype(W)); }
  Value *unfp(Value *V, unsigned W) {
    return B.CreateBitCast(V, B.getIntNTy(W));
  }

  /// args at one width: Want's, or that of those that have one.
  pair<vector<Val>, unsigned> same(ArrayRef<SExpr> As, Env &En,
                                   unsigned Want) {
    vector<optional<Val>> Vs(As.size());
    unsigned W = Want;
    for (size_t I = 0; I < As.size(); ++I) {
      try {
        Vs[I] = ev(As[I], En, Want);
        W = Vs[I]->W;
      } catch (NeedWidth &) {
      }
    }
    if (!W)
      throw NeedWidth();
    vector<Val> Out;
    for (size_t I = 0; I < As.size(); ++I) {
      if (!Vs[I])
        Vs[I] = ev(As[I], En, W);
      if (Vs[I]->W != W)
        throw FormalError("operands of " + to_string(Vs[I]->W) + " and " +
                          to_string(W) + " bits");
      Out.push_back(*Vs[I]);
    }
    return {Out, W};
  }

  Val thunk(Thunk &T, unsigned Want) {
    if (auto It = T.Cache.find(0); It != T.Cache.end())
      return It->second;
    if (auto It = T.Cache.find(Want); It != T.Cache.end())
      return It->second;
    try {
      Val V = ev(T.Expr, T.Env, 0);
      T.Cache[0] = V;
      return V;
    } catch (NeedWidth &) {
      if (!Want)
        throw;
    }
    Val V = ev(T.Expr, T.Env, Want);
    T.Cache[Want] = V;
    return V;
  }

  Env bind(const SExpr &Bs, const Env &En) {
    Env E2 = En;
    for (const SExpr &Bnd : Bs.L) {
      auto T = make_shared<Thunk>();
      T->Expr = Bnd.L[1];
      T->Env = E2;
      E2[Bnd.L[0].S] = T;
    }
    return E2;
  }

  Value *intr(Intrinsic::ID ID, ArrayRef<Type *> Tys, ArrayRef<Value *> As) {
    return B.CreateIntrinsic(ID, Tys, As);
  }

  Val ev(const SExpr &X, Env &En, unsigned Want) {
    LLVMContext &C = B.getContext();
    if (X.K == SExpr::Float) {
      if (!Want)
        throw NeedWidth();
      APFloat F(X.F);
      bool Lost;
      F.convert(ftype(Want)->getFltSemantics(),
                APFloat::rmNearestTiesToEven, &Lost);
      return {ConstantInt::get(C, F.bitcastToAPInt()), Want};
    }
    if (optional<int64_t> S = stat(X)) {
      if (!Want)
        throw NeedWidth();
      return lit(*S, Want);
    }
    if (X.K == SExpr::Sym) {
      if (auto It = En.find(X.S); It != En.end())
        return thunk(*It->second, Want);
      if (Ops.count(X.S)) {
        if (!OpCache.count(X.S))
          OpCache.emplace(X.S, operand(X.S));
        return OpCache.at(X.S);
      }
      throw FormalError("unknown name " + X.S);
    }
    if (X.K != SExpr::List || X.L.empty())
      throw FormalError("expression");
    StringRef H = X.head();
    ArrayRef<SExpr> A = ArrayRef(X.L).drop_front();

    // Macros, bindings and static choices.
    if (auto It = Defines.find(H.str()); It != Defines.end()) {
      const auto &[Params, Body] = It->second;
      if (Params.size() != A.size())
        throw FormalError(H.str() + ": argument count");
      map<string, SExpr> M;
      for (size_t I = 0; I < Params.size(); ++I)
        M[Params[I]] = A[I];
      return ev(subst(Body, M), En, Want);
    }
    if (H == "let") {
      Env E2 = bind(A[0], En);
      return ev(A[1], E2, Want);
    }
    if (H == "case") {
      auto It = Bd.Vals.find(A[0].S);
      string Key = It == Bd.Vals.end() ? "" : It->second;
      for (const SExpr &Alt : A.drop_front())
        if (Alt.L[0].S == Key)
          return ev(Alt.L[1], En, Want);
      throw FormalError("case " + A[0].S + ": no " + Key);
    }
    if (H == "if-signed")
      return ev(A[isSigned("") ? 0 : 1], En, Want);
    if (H == "has")
      return lit(Bd.Vals.count("has:" + A[0].S) ? 1 : 0, 1);
    if (H == "if-has")
      return ev(A[Bd.Vals.count("has:" + A[0].S) ? 1 : 2], En, Want);

    // Directed roundings: LLVM's constrained intrinsics.
    if (H == "fp.rounded") {
      auto It = Bd.Vals.find(A[0].S);
      string Mode = It == Bd.Vals.end() ? A[0].S : It->second;
      SExpr Op = A[1];
      ArrayRef<SExpr> Rest = A.drop_front(2);
      if (Mode == "rn") {
        SExpr Call;
        Call.L.push_back(Op);
        Call.L.insert(Call.L.end(), Rest.begin(), Rest.end());
        return ev(Call, En, Want);
      }
      optional<RoundingMode> RM = StringSwitch<optional<RoundingMode>>(Mode)
                                      .Case("rz", RoundingMode::TowardZero)
                                      .Case("rm", RoundingMode::TowardNegative)
                                      .Case("rp", RoundingMode::TowardPositive)
                                      .Default(nullopt);
      if (!RM)
        throw FormalError("rounding " + Mode);
      IRBuilder<>::FastMathFlagGuard Guard(B);
      B.setIsFPConstrained(true);
      B.setDefaultConstrainedRounding(*RM);
      B.setDefaultConstrainedExcept(fp::ebIgnore);
      if (Op.S == "fp.narrow" || Op.S == "fp.from-sint" ||
          Op.S == "fp.from-uint") {
        Val X = ev(Rest[0], En, 0);
        unsigned N = statOrFail(Rest[1]);
        Value *R;
        if (Op.S == "fp.narrow")
          R = B.CreateFPTrunc(fp(X.V, X.W), ftype(N));
        else if (Op.S == "fp.from-sint")
          R = B.CreateSIToFP(X.V, ftype(N));
        else
          R = B.CreateUIToFP(X.V, ftype(N));
        return {unfp(R, N), N};
      }
      auto [Vs, W] = same(Rest, En, Want);
      SmallVector<Value *, 3> As;
      for (Val &V : Vs)
        As.push_back(fp(V.V, W));
      Value *R;
      if (Op.S == "fp.add")
        R = B.CreateFAdd(As[0], As[1]);
      else if (Op.S == "fp.sub")
        R = B.CreateFSub(As[0], As[1]);
      else if (Op.S == "fp.mul")
        R = B.CreateFMul(As[0], As[1]);
      else if (Op.S == "fp.div")
        R = B.CreateFDiv(As[0], As[1]);
      else if (Op.S == "fp.fma")
        R = B.CreateConstrainedFPCall(
            Intrinsic::getOrInsertDeclaration(
                B.GetInsertBlock()->getModule(),
                Intrinsic::experimental_constrained_fma, {As[0]->getType()}),
            As);
      else if (Op.S == "fp.sqrt")
        R = B.CreateConstrainedFPCall(
            Intrinsic::getOrInsertDeclaration(
                B.GetInsertBlock()->getModule(),
                Intrinsic::experimental_constrained_sqrt, {As[0]->getType()}),
            As);
      else
        throw FormalError("fp.rounded of " + Op.S);
      return {unfp(R, W), W};
    }
    if (H == "fp.ftz") {
      Val X = ev(A[0], En, Want);
      Type *F = ftype(X.W);
      APInt E = APInt::getZero(X.W), Mt = APInt::getZero(X.W);
      if (F->isHalfTy())
        E = APInt(16, 0x7c00), Mt = APInt(16, 0x3ff);
      else if (F->isBFloatTy())
        E = APInt(16, 0x7f80), Mt = APInt(16, 0x7f);
      else if (F->isFloatTy())
        E = APInt(32, 0x7f800000), Mt = APInt(32, 0x7fffff);
      else
        E = APInt(64, 0x7ff0000000000000ull), Mt = APInt(64, 0xfffffffffffffull);
      Value *Sub = B.CreateAnd(
          B.CreateICmpEQ(B.CreateAnd(X.V, ConstantInt::get(C, E)),
                         ConstantInt::get(C, APInt::getZero(X.W))),
          B.CreateICmpNE(B.CreateAnd(X.V, ConstantInt::get(C, Mt)),
                         ConstantInt::get(C, APInt::getZero(X.W))));
      Value *Sign = B.CreateAnd(
          X.V, ConstantInt::get(C, APInt::getSignMask(X.W)));
      return {B.CreateSelect(Sub, Sign, X.V), X.W};
    }
    if (H == "fp.sat" || H == "fp.relu") {
      Val X = ev(A[0], En, Want);
      Type *F = ftype(X.W);
      Value *Fv = fp(X.V, X.W);
      Value *R;
      if (H == "fp.sat") {
        // To [0, 1]; NaN to 0.
        R = B.CreateMinNum(B.CreateMaxNum(Fv, ConstantFP::get(F, 0.0)),
                           ConstantFP::get(F, 1.0));
      } else {
        // A negative result 0, a NaN the canonical NaN.
        Value *Canon = ConstantFP::get(
            F, APFloat(F->getFltSemantics(),
                       APInt::getSignedMaxValue(X.W)));
        Value *S1 = B.CreateSelect(
            B.CreateFCmpOLT(Fv, ConstantFP::get(F, 0.0)),
            ConstantFP::get(F, 0.0), Fv);
        R = B.CreateSelect(B.CreateFCmpUNO(Fv, Fv), Canon, S1);
      }
      return {unfp(R, X.W), X.W};
    }

    // Bit-vectors.
    optional<Instruction::BinaryOps> Bin =
        StringSwitch<optional<Instruction::BinaryOps>>(H)
            .Case("bvadd", Instruction::Add)
            .Case("bvsub", Instruction::Sub)
            .Case("bvmul", Instruction::Mul)
            .Case("bvand", Instruction::And)
            .Case("bvor", Instruction::Or)
            .Case("bvxor", Instruction::Xor)
            .Case("bvshl", Instruction::Shl)
            .Case("bvlshr", Instruction::LShr)
            .Case("bvashr", Instruction::AShr)
            .Default(nullopt);
    if (Bin) {
      auto [Vs, W] = same(A, En, Want);
      return {B.CreateBinOp(*Bin, Vs[0].V, Vs[1].V), W};
    }
    if (H == "udiv" || H == "sdiv" || H == "urem" || H == "srem") {
      auto [Vs, W] = same(A, En, Want);
      // The divisor where LLVM's division is defined: the entry guards the
      // others.
      Value *Bad = B.CreateICmpEQ(Vs[1].V, lit(0, W).V);
      if (H[0] == 's')
        Bad = B.CreateOr(
            Bad, B.CreateAnd(B.CreateICmpEQ(Vs[1].V, lit(-1, W).V),
                             B.CreateICmpEQ(
                                 Vs[0].V,
                                 ConstantInt::get(
                                     C, APInt::getSignedMinValue(W)))));
      Value *D = B.CreateSelect(Bad, lit(1, W).V, Vs[1].V);
      Instruction::BinaryOps Op = StringSwitch<Instruction::BinaryOps>(H)
                                      .Case("udiv", Instruction::UDiv)
                                      .Case("sdiv", Instruction::SDiv)
                                      .Case("urem", Instruction::URem)
                                      .Default(Instruction::SRem);
      return {B.CreateBinOp(Op, Vs[0].V, D), W};
    }
    optional<Intrinsic::ID> Int2 = StringSwitch<optional<Intrinsic::ID>>(H)
                                       .Case("umin", Intrinsic::umin)
                                       .Case("umax", Intrinsic::umax)
                                       .Case("smin", Intrinsic::smin)
                                       .Case("smax", Intrinsic::smax)
                                       .Case("fshl", Intrinsic::fshl)
                                       .Case("fshr", Intrinsic::fshr)
                                       .Default(nullopt);
    if (Int2) {
      auto [Vs, W] = same(A, En, Want);
      SmallVector<Value *, 3> As;
      for (Val &V : Vs)
        As.push_back(V.V);
      return {intr(*Int2, {B.getIntNTy(W)}, As), W};
    }
    if (H == "bvnot" || H == "bvneg") {
      Val V = ev(A[0], En, Want);
      return {H == "bvnot" ? B.CreateNot(V.V) : B.CreateNeg(V.V), V.W};
    }
    if (H == "abs" || H == "popcount" || H == "clz" || H == "bitreverse") {
      Val V = ev(A[0], En, Want);
      Type *T = B.getIntNTy(V.W);
      if (H == "abs")
        return {intr(Intrinsic::abs, {T}, {V.V, B.getFalse()}), V.W};
      if (H == "clz")
        return {intr(Intrinsic::ctlz, {T}, {V.V, B.getFalse()}), V.W};
      return {intr(H == "popcount" ? Intrinsic::ctpop : Intrinsic::bitreverse,
                   {T}, {V.V}),
              V.W};
    }
    optional<CmpInst::Predicate> ICmp =
        StringSwitch<optional<CmpInst::Predicate>>(H)
            .Case("=", CmpInst::ICMP_EQ)
            .Case("distinct", CmpInst::ICMP_NE)
            .Case("bvult", CmpInst::ICMP_ULT)
            .Case("bvule", CmpInst::ICMP_ULE)
            .Case("bvugt", CmpInst::ICMP_UGT)
            .Case("bvuge", CmpInst::ICMP_UGE)
            .Case("bvslt", CmpInst::ICMP_SLT)
            .Case("bvsle", CmpInst::ICMP_SLE)
            .Case("bvsgt", CmpInst::ICMP_SGT)
            .Case("bvsge", CmpInst::ICMP_SGE)
            .Default(nullopt);
    if (ICmp) {
      auto [Vs, W] = same(A, En, 0);
      return {B.CreateICmp(*ICmp, Vs[0].V, Vs[1].V), 1};
    }
    if (H == "ite") {
      Val Cond = ev(A[0], En, 1);
      auto [Vs, W] = same(A.drop_front(), En, Want);
      return {B.CreateSelect(Cond.V, Vs[0].V, Vs[1].V), W};
    }
    if (H == "zext" || H == "sext" || H == "trunc" || H == "zext-or-trunc" ||
        H == "ext" || H == "ext-by") {
      bool Sg;
      if (H == "ext-by") {
        Sg = isSigned(A[0].S);
        A = A.drop_front();
      } else {
        Sg = H == "sext" || (H == "ext" && isSigned(""));
      }
      unsigned N = statOrFail(A[1]);
      Val V = ev(A[0], En, 0);
      if ((H == "zext" || H == "sext" || H == "ext") && N < V.W)
        throw FormalError(H.str() + " to fewer bits");
      if (H == "trunc" && N > V.W)
        throw FormalError("trunc to more bits");
      return {resize(V.V, V.W, N, Sg), N};
    }
    if (H == "extract") {
      unsigned Hi = statOrFail(A[0]), Lo = statOrFail(A[1]);
      Val V = ev(A[2], En, 0);
      Value *R = Lo ? B.CreateLShr(V.V, Lo) : V.V;
      return {resize(R, V.W, Hi - Lo + 1, false), Hi - Lo + 1};
    }
    if (H == "concat") {
      Val Hi = ev(A[0], En, 0), Lo = ev(A[1], En, 0);
      unsigned W = Hi.W + Lo.W;
      Value *HZ = B.CreateShl(resize(Hi.V, Hi.W, W, false), Lo.W);
      return {B.CreateOr(HZ, resize(Lo.V, Lo.W, W, false)), W};
    }
    if (H == "any") {
      unsigned N = statOrFail(A[0]);
      return {B.CreateFreeze(PoisonValue::get(B.getIntNTy(N))), N};
    }

    // Floats.
    bool Contract = H == "fp.contract";
    StringRef FH = Contract ? StringRef(A[0].S) : H;
    if (Contract)
      A = A.drop_front();
    optional<Instruction::BinaryOps> FBin =
        StringSwitch<optional<Instruction::BinaryOps>>(FH)
            .Case("fp.add", Instruction::FAdd)
            .Case("fp.sub", Instruction::FSub)
            .Case("fp.mul", Instruction::FMul)
            .Case("fp.div", Instruction::FDiv)
            .Default(nullopt);
    if (FBin) {
      auto [Vs, W] = same(A, En, Want);
      Value *R = B.CreateBinOp(*FBin, fp(Vs[0].V, W), fp(Vs[1].V, W));
      if (Contract)
        if (auto *I = dyn_cast<Instruction>(R)) {
          FastMathFlags FMF;
          FMF.setAllowContract();
          I->setFastMathFlags(FMF);
        }
      return {unfp(R, W), W};
    }
    if (Contract)
      throw FormalError("fp.contract of " + FH.str());
    if (H == "fp.neg") {
      Val V = ev(A[0], En, Want);
      return {unfp(B.CreateFNeg(fp(V.V, V.W)), V.W), V.W};
    }
    optional<Intrinsic::ID> FInt = StringSwitch<optional<Intrinsic::ID>>(H)
                                       .Case("fp.sqrt", Intrinsic::sqrt)
                                       .Case("fp.abs", Intrinsic::fabs)
                                       .Case("fp.roundeven", Intrinsic::roundeven)
                                       .Case("fp.trunc", Intrinsic::trunc)
                                       .Case("fp.floor", Intrinsic::floor)
                                       .Case("fp.ceil", Intrinsic::ceil)
                                       .Case("fp.minnum", Intrinsic::minnum)
                                       .Case("fp.maxnum", Intrinsic::maxnum)
                                       .Case("fp.minimumnum", Intrinsic::minimumnum)
                                       .Case("fp.maximumnum", Intrinsic::maximumnum)
                                       .Case("fp.minimum", Intrinsic::minimum)
                                       .Case("fp.maximum", Intrinsic::maximum)
                                       .Case("fp.copysign", Intrinsic::copysign)
                                       .Case("fp.fma", Intrinsic::fma)
                                       .Default(nullopt);
    if (FInt) {
      auto [Vs, W] = same(A, En, Want);
      SmallVector<Value *, 3> As;
      for (Val &V : Vs)
        As.push_back(fp(V.V, W));
      return {unfp(intr(*FInt, {ftype(W)}, As), W), W};
    }
    if (H == "fcmp") {
      auto [Vs, W] = same(A.drop_front(), En, 0);
      CmpInst::Predicate P = StringSwitch<CmpInst::Predicate>(A[0].S)
                                 .Case("oeq", CmpInst::FCMP_OEQ)
                                 .Case("one", CmpInst::FCMP_ONE)
                                 .Case("olt", CmpInst::FCMP_OLT)
                                 .Case("ole", CmpInst::FCMP_OLE)
                                 .Case("ogt", CmpInst::FCMP_OGT)
                                 .Case("oge", CmpInst::FCMP_OGE)
                                 .Case("ueq", CmpInst::FCMP_UEQ)
                                 .Case("une", CmpInst::FCMP_UNE)
                                 .Case("ult", CmpInst::FCMP_ULT)
                                 .Case("ule", CmpInst::FCMP_ULE)
                                 .Case("ugt", CmpInst::FCMP_UGT)
                                 .Case("uge", CmpInst::FCMP_UGE)
                                 .Case("ord", CmpInst::FCMP_ORD)
                                 .Case("uno", CmpInst::FCMP_UNO)
                                 .Default(CmpInst::BAD_FCMP_PREDICATE);
      if (P == CmpInst::BAD_FCMP_PREDICATE)
        throw FormalError("fcmp " + A[0].S);
      return {B.CreateFCmp(P, fp(Vs[0].V, W), fp(Vs[1].V, W)), 1};
    }
    if (H == "fp.extend" || H == "fp.narrow") {
      Val V = ev(A[0], En, 0);
      unsigned N = statOrFail(A[1]);
      if ((N > V.W) != (H == "fp.extend") || N == V.W)
        throw FormalError(H.str() + ": widths");
      Value *F = fp(V.V, V.W);
      Value *R = H == "fp.extend" ? B.CreateFPExt(F, ftype(N))
                                  : B.CreateFPTrunc(F, ftype(N));
      return {unfp(R, N), N};
    }
    if (H == "fp.from-sint" || H == "fp.from-uint") {
      Val V = ev(A[0], En, 0);
      unsigned N = statOrFail(A[1]);
      Value *R = H == "fp.from-sint" ? B.CreateSIToFP(V.V, ftype(N))
                                     : B.CreateUIToFP(V.V, ftype(N));
      return {unfp(R, N), N};
    }
    if (H == "fp.to-int-sat") {
      string Var = A[0].S;
      Val V = ev(A[1], En, 0);
      unsigned N = varWidth(Var);
      Value *R = intr(isSigned(Var) ? Intrinsic::fptosi_sat
                                    : Intrinsic::fptoui_sat,
                      {B.getIntNTy(N), ftype(V.W)}, {fp(V.V, V.W)});
      return {R, N};
    }
    if (H == "uninterpreted") {
      vector<Val> Vs;
      for (const SExpr &Y : A.drop_front())
        Vs.push_back(ev(Y, En, Want));
      unsigned W = Vs[0].W;
      Type *F = ftype(W);
      SmallVector<Type *, 3> Tys(Vs.size(), F);
      SmallVector<Value *, 3> As;
      for (Val &V : Vs)
        As.push_back(fp(V.V, W));
      Module *M = B.GetInsertBlock()->getModule();
      FunctionCallee Fn =
          M->getOrInsertFunction(A[0].S, FunctionType::get(F, Tys, false));
      if (auto *Fun = dyn_cast<Function>(Fn.getCallee())) {
        Fun->setDoesNotThrow();
        Fun->setDoesNotAccessMemory();
      }
      return {unfp(B.CreateCall(Fn, As), W), W};
    }
    throw FormalError("unknown operation " + H.str());
  }

  Value *run() {
    // The body: (set DEST EXPR), possibly in lets.
    Env En;
    const SExpr *S = &E.Body;
    while (S->head() == "let") {
      En = bind(S->L[1], En);
      S = &S->L[2];
    }
    if (S->head() != "set")
      throw FormalError("only one (set ...) per entry is interpreted here");
    auto [I, Ann] = Ops.at(S->L[1].S);
    if (I != 0)
      throw FormalError("set of a source operand");
    unsigned W = annWidth(Ann).first;
    Val V = ev(S->L[2], En, W);
    if (V.W != W)
      throw FormalError("the destination is " + to_string(W) +
                        " bits, its value " + to_string(V.W));
    return V.V;
  }
};

} // namespace

struct PTXFormal::Impl {
  vector<Entry> Entries;
  map<string, pair<vector<string>, SExpr>> Defines;
};

PTXFormal::PTXFormal() : P(make_unique<Impl>()) {}
PTXFormal::~PTXFormal() = default;

unique_ptr<PTXFormal> PTXFormal::parse(StringRef Text, string &Err) {
  unique_ptr<PTXFormal> F(new PTXFormal());
  try {
    for (SExpr &Form : parseAll(Text)) {
      StringRef H = Form.head();
      if (H == "define") {
        const SExpr &Sig = Form.L[1];
        vector<string> Params;
        for (size_t I = 1; I < Sig.L.size(); ++I)
          Params.push_back(Sig.L[I].S);
        F->P->Defines[Sig.L[0].S] = {Params, Form.L[2]};
      } else if (H == "instruction") {
        Entry E;
        E.Pattern = Form.L[1].L;
        for (const SExpr &O : Form.L[2].L)
          E.Operands.push_back(O.S);
        E.Body = Form.L[3];
        F->P->Entries.push_back(std::move(E));
      } else {
        throw FormalError("top-level form " + H.str());
      }
    }
  } catch (FormalError &X) {
    Err = X.what();
    return nullptr;
  }
  return F;
}

const PTXFormal &PTXFormal::builtin() {
  static unique_ptr<PTXFormal> F = [] {
    string Err;
    unique_ptr<PTXFormal> R = parse(BuiltinText, Err);
    if (!R)
      report_fatal_error(Twine("ptx.formal: ") + Err);
    return R;
  }();
  return *F;
}

Value *PTXFormal::lift(IRBuilder<> &B, StringRef Opcode,
                       ArrayRef<Value *> Args, string &Err) const {
  SmallVector<StringRef, 6> Split;
  Opcode.split(Split, '.');
  vector<string> Parts(Split.begin(), Split.end());
  for (const Entry &E : P->Entries) {
    if (E.Operands.size() != Args.size() + 1)
      continue;
    Bindings Bd;
    try {
      if (!matchFrom(E.Pattern, 0, Parts, 0, Bd))
        continue;
      return Interp(P->Defines, B, E, Bd, Args).run();
    } catch (FormalError &X) {
      Err = string("ptx.formal, ") + Opcode.str() + ": " + X.what();
      return nullptr;
    } catch (NeedWidth &) {
      Err = string("ptx.formal, ") + Opcode.str() + ": a literal's width";
      return nullptr;
    }
  }
  return nullptr;
}

string ptx_instruction_of_nvvm(StringRef N) {
  // Float rounding to integer and back: f2i.rz, d2ull.rn, i2f.rn, ll2d.rn,
  // d2f.rn (cvt; the integer roundings of cvt are rni, rzi, rmi, rpi).
  auto [Conv, Rnd] = N.split('.');
  auto intType = [](StringRef S) -> string {
    bool U = S.consume_front("u");
    if (S == "i")
      return U ? "u32" : "s32";
    if (S == "ll")
      return U ? "u64" : "s64";
    return "";
  };
  auto fType = [](StringRef S) -> string {
    return S == "f" ? "f32" : S == "d" ? "f64" : "";
  };
  if (Rnd == "rn" || Rnd == "rz" || Rnd == "rm" || Rnd == "rp") {
    auto [From, To] = Conv.split('2');
    if (!To.empty()) {
      if (string F = fType(From), I = intType(To); !F.empty() && !I.empty())
        return "cvt." + Rnd.str() + "i." + I + "." + F;
      if (string I = intType(From), F = fType(To); !F.empty() && !I.empty())
        return "cvt." + Rnd.str() + "." + F + "." + I;
      if (From == "d" && To == "f")
        return "cvt." + Rnd.str() + ".f32.f64";
    }
  }
  if (N == "prmt")
    return "prmt.b32";
  // div, rcp, sqrt with a rounding: .f and .d (not .ftz).
  for (StringRef Op : {"div", "rcp", "sqrt", "add", "mul", "fma"}) {
    StringRef R = N;
    if (!R.consume_front(Op) || !R.consume_front("."))
      continue;
    auto [Rn, Ty] = R.split('.');
    if ((Rn == "rn" || Rn == "rz" || Rn == "rm" || Rn == "rp") &&
        !fType(Ty).empty())
      return Op.str() + "." + Rn.str() + "." + fType(Ty);
  }
  if (N == "fmin.f" || N == "fmax.f" || N == "fmin.d" || N == "fmax.d")
    return N.substr(1, 3).str() + "." + fType(N.substr(5));
  if (N == "mulhi.i" || N == "mulhi.ui" || N == "mulhi.ll" ||
      N == "mulhi.ull")
    return "mul.hi." + intType(N.substr(6));
  return "";
}

} // namespace llvm_util::gpu

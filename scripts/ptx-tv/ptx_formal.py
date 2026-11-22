# Copyright (c) 2018-present The Alive2 Authors.
# Distributed under the MIT license that can be found in the LICENSE file.
"""The PTX formalization (ptx.formal): its reader, and its interpreter,
which gives an instruction the meaning its entry says, as LLVM IR.

The format is described at the top of ptx.formal. This file knows only the
format: which instructions there are, and what they compute, is all
ptx.formal's.
"""

import copy
import os
import re
import struct


class FormalError(Exception):
  pass


# ---------------------------------------------------------------------------
# S-expressions.


def tokenize(text):
  text = re.sub(r';[^\n]*', '', text)
  return re.findall(r'"[^"]*"|[()]|[^\s()]+', text)


def parse(tokens):
  out, stack = [], []
  for t in tokens:
    if t == '(':
      stack.append([])
    elif t == ')':
      if not stack:
        raise FormalError('unbalanced )')
      e = stack.pop()
      (stack[-1] if stack else out).append(e)
    else:
      a = atom(t)
      (stack[-1] if stack else out).append(a)
  if stack:
    raise FormalError('unbalanced (')
  return out


class Str(str):
  pass


class FloatLit(float):
  pass


def atom(t):
  if t.startswith('"'):
    return Str(t[1:-1])
  if re.fullmatch(r'-?\d+', t):
    return int(t)
  if re.fullmatch(r'-?\d+\.\d*([eE][-+]?\d+)?', t):
    return FloatLit(float(t))
  return t


# ---------------------------------------------------------------------------
# Types.

INT = ['s8', 's16', 's32', 's64', 'u8', 'u16', 'u32', 'u64']
BITS = ['b8', 'b16', 'b32', 'b64']
FLOAT = ['f16', 'bf16', 'f32', 'f64']
CLASSES = {
    'int': INT,
    'sint': [t for t in INT if t[0] == 's'],
    'uint': [t for t in INT if t[0] == 'u'],
    'int16-32': ['s16', 'u16', 's32', 'u32'],
    'int32-64': ['s32', 'u32', 's64', 'u64'],
    'bits': BITS,
    'b32': ['b32'],
    'f32': ['f32'],
    'f64': ['f64'],
    'bits32-64': ['b32', 'b64'],
    'bitsp': BITS + ['pred'],
    'ubits': [t for t in INT if t[0] == 'u'] + BITS,
    'cmpint': INT + BITS,
    'float': FLOAT,
    'any': INT + BITS + FLOAT + ['pred'],
}


def width(ty):
  if ty == 'pred':
    return 1
  return int(re.search(r'\d+', ty).group(0))


def ftype(bits):
  return {16: 'half', 32: 'float', 64: 'double'}[bits]


# ---------------------------------------------------------------------------
# Entries.


class Entry:
  def __init__(self, pattern, operands, body, defines):
    self.pattern = pattern
    self.operands = operands
    self.body = body
    self.defines = defines

  def match(self, parts):
    """The bindings if the opcode's parts match the pattern, or None."""
    def go(i, j, b):
      if i == len(self.pattern):
        return b if j == len(parts) else None
      p = self.pattern[i]
      if isinstance(p, list) and p and p[0] == 'opt':
        if j < len(parts) and parts[j] == p[1]:
          r = go(i + 1, j + 1, dict(b, **{'has:' + p[1]: True}))
          if r is not None:
            return r
        return go(i + 1, j, b)
      if j >= len(parts):
        return None
      t = parts[j]
      if isinstance(p, list) and p and p[0] == 'one':
        if t not in p[2:]:
          return None
        return go(i + 1, j + 1, dict(b, **{p[1]: t}))
      if isinstance(p, str) and p.startswith('$'):
        var, cls = p[1:].split(':')
        if t not in CLASSES[cls]:
          return None
        if var in b and b[var] != t:
          return None
        nb = dict(b)
        nb[var] = t
        nb.setdefault('vars', [])
        nb['vars'] = b.get('vars', []) + ([var] if var not in b else [])
        return go(i + 1, j + 1, nb)
      return go(i + 1, j + 1, b) if p == t else None
    return go(0, 0, {})


class Formal:
  def __init__(self, path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'ptx.formal')
    self.entries = []
    self.defines = {}
    for form in parse(tokenize(open(path).read())):
      if not isinstance(form, list) or not form:
        raise FormalError('top-level form: %r' % (form,))
      if form[0] == 'define':
        sig, body = form[1], form[2]
        self.defines[sig[0]] = (sig[1:], body)
      elif form[0] == 'instruction':
        _, pattern, operands, body = form
        self.entries.append(Entry(pattern, operands, body, self.defines))
      else:
        raise FormalError('top-level form: %s' % form[0])

  def find(self, x):
    parts = x.op.split('.')
    nops = len(x.operands)
    for e in self.entries:
      if len(e.operands) != nops:
        continue
      b = e.match(parts)
      if b is not None:
        return e, b
    return None, None

  def lift(self, lifter, x):
    """Lift x by its entry: whether there was one. A packed instruction
    (add.rn.f16x2, all its types of two lanes) is its scalar entry's,
    lane by lane."""
    e, b = self.find(x)
    if e is not None:
      Interp(self, lifter, x, e, b).run()
      return True
    parts = x.op.split('.')
    packed = [p for p in parts[1:] if re.fullmatch(r'(f16|bf16|f32|[sub]16)x2', p)]
    if not packed or len(set(packed)) != 1 or any(
        p in ('f16', 'bf16', 'f32', 'f64') or re.fullmatch(r'[sub]\d+', p)
        for p in parts[1:]):
      return False
    scalar = copy.copy(x)
    scalar.op = '.'.join(p[:-2] if p == packed[0] else p for p in parts)
    scalar.mods = scalar.op.split('.')
    e, b = self.find(scalar)
    if e is None:
      return False
    lanes = [Interp(self, lifter, scalar, e, b, lane=i).value() for i in range(2)]
    w = lanes[0][1]
    lo = lifter.resize(lanes[0][0], w, 2 * w, False)
    hi = lifter.resize(lanes[1][0], w, 2 * w, False)
    sh, r = lifter.t('f'), lifter.t('f')
    lifter.emit('%s = shl i%d %s, %d' % (sh, 2 * w, hi, w))
    lifter.emit('%s = or i%d %s, %s' % (r, 2 * w, sh, lo))
    lifter.write_reg(x.operands[0], r, 'i%d' % (2 * w))
    return True


# ---------------------------------------------------------------------------
# The interpreter.


class NeedWidth(Exception):
  """A literal without a width from its context."""


class Thunk:
  def __init__(self, interp, expr, env):
    self.interp, self.expr, self.env = interp, expr, env
    self.cache = {}

  def get(self, want):
    if None in self.cache:
      return self.cache[None]
    if want in self.cache:
      return self.cache[want]
    try:
      v = self.interp.ev(self.expr, self.env, None)
      self.cache[None] = v
      return v
    except NeedWidth:
      if want is None:
        raise
    v = self.interp.ev(self.expr, self.env, want)
    self.cache[want] = v
    return v


SAME = {'bvadd': 'add', 'bvsub': 'sub', 'bvmul': 'mul', 'bvand': 'and',
        'bvor': 'or', 'bvxor': 'xor', 'bvshl': 'shl', 'bvlshr': 'lshr',
        'bvashr': 'ashr'}
CMP = {'=': 'eq', 'distinct': 'ne', 'bvult': 'ult', 'bvule': 'ule',
       'bvugt': 'ugt', 'bvuge': 'uge', 'bvslt': 'slt', 'bvsle': 'sle',
       'bvsgt': 'sgt', 'bvsge': 'sge'}
INTRINSIC2 = {'umin': 'umin', 'umax': 'umax', 'smin': 'smin',
              'smax': 'smax'}
FP2 = {'fp.add': 'fadd', 'fp.sub': 'fsub', 'fp.mul': 'fmul',
       'fp.div': 'fdiv'}
FP_INTR1 = {'fp.sqrt': 'sqrt', 'fp.abs': 'fabs', 'fp.roundeven': 'roundeven',
            'fp.trunc': 'trunc', 'fp.floor': 'floor', 'fp.ceil': 'ceil'}
FP_INTR2 = {'fp.minnum': 'minnum', 'fp.maxnum': 'maxnum',
            'fp.copysign': 'copysign'}


class Interp:
  def __init__(self, formal, lifter, x, entry, bindings, lane=None):
    self.f, self.L, self.x, self.e, self.b = formal, lifter, x, entry, bindings
    self.lane = lane
    vars_ = bindings.get('vars', [])
    self.first = vars_[0] if vars_ else None
    self.ops = {}
    for i, o in enumerate(entry.operands):
      name, _, ann = o.partition(':')
      self.ops[name] = (i, ann)

  # Static values.

  def var_width(self, v):
    if v not in self.b:
      raise FormalError('type variable %s unbound in %s' % (v, self.x))
    return width(self.b[v])

  def signed(self, v=None):
    v = v or self.first
    return self.b.get(v, 'u')[0] == 's'

  def static(self, e, env):
    if isinstance(e, bool):
      return int(e)
    if isinstance(e, int) and not isinstance(e, FloatLit):
      return e
    if isinstance(e, str) and not isinstance(e, Str):
      if e == 'W':
        return self.var_width(self.first)
      if e in ('WD', 'WS'):
        return self.var_width(e[1])
      if e in env and isinstance(env[e], int):
        return env[e]
      return None
    if isinstance(e, list) and e and e[0] in ('+', '-', '*'):
      a = [self.static(y, env) for y in e[1:]]
      if None in a:
        return None
      r = a[0]
      for y in a[1:]:
        r = r + y if e[0] == '+' else r - y if e[0] == '-' else r * y
      return r
    return None

  def ann_width(self, ann):
    if not ann:
      return self.var_width(self.first), self.signed()
    if ann == 'pred':
      return 1, False
    if ann in self.b:
      return self.var_width(ann), self.signed(ann)
    m = re.fullmatch(r'(\d*)W([DS]?)', ann)
    if m:
      k = int(m.group(1) or 1)
      v = m.group(2) or self.first
      return k * self.var_width(v), self.signed(v)
    if re.fullmatch(r'\d+', ann):
      return int(ann), False
    raise FormalError('operand annotation ' + ann)

  # Values: (text, width).

  def lit(self, v, w):
    v &= (1 << w) - 1
    if w == 1:
      return ('true' if v else 'false', 1)
    if v >> (w - 1):
      v -= 1 << w
    return (str(v), w)

  def operand(self, name):
    i, ann = self.ops[name]
    w, sg = self.ann_width(ann)
    tok = self.x.operands[i]
    if self.lane is not None:
      # A lane of a packed operand (lane 0 the low half).
      full = self.L.operand(tok, 2 * w)
      if self.lane:
        sh = self.L.t('f')
        self.L.emit('%s = lshr i%d %s, %d' % (sh, 2 * w, full, w))
        full = sh
      return (self.L.resize(full, 2 * w, w, False), w)
    return (self.L.operand(tok, w, sg), w)

  def emit(self, text):
    r = self.L.t('f')
    self.L.emit('%s = %s' % (r, text))
    return r

  def same(self, args, env, want):
    """Evaluate args at one width: want's, or that of those that have
    one."""
    vals = [None] * len(args)
    w = want
    for i, a in enumerate(args):
      try:
        vals[i] = self.ev(a, env, want)
        w = vals[i][1]
      except NeedWidth:
        pass
    if w is None:
      raise NeedWidth()
    for i, a in enumerate(args):
      if vals[i] is None:
        vals[i] = self.ev(a, env, w)
      if vals[i][1] != w:
        raise FormalError('widths %d and %d in %s' % (vals[i][1], w,
                                                      self.x))
    return vals, w

  def ftype(self, w):
    """The float type of w bits: bf16 for 16 bits where the instruction's
    types are bf16 (and not f16)."""
    if w == 16:
      tys = set(self.b.get(v) for v in self.b.get('vars', []))
      if 'bf16' in tys and 'f16' not in tys:
        return 'bfloat'
    return ftype(w)

  def fp(self, v, w):
    return self.L.cast('bitcast', v, 'i%d' % w, self.ftype(w))

  def unfp(self, v, w):
    return self.L.cast('bitcast', v, self.ftype(w), 'i%d' % w)

  def ev(self, e, env, want):
    if isinstance(e, FloatLit):
      if want is None:
        raise NeedWidth()
      fmt = {16: '<e', 32: '<f', 64: '<d'}[want]
      bits = int.from_bytes(struct.pack(fmt, e), 'little')
      return self.lit(bits, want)
    s = self.static(e, env)
    if s is not None:
      if want is None:
        raise NeedWidth()
      return self.lit(s, want)
    if isinstance(e, str):
      if e in env:
        v = env[e]
        if isinstance(v, Thunk):
          return v.get(want)
        return v
      if e in self.ops:
        if e not in self.cache:
          self.cache[e] = self.operand(e)
        return self.cache[e]
      raise FormalError('unknown name %s' % e)
    if not isinstance(e, list) or not e:
      raise FormalError('expression %r' % (e,))
    h, a = e[0], e[1:]
    L = self.L

    # Macros, bindings and static choices.
    if h in self.f.defines:
      params, body = self.f.defines[h]
      if len(params) != len(a):
        raise FormalError('%s takes %d arguments' % (h, len(params)))
      return self.ev(subst(body, dict(zip(params, a))), env, want)
    if h == 'let':
      env2 = dict(env)
      for name, ex in a[0]:
        env2[name] = Thunk(self, ex, dict(env2))
      return self.ev(a[1], env2, want)
    if h == 'case':
      key = self.b.get(a[0])
      for alt in a[1:]:
        if alt[0] == key:
          return self.ev(alt[1], env, want)
      raise FormalError('case %s: no %s' % (a[0], key))
    if h == 'if-signed':
      return self.ev(a[0] if self.signed() else a[1], env, want)
    if h == 'has':
      return self.lit(int(self.b.get('has:' + a[0], False)), 1)
    if h == 'if-has':
      return self.ev(a[1] if self.b.get('has:' + a[0]) else a[2], env, want)

    # Bit-vectors.
    if h in SAME:
      (x, y), w = self.same(a, env, want)
      return (self.emit('%s i%d %s, %s' % (SAME[h], w, x[0], y[0])), w)
    if h in ('udiv', 'sdiv', 'urem', 'srem'):
      (x, y), w = self.same(a, env, want)
      T = 'i%d' % w
      # The divisor where LLVM's division is defined: the entry guards the
      # others.
      bad = self.emit('icmp eq %s %s, 0' % (T, y[0]))
      if h[0] == 's':
        m1 = self.emit('icmp eq %s %s, -1' % (T, y[0]))
        mn = self.emit('icmp eq %s %s, %d' % (T, x[0], -(1 << (w - 1))))
        ov = self.emit('and i1 %s, %s' % (m1, mn))
        bad = self.emit('or i1 %s, %s' % (bad, ov))
      d = self.emit('select i1 %s, %s 1, %s %s' % (bad, T, T, y[0]))
      return (self.emit('%s %s %s, %s' % (h, T, x[0], d)), w)
    if h in INTRINSIC2:
      (x, y), w = self.same(a, env, want)
      T = 'i%d' % w
      return (L.intr(INTRINSIC2[h], T, [(T, x[0]), (T, y[0])]), w)
    if h in ('fshl', 'fshr'):
      vals, w = self.same(a, env, want)
      T = 'i%d' % w
      return (L.intr(h, T, [(T, v[0]) for v in vals]), w)
    if h == 'bvnot':
      x = self.ev(a[0], env, want)
      return (self.emit('xor i%d %s, -1' % (x[1], x[0])), x[1])
    if h == 'bvneg':
      x = self.ev(a[0], env, want)
      return (self.emit('sub i%d 0, %s' % (x[1], x[0])), x[1])
    if h == 'abs':
      x = self.ev(a[0], env, want)
      T = 'i%d' % x[1]
      return (L.intr('abs', T, [(T, x[0]), ('i1', 'false')]), x[1])
    if h in ('popcount', 'clz', 'bitreverse'):
      x = self.ev(a[0], env, want)
      T = 'i%d' % x[1]
      args = [(T, x[0])] + ([('i1', 'false')] if h == 'clz' else [])
      name = {'popcount': 'ctpop', 'clz': 'ctlz'}.get(h, h)
      return (L.intr(name, T, args), x[1])
    if h in CMP:
      (x, y), w = self.same(a, env, None)
      return (self.emit('icmp %s i%d %s, %s' % (CMP[h], w, x[0], y[0])), 1)
    if h == 'ite':
      c = self.ev(a[0], env, 1)
      (x, y), w = self.same(a[1:], env, want)
      T = 'i%d' % w
      return (self.emit('select i1 %s, %s %s, %s %s' % (c[0], T, x[0], T,
                                                         y[0])), w)
    if h in ('zext', 'sext', 'trunc', 'zext-or-trunc', 'ext', 'ext-by'):
      if h == 'ext-by':
        sg, a = self.signed(a[0]), a[1:]
      else:
        sg = h == 'sext' or (h == 'ext' and self.signed())
      n = self.static(a[1], env)
      x = self.ev(a[0], env, None)
      if h in ('zext', 'sext', 'ext') and n < x[1]:
        raise FormalError('%s to fewer bits in %s' % (h, self.x))
      if h == 'trunc' and n > x[1]:
        raise FormalError('trunc to more bits in %s' % self.x)
      return (L.resize(x[0], x[1], n, sg), n)
    if h == 'extract':
      hi, lo = self.static(a[0], env), self.static(a[1], env)
      x = self.ev(a[2], env, None)
      v = x[0]
      if lo:
        v = self.emit('lshr i%d %s, %d' % (x[1], v, lo))
      return (L.resize(v, x[1], hi - lo + 1, False), hi - lo + 1)
    if h == 'concat':
      hi = self.ev(a[0], env, None)
      lo = self.ev(a[1], env, None)
      w = hi[1] + lo[1]
      hz = L.resize(hi[0], hi[1], w, False)
      lz = L.resize(lo[0], lo[1], w, False)
      hs = self.emit('shl i%d %s, %d' % (w, hz, lo[1]))
      return (self.emit('or i%d %s, %s' % (w, hs, lz)), w)
    if h == 'any':
      n = self.static(a[0], env)
      return (self.emit('freeze i%d poison' % n), n)

    # Floats.
    if h == 'fp.rounded':
      # An operation rounding as RND says: rn to nearest even, rz toward
      # zero, rm down, rp up (LLVM's constrained intrinsics).
      mode = self.b.get(a[0], a[0])
      op, args = a[1], a[2:]
      if mode == 'rn':
        return self.ev([op] + args, env, want)
      md = {'rz': 'round.towardzero', 'rm': 'round.downward',
            'rp': 'round.upward'}[mode]
      sfxs = {'half': 'f16', 'bfloat': 'bf16', 'float': 'f32',
              'double': 'f64'}
      if op in ('fp.narrow', 'fp.from-sint', 'fp.from-uint'):
        x = self.ev(args[0], env, None)
        n = self.static(args[1], env)
        F = self.ftype(n)
        if op == 'fp.narrow':
          S = self.ftype(x[1])
          name = 'llvm.experimental.constrained.fptrunc.%s.%s' % (sfxs[F],
                                                                   sfxs[S])
          arg = (S, self.fp(x[0], x[1]))
        else:
          name = 'llvm.experimental.constrained.%s.%s.i%d' % (
              'sitofp' if op == 'fp.from-sint' else 'uitofp', sfxs[F], x[1])
          arg = ('i%d' % x[1], x[0])
        r = L.call(name, F, [arg, ('metadata', '!"%s"' % md),
                             ('metadata', '!"fpexcept.ignore"')])
        return (self.unfp(r, n), n)
      name = {'fp.add': 'fadd', 'fp.sub': 'fsub', 'fp.mul': 'fmul',
              'fp.div': 'fdiv', 'fp.fma': 'fma', 'fp.sqrt': 'sqrt'}[op]
      vals, w = self.same(args, env, want)
      F = self.ftype(w)
      sfx = {'half': 'f16', 'bfloat': 'bf16', 'float': 'f32',
             'double': 'f64'}[F]
      r = L.call('llvm.experimental.constrained.%s.%s' % (name, sfx), F,
                 [(F, self.fp(v[0], w)) for v in vals] +
                 [('metadata', '!"%s"' % md),
                  ('metadata', '!"fpexcept.ignore"')])
      return (self.unfp(r, w), w)
    if h in ('fp.ftz', 'fp.sat', 'fp.relu'):
      x = self.ev(a[0], env, want)
      w = x[1]
      F = self.ftype(w)
      if h == 'fp.ftz':
        # A subnormal is a zero of its sign.
        e_bits, m_bits = {'half': (0x7c00, 0x3ff), 'bfloat': (0x7f80, 0x7f),
                          'float': (0x7f800000, 0x7fffff),
                          'double': (0x7ff0000000000000,
                                     0xfffffffffffff)}[F]
        T = 'i%d' % w
        e_, m_, z, nz, sub, sign, r = [self.L.t('f') for _ in range(7)]
        self.L.emit('%s = and %s %s, %d' % (e_, T, x[0], signed_(e_bits, w)))
        self.L.emit('%s = and %s %s, %d' % (m_, T, x[0], m_bits))
        self.L.emit('%s = icmp eq %s %s, 0' % (z, T, e_))
        self.L.emit('%s = icmp ne %s %s, 0' % (nz, T, m_))
        self.L.emit('%s = and i1 %s, %s' % (sub, z, nz))
        self.L.emit('%s = and %s %s, %d' % (sign, T, x[0], -(1 << (w - 1))))
        self.L.emit('%s = select i1 %s, %s %s, %s %s' % (r, sub, T, sign, T, x[0]))
        return (r, w)
      f = self.fp(x[0], w)
      if h == 'fp.sat':
        # To [0, 1]; NaN to 0.
        zero = L.intr('maxnum', F, [(F, f), (F, '0.0')])
        r = L.intr('minnum', F, [(F, zero), (F, '1.0')])
        return (self.unfp(r, w), w)
      # relu: a negative result is 0, a NaN the canonical NaN.
      nan, neg, s1, r = [self.L.t('f') for _ in range(4)]
      canon = {'half': '0xH7FFF', 'bfloat': '0xR7FFF', 'float': '0x7FFFFFFFE0000000',
               'double': '0x7FFFFFFFFFFFFFFF'}[F]
      self.L.emit('%s = fcmp uno %s %s, %s' % (nan, F, f, f))
      self.L.emit('%s = fcmp olt %s %s, 0.0' % (neg, F, f))
      self.L.emit('%s = select i1 %s, %s 0.0, %s %s' % (s1, neg, F, F, f))
      self.L.emit('%s = select i1 %s, %s %s, %s %s' % (r, nan, F, canon, F, s1))
      return (self.unfp(r, w), w)
    if h in ('fp.minimum', 'fp.maximum', 'fp.minimumnum', 'fp.maximumnum'):
      (x, y), w = self.same(a, env, want)
      F = self.ftype(w)
      r = L.intr(h[3:], F, [(F, self.fp(x[0], w)), (F, self.fp(y[0], w))])
      return (self.unfp(r, w), w)
    if h in FP2 or (h == 'fp.contract' and a[0] in FP2):
      flags = ''
      if h == 'fp.contract':
        h, a, flags = a[0], a[1:], 'contract '
      (x, y), w = self.same(a, env, want)
      r = self.emit('%s %s%s %s, %s' % (FP2[h], flags, self.ftype(w),
                                        self.fp(x[0], w), self.fp(y[0], w)))
      return (self.unfp(r, w), w)
    if h == 'fp.neg':
      x = self.ev(a[0], env, want)
      r = self.emit('fneg %s %s' % (self.ftype(x[1]), self.fp(x[0], x[1])))
      return (self.unfp(r, x[1]), x[1])
    if h in FP_INTR1 or h in FP_INTR2 or h == 'fp.fma':
      vals, w = self.same(a, env, want)
      F = self.ftype(w)
      name = FP_INTR1.get(h) or FP_INTR2.get(h) or 'fma'
      r = L.intr(name, F, [(F, self.fp(v[0], w)) for v in vals])
      return (self.unfp(r, w), w)
    if h == 'fcmp':
      (x, y), w = self.same(a[1:], env, None)
      F = self.ftype(w)
      return (self.emit('fcmp %s %s %s, %s' % (a[0], F, self.fp(x[0], w),
                                               self.fp(y[0], w))), 1)
    if h in ('fp.extend', 'fp.narrow'):
      x = self.ev(a[0], env, None)
      n = self.static(a[1], env)
      if (n > x[1]) != (h == 'fp.extend') or n == x[1]:
        raise FormalError('%s from %d to %d bits in %s' % (h, x[1], n,
                                                           self.x))
      op = 'fpext' if h == 'fp.extend' else 'fptrunc'
      r = L.cast(op, self.fp(x[0], x[1]), self.ftype(x[1]), self.ftype(n))
      return (self.unfp(r, n), n)
    if h in ('fp.from-sint', 'fp.from-uint'):
      x = self.ev(a[0], env, None)
      n = self.static(a[1], env)
      op = 'sitofp' if h == 'fp.from-sint' else 'uitofp'
      r = L.cast(op, x[0], 'i%d' % x[1], self.ftype(n))
      return (self.unfp(r, n), n)
    if h == 'fp.to-int-sat':
      v = a[0]
      x = self.ev(a[1], env, None)
      n = self.var_width(v)
      F = self.ftype(x[1])
      name = 'llvm.fpto%si.sat.i%d.%s' % ('s' if self.signed(v) else 'u', n,
                                          {'half': 'f16', 'float': 'f32',
                                           'bfloat': 'bf16',
                                           'double': 'f64'}[F])
      return (L.call(name, 'i%d' % n, [(F, self.fp(x[0], x[1]))]), n)
    if h == 'uninterpreted':
      vals = [self.ev(y, env, want) for y in a[1:]]
      w = vals[0][1]
      F = self.ftype(w)
      r = L.call(str(a[0]), F, [(F, self.fp(v[0], w)) for v in vals],
                 ' nounwind readnone')
      return (self.unfp(r, w), w)
    raise FormalError('unknown operation %s in %s' % (h, self.x))

  def run(self):
    self.cache = {}
    self.stmt(self.e.body, {})

  def value(self):
    """The destination's value (for a lane of a packed instruction)."""
    self.cache = {}
    s, env = self.e.body, {}
    while s[0] == 'let':
      env = dict(env)
      for name, ex in s[1]:
        env[name] = Thunk(self, ex, dict(env))
      s = s[2]
    i, ann = self.ops[s[1]]
    w, _ = self.ann_width(ann)
    return self.ev(s[2], env, w)

  def stmt(self, s, env):
    if s[0] == 'let':
      env2 = dict(env)
      for name, ex in s[1]:
        env2[name] = Thunk(self, ex, dict(env2))
      return self.stmt(s[2], env2)
    if s[0] == 'seq':
      for t in s[1:]:
        self.stmt(t, env)
      return
    if s[0] != 'set':
      raise FormalError('statement %s' % s[0])
    name = s[1]
    i, ann = self.ops[name]
    w, sg = self.ann_width(ann)
    v = self.ev(s[2], env, w)
    if v[1] != w:
      raise FormalError('%s is %d bits, its value %d in %s' % (name, w, v[1],
                                                              self.x))
    dest = self.x.operands[i]
    rb = self.L.reg(dest)
    val = v[0]
    if rb > w and w > 1:
      # The ISA's relaxed rule for destinations: extended by the type.
      val = self.L.resize(val, w, rb, sg)
      w = rb
    self.L.write_reg(dest, val, 'i%d' % w)


def signed_(v, w):
  return v - (1 << w) if v >> (w - 1) else v


def subst(e, m):
  if isinstance(e, list):
    return [subst(y, m) for y in e]
  if isinstance(e, str) and not isinstance(e, Str) and e in m:
    return m[e]
  return e

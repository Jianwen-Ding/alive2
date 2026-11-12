#!/usr/bin/env python3
# Copyright (c) 2018-present The Alive2 Authors.
# Distributed under the MIT license that can be found in the LICENSE file.
"""Lift PTX to LLVM IR, for translation validation of the NVPTX backend (an
arm-tv for NVIDIA GPUs).

The input is a module in LLVM IR for the NVPTX target and the PTX the NVPTX
backend prints for it (llc -mtriple=nvptx64-nvidia-cuda). The output is an
LLVM IR function with the IR function's signature whose body is the
per-thread meaning of the PTX, so that alive-tv can check that the PTX
refines the IR (see README.md).

The meaning of each arithmetic, logic, comparison and conversion
instruction is its entry in the PTX formalization, ptx.formal, after the PTX
ISA ("Instructions"), not the backend's patterns: shifts by the type's width
or more give 0 (or the sign), a division by zero gives some value, cvt to an
integer saturates and takes NaN to 0, setp's ne is ordered, and so on. This
file models the machine those instructions run on (registers, parameters,
memory, control flow) and lifts the instructions that act on it. What the
lifted code says:

* Registers hold bits, never poison: they start as unknown values, a
  parameter that is poison in the IR is some value in the parameter space,
  and every load's value is frozen. A .b32 register that holds a float holds
  its bits.
* A 64-bit register that the analysis below finds holds a pointer is a
  pointer, so that alive2 sees its provenance: one loaded from a pointer
  parameter, a variable's address (mov of .shared/.global/.const/.local
  symbols, the local depot), and sums of such with integers. cvta and
  cvta.to are the identity (the generic window of each state space maps
  addresses one to one; the lifted code has one address space, as alive2's
  canonical GPU form does, llvm_util/gpu.h).
* Every access is aligned to its size (vectors: the vector's), as the GPU
  requires.
* The special registers are the llvm.nvvm.read.ptx.sreg.* intrinsics,
  bar.sync / barrier.sync the CTA barrier intrinsic, membar and fence LLVM
  fences: alive2's GPU semantics gives them their meaning.
* The approximate instructions (ex2.approx, rsqrt.approx, ...) are calls of
  the NVVM intrinsics of the same name: uninterpreted, equal only to
  themselves.
"""

import argparse
import os
import re
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, '..', 'gpu-tv'))
import ptx_formal  # noqa: E402
# The lifters' common part: reading the source's headers and parameters.
from gpu_lift import (LiftError, Param, function_header,  # noqa: E402
                      one_line_headers, split_top, without_captures)


# The semantics of the instructions that compute (ptx.formal).
FORMAL = ptx_formal.Formal()


# ---------------------------------------------------------------------------
# The input IR: the function's signature and what the lifted module needs.


def parse_signature(ll, name):
  """(return type, calling convention, params) of @name's definition."""
  prefix, params, _ = function_header(ll, name)
  cc = 'ptx_kernel ' if re.search(r'\bptx_kernel\b', prefix) else ''
  # The return type: the last type among the prefix's words (vectors,
  # structures and arrays are one word; attributes are not types).
  words = split_top(prefix.replace('addrspace (', 'addrspace('), ' ')
  joined = []
  for w in words:
    if w.startswith('addrspace(') and joined and joined[-1] == 'ptr':
      joined[-1] = 'ptr ' + w
    elif w:
      joined.append(w)
  types = [w for w in joined if re.fullmatch(
      r'void|i\d+|half|bfloat|float|double|fp128|x86_fp80|ptr( addrspace\(\d+\))?|'
      r'<.*>|\{.*\}|\[.*\]|%[\w.$"]+', w)]
  ret = types[-1] if types else 'void'
  return ret, cc, params


def lifted_param(p):
  """A parameter's text in the lifted function."""
  return without_captures(p.text)


def global_definitions(ll):
  """The module's global variables: name -> (definition line, address
  space)."""
  out = {}
  for m in re.finditer(r'^@("[^"]*"|[\w.$]+)\s*=\s*(.*)$', ll, re.M):
    name = m.group(1).strip('"')
    text = m.group(0)
    if re.search(r'\b(global|constant)\b', m.group(2)) is None:
      continue
    a = re.search(r'\baddrspace\((\d+)\)', m.group(2).split('global')[0]
                  .split('constant')[0])
    out[name] = (text, int(a.group(1)) if a else 0)
  return out


# ---------------------------------------------------------------------------
# PTX.


def strip_comments(s):
  s = re.sub(r'/\*.*?\*/', ' ', s, flags=re.S)
  return re.sub(r'//[^\n]*', '', s)


def ptx_function(ptx, name):
  """(kind, return parameter text, parameter text, body) of the PTX
  function name."""
  ptx = strip_comments(ptx)
  m = re.search(r'\.(entry|func)\s+(?:\(([^)]*)\)\s*)?' + re.escape(name) +
                r'\s*\(', ptx)
  if not m:
    # A function without parameters may print no parentheses.
    m = re.search(r'\.(entry|func)\s+(?:\(([^)]*)\)\s*)?' + re.escape(name) +
                  r'\s*(?=[{.])', ptx)
    if not m:
      raise LiftError('no PTX function ' + name)
    params, i = '', m.end()
  else:
    i, depth = m.end(), 1
    while depth:
      depth += {'(': 1, ')': -1}.get(ptx[i], 0)
      i += 1
    params = ptx[m.end():i - 1]
  j = ptx.index('{', i)
  k, depth = j + 1, 1
  while depth:
    depth += {'{': 1, '}': -1}.get(ptx[k], 0)
    k += 1
  return m.group(1), m.group(2) or '', params, ptx[j + 1:k - 1]


def module_symbols(ptx):
  """Module-scope variables of the PTX: name -> state space."""
  out = {}
  ptx = strip_comments(ptx)
  # Function bodies are not module scope: drop them.
  depth, top = 0, ''
  for c in ptx:
    if c == '{':
      depth += 1
    elif c == '}':
      depth -= 1
    elif depth == 0:
      top += c
  for m in re.finditer(r'\.(global|shared|const)\b[^;]*?([\w$]+)\s*(\[[^;]*)?'
                       r'(=[^;]*)?;', top):
    out[m.group(2)] = m.group(1)
  return out


def statements(body):
  """The body's statements: ('label', name), ('{',), ('}',) or
  ('stmt', text) without its semicolon."""
  out, i, n = [], 0, len(body)
  while i < n:
    c = body[i]
    if c.isspace():
      i += 1
      continue
    if c in '{}':
      out.append((c,))
      i += 1
      continue
    # (Not ld.param::func, a state space's sub-qualifier.)
    m = re.compile(r'([$\w.]+)\s*:(?!:)').match(body, i)
    if m and not body[i] == '@':
      out.append(('label', m.group(1)))
      i = m.end()
      continue
    j = body.index(';', i)
    out.append(('stmt', ' '.join(body[i:j].split())))
    i = j + 1
  return out


class Instr:
  def __init__(self, text):
    self.text = text
    self.guard = None   # (register, negated)
    m = re.match(r'@(!?)(%[\w$]+)\s+(.*)$', text)
    if m:
      self.guard = (m.group(2), m.group(1) == '!')
      text = m.group(3)
    parts = text.split(None, 1)
    # The sub-qualifiers of the parameter and shared spaces that are the
    # space itself here (ld.param::func, st.shared::cta).
    self.op = re.sub(r'\.(param|shared)::(func|entry|cta)\b', r'.\1', parts[0])
    self.mods = self.op.split('.')
    self.base = self.mods[0]
    self.operands = split_top(parts[1]) if len(parts) > 1 else []

  def has(self, *mods):
    return any(m in self.mods[1:] for m in mods)

  def __str__(self):
    return self.text


TYPES = {}
for _w in (8, 16, 32, 64, 128):
  for _k in 'bus':
    TYPES['%s%d' % (_k, _w)] = (_k, _w)
TYPES.update({'f16': ('f', 16), 'bf16': ('bf', 16), 'f32': ('f', 32),
              'f64': ('f', 64), 'pred': ('pred', 1), 'f16x2': ('fv', 32),
              'bf16x2': ('bfv', 32)})

FLOAT_TYPE = {('f', 16): 'half', ('bf', 16): 'bfloat', ('f', 32): 'float',
              ('f', 64): 'double'}

SREG = {}
for _r in ('tid', 'ntid', 'ctaid', 'nctaid'):
  for _c in 'xyz':
    SREG['%%%s.%s' % (_r, _c)] = 'llvm.nvvm.read.ptx.sreg.%s.%s' % (_r, _c)
SREG['%laneid'] = 'llvm.nvvm.read.ptx.sreg.laneid'
SREG['%warpsize'] = 'llvm.nvvm.read.ptx.sreg.warpsize'

ATOMIC_ORDER = {'relaxed': 'monotonic', 'acquire': 'acquire',
                'release': 'release', 'acq_rel': 'acq_rel'}


def imm_value(tok, bits):
  """An immediate's bits as an unsigned integer of the given width, or
  None."""
  t = tok.strip()
  try:
    if re.fullmatch(r'0[fF][0-9a-fA-F]{8}', t):
      v = int(t[2:], 16)
    elif re.fullmatch(r'0[dD][0-9a-fA-F]{16}', t):
      v = int(t[2:], 16)
    elif re.fullmatch(r'-?0[xX][0-9a-fA-F]+U?', t):
      v = int(t.rstrip('U'), 16)
    elif re.fullmatch(r'-?0[0-7]+U?', t):
      v = int(t.rstrip('U'), 8)
    elif re.fullmatch(r'-?[0-9]+U?', t):
      v = int(t.rstrip('U'), 10)
    elif re.fullmatch(r'-?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?', t):
      v = struct.unpack('<Q', struct.pack('<d', float(t)))[0]
      if bits == 32:
        v = struct.unpack('<I', struct.pack('<f', float(t)))[0]
    else:
      return None
  except ValueError:
    return None
  return v & ((1 << bits) - 1)


def signed(v, bits):
  return v - (1 << bits) if v >> (bits - 1) else v


# ---------------------------------------------------------------------------
# The lifter.


class Lifter:
  def __init__(self, ll, ptx, name):
    self.ll = one_line_headers(ll)
    self.name = name
    self.ret, self.cc, self.params = parse_signature(self.ll, name)
    self.globals = global_definitions(self.ll)
    kind, retparam, ptxparams, body = ptx_function(ptx, name)
    self.kind = kind
    self.syms = module_symbols(ptx)   # name -> space
    self.ptx_params = self.parse_ptx_params(ptxparams)
    if len(self.ptx_params) != len(self.params):
      raise LiftError('%d PTX parameters for %d IR parameters' %
                      (len(self.ptx_params), len(self.params)))
    self.retparam = None
    if retparam.strip():
      m = re.search(r'\.param\s+(?:\.align\s+\d+\s+)?\.(\w+)\s+([\w$]+)'
                    r'(\[\d+\])?', retparam)
      if not m:
        raise LiftError('return parameter: ' + retparam)
      # (name, bits of a scalar, or None, and bytes of an array).
      if m.group(3):
        self.retparam = (m.group(2), None, int(m.group(3)[1:-1]))
      else:
        self.retparam = (m.group(2), TYPES[m.group(1)][1], None)
    self.regs = {}      # name -> bits
    self.locals = {}    # local symbol -> (space, size, align)
    self.body = []      # ('label', name) | ('instr', Instr)
    for s in statements(body):
      if s[0] in '{}':
        continue
      if s[0] == 'label':
        self.body.append(s)
        continue
      self.statement(s[1])
    self.lines = []
    self.decls = {}
    self.tmp = 0

  # -- Declarations ---------------------------------------------------------

  def parse_ptx_params(self, text):
    out = []
    for p in split_top(text):
      m = re.match(r'\.param\s+(.*?)([\w$]+)\s*(\[(\d+)\])?$', p.strip())
      if not m:
        raise LiftError('parameter: ' + p)
      t = re.search(r'\.(b8|b16|b32|b64|u8|u16|u32|u64|s8|s16|s32|s64|f16|'
                    r'f32|f64|b128)\b', m.group(1))
      size = (int(m.group(4)) if m.group(4)
              else (TYPES[t.group(1)][1] // 8 if t else 0))
      out.append((m.group(2), size, m.group(4) is not None))
    return out

  def statement(self, text):
    if text.startswith('.reg'):
      m = re.match(r'\.reg\s+(?:\.v\d\s+)?\.(\w+)\s+(.*)$', text)
      if not m or m.group(1) not in TYPES:
        raise LiftError('register declaration: ' + text)
      bits = TYPES[m.group(1)][1]
      if m.group(1) == 'pred':
        bits = 1
      for d in split_top(m.group(2)):
        r = re.fullmatch(r'(%[\w$]+)<(\d+)>', d)
        if r:
          for i in range(int(r.group(2))):
            self.regs['%s%d' % (r.group(1), i)] = bits
        else:
          self.regs[d] = bits
      return
    m = re.match(r'\.(shared|local|global|const)\s+(?:\.align\s+(\d+)\s+)?'
                 r'\.(\w+)\s+([\w$]+)\s*(?:\[(\d*)\])?\s*$', text)
    if m:
      count = int(m.group(5)) if m.group(5) else 1
      size = TYPES[m.group(3)][1] // 8 * count
      self.locals[m.group(4)] = (m.group(1), size, int(m.group(2) or 1))
      return
    if text.startswith('.'):
      # .maxntid, .reqntid, .pragma, .loc, ...: nothing per thread.
      if re.match(r'\.(maxntid|reqntid|minnctapersm|maxnreg|pragma|loc|'
                  r'file|section|noreturn|explicitcluster|reqnctapercluster|'
                  r'maxclusterrank)\b', text):
        return
      raise LiftError('directive: ' + text)
    self.body.append(('instr', Instr(text)))

  # -- Output ---------------------------------------------------------------

  def emit(self, s):
    self.lines.append('  ' + s)

  def t(self, hint='t'):
    self.tmp += 1
    return '%%%s.%d' % (hint, self.tmp)

  def declare(self, name, ret, args, attrs=''):
    self.decls[name] = 'declare %s @%s(%s)%s' % (ret, name, ', '.join(args),
                                                 attrs)

  def call(self, name, ret, args, attrs=''):
    """A call of name with args [(type, value)]; its value or None."""
    self.declare(name, ret, [a[0] for a in args], attrs)
    a = ', '.join('%s %s' % x for x in args)
    if ret == 'void':
      self.emit('call void @%s(%s)' % (name, a))
      return None
    v = self.t()
    self.emit('%s = call %s @%s(%s)' % (v, ret, name, a))
    return v

  def intr(self, base, ty, args):
    """A call of the overloaded LLVM intrinsic llvm.<base> at type ty."""
    suffix = {'float': 'f32', 'double': 'f64', 'half': 'f16',
              'bfloat': 'bf16'}.get(ty, ty)
    return self.call('llvm.%s.%s' % (base, suffix), ty, args)

  # -- Pointer analysis -----------------------------------------------------

  def ptr_param(self, operand):
    """The IR parameter an ld.param operand reads at offset 0, if it is a
    pointer parameter."""
    m = re.fullmatch(r'\[\s*([\w$]+)\s*\]', operand)
    if not m:
      return None
    for i, (n, _, _) in enumerate(self.ptx_params):
      if n == m.group(1) and self.params[i].is_ptr and not self.params[i].byval:
        return self.params[i]
    return None

  def is_symbol(self, tok):
    return tok in self.locals or tok in self.syms

  def analyze_pointers(self):
    """The 64-bit registers that hold pointers: those whose definitions
    are pointers (a pointer parameter, a variable's address, cvta of a
    pointer, a sum of a pointer and an integer), and registers used as
    addresses that are loaded from memory."""
    ptr = set()
    addr_uses = set()
    for kind, x in self.body:
      if kind != 'instr':
        continue
      for o in x.operands:
        m = re.fullmatch(r'\[\s*(%[\w$]+)\s*(?:\+\s*-?\w+)?\s*\]', o)
        if m:
          addr_uses.add(m.group(1))
    changed = True
    while changed:
      changed = False
      for kind, x in self.body:
        if kind != 'instr' or not x.operands:
          continue
        d = x.operands[0]
        if d in ptr or self.regs.get(d) != 64:
          continue
        src = x.operands[1:]
        p = False
        if x.base == 'ld' and x.has('param'):
          p = self.ptr_param(src[0]) is not None
        elif x.base == 'ld' and d in addr_uses and not x.has('param'):
          p = True
        elif x.base in ('mov', 'cvta') and len(src) == 1:
          p = src[0] in ptr or self.is_symbol(src[0])
        elif x.base == 'add' and len(src) == 2 and x.mods[-1] in ('s64', 'u64'):
          p = (src[0] in ptr) != (src[1] in ptr) or any(
              self.is_symbol(s) for s in src)
        elif x.base == 'sub' and len(src) == 2:
          p = src[0] in ptr and src[1] not in ptr
        elif x.base == 'selp' and len(src) == 3:
          p = src[0] in ptr and src[1] in ptr
        if p:
          ptr.add(d)
          changed = True
    return ptr

  # -- Registers ------------------------------------------------------------

  def reg(self, name):
    if name not in self.regs:
      raise LiftError('unknown register ' + name)
    return self.regs[name]

  def slot(self, name):
    return '%%R.%s' % name.lstrip('%').replace('$', '_')

  def rtype(self, name):
    return 'ptr' if name in self.ptr else 'i%d' % self.reg(name)

  def read_reg(self, name):
    v = self.t(name.lstrip('%').replace('$', '_'))
    self.emit('%s = load %s, ptr %s' % (v, self.rtype(name), self.slot(name)))
    return v

  def write_reg(self, name, val, ty):
    """Write val of type ty (iN or ptr) to the register."""
    if name in ('_',):
      return
    rt = self.rtype(name)
    if ty != rt:
      if rt == 'ptr':
        val = self.cast('inttoptr', val, ty, 'ptr')
      elif ty == 'ptr':
        val = self.cast('ptrtoint', val, 'ptr', 'i64')
        val = self.resize(val, 64, self.reg(name), False)
      else:
        val = self.resize(val, int(ty[1:]), self.reg(name), False)
    self.emit('store %s %s, ptr %s' % (rt, val, self.slot(name)))

  def cast(self, op, v, a, b):
    if a == b:
      return v
    r = self.t()
    self.emit('%s = %s %s %s to %s' % (r, op, a, v, b))
    return r

  def resize(self, v, a, b, sign):
    """iA -> iB: truncate, or extend by sign."""
    if a == b:
      return v
    if a > b:
      return self.cast('trunc', v, 'i%d' % a, 'i%d' % b)
    return self.cast('sext' if sign else 'zext', v, 'i%d' % a, 'i%d' % b)

  def operand(self, tok, bits, sign=False):
    """A source operand as an i<bits> value: a register (a wider one
    truncated, PTX's relaxed type rule for sources), an immediate, a special
    register, or a symbol's address."""
    tok = tok.strip()
    if tok.startswith('%') and tok in self.regs:
      rb = self.reg(tok)
      v = self.read_reg(tok)
      if tok in self.ptr:
        v = self.cast('ptrtoint', v, 'ptr', 'i64')
      if rb < bits and not (rb == 1 and bits == 1):
        if rb == 1:
          raise LiftError('predicate as a %d-bit operand' % bits)
        return self.resize(v, rb, bits, sign)
      return self.resize(v, rb, bits, sign)
    if tok in SREG:
      v = self.call(SREG[tok], 'i32', [], ' nounwind readnone')
      return self.resize(v, 32, bits, False)
    if self.is_symbol(tok):
      return self.resize(self.cast('ptrtoint', self.symbol(tok), 'ptr',
                                   'i64'), 64, bits, False)
    m = re.fullmatch(r'([\w$]+)\s*\+\s*(-?\d+)', tok)
    if m and self.is_symbol(m.group(1)):
      p = self.gep(self.symbol(m.group(1)), str(int(m.group(2))))
      return self.resize(self.cast('ptrtoint', p, 'ptr', 'i64'), 64, bits,
                         False)
    v = imm_value(tok, bits)
    if v is None:
      raise LiftError('operand ' + tok)
    return str(signed(v, bits)) if bits > 1 else ('true' if v else 'false')

  def pointer(self, tok):
    """A source operand as a pointer (a pointer register, a symbol), or
    None."""
    tok = tok.strip()
    if tok in self.ptr:
      return self.read_reg(tok)
    if self.is_symbol(tok):
      return self.symbol(tok)
    return None

  def address(self, tok):
    """An address operand [base+offset] as a pointer."""
    m = re.fullmatch(r'\[\s*([%\w$]+)\s*(?:\+\s*(-?\w+))?\s*\]', tok.strip())
    if not m:
      raise LiftError('address ' + tok)
    base, off = m.group(1), m.group(2)
    p = self.pointer(base)
    if p is None:
      if base not in self.regs:
        v = imm_value(base, 64)
        if v is None:
          raise LiftError('address ' + tok)
        p = self.cast('inttoptr', str(v), 'i64', 'ptr')
      else:
        p = self.cast('inttoptr', self.operand(base, 64), 'i64', 'ptr')
    if off:
      o = imm_value(off, 64)
      if o is None:
        raise LiftError('address offset ' + tok)
      p = self.gep(p, str(signed(o, 64)))
    return p

  def gep(self, p, off):
    if off == '0':
      return p
    r = self.t('a')
    self.emit('%s = getelementptr i8, ptr %s, i64 %s' % (r, p, off))
    return r

  def symbol(self, name):
    """A variable's address, as a pointer of address space 0."""
    if name in self.locals and self.locals[name][0] == 'local':
      return '%%L.%s' % name.replace('$', '_')
    if name not in self.globals:
      raise LiftError('no IR variable for the PTX variable ' + name)
    _, a = self.globals[name]
    if a == 0:
      return '@%s' % name
    return '%%G.%s' % name.replace('$', '_')

  def fp(self, v, bits, ftype):
    return self.cast('bitcast', v, 'i%d' % bits, ftype)

  def unfp(self, v, ftype, bits):
    return self.cast('bitcast', v, ftype, 'i%d' % bits)

  def freeze(self, v, ty):
    r = self.t('fr')
    self.emit('%s = freeze %s %s' % (r, ty, v))
    return r

  # -- The function ---------------------------------------------------------

  def float_registers(self):
    """Registers some float instruction reads: their loads are of whole
    words (a float that is partly poison is rare, and bytes of floats are
    slow for the solver)."""
    out = set()
    for kind, x in self.body:
      if kind != 'instr':
        continue
      if any(m in ('f16', 'bf16', 'f32', 'f64') for m in x.mods[1:]):
        for o in x.operands[1:]:
          out.update(re.findall(r'%[\w$]+', o))
    return out

  def lift(self):
    self.ptr = self.analyze_pointers()
    self.fregs = self.float_registers()
    out = []
    params = ', '.join(lifted_param(p) for p in self.params)
    out.append('define %s%s @%s(%s)%s {' % (self.cc, self.ret, self.name,
                                          params, self.fp_env()))
    out.append('entry:')
    # Registers start as unknown values.
    for r, bits in sorted(self.regs.items()):
      ty = self.rtype(r)
      out.append('  %s = alloca %s' % (self.slot(r), ty))
      if ty == 'ptr':
        out.append('  store ptr null, ptr %s' % self.slot(r))
      else:
        out.append('  %s.init = freeze %s poison' % (self.slot(r), ty))
        out.append('  store %s %s.init, ptr %s' % (ty, self.slot(r),
                                                   self.slot(r)))
    if self.retparam and self.retparam[1] is None:
      # An array: the bytes the IR's return value is laid out in.
      out.append('  %%retval = alloca [%d x i8], align 16' % self.retparam[2])
    elif self.retparam and self.ret.startswith('ptr'):
      # A returned pointer stays a pointer (its provenance with it).
      out.append('  %retval = alloca ptr')
      out.append('  store ptr null, ptr %retval')
    elif self.retparam:
      out.append('  %%retval = alloca i%d' % self.retparam[1])
      out.append('  %%retval.init = freeze i%d poison' % self.retparam[1])
      out.append('  store i%d %%retval.init, ptr %%retval' %
                 self.retparam[1])
    # Local variables (the stack's depot) and global ones.
    for n, (space, size, align) in self.locals.items():
      if space == 'local':
        out.append('  %%L.%s = alloca [%d x i8], align %d' %
                   (n.replace('$', '_'), size, align))
    used = set()
    for kind, x in self.body:
      if kind == 'instr':
        for o in x.operands:
          for w in re.findall(r'[A-Za-z_$][\w$]*', o):
            if w in self.globals and (w in self.syms or w in self.locals):
              used.add(w)
    for n in sorted(used):
      _, a = self.globals[n]
      if a:
        out.append('  %%G.%s = addrspacecast ptr addrspace(%d) @%s to ptr' %
                   (n.replace('$', '_'), a, n))
    self.globals_used = used

    # Basic blocks: at labels, and after branches.
    blocks, cur = [], ['b0', []]
    for item in self.body:
      if item[0] == 'label':
        blocks.append(cur)
        cur = [item[1], []]
      else:
        cur[1].append(item[1])
        if item[1].base in ('bra', 'ret', 'exit', 'trap') and not item[1].guard:
          blocks.append(cur)
          cur = ['b%d' % len(blocks), []]
    blocks.append(cur)
    # Code after an unconditional branch or return with no label is never
    # run: dropped (it would keep reads of registers that are no more).
    blocks = [b for i, b in enumerate(blocks)
              if i == 0 or not re.fullmatch(r'b\d+', b[0])]
    self.labels = {b[0]: 'L.' + b[0].replace('$', '_') for b in blocks}
    out.append('  br label %%%s' % self.labels[blocks[0][0]])
    for bi, (label, instrs) in enumerate(blocks):
      nxt = self.labels[blocks[bi + 1][0]] if bi + 1 < len(blocks) else None
      self.lines = []
      self.lines.append('%s:' % self.labels[label])
      terminated = False
      for x in instrs:
        terminated = self.instr(x)
      if not terminated:
        if nxt:
          self.emit('br label %%%s' % nxt)
        else:
          self.ret_instr()
      out.extend(self.lines)
    out.append('}')
    head = ['target triple = "nvptx64-nvidia-cuda"']
    # The source's datalayout (its pointers' sizes are alive2's).
    dl = re.search(r'^target datalayout = .*$', self.ll, re.M)
    if dl:
      head.append(dl.group(0))
    head.append('')
    # The module's named types (of byval parameters, of variables).
    head += re.findall(r'^%[\w.$"]+ = type .*$', self.ll, re.M)
    for n in sorted(self.globals_used):
      head.append(self.globals[n][0])
    head.append('')
    return '\n'.join(head + out + [''] + sorted(self.decls.values())) + '\n'

  def fp_env(self):
    """The source function's floating-point environment attributes (its
    denormal mode), which the lifted function declares too."""
    m = re.search(r'^define[^\n]*@"?%s"?\(.*$' % re.escape(self.name), self.ll,
                  re.M)
    tail = m.group(0)[m.group(0).rindex(')'):] if m else ''
    texts = [tail]
    for g in re.findall(r'#(\d+)', tail):
      a = re.search(r'^attributes #%s = \{(.*)\}' % g, self.ll, re.M)
      if a:
        texts.append(a.group(1))
    out = []
    for t in texts:
      out += re.findall(r'denormal_fpenv\([^)]*\)', t)
      out += re.findall(r'"denormal-fp-math(?:-f32)?"="[^"]*"', t)
    return ''.join(' ' + a for a in dict.fromkeys(out))

  def ret_instr(self):
    if self.ret == 'void':
      self.emit('ret void')
      return
    if not self.retparam:
      raise LiftError('a value returned without a return parameter')
    bits = self.retparam[1]
    if bits is None:
      v = self.t()
      self.emit('%s = load %s, ptr %%retval, align 1' % (v, self.ret))
      self.emit('ret %s %s' % (self.ret, v))
      return
    v = self.t()
    if self.ret.startswith('ptr'):
      self.emit('%s = load ptr, ptr %%retval' % v)
      if self.ret != 'ptr':
        v = self.cast('addrspacecast', v, 'ptr', self.ret)
      self.emit('ret %s %s' % (self.ret, v))
      return
    self.emit('%s = load i%d, ptr %%retval' % (v, bits))
    self.emit('ret %s %s' % (self.ret, self.to_ir(v, bits, self.ret)))

  def to_ir(self, v, bits, ty):
    """iBITS bits -> a value of the IR type ty."""
    if ty in ('float', 'double', 'half', 'bfloat'):
      fb = {'float': 32, 'double': 64, 'half': 16, 'bfloat': 16}[ty]
      return self.cast('bitcast', self.resize(v, bits, fb, False),
                       'i%d' % fb, ty)
    if ty == 'ptr':
      return self.cast('inttoptr', self.resize(v, bits, 64, False), 'i64',
                       'ptr')
    m = re.fullmatch(r'i(\d+)', ty)
    if m:
      return self.resize(v, bits, int(m.group(1)), False)
    raise LiftError('return type ' + ty)

  # -- Instructions ---------------------------------------------------------

  def instr(self, x):
    """Lift x; whether it ended the block."""
    if x.guard and x.base != 'bra':
      # @p x (@!p x): x in a block of its own, run when p (not p).
      p = self.operand(x.guard[0], 1)
      exec_block, cont = self.t('do')[1:], self.t('skip')[1:]
      t, f = (cont, exec_block) if x.guard[1] else (exec_block, cont)
      self.emit('br i1 %s, label %%%s, label %%%s' % (p, t, f))
      self.lines.append('%s:' % exec_block)
      g, x.guard = x.guard, None
      ended = self.instr(x)
      x.guard = g
      if not ended:
        self.emit('br label %%%s' % cont)
      self.lines.append('%s:' % cont)
      return False
    # Address arithmetic on pointers keeps their provenance.
    if (x.base in ('add', 'sub', 'selp') and x.operands and
        x.operands[0] in self.ptr and self.pointer_arith(x)):
      return False
    try:
      if FORMAL.lift(self, x):
        return False
    except ptx_formal.FormalError as e:
      raise LiftError('ptx.formal: %s' % e)
    f = getattr(self, 'i_' + x.base, None)
    if f is None:
      raise LiftError('no meaning in ptx.formal for ' + str(x))
    return bool(f(x))

  def itype(self, x):
    """The instruction's type: its last type modifier."""
    for m in reversed(x.mods[1:]):
      if m in TYPES:
        return m
    raise LiftError('no type: ' + str(x))

  def no_mods(self, x, allowed):
    for m in x.mods[1:]:
      if m not in allowed and m not in TYPES:
        raise LiftError('modifier .%s not lifted: %s' % (m, x))

  # Control flow.

  def i_bra(self, x):
    target = self.labels.get(x.operands[0])
    if target is None:
      raise LiftError('branch to ' + x.operands[0])
    if x.guard:
      p = self.operand(x.guard[0], 1)
      nxt = self.t('fall')[1:]
      t, f = (target, nxt) if not x.guard[1] else (nxt, target)
      self.emit('br i1 %s, label %%%s, label %%%s' % (p, t, f))
      self.lines.append('%s:' % nxt)
      return False
    self.emit('br label %%%s' % target)
    return True

  def i_ret(self, x):
    self.ret_instr()
    return True

  def i_exit(self, x):
    if self.kind != 'entry':
      raise LiftError('exit in a function')
    self.emit('ret void')
    return True

  def i_trap(self, x):
    self.call('llvm.trap', 'void', [], ' cold noreturn nounwind')
    self.emit('unreachable')
    return True

  def i_bar(self, x):
    # bar.sync a / bar.sync a, b / barrier.sync[.aligned] a.
    if not (x.has('sync') or len(x.mods) == 1) or len(x.operands) != 1:
      raise LiftError('barrier not lifted: ' + str(x))
    self.call('llvm.nvvm.barrier.cta.sync.aligned.all', 'void',
              [('i32', self.operand(x.operands[0], 32))],
              ' convergent nounwind')

  i_barrier = i_bar

  def i_membar(self, x):
    self.emit('fence seq_cst')

  def i_fence(self, x):
    if x.has('sc'):
      self.emit('fence seq_cst')
    elif x.has('acq_rel') or len(x.mods) == 2:
      self.emit('fence acq_rel')
    elif x.has('acquire'):
      self.emit('fence acquire')
    elif x.has('release'):
      self.emit('fence release')
    else:
      raise LiftError('fence not lifted: ' + str(x))

  # Moves.

  def i_mov(self, x):
    d, s = x.operands
    ty = self.itype(x)
    bits = TYPES[ty][1]
    if d.startswith('{'):
      # Unpack: mov.b64 {lo, hi}, x.
      parts = split_top(d.strip('{}'))
      v = self.operand(s, bits)
      w = bits // len(parts)
      for i, p in enumerate(parts):
        if p == '_':
          continue
        sh = v
        if i:
          sh = self.t()
          self.emit('%s = lshr i%d %s, %d' % (sh, bits, v, w * i))
        self.write_reg(p, self.resize(sh, bits, w, False), 'i%d' % w)
      return
    if s.startswith('{'):
      parts = split_top(s.strip('{}'))
      w = bits // len(parts)
      acc = '0'
      for i, p in enumerate(parts):
        v = self.resize(self.operand(p, w), w, bits, False)
        if i:
          sh = self.t()
          self.emit('%s = shl i%d %s, %d' % (sh, bits, v, w * i))
          v = sh
        o = self.t()
        self.emit('%s = or i%d %s, %s' % (o, bits, acc, v))
        acc = o
      self.write_reg(d, acc, 'i%d' % bits)
      return
    p = self.pointer(s)
    if p is not None and d in self.ptr:
      self.write_reg(d, p, 'ptr')
      return
    self.write_reg(d, self.operand(s, bits), 'i%d' % bits)

  def i_cvta(self, x):
    d, s = x.operands
    p = self.pointer(s)
    if p is not None:
      self.write_reg(d, p, 'ptr')
    else:
      self.write_reg(d, self.operand(s, 64), 'i64')

  # Pointer arithmetic (what ptx.formal says of add, sub and selp, on
  # pointers, so that the result keeps the pointer's provenance).

  def pointer_arith(self, x):
    """add of a pointer and an integer, sub of an integer from a pointer,
    selp of two pointers: whether x was one."""
    d = x.operands[0]
    if x.base == 'add' and len(x.operands) == 3:
      pa, pb = self.pointer(x.operands[1]), self.pointer(x.operands[2])
      if (pa is None) != (pb is None):
        p, o = (pa, x.operands[2]) if pa is not None else (pb, x.operands[1])
        self.write_reg(d, self.gep(p, self.operand(o, 64)), 'ptr')
        return True
    if x.base == 'sub' and len(x.operands) == 3:
      pa = self.pointer(x.operands[1])
      if pa is not None and self.pointer(x.operands[2]) is None:
        n = self.t()
        self.emit('%s = sub i64 0, %s' % (n, self.operand(x.operands[2], 64)))
        self.write_reg(d, self.gep(pa, n), 'ptr')
        return True
    if x.base == 'selp' and len(x.operands) == 4:
      pa, pb = self.pointer(x.operands[1]), self.pointer(x.operands[2])
      if pa is not None and pb is not None:
        r = self.t()
        self.emit('%s = select i1 %s, ptr %s, ptr %s' %
                  (r, self.operand(x.operands[3], 1), pa, pb))
        self.write_reg(d, r, 'ptr')
        return True
    return False

  def fval(self, tok, bits, ft):
    return self.fp(self.operand(tok, bits), bits, ft)

  # Memory.

  def access(self, x):
    """(space, vector count, type kind, bits) of ld/st."""
    space = 'generic'
    for m in x.mods[1:]:
      if m in ('global', 'shared', 'local', 'const', 'param'):
        space = m
    n = 1
    for m in x.mods[1:]:
      if m in ('v2', 'v4', 'v8'):
        n = int(m[1:])
    kind, bits = TYPES[self.itype(x)]
    # The access's kind: plain (.weak too), volatile (as LLVM's), or atomic
    # in an order (.relaxed: monotonic). Its scope (.cta, .gpu, .sys) is no
    # matter in a sequential memory model.
    self.order = None
    for m in x.mods[1:]:
      if m == 'volatile':
        self.order = 'volatile'
      elif m in ('relaxed', 'acquire', 'release'):
        self.order = {'relaxed': 'monotonic'}.get(m, m)
      elif m == 'mmio':
        raise LiftError('mmio access not lifted: ' + str(x))
    if self.order and self.order != 'volatile' and n > 1:
      raise LiftError('vector atomic access not lifted: ' + str(x))
    return space, n, kind, bits

  def load_text(self, ty, a, align):
    if self.order == 'volatile':
      return 'load volatile %s, ptr %s, align %d' % (ty, a, align)
    if self.order:
      return 'load atomic %s, ptr %s %s, align %d' % (ty, a, self.order,
                                                       align)
    return 'load %s, ptr %s, align %d' % (ty, a, align)

  def store_text(self, ty, v, a, align):
    if self.order == 'volatile':
      return 'store volatile %s %s, ptr %s, align %d' % (ty, v, a, align)
    if self.order:
      return 'store atomic %s %s, ptr %s %s, align %d' % (ty, v, a,
                                                          self.order, align)
    return 'store %s %s, ptr %s, align %d' % (ty, v, a, align)

  def elems(self, tok, n):
    if n == 1:
      return [tok]
    if not tok.startswith('{'):
      raise LiftError('vector operand ' + tok)
    return split_top(tok.strip()[1:-1])

  def i_ld(self, x):
    space, n, kind, bits = self.access(x)
    if space == 'param':
      return self.ld_param(x, n, kind, bits)
    p = self.address(x.operands[1])
    regs = self.elems(x.operands[0], n)
    align = bits // 8 * n
    for i, d in enumerate(regs):
      a = self.gep(p, str(i * bits // 8))
      al = align if i == 0 else bits // 8
      if d in self.ptr and bits == 64:
        v = self.t('ld')
        self.emit('%s = %s' % (v, self.load_text('ptr', a, al)))
        self.write_reg(d, self.freeze(v, 'ptr'), 'ptr')
        continue
      v = self.t('ld')
      if bits > 8 and not self.order and d not in self.fregs:
        # Memory holds bytes: a byte that is poison in the IR is some value
        # in memory, and the others are what they are (an integer load of a
        # poison byte would be all poison).
        vt = '<%d x i8>' % (bits // 8)
        self.emit('%s = %s' % (v, self.load_text(vt, a, al)))
        v = self.cast('bitcast', self.freeze(v, vt), vt, 'i%d' % bits)
      else:
        self.emit('%s = %s' % (v, self.load_text('i%d' % bits, a, al)))
        v = self.freeze(v, 'i%d' % bits)
        if self.order in ('monotonic', 'volatile') and space != 'const':
          # The bits read are in memory: a later read sees them too.
          self.emit('store i%d %s, ptr %s, align %d' % (bits, v, a, al))
      if d == '_':
        continue
      rb = self.reg(d)
      self.write_reg(d, self.resize(v, bits, max(bits, rb), kind == 's'),
                     'i%d' % max(bits, rb))

  def ld_param(self, x, n, kind, bits):
    m = re.fullmatch(r'\[\s*([\w$]+)\s*(?:\+\s*(\d+))?\s*\]',
                     x.operands[1].strip())
    if not m:
      raise LiftError('parameter access ' + str(x))
    idx = [i for i, p in enumerate(self.ptx_params) if p[0] == m.group(1)]
    if not idx:
      raise LiftError('parameter ' + m.group(1))
    i = idx[0]
    P = self.params[i]
    off = int(m.group(2) or 0)
    regs = self.elems(x.operands[0], n)
    for j, d in enumerate(regs):
      o = off + j * bits // 8
      if P.byval:
        a = self.gep(P.name, str(o))
        v = self.t('ld')
        self.emit('%s = load i%d, ptr %s, align 1' % (v, bits, a))
        v = self.freeze(v, 'i%d' % bits)
      elif d in self.ptr and P.is_ptr and o == 0 and bits == 64:
        # A pointer stays a pointer (with its provenance; not frozen: an
        # access through a poison pointer is UB in the IR too).
        v = P.name
        if P.type != 'ptr':
          v = self.cast('addrspacecast', v, P.type, 'ptr')
        self.write_reg(d, v, 'ptr')
        continue
      else:
        v = self.param_bits(P, o, bits)
      if d == '_':
        continue
      rb = self.reg(d)
      self.write_reg(d, self.resize(v, bits, max(bits, rb), kind == 's'),
                     'i%d' % max(bits, rb))

  def param_bits(self, P, off, bits):
    """bits bits of a scalar parameter's value at byte offset off (a value
    that is poison is some value)."""
    t = P.type
    if t.startswith('ptr'):
      v = P.name
      if t != 'ptr':
        v = self.cast('addrspacecast', v, t, 'ptr')
      v, w = self.cast('ptrtoint', v, 'ptr', 'i64'), 64
    elif t in ('float', 'double', 'half', 'bfloat'):
      w = {'float': 32, 'double': 64, 'half': 16, 'bfloat': 16}[t]
      v = self.cast('bitcast', self.freeze(P.name, t), t, 'i%d' % w)
    elif re.fullmatch(r'i\d+', t):
      w = int(t[1:])
      v = P.name
    else:
      m = re.fullmatch(r'<(\d+) x (\w+)>', t)
      if not m:
        raise LiftError('parameter type ' + t)
      eb = {'float': 32, 'double': 64, 'half': 16, 'bfloat': 16}.get(
          m.group(2)) or int(m.group(2)[1:])
      w = int(m.group(1)) * eb
      # Each element frozen (a poison element is some value there), not the
      # whole: a bitcast of a vector with a poison element is all poison.
      v = self.cast('bitcast', self.freeze(P.name, t), t, 'i%d' % w)
    v = self.freeze(v, 'i%d' % w)
    if off:
      s = self.t()
      self.emit('%s = lshr i%d %s, %d' % (s, w, v, 8 * off))
      v = s
    return self.resize(v, w, bits, False) if bits <= w else self.resize(
        v, w, bits, False)

  def i_st(self, x):
    space, n, kind, bits = self.access(x)
    if space == 'param':
      return self.st_param(x, n, bits)
    p = self.address(x.operands[0])
    vals = self.elems(x.operands[1], n)
    align = bits // 8 * n
    for i, s in enumerate(vals):
      a = self.gep(p, str(i * bits // 8))
      al = align if i == 0 else bits // 8
      if s in self.ptr and bits == 64:
        self.emit(self.store_text('ptr', self.read_reg(s), a, al))
        continue
      v = self.operand(s, bits)
      self.emit(self.store_text('i%d' % bits, v, a, al))

  def st_param(self, x, n, bits):
    m = re.fullmatch(r'\[\s*([\w$]+)\s*(?:\+\s*(\d+))?\s*\]',
                     x.operands[0].strip())
    if not m or not self.retparam or m.group(1) != self.retparam[0]:
      raise LiftError('parameter store not lifted: ' + str(x))
    if self.retparam[1] is None:
      # Into the return value's bytes, at its offset.
      off = int(m.group(2) or 0)
      for i, s in enumerate(self.elems(x.operands[1], n)):
        a = self.gep('%retval', str(off + i * bits // 8))
        self.emit('store i%d %s, ptr %s, align 1' %
                  (bits, self.operand(s, bits), a))
      return
    if m.group(2) or n != 1:
      raise LiftError('parameter store not lifted: ' + str(x))
    if self.ret.startswith('ptr'):
      p = self.pointer(x.operands[1])
      if p is None:
        p = self.cast('inttoptr', self.operand(x.operands[1], 64), 'i64',
                      'ptr')
      self.emit('store ptr %s, ptr %%retval' % p)
      return
    rb = self.retparam[1]
    v = self.operand(x.operands[1], bits)
    self.emit('store i%d %s, ptr %%retval' % (rb, self.resize(v, bits, rb,
                                                                False)))

  def atomic(self, x, red):
    order = 'monotonic'
    space = 'generic'
    op = None
    for m in x.mods[1:]:
      if m in ATOMIC_ORDER:
        order = ATOMIC_ORDER[m]
      elif m in ('global', 'shared'):
        space = m
      elif m in ('add', 'min', 'max', 'inc', 'dec', 'and', 'or', 'xor',
                 'exch', 'cas'):
        op = m
      elif m in ('cta', 'gpu', 'sys', 'cluster', 'noftz') or m in TYPES:
        pass
      else:
        raise LiftError('atomic modifier .%s not lifted: %s' % (m, x))
    kind, bits = TYPES[self.itype(x)]
    ops = x.operands if red else x.operands[1:]
    p = self.address(ops[0])
    T = 'i%d' % bits
    # The operation as the GPU does it, between fences if it is ordered: it
    # reads the location's bits (each byte frozen: memory holds bits, and a
    # byte that is poison in the IR is some value there), and writes what
    # the operation makes of them (a CAS that fails writes back what it read:
    # no matter in a sequential model, and two reads then agree).
    ordered = order != 'monotonic'
    if ordered:
      self.emit('fence acq_rel')
    old = self.read_bits(p, bits)
    if kind in ('f', 'bf'):
      if op not in ('add', 'min', 'max'):
        raise LiftError('atomic not lifted: ' + str(x))
      ft = FLOAT_TYPE[(kind, bits)]
      v = self.operand(ops[1], bits)
      # atom.add.f32 "flushes subnormal inputs and results to
      # sign-preserving zero" (PTX ISA, atom); .noftz and f64 do not.
      ftz = op == 'add' and bits == 32 and not x.has('noftz')
      a, b = (self.flush32(old), self.flush32(v)) if ftz else (old, v)
      fa, fb = self.fp(a, bits, ft), self.fp(b, bits, ft)
      r = self.t()
      if op == 'add':
        self.emit('%s = fadd %s %s, %s' % (r, ft, fa, fb))
      else:
        r = self.intr({'min': 'minnum', 'max': 'maxnum'}[op], ft,
                      [(ft, fa), (ft, fb)])
      new = self.unfp(r, ft, bits)
      if ftz:
        new = self.flush32(new)
    elif op == 'cas':
      c = self.operand(ops[1], bits)
      v = self.operand(ops[2], bits)
      eq, new = self.t(), self.t()
      self.emit('%s = icmp eq %s %s, %s' % (eq, T, old, c))
      self.emit('%s = select i1 %s, %s %s, %s %s' % (new, eq, T, v, T, old))
    else:
      v = self.operand(ops[1], bits)
      new = self.t()
      if op in ('add', 'and', 'or', 'xor'):
        self.emit('%s = %s %s %s, %s' % (new, op, T, old, v))
      elif op == 'exch':
        new = v
      elif op in ('min', 'max'):
        new = self.intr(('s' if kind == 's' else 'u') + op, T,
                        [(T, old), (T, v)])
      elif op == 'inc':
        # (old >= b) ? 0 : old + 1, unsigned.
        ge, inc = self.t(), self.t()
        self.emit('%s = icmp uge %s %s, %s' % (ge, T, old, v))
        self.emit('%s = add %s %s, 1' % (inc, T, old))
        self.emit('%s = select i1 %s, %s 0, %s %s' % (new, ge, T, T, inc))
      elif op == 'dec':
        # (old == 0 || old > b) ? b : old - 1, unsigned.
        z, gt, o, dec = self.t(), self.t(), self.t(), self.t()
        self.emit('%s = icmp eq %s %s, 0' % (z, T, old))
        self.emit('%s = icmp ugt %s %s, %s' % (gt, T, old, v))
        self.emit('%s = or i1 %s, %s' % (o, z, gt))
        self.emit('%s = sub %s %s, 1' % (dec, T, old))
        self.emit('%s = select i1 %s, %s %s, %s %s' % (new, o, T, v, T, dec))
      else:
        raise LiftError('atomic not lifted: ' + str(x))
    self.emit('store %s %s, ptr %s, align %d' % (T, new, p, bits // 8))
    if ordered:
      self.emit('fence acq_rel')
    if not red:
      self.write_reg(x.operands[0], old, T)

  def read_bits(self, p, bits):
    """The bits at p, each byte frozen."""
    v = self.t('ld')
    if bits == 8:
      self.emit('%s = load i8, ptr %s, align 1' % (v, p))
      return self.freeze(v, 'i8')
    vt = '<%d x i8>' % (bits // 8)
    self.emit('%s = load %s, ptr %s, align %d' % (v, vt, p, bits // 8))
    return self.cast('bitcast', self.freeze(v, vt), vt, 'i%d' % bits)

  def flush32(self, v):
    """An f32's bits with a subnormal flushed to zero of its sign."""
    e, m, sub, sign, r = self.t(), self.t(), self.t(), self.t(), self.t()
    self.emit('%s = and i32 %s, 2139095040' % (e, v))
    self.emit('%s = and i32 %s, 8388607' % (m, v))
    z, nz = self.t(), self.t()
    self.emit('%s = icmp eq i32 %s, 0' % (z, e))
    self.emit('%s = icmp ne i32 %s, 0' % (nz, m))
    self.emit('%s = and i1 %s, %s' % (sub, z, nz))
    self.emit('%s = and i32 %s, -2147483648' % (sign, v))
    self.emit('%s = select i1 %s, i32 %s, i32 %s' % (r, sub, sign, v))
    return r

  def i_atom(self, x):
    self.atomic(x, False)

  def i_red(self, x):
    self.atomic(x, True)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('ir', help='the LLVM IR module (NVPTX)')
  ap.add_argument('ptx', help="llc's PTX for it")
  ap.add_argument('function', help='the function to lift')
  ap.add_argument('-o', '--output', default='-')
  args = ap.parse_args()
  ll = open(args.ir).read()
  ptx = open(args.ptx).read()
  try:
    out = Lifter(ll, ptx, args.function).lift()
  except LiftError as e:
    print('ptx_lift: cannot lift %s: %s' % (args.function, e),
          file=sys.stderr)
    return 2
  if args.output == '-':
    sys.stdout.write(out)
  else:
    open(args.output, 'w').write(out)
  return 0


if __name__ == '__main__':
  sys.exit(main())

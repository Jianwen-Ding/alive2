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

The meaning of each instruction is the PTX ISA's ("Instructions"), not the
backend's patterns: shifts by the type's width or more give 0 (or the sign),
a division by zero gives some value, cvt to an integer saturates and takes
NaN to 0, setp's ne is ordered, and so on. What the lifted code says:

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
import re
import struct
import sys


class LiftError(Exception):
  pass


# ---------------------------------------------------------------------------
# The input IR: the function's signature and what the lifted module needs.


def split_top(s, sep=','):
  """Split s at sep outside of parentheses, brackets and braces."""
  out, depth, cur = [], 0, ''
  for c in s:
    if c in '([{<':
      depth += 1
    elif c in ')]}>':
      depth -= 1
    if c == sep and depth == 0:
      out.append(cur)
      cur = ''
    else:
      cur += c
  if cur.strip():
    out.append(cur)
  return [x.strip() for x in out]


class Param:
  def __init__(self, text):
    self.text = text
    m = re.match(r'(ptr(?: addrspace\(\d+\))?|<[^>]*>|\[[^\]]*\]|\{[^}]*\}|'
                 r'%[\w.$]+|\S+)', text)
    self.type = m.group(1)
    self.name = text.split()[-1]
    if not self.name.startswith('%'):
      raise LiftError('parameter without a name: ' + text)
    self.byval = re.search(r'\bbyval\(', text) is not None
    self.is_ptr = self.type.startswith('ptr')


def one_line_headers(ll):
  """The module with each function definition's header on one line."""
  out, lines, i = [], ll.split('\n'), 0
  while i < len(lines):
    l = lines[i]
    if l.startswith('define ') and '{' not in l:
      while i + 1 < len(lines) and '{' not in l:
        i += 1
        l = l.rstrip() + ' ' + lines[i].strip()
    out.append(l)
    i += 1
  return '\n'.join(out)


def parse_signature(ll, name):
  """(text before @name, return type, cc, params) of @name's definition."""
  m = re.search(r'^define\s+(.*?)@"?' + re.escape(name) + r'"?\(', ll, re.M)
  if not m:
    raise LiftError('no definition of @' + name)
  i, depth = m.end(), 1
  while depth:
    depth += {'(': 1, ')': -1}.get(ll[i], 0)
    i += 1
  params_text = ll[m.end():i - 1]
  params = ([Param(p) for p in split_top(params_text)]
            if params_text.strip() else [])
  prefix = m.group(1)
  cc = 'ptx_kernel ' if re.search(r'\bptx_kernel\b', prefix) else ''
  # The return type: the last type-looking word of the prefix.
  words = re.sub(r'\b(ptx_kernel|ptx_device|dso_local|internal|private|'
                 r'linkonce_odr|weak_odr|hidden|protected|default|'
                 r'local_unnamed_addr|unnamed_addr|noundef|signext|zeroext|'
                 r'inreg|nonnull)\b', ' ', prefix).split()
  ret = words[-1] if words else 'void'
  return ret, cc, params


def lifted_param(p):
  """A parameter's text in the lifted function: without capture attributes
  (the lifted code takes pointers apart, which they would make poison;
  dropping them only weakens the target's assumptions)."""
  return re.sub(r'\s*\b(captures\([^)]*\)|nocapture)', '', p.text)


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
    m = re.compile(r'([$\w.]+)\s*:').match(body, i)
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
    self.op = parts[0]
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

# The approximate instructions: the NVVM intrinsics of the same meaning
# (uninterpreted), by (opcode, ftz, type).
APPROX = {
    ('rsqrt', False, 'f32'): 'llvm.nvvm.rsqrt.approx.f',
    ('rsqrt', True, 'f32'): 'llvm.nvvm.rsqrt.approx.ftz.f',
    ('rsqrt', False, 'f64'): 'llvm.nvvm.rsqrt.approx.d',
    ('rsqrt', True, 'f64'): 'llvm.nvvm.rsqrt.approx.ftz.d',
    ('sqrt', False, 'f32'): 'llvm.nvvm.sqrt.approx.f',
    ('sqrt', True, 'f32'): 'llvm.nvvm.sqrt.approx.ftz.f',
    ('div', False, 'f32'): 'llvm.nvvm.div.approx.f',
    ('div', True, 'f32'): 'llvm.nvvm.div.approx.ftz.f',
    ('lg2', False, 'f32'): 'llvm.nvvm.lg2.approx.f',
    ('lg2', True, 'f32'): 'llvm.nvvm.lg2.approx.ftz.f',
    ('lg2', False, 'f64'): 'llvm.nvvm.lg2.approx.d',
    ('ex2', False, 'f32'): 'llvm.nvvm.ex2.approx.f32',
    ('ex2', True, 'f32'): 'llvm.nvvm.ex2.approx.ftz.f32',
    ('ex2', False, 'f64'): 'llvm.nvvm.ex2.approx.f64',
    ('rcp', True, 'f64'): 'llvm.nvvm.rcp.approx.ftz.d',
}

FCMP = {'eq': 'oeq', 'ne': 'one', 'lt': 'olt', 'le': 'ole', 'gt': 'ogt',
        'ge': 'oge', 'equ': 'ueq', 'neu': 'une', 'ltu': 'ult', 'leu': 'ule',
        'gtu': 'ugt', 'geu': 'uge', 'num': 'ord', 'nan': 'uno'}

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
      if m.group(3):
        raise LiftError('aggregate return values are not lifted')
      self.retparam = (m.group(2), TYPES[m.group(1)][1])
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

  def lift(self):
    self.ptr = self.analyze_pointers()
    out = []
    params = ', '.join(lifted_param(p) for p in self.params)
    out.append('define %s%s @%s(%s) {' % (self.cc, self.ret, self.name,
                                          params))
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
    if self.retparam:
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
    head = ['target triple = "nvptx64-nvidia-cuda"', '']
    # The module's named types (of byval parameters, of variables).
    head += re.findall(r'^%[\w.$"]+ = type .*$', self.ll, re.M)
    for n in sorted(self.globals_used):
      head.append(self.globals[n][0])
    head.append('')
    return '\n'.join(head + out + [''] + sorted(self.decls.values())) + '\n'

  def ret_instr(self):
    if self.ret == 'void':
      self.emit('ret void')
      return
    bits = self.retparam[1] if self.retparam else None
    if bits is None:
      raise LiftError('a value returned without a return parameter')
    v = self.t()
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
    try:
      f = getattr(self, 'i_' + x.base)
    except AttributeError:
      raise LiftError('instruction not lifted: ' + str(x))
    r = f(x)
    return bool(r)

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

  # Integer arithmetic.

  def binop(self, x, op, sign=False):
    ty = self.itype(x)
    kind, bits = TYPES[ty]
    a = self.operand(x.operands[1], bits, sign)
    b = self.operand(x.operands[2], bits, sign)
    r = self.t()
    self.emit('%s = %s i%d %s, %s' % (r, op, bits, a, b))
    self.write_reg(x.operands[0], r, 'i%d' % bits)

  def is_float(self, x):
    return TYPES[self.itype(x)][0] in ('f', 'bf')

  def i_add(self, x):
    if self.is_float(x):
      return self.fbinop(x, 'fadd')
    self.no_mods(x, ())
    d, a, b = x.operands
    if d in self.ptr:
      pa, pb = self.pointer(a), self.pointer(b)
      if pa is not None and pb is None:
        return self.write_reg(d, self.gep(pa, self.operand(b, 64)), 'ptr')
      if pb is not None and pa is None:
        return self.write_reg(d, self.gep(pb, self.operand(a, 64)), 'ptr')
    self.binop(x, 'add')

  def i_sub(self, x):
    if self.is_float(x):
      return self.fbinop(x, 'fsub')
    self.no_mods(x, ())
    d, a, b = x.operands
    if d in self.ptr:
      pa = self.pointer(a)
      if pa is not None and self.pointer(b) is None:
        n = self.t()
        self.emit('%s = sub i64 0, %s' % (n, self.operand(b, 64)))
        return self.write_reg(d, self.gep(pa, n), 'ptr')
    self.binop(x, 'sub')

  def i_and(self, x):
    self.binop(x, 'and')

  def i_or(self, x):
    self.binop(x, 'or')

  def i_xor(self, x):
    self.binop(x, 'xor')

  def i_not(self, x):
    ty = self.itype(x)
    bits = TYPES[ty][1]
    a = self.operand(x.operands[1], bits)
    r = self.t()
    self.emit('%s = xor i%d %s, -1' % (r, bits, a))
    self.write_reg(x.operands[0], r, 'i%d' % bits)

  def i_cnot(self, x):
    bits = TYPES[self.itype(x)][1]
    a = self.operand(x.operands[1], bits)
    c, r = self.t(), self.t()
    self.emit('%s = icmp eq i%d %s, 0' % (c, bits, a))
    self.emit('%s = zext i1 %s to i%d' % (r, c, bits))
    self.write_reg(x.operands[0], r, 'i%d' % bits)

  def mul_parts(self, x):
    """(mode, signed, bits) of mul/mad: mode lo, hi or wide."""
    ty = self.itype(x)
    kind, bits = TYPES[ty]
    mode = 'lo'
    for m in ('lo', 'hi', 'wide'):
      if x.has(m):
        mode = m
    return mode, kind == 's', bits

  def product(self, x, a, b):
    mode, sg, bits = self.mul_parts(x)
    av = self.operand(a, bits, sg)
    bv = self.operand(b, bits, sg)
    if mode == 'lo':
      r = self.t()
      self.emit('%s = mul i%d %s, %s' % (r, bits, av, bv))
      return r, bits
    w = 2 * bits
    ea, eb = self.resize(av, bits, w, sg), self.resize(bv, bits, w, sg)
    p = self.t()
    self.emit('%s = mul i%d %s, %s' % (p, w, ea, eb))
    if mode == 'wide':
      return p, w
    h = self.t()
    self.emit('%s = lshr i%d %s, %d' % (h, w, p, bits))
    return self.resize(h, w, bits, False), bits

  def i_mul(self, x):
    if self.is_float(x):
      return self.fbinop(x, 'fmul')
    if x.has('24'):
      raise LiftError('mul24 not lifted: ' + str(x))
    p, bits = self.product(x, x.operands[1], x.operands[2])
    self.write_reg(x.operands[0], p, 'i%d' % bits)

  def i_mad(self, x):
    if self.is_float(x):
      return self.ffma(x)
    if x.has('24', 'cc', 'sat'):
      raise LiftError('mad modifier not lifted: ' + str(x))
    p, bits = self.product(x, x.operands[1], x.operands[2])
    c = self.operand(x.operands[3], bits)
    r = self.t()
    self.emit('%s = add i%d %s, %s' % (r, bits, p, c))
    self.write_reg(x.operands[0], r, 'i%d' % bits)

  def i_div(self, x):
    if self.is_float(x):
      return self.fdiv(x)
    self.divrem(x, 'div')

  def i_rem(self, x):
    self.divrem(x, 'rem')

  def divrem(self, x, what):
    """A division by zero (or of the least value by -1) gives some value
    (PTX ISA: "unspecified, machine-specific")."""
    kind, bits = TYPES[self.itype(x)]
    sg = kind == 's'
    a = self.operand(x.operands[1], bits)
    b = self.operand(x.operands[2], bits)
    T = 'i%d' % bits
    bad = self.t()
    self.emit('%s = icmp eq %s %s, 0' % (bad, T, b))
    if sg:
      m1, mn, both, bad2 = self.t(), self.t(), self.t(), self.t()
      self.emit('%s = icmp eq %s %s, -1' % (m1, T, b))
      self.emit('%s = icmp eq %s %s, %d' % (mn, T, a, -(1 << (bits - 1))))
      self.emit('%s = and i1 %s, %s' % (both, m1, mn))
      self.emit('%s = or i1 %s, %s' % (bad2, bad, both))
      bad = bad2
    sb, q, any_, r = self.t(), self.t(), self.t(), self.t()
    self.emit('%s = select i1 %s, %s 1, %s %s' % (sb, bad, T, T, b))
    op = ('s' if sg else 'u') + what
    self.emit('%s = %s %s %s, %s' % (q, op, T, a, sb))
    self.emit('%s = freeze %s poison' % (any_, T))
    self.emit('%s = select i1 %s, %s %s, %s %s' % (r, bad, T, any_, T, q))
    self.write_reg(x.operands[0], r, T)

  def i_neg(self, x):
    if self.is_float(x):
      return self.funop(x, 'fneg')
    bits = TYPES[self.itype(x)][1]
    a = self.operand(x.operands[1], bits)
    r = self.t()
    self.emit('%s = sub i%d 0, %s' % (r, bits, a))
    self.write_reg(x.operands[0], r, 'i%d' % bits)

  def i_abs(self, x):
    if self.is_float(x):
      return self.funop(x, 'fabs')
    bits = TYPES[self.itype(x)][1]
    a = self.operand(x.operands[1], bits)
    r = self.intr('abs', 'i%d' % bits, [('i%d' % bits, a), ('i1', 'false')])
    self.write_reg(x.operands[0], r, 'i%d' % bits)

  def minmax(self, x, which):
    if self.is_float(x):
      return self.fminmax(x, which)
    kind, bits = TYPES[self.itype(x)]
    T = 'i%d' % bits
    a = self.operand(x.operands[1], bits)
    b = self.operand(x.operands[2], bits)
    r = self.intr(('s' if kind == 's' else 'u') + which, T,
                  [(T, a), (T, b)])
    self.write_reg(x.operands[0], r, T)

  def i_min(self, x):
    self.minmax(x, 'min')

  def i_max(self, x):
    self.minmax(x, 'max')

  def shift(self, x, op):
    """Shifts by the width or more: 0, or the sign for shr.s (PTX ISA:
    "Shift amounts greater than the register width N are clamped to N")."""
    kind, bits = TYPES[self.itype(x)]
    T = 'i%d' % bits
    a = self.operand(x.operands[1], bits)
    s = self.operand(x.operands[2], 32)
    big, cl = self.t(), self.t()
    self.emit('%s = icmp uge i32 %s, %d' % (big, s, bits))
    self.emit('%s = select i1 %s, i32 %d, i32 %s' % (cl, big, bits - 1, s))
    amt = self.resize(cl, 32, bits, False)
    r = self.t()
    if op == 'shr' and kind == 's':
      self.emit('%s = ashr %s %s, %s' % (r, T, a, amt))
    else:
      v = self.t()
      self.emit('%s = %s %s %s, %s' % (v, 'shl' if op == 'shl' else 'lshr',
                                       T, a, amt))
      self.emit('%s = select i1 %s, %s 0, %s %s' % (r, big, T, T, v))
    self.write_reg(x.operands[0], r, T)

  def i_shl(self, x):
    self.shift(x, 'shl')

  def i_shr(self, x):
    self.shift(x, 'shr')

  def i_shf(self, x):
    """shf.{l,r}.{wrap,clamp}.b32 d, a, b, c: the funnel shift of {b, a}
    (b the high word) by c mod 32 (wrap) or min(c, 32) (clamp)."""
    left = x.has('l')
    if not (left or x.has('r')) or not (x.has('wrap') or x.has('clamp')):
      raise LiftError('shf not lifted: ' + str(x))
    a = self.operand(x.operands[1], 32)
    b = self.operand(x.operands[2], 32)
    c = self.operand(x.operands[3], 32)
    if x.has('wrap'):
      n = self.t()
      self.emit('%s = and i32 %s, 31' % (n, c))
      # fshl/fshr take the amount mod 32.
      r = self.intr('fshl' if left else 'fshr', 'i32',
                    [('i32', b), ('i32', a), ('i32', n)])
    else:
      big = self.t()
      self.emit('%s = icmp uge i32 %s, 32' % (big, c))
      f = self.intr('fshl' if left else 'fshr', 'i32',
                    [('i32', b), ('i32', a), ('i32', c)])
      r = self.t()
      # Shifted by 32 or more: the whole other word.
      self.emit('%s = select i1 %s, i32 %s, i32 %s' % (r, big,
                                                       a if left else b, f))
    self.write_reg(x.operands[0], r, 'i32')

  def i_popc(self, x):
    bits = TYPES[self.itype(x)][1]
    T = 'i%d' % bits
    r = self.intr('ctpop', T, [(T, self.operand(x.operands[1], bits))])
    self.write_reg(x.operands[0], self.resize(r, bits, 32, False), 'i32')

  def i_clz(self, x):
    bits = TYPES[self.itype(x)][1]
    T = 'i%d' % bits
    r = self.intr('ctlz', T, [(T, self.operand(x.operands[1], bits)),
                              ('i1', 'false')])
    self.write_reg(x.operands[0], self.resize(r, bits, 32, False), 'i32')

  def i_brev(self, x):
    bits = TYPES[self.itype(x)][1]
    T = 'i%d' % bits
    r = self.intr('bitreverse', T, [(T, self.operand(x.operands[1], bits))])
    self.write_reg(x.operands[0], r, T)

  def field_mask(self, n, bits):
    """An i<bits> mask of the n (i<bits>, at most bits) lowest bits."""
    T = 'i%d' % bits
    full, sh, m1, m = self.t(), self.t(), self.t(), self.t()
    self.emit('%s = icmp uge %s %s, %d' % (full, T, n, bits))
    self.emit('%s = shl %s 1, %s' % (sh, T, n))
    self.emit('%s = sub %s %s, 1' % (m1, T, sh))
    self.emit('%s = select i1 %s, %s -1, %s %s' % (m, full, T, T, m1))
    return m

  def i_bfe(self, x):
    """PTX ISA, bfe: d[i] = (i < len && pos + i <= msb) ? a[pos + i] : sbit,
    sbit 0 for .u or len 0, else a[min(pos + len - 1, msb)]."""
    kind, bits = TYPES[self.itype(x)]
    T = 'i%d' % bits
    a = self.operand(x.operands[1], bits)
    pos32, len32 = (self.operand(x.operands[2], 32),
                    self.operand(x.operands[3], 32))
    pos, ln = self.t(), self.t()
    self.emit('%s = and i32 %s, 255' % (pos, pos32))
    self.emit('%s = and i32 %s, 255' % (ln, len32))
    out = self.t()
    self.emit('%s = icmp ugt i32 %s, %d' % (out, pos, bits - 1))
    avail, n = self.t(), self.t()
    self.emit('%s = sub i32 %d, %s' % (avail, bits, pos))
    self.emit('%s = call i32 @llvm.umin.i32(i32 %s, i32 %s)' % (n, ln, avail))
    self.declare('llvm.umin.i32', 'i32', ['i32', 'i32'])
    n2 = self.t()
    self.emit('%s = select i1 %s, i32 0, i32 %s' % (n2, out, n))
    ps, sh = self.t(), self.t()
    self.emit('%s = select i1 %s, i32 0, i32 %s' % (ps, out, pos))
    self.emit('%s = lshr %s %s, %s' % (sh, T, a, self.resize(ps, 32, bits,
                                                              False)))
    m = self.field_mask(self.resize(n2, 32, bits, False), bits)
    f = self.t()
    self.emit('%s = and %s %s, %s' % (f, T, sh, m))
    if kind == 's':
      top, ti, tc, bit, z, sb = (self.t(), self.t(), self.t(), self.t(),
                                 self.t(), self.t())
      self.emit('%s = add i32 %s, %s' % (top, pos, ln))
      self.emit('%s = sub i32 %s, 1' % (ti, top))
      self.emit('%s = call i32 @llvm.umin.i32(i32 %s, i32 %d)' %
                (tc, ti, bits - 1))
      self.emit('%s = lshr %s %s, %s' % (bit, T, a, self.resize(tc, 32, bits,
                                                                 False)))
      self.emit('%s = icmp eq i32 %s, 0' % (z, ln))
      b1, inv, sx, r = self.t(), self.t(), self.t(), self.t()
      self.emit('%s = trunc %s %s to i1' % (b1, T, bit))
      self.emit('%s = select i1 %s, i1 false, i1 %s' % (sb, z, b1))
      self.emit('%s = xor %s %s, -1' % (inv, T, m))
      self.emit('%s = select i1 %s, %s %s, %s 0' % (sx, sb, T, inv, T))
      self.emit('%s = or %s %s, %s' % (r, T, f, sx))
      f = r
    self.write_reg(x.operands[0], f, T)

  def i_bfi(self, x):
    """PTX ISA, bfi: f = b; f[pos + i] = a[i] for i < len, pos + i <= msb."""
    bits = TYPES[self.itype(x)][1]
    T = 'i%d' % bits
    a = self.operand(x.operands[1], bits)
    b = self.operand(x.operands[2], bits)
    pos32, len32 = (self.operand(x.operands[3], 32),
                    self.operand(x.operands[4], 32))
    pos, ln, out, avail, n, n2, ps = (self.t(), self.t(), self.t(), self.t(),
                                      self.t(), self.t(), self.t())
    self.emit('%s = and i32 %s, 255' % (pos, pos32))
    self.emit('%s = and i32 %s, 255' % (ln, len32))
    self.emit('%s = icmp ugt i32 %s, %d' % (out, pos, bits - 1))
    self.emit('%s = sub i32 %d, %s' % (avail, bits, pos))
    self.declare('llvm.umin.i32', 'i32', ['i32', 'i32'])
    self.emit('%s = call i32 @llvm.umin.i32(i32 %s, i32 %s)' % (n, ln, avail))
    self.emit('%s = select i1 %s, i32 0, i32 %s' % (n2, out, n))
    self.emit('%s = select i1 %s, i32 0, i32 %s' % (ps, out, pos))
    psT = self.resize(ps, 32, bits, False)
    m = self.field_mask(self.resize(n2, 32, bits, False), bits)
    am, ash, msh, inv, bm, r = (self.t(), self.t(), self.t(), self.t(),
                                self.t(), self.t())
    self.emit('%s = and %s %s, %s' % (am, T, a, m))
    self.emit('%s = shl %s %s, %s' % (ash, T, am, psT))
    self.emit('%s = shl %s %s, %s' % (msh, T, m, psT))
    self.emit('%s = xor %s %s, -1' % (inv, T, msh))
    self.emit('%s = and %s %s, %s' % (bm, T, b, inv))
    self.emit('%s = or %s %s, %s' % (r, T, bm, ash))
    self.write_reg(x.operands[0], r, T)

  def i_prmt(self, x):
    """prmt.b32 (default mode): byte i of d is byte c[4i+2:4i] of {b, a},
    its sign replicated when c[4i+3]."""
    if len(x.mods) != 2:
      raise LiftError('prmt mode not lifted: ' + str(x))
    a = self.operand(x.operands[1], 32)
    b = self.operand(x.operands[2], 32)
    c = self.operand(x.operands[3], 32)
    a64, b64, bh, ab = self.t(), self.t(), self.t(), self.t()
    self.emit('%s = zext i32 %s to i64' % (a64, a))
    self.emit('%s = zext i32 %s to i64' % (b64, b))
    self.emit('%s = shl i64 %s, 32' % (bh, b64))
    self.emit('%s = or i64 %s, %s' % (ab, bh, a64))
    acc = '0'
    for i in range(4):
      s, sel, s8, s64, by, sg, rep, sx, b32, bsh, o = [self.t()
                                                       for _ in range(11)]
      self.emit('%s = lshr i32 %s, %d' % (s, c, 4 * i))
      self.emit('%s = and i32 %s, 7' % (sel, s))
      self.emit('%s = mul i32 %s, 8' % (s8, sel))
      self.emit('%s = zext i32 %s to i64' % (s64, s8))
      sh = self.t()
      self.emit('%s = lshr i64 %s, %s' % (sh, ab, s64))
      self.emit('%s = trunc i64 %s to i8' % (by, sh))
      self.emit('%s = and i32 %s, 8' % (sg, s))
      self.emit('%s = ashr i8 %s, 7' % (rep, by))
      c8 = self.t()
      self.emit('%s = icmp ne i32 %s, 0' % (c8, sg))
      self.emit('%s = select i1 %s, i8 %s, i8 %s' % (sx, c8, rep, by))
      self.emit('%s = zext i8 %s to i32' % (b32, sx))
      self.emit('%s = shl i32 %s, %d' % (bsh, b32, 8 * i))
      self.emit('%s = or i32 %s, %s' % (o, acc, bsh))
      acc = o
    self.write_reg(x.operands[0], acc, 'i32')

  # Comparisons and selection.

  def i_setp(self, x):
    """setp.cmp[.bop].type p, a, b[, [!]c]: p = (a cmp b) [bop c]."""
    bop = [m for m in x.mods[1:] if m in ('and', 'or', 'xor')]
    if (len(x.operands) != (4 if bop else 3) or '|' in x.operands[0]):
      raise LiftError('setp form not lifted: ' + str(x))
    ty = self.itype(x)
    kind, bits = TYPES[ty]
    cmp = x.mods[1]
    if x.has('ftz'):
      raise LiftError('setp modifier not lifted: ' + str(x))
    r = self.t()
    if kind in ('f', 'bf'):
      ft = FLOAT_TYPE[(kind, bits)]
      a = self.fp(self.operand(x.operands[1], bits), bits, ft)
      b = self.fp(self.operand(x.operands[2], bits), bits, ft)
      if cmp not in FCMP:
        raise LiftError('comparison ' + str(x))
      self.emit('%s = fcmp %s %s %s, %s' % (r, FCMP[cmp], ft, a, b))
    else:
      a = self.operand(x.operands[1], bits)
      b = self.operand(x.operands[2], bits)
      pred = {'eq': 'eq', 'ne': 'ne', 'lt': 'lt', 'le': 'le', 'gt': 'gt',
              'ge': 'ge', 'lo': 'ult', 'ls': 'ule', 'hi': 'ugt',
              'hs': 'uge'}.get(cmp)
      if pred is None:
        raise LiftError('comparison ' + str(x))
      if pred in ('lt', 'le', 'gt', 'ge'):
        pred = ('s' if kind == 's' else 'u') + pred
      self.emit('%s = icmp %s i%d %s, %s' % (r, pred, bits, a, b))
    if bop:
      c = x.operands[3].strip()
      cv = self.operand(c.lstrip('!'), 1)
      if c.startswith('!'):
        n = self.t()
        self.emit('%s = xor i1 %s, true' % (n, cv))
        cv = n
      r2 = self.t()
      self.emit('%s = %s i1 %s, %s' % (r2, bop[0], r, cv))
      r = r2
    self.write_reg(x.operands[0], r, 'i1')

  def i_selp(self, x):
    d, a, b, c = x.operands
    bits = TYPES[self.itype(x)][1]
    p = self.operand(c, 1)
    r = self.t()
    if d in self.ptr:
      pa, pb = self.pointer(a), self.pointer(b)
      if pa is not None and pb is not None:
        self.emit('%s = select i1 %s, ptr %s, ptr %s' % (r, p, pa, pb))
        return self.write_reg(d, r, 'ptr')
    av, bv = self.operand(a, bits), self.operand(b, bits)
    self.emit('%s = select i1 %s, i%d %s, i%d %s' % (r, p, bits, av, bits, bv))
    self.write_reg(d, r, 'i%d' % bits)

  # Floating point.

  def fparts(self, x):
    """(LLVM type, bits) of a float instruction; rejects what is not
    lifted: other roundings than to nearest, ftz, sat."""
    ty = self.itype(x)
    kind, bits = TYPES[ty]
    if (kind, bits) not in FLOAT_TYPE:
      raise LiftError('float type not lifted: ' + str(x))
    for m in x.mods[1:]:
      if m in ('rz', 'rm', 'rp'):
        raise LiftError('rounding .%s not lifted: %s' % (m, x))
      if m in ('ftz', 'sat', 'relu'):
        raise LiftError('modifier .%s not lifted: %s' % (m, x))
    return FLOAT_TYPE[(kind, bits)], bits

  def flags(self, x):
    """add/sub/mul without a rounding modifier may be contracted into fma
    (PTX ISA, "Floating-Point Instructions")."""
    return '' if x.has('rn') else 'contract '

  def fval(self, tok, bits, ft):
    return self.fp(self.operand(tok, bits), bits, ft)

  def fresult(self, x, v, ft, bits):
    self.write_reg(x.operands[0], self.unfp(v, ft, bits), 'i%d' % bits)

  def fbinop(self, x, op):
    ft, bits = self.fparts(x)
    a = self.fval(x.operands[1], bits, ft)
    b = self.fval(x.operands[2], bits, ft)
    r = self.t()
    self.emit('%s = %s %s%s %s, %s' % (r, op, self.flags(x), ft, a, b))
    self.fresult(x, r, ft, bits)

  def funop(self, x, op):
    ft, bits = self.fparts(x)
    a = self.fval(x.operands[1], bits, ft)
    r = self.t()
    if op == 'fneg':
      self.emit('%s = fneg %s %s' % (r, ft, a))
    else:
      r = self.intr(op, ft, [(ft, a)])
    self.fresult(x, r, ft, bits)

  def ffma(self, x):
    if not x.has('rn'):
      raise LiftError('fma/mad without rounding not lifted: ' + str(x))
    ft, bits = self.fparts(x)
    a, b, c = [self.fval(o, bits, ft) for o in x.operands[1:4]]
    r = self.intr('fma', ft, [(ft, a), (ft, b), (ft, c)])
    self.fresult(x, r, ft, bits)

  i_fma = ffma

  def approx(self, x, n):
    ty = self.itype(x)
    name = APPROX.get((x.base, x.has('ftz'), ty))
    if name is None:
      raise LiftError('approximation not lifted: ' + str(x))
    bits = TYPES[ty][1]
    ft = FLOAT_TYPE[TYPES[ty]]
    args = [(ft, self.fval(o, bits, ft)) for o in x.operands[1:1 + n]]
    r = self.call(name, ft, args, ' nounwind readnone')
    self.fresult(x, r, ft, bits)

  def fdiv(self, x):
    if x.has('approx'):
      return self.approx(x, 2)
    if x.has('full') or not x.has('rn'):
      raise LiftError('division not lifted: ' + str(x))
    self.fbinop(x, 'fdiv')

  def i_sqrt(self, x):
    if x.has('approx'):
      return self.approx(x, 1)
    if not x.has('rn'):
      raise LiftError('sqrt not lifted: ' + str(x))
    self.funop(x, 'sqrt')

  def i_rcp(self, x):
    if x.has('approx'):
      return self.approx(x, 1)
    if not x.has('rn'):
      raise LiftError('rcp not lifted: ' + str(x))
    ft, bits = self.fparts(x)
    a = self.fval(x.operands[1], bits, ft)
    r = self.t()
    self.emit('%s = fdiv %s 1.0, %s' % (r, ft, a))
    self.fresult(x, r, ft, bits)

  def i_rsqrt(self, x):
    self.approx(x, 1)

  def i_ex2(self, x):
    self.approx(x, 1)

  def i_lg2(self, x):
    self.approx(x, 1)

  def fminmax(self, x, which):
    """min/max.f: a NaN operand gives the other (both NaN: NaN), as
    minnum/maxnum."""
    if x.has('NaN', 'xorsign', 'abs'):
      raise LiftError('min/max modifier not lifted: ' + str(x))
    ft, bits = self.fparts(x)
    a = self.fval(x.operands[1], bits, ft)
    b = self.fval(x.operands[2], bits, ft)
    r = self.intr(which + 'num', ft, [(ft, a), (ft, b)])
    self.fresult(x, r, ft, bits)

  def i_copysign(self, x):
    # copysign.f d, a, b: b with a's sign.
    ft, bits = self.fparts(x)
    a = self.fval(x.operands[1], bits, ft)
    b = self.fval(x.operands[2], bits, ft)
    r = self.intr('copysign', ft, [(ft, b), (ft, a)])
    self.fresult(x, r, ft, bits)

  def i_testp(self, x):
    raise LiftError('testp not lifted: ' + str(x))

  # Conversions.

  def i_cvt(self, x):
    tys = [m for m in x.mods[1:] if m in TYPES]
    if len(tys) != 2:
      raise LiftError('cvt types: ' + str(x))
    dt, st = tys
    dk, db = TYPES[dt]
    sk, sb = TYPES[st]
    rnd = [m for m in x.mods[1:] if m in ('rn', 'rz', 'rm', 'rp', 'rni',
                                          'rzi', 'rmi', 'rpi', 'rna')]
    rnd = rnd[0] if rnd else None
    for m in x.mods[1:]:
      if m in ('ftz', 'sat', 'relu', 'satfinite'):
        raise LiftError('cvt modifier .%s not lifted: %s' % (m, x))
    dfl, sfl = dk in ('f', 'bf'), sk in ('f', 'bf')
    d, s = x.operands
    if not dfl and not sfl:
      # Integers: extend by the source's type, or truncate.
      v = self.operand(s, sb, sk == 's')
      v = self.resize(v, sb, db, sk == 's')
      dbits = self.reg(d)
      v = self.resize(v, db, max(db, dbits) if dbits > db else db,
                      dk == 's')
      self.write_reg(d, v, 'i%d' % (max(db, dbits) if dbits > db else db))
      return
    if sfl:
      sft = FLOAT_TYPE[(sk, sb)]
      a = self.fval(s, sb, sft)
    if not sfl:
      a = self.operand(s, sb)
    if sfl and dfl:
      dft = FLOAT_TYPE[(dk, db)]
      if rnd in ('rni', 'rzi', 'rmi', 'rpi'):
        if dt != st:
          raise LiftError('cvt: ' + str(x))
        fn = {'rni': 'roundeven', 'rzi': 'trunc', 'rmi': 'floor',
              'rpi': 'ceil'}[rnd]
        r = self.intr(fn, sft, [(sft, a)])
      elif db > sb:
        if rnd:
          raise LiftError('cvt: ' + str(x))
        r = self.cast('fpext', a, sft, dft)
      elif db < sb:
        if rnd != 'rn':
          raise LiftError('cvt rounding not lifted: ' + str(x))
        r = self.cast('fptrunc', a, sft, dft)
      else:
        raise LiftError('cvt: ' + str(x))
      self.write_reg(d, self.unfp(r, dft, db), 'i%d' % db)
      return
    if sfl:
      # Float to integer: rounded, saturating, NaN to 0 (PTX ISA, cvt).
      fn = {'rni': 'roundeven', 'rzi': None, 'rmi': 'floor',
            'rpi': 'ceil'}.get(rnd, 'bad')
      if fn == 'bad':
        raise LiftError('cvt rounding not lifted: ' + str(x))
      if fn:
        a = self.intr(fn, sft, [(sft, a)])
      it = 'i%d' % db
      name = 'llvm.fpto%si.sat.%s.%s' % ('s' if dk == 's' else 'u', it,
                                         {'float': 'f32', 'double': 'f64',
                                          'half': 'f16',
                                          'bfloat': 'bf16'}[sft])
      r = self.call(name, it, [(sft, a)])
      dbits = self.reg(d)
      self.write_reg(d, self.resize(r, db, max(db, dbits), dk == 's'),
                     'i%d' % max(db, dbits))
      return
    # Integer to float.
    dft = FLOAT_TYPE[(dk, db)]
    if rnd != 'rn':
      raise LiftError('cvt rounding not lifted: ' + str(x))
    r = self.cast('sitofp' if sk == 's' else 'uitofp', a, 'i%d' % sb, dft)
    self.write_reg(d, self.unfp(r, dft, db), 'i%d' % db)

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
    for m in x.mods[1:]:
      if m in ('volatile', 'relaxed', 'acquire', 'release', 'weak', 'cta',
               'gpu', 'sys', 'cluster', 'mmio'):
        if m not in ('weak',):
          raise LiftError('ordered or volatile access not lifted: ' + str(x))
    return space, n, kind, bits

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
        self.emit('%s = load ptr, ptr %s, align %d' % (v, a, al))
        self.write_reg(d, self.freeze(v, 'ptr'), 'ptr')
        continue
      v = self.t('ld')
      self.emit('%s = load i%d, ptr %s, align %d' % (v, bits, a, al))
      v = self.freeze(v, 'i%d' % bits)
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
      v = self.cast('bitcast', P.name, t, 'i%d' % w)
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
      v = self.cast('bitcast', P.name, t, 'i%d' % w)
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
        self.emit('store ptr %s, ptr %s, align %d' % (self.read_reg(s), a, al))
        continue
      v = self.operand(s, bits)
      self.emit('store i%d %s, ptr %s, align %d' % (bits, v, a, al))

  def st_param(self, x, n, bits):
    m = re.fullmatch(r'\[\s*([\w$]+)\s*(?:\+\s*(\d+))?\s*\]',
                     x.operands[0].strip())
    if (not m or not self.retparam or m.group(1) != self.retparam[0] or
        m.group(2) or n != 1):
      raise LiftError('parameter store not lifted: ' + str(x))
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
      elif m in ('cta', 'gpu', 'sys', 'cluster') or m in TYPES:
        pass
      else:
        raise LiftError('atomic modifier .%s not lifted: %s' % (m, x))
    kind, bits = TYPES[self.itype(x)]
    ops = x.operands if red else x.operands[1:]
    p = self.address(ops[0])
    T = 'i%d' % bits
    if kind in ('f', 'bf'):
      if op not in ('add', 'min', 'max'):
        raise LiftError('atomic not lifted: ' + str(x))
      ft = FLOAT_TYPE[(kind, bits)]
      v = self.fval(ops[1], bits, ft)
      rmw = {'add': 'fadd', 'min': 'fmin', 'max': 'fmax'}[op]
      r = self.t()
      self.emit('%s = atomicrmw %s ptr %s, %s %s %s, align %d' %
                (r, rmw, p, ft, v, order, bits // 8))
      r = self.unfp(r, ft, bits)
    elif op == 'cas':
      c = self.operand(ops[1], bits)
      v = self.operand(ops[2], bits)
      fail = {'acq_rel': 'acquire', 'release': 'monotonic'}.get(order, order)
      pr, r = self.t(), self.t()
      self.emit('%s = cmpxchg ptr %s, %s %s, %s %s %s %s, align %d' %
                (pr, p, T, c, T, v, order, fail, bits // 8))
      self.emit('%s = extractvalue { %s, i1 } %s, 0' % (r, T, pr))
    else:
      rmw = {'add': 'add', 'and': 'and', 'or': 'or', 'xor': 'xor',
             'exch': 'xchg', 'inc': 'uinc_wrap', 'dec': 'udec_wrap',
             'min': 'min' if kind == 's' else 'umin',
             'max': 'max' if kind == 's' else 'umax'}.get(op)
      if rmw is None:
        raise LiftError('atomic not lifted: ' + str(x))
      v = self.operand(ops[1], bits)
      r = self.t()
      self.emit('%s = atomicrmw %s ptr %s, %s %s %s, align %d' %
                (r, rmw, p, T, v, order, bits // 8))
    if not red:
      self.write_reg(x.operands[0], self.freeze(r, T), T)

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

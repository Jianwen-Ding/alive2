# Copyright (c) 2018-present The Alive2 Authors.
# Distributed under the MIT license that can be found in the LICENSE file.
"""What the GPU machine-code lifters share (agx-tv's agx_lift.py, ptx-tv's
ptx_lift.py): reading the source IR's function headers and parameters.

A lifter's output is a function with the source function's signature whose
body is the per-thread meaning of the machine code; alive2's GPU semantics
(llvm_util/gpu.h) gives both the same canonical form, whichever GPU target
they are for. What is target-specific (where arguments are, what each
instruction means) stays in each lifter.
"""

import re


class LiftError(Exception):
  """Code a lifter does not lift (what it is, in the message)."""


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


class Param:
  """A parameter of an IR function definition: its text, type and name, and
  the attributes that say how it is passed."""

  def __init__(self, text):
    self.text = text
    m = re.match(r'(ptr(?: addrspace\(\d+\))?|<[^>]*>|\[[^\]]*\]|\{[^}]*\}|'
                 r'%[\w.$"]+|\S+)', text)
    self.type = m.group(1)
    self.name = text.split()[-1]
    if not self.name.startswith('%'):
      raise LiftError('parameter without a name: ' + text)
    b = re.search(r'byref\(([^)]*)\)', text)
    self.byref = b.group(1) if b else None
    self.byval = re.search(r'\bbyval\(', text) is not None
    self.is_ptr = self.type.startswith('ptr')


def function_header(ll, name):
  """(the text between define and @name, the parameters, the text between
  the parameters and the body) of @name's definition in ll (headers on one
  line: one_line_headers)."""
  m = re.search(r'^define\s+(.*?)@"?' + re.escape(name) + r'"?\(', ll, re.M)
  if not m:
    raise LiftError('no definition of @' + name)
  i, depth = m.end(), 1
  while depth:
    depth += {'(': 1, ')': -1}.get(ll[i], 0)
    i += 1
  params_text = ll[m.end():i - 1]
  suffix = ll[i:ll.index('{', i)]
  params = ([Param(p) for p in split_top(params_text)]
            if params_text.strip() else [])
  return m.group(1), params, suffix


def without_captures(text):
  """A parameter's text without capture attributes (a lifted function takes
  pointers apart, which they would make poison; dropping them only weakens
  the target's assumptions)."""
  return re.sub(r'\s*\b(captures\([^)]*\)|nocapture)', '', text)

#!/usr/bin/env python3
# Copyright (c) 2018-present The Alive2 Authors.
# Distributed under the MIT license that can be found in the LICENSE file.
"""Random straight-line integer functions for NVPTX: what instruction
selection specializes (narrow types, small vectors, shifts, division by
constants, min/max, bit counting, funnel shifts, byte swaps, selects)."""
import random, sys

SC = ['i8', 'i16', 'i32', 'i64']
VEC = ['<2 x i16>', '<4 x i8>', '<2 x i32>', '<2 x i8>']


def bits(t):
  return int(t.split('i')[-1].rstrip('>'))


def lanes(t):
  return int(t[1:].split(' x ')[0]) if t.startswith('<') else 1


def const(t, rnd):
  b = bits(t)
  v = rnd.choice([0, 1, 2, 3, 7, 8, 15, 16, 31, 255, (1 << (b - 1)) - 1,
                  1 << (b - 1), (1 << b) - 1, rnd.randrange(1 << b)])
  v &= (1 << b) - 1
  if v >> (b - 1):
    v -= 1 << b
  if t.startswith('<'):
    return '<' + ', '.join('%s %d' % (t.split(' x ')[1].rstrip('>'), v)
                           for _ in range(lanes(t))) + '>'
  return str(v)


def gen_func(name, rnd):
  ty = rnd.choice(SC * 3 + VEC)
  et = ty.split(' x ')[1].rstrip('>') if ty.startswith('<') else ty
  n = rnd.randint(2, 3)
  args = ['%%a%d' % i for i in range(n)]
  vals = list(args)
  body = []
  cnt = [0]

  def new():
    cnt[0] += 1
    return '%%v%d' % cnt[0]

  def pick():
    return rnd.choice(vals) if rnd.random() < 0.85 else const(ty, rnd)

  b = bits(ty)
  for _ in range(rnd.randint(3, 8)):
    v = new()
    k = rnd.random()
    if k < 0.30:
      op = rnd.choice(['add', 'sub', 'mul', 'and', 'or', 'xor'])
      flags = rnd.choice(['', '', 'nsw ', 'nuw '] if op in ('add', 'sub', 'mul') else [''])
      body.append('%s = %s %s%s %s, %s' % (v, op, flags, ty, pick(), pick()))
    elif k < 0.42:
      op = rnd.choice(['shl', 'lshr', 'ashr'])
      amt = rnd.randrange(b)
      amt = '<' + ', '.join('%s %d' % (et, amt) for _ in range(lanes(ty))) + '>' \
          if ty.startswith('<') else str(amt)
      if rnd.random() < 0.4:
        m = new()
        mask = b - 1
        mk = '<' + ', '.join('%s %d' % (et, mask) for _ in range(lanes(ty))) + '>' \
            if ty.startswith('<') else str(mask)
        body.append('%s = and %s %s, %s' % (m, ty, pick(), mk))
        amt = m
      body.append('%s = %s %s %s, %s' % (v, op, ty, pick(), amt))
    elif k < 0.50:
      op = rnd.choice(['udiv', 'sdiv', 'urem', 'srem'])
      d = rnd.choice([3, 5, 7, 10, 12, 16, 100, 255])
      d = d % (1 << (b - 1)) or 3
      dk = '<' + ', '.join('%s %d' % (et, d) for _ in range(lanes(ty))) + '>' \
          if ty.startswith('<') else str(d)
      body.append('%s = %s %s %s, %s' % (v, op, ty, pick(), dk))
    elif k < 0.62:
      c = new()
      pred = rnd.choice(['eq', 'ne', 'slt', 'sle', 'sgt', 'sge', 'ult', 'ule',
                         'ugt', 'uge'])
      body.append('%s = icmp %s %s %s, %s' % (c, pred, ty, pick(), pick()))
      ct = 'i1' if not ty.startswith('<') else '<%d x i1>' % lanes(ty)
      body.append('%s = select %s %s, %s %s, %s %s' % (v, ct, c, ty, pick(), ty, pick()))
    elif k < 0.75:
      f = rnd.choice(['smin', 'smax', 'umin', 'umax'])
      sfx = ('v%di%d' % (lanes(ty), bits(ty))) if ty.startswith('<') else et
      body.append('%s = call %s @llvm.%s.%s(%s %s, %s %s)' % (v, ty, f, sfx, ty, pick(), ty, pick()))
    elif k < 0.85:
      f = rnd.choice(['ctpop', 'bitreverse', 'abs', 'ctlz', 'cttz', 'bswap'])
      if f == 'bswap' and bits(ty) == 8:
        f = 'ctpop'
      sfx = ('v%di%d' % (lanes(ty), bits(ty))) if ty.startswith('<') else et
      extra = ', i1 false' if f in ('abs', 'ctlz', 'cttz') else ''
      body.append('%s = call %s @llvm.%s.%s(%s %s%s)' % (v, ty, f, sfx, ty, pick(), extra))
    elif k < 0.93:
      f = rnd.choice(['fshl', 'fshr'])
      sfx = ('v%di%d' % (lanes(ty), bits(ty))) if ty.startswith('<') else et
      body.append('%s = call %s @llvm.%s.%s(%s %s, %s %s, %s %s)' % (v, ty, f, sfx, ty, pick(), ty, pick(), ty, pick()))
    else:
      if ty.startswith('<'):
        body.append('%s = add %s %s, %s' % (v, ty, pick(), pick()))
      else:
        # A round trip through another width.
        o = rnd.choice([t for t in SC if t != ty])
        w = new()
        if bits(o) > bits(ty):
          body.append('%s = %s %s %s to %s' % (w, rnd.choice(['zext', 'sext']), ty, pick(), o))
          body.append('%s = trunc %s %s to %s' % (v, o, w, ty))
        else:
          body.append('%s = trunc %s %s to %s' % (w, ty, pick(), o))
          body.append('%s = %s %s %s to %s' % (v, rnd.choice(['zext', 'sext']), o, w, ty))
    vals.append(v)
  head = 'define %s @%s(%s) {' % (ty, name, ', '.join('%s %s' % (ty, a) for a in args))
  return '\n'.join([head] + ['  ' + l for l in body] + ['  ret %s %s' % (ty, vals[-1]), '}'])


def main():
  seed, count = int(sys.argv[1]), int(sys.argv[2])
  rnd = random.Random(seed)
  print('; RUN: llc -mtriple=nvptx64 -mcpu=%s' % ['sm_70', 'sm_80', 'sm_90'][seed % 3])
  print('target triple = "nvptx64-nvidia-cuda"\n')
  for i in range(count):
    print(gen_func('f%d_%d' % (seed, i), rnd))
    print()


main()

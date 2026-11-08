#!/usr/bin/env bash
# agx-tv's tests: kernels whose machine code verifies, and machine code with
# a seeded mistake that alive2 must reject. LLC, OPT and ALIVE_TV as agx-tv.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tv=$here/../agx-tv
fail=0
expect() {  # expect correct|incorrect DESCRIPTION agx-tv-args...
  want=$1; what=$2; shift 2
  out=$("$tv" "$@" 2>&1)
  if [[ $want == correct ]]; then
    grep -q 'Transformation seems to be correct' <<<"$out" &&
      ! grep -qE "doesn't verify|cannot lift|ERROR" <<<"$out" && r=ok || r=FAIL
  else
    grep -q "doesn't verify" <<<"$out" && r=ok || r=FAIL
  fi
  [[ $r == ok ]] || { fail=1; echo "$out" | tail -20; }
  echo "$r: $what"
}
tmp=$(mktemp -d)
expect correct "guarded verifies" --keep "$tmp/guarded" "$here/guarded.ll"
# alive2 checks loops unrolled to a bound.
unroll=(-src-unroll=1 -tgt-unroll=1 -smt-to=300000)
expect correct "loop verifies (one iteration)" --keep "$tmp/loop" \
  "$here/loop.ll" "${unroll[@]}"
mutate() {  # mutate NAME KERNEL SED [alive-tv options]
  local name=$1 kernel=$2 edit=$3; shift 3
  sed "$edit" "$tmp/$kernel/out.s" > "$tmp/$name.s"
  if cmp -s "$tmp/$name.s" "$tmp/$kernel/out.s"; then
    echo "FAIL: $name: the mutation did not apply"; fail=1; return
  fi
  expect incorrect "$name is rejected" --asm "$tmp/$name.s" "$here/$kernel.ll" "$@"
}
mutate shift guarded 's/lsl 1/lsl 2/'
mutate condition guarded 's/if_icmp r0l, slt/if_icmp r0l, sgt/'
mutate immediate guarded 's/xor\t\(r[0-9]*\), \(r[0-9]*\), 255/xor\t\1, \2, 254/'
mutate buffer guarded 's/u0_u1, r1, signed/u2_u3, r1, signed/'
mutate wait guarded '/^\twait 0$/d'
mutate loop-body loop 's/iadd[ \t]*\(r[0-9]*\), \1, \(r[0-9]*\)$/isub \1, \1, \2/' "${unroll[@]}"
rm -rf "$tmp"
exit $fail

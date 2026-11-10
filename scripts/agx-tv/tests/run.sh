#!/usr/bin/env bash
# agx-tv's tests: kernels whose machine code verifies, and machine code with
# a seeded mistake that alive2 must reject. LLC, OPT and ALIVE_TV as agx-tv;
# JOBS checks run at once (default: half the processors); each query may take
# 120 s unless a case says otherwise. EXACT=1 adds a check without the buffer
# alignment contract (agx-tv --exact-alignment: minutes).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tv=$here/../agx-tv
jobs=${JOBS:-$(( ($(nproc) + 1) / 2 ))}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
n=0

check() {  # check correct|incorrect DESCRIPTION agx-tv-args...
  local want=$1 what=$2 out r start=$SECONDS
  shift 2
  [[ " $* " == *" -smt-to="* ]] || set -- "$@" -smt-to=120000
  out=$("$tv" "$@" 2>&1)
  if [[ $want == correct ]]; then
    grep -q 'Transformation seems to be correct' <<<"$out" &&
      ! grep -qE "doesn't verify|cannot lift|ERROR" <<<"$out" && r=ok || r=FAIL
  else
    grep -q "doesn't verify" <<<"$out" && r=ok || r=FAIL
  fi
  [[ $r == ok ]] || echo "$out" | tail -20
  echo "$r: $what ($((SECONDS - start)) s)"
}
expect() {  # in the background, at most $jobs at a time
  while (( $(jobs -rp | wc -l) >= jobs )); do wait -n; done
  n=$((n + 1))
  check "$@" > "$tmp/result.$n" &
}

expect correct "guarded verifies" --keep "$tmp/guarded" "$here/guarded.ll"
if [[ -n ${EXACT:-} ]]; then
  expect correct "guarded verifies (exact alignment)" --exact-alignment \
    "$here/guarded.ll" -smt-to=900000
fi
expect correct "scalar64 verifies" "$here/scalar64.ll"
expect correct "threadgroup verifies" --keep "$tmp/threadgroup" \
  "$here/threadgroup.ll"
# alive2 checks loops unrolled to a bound.
unroll=(-src-unroll=1 -tgt-unroll=1 -smt-to=300000)
expect correct "loop verifies (one iteration)" --keep "$tmp/loop" \
  "$here/loop.ll" "${unroll[@]}"
wait  # the mutations edit those kernels' code

mutate() {  # mutate NAME KERNEL SED [alive-tv options]
  local name=$1 kernel=$2 edit=$3; shift 3
  sed "$edit" "$tmp/$kernel/out.s" > "$tmp/$name.s"
  if cmp -s "$tmp/$name.s" "$tmp/$kernel/out.s"; then
    n=$((n + 1))
    echo "FAIL: $name: the mutation did not apply" > "$tmp/result.$n"
    return
  fi
  expect incorrect "$name is rejected" --asm "$tmp/$name.s" "$here/$kernel.ll" "$@"
}
mutate shift guarded 's/lsl 1/lsl 2/'
mutate condition guarded 's/if_icmp r0l, slt/if_icmp r0l, sgt/'
mutate immediate guarded 's/xor\t\(r[0-9]*\), \(r[0-9]*\), 255/xor\t\1, \2, 254/'
mutate buffer guarded 's/u0_u1, r1, signed/u2_u3, r1, signed/'
mutate wait guarded '/^\twait 0$/d'
mutate tg-index threadgroup 's/threadgroup_store i32, x, \(r[0-9]*\), \(r[0-9]*l\), 0/threadgroup_store i32, x, \1, \2, 1/'
mutate loop-body loop 's/iadd[ \t]*\(r[0-9]*\), \1, \(r[0-9]*\)$/isub \1, \1, \2/' "${unroll[@]}"
wait

fail=0
for i in $(seq 1 $n); do
  cat "$tmp/result.$i"
  grep -q '^FAIL' "$tmp/result.$i" && fail=1
done
exit $fail

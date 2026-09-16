#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
: "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to the Android NDK directory}"
compiler="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang++"
test -x "$compiler"
mkdir -p "$repo/probe/artifacts/candidates/stale-recovery"
exec "$compiler" -std=c++17 -O2 -fPIC -fvisibility=hidden -shared \
  -Wl,-z,max-page-size=16384 -DCR_EXPERIMENTAL_PRODUCER_ORIGINS=0 \
  -o "$repo/probe/artifacts/candidates/stale-recovery/libscid_sdk.so" \
  "$repo/probe/nulls_probe.cpp" "$repo/probe/card_selection_arm64.S" \
  "$repo/probe/spawn_relations_arm64.S" "$repo/probe/attack_start_arm64.S" \
  "$repo/probe/heal_events_arm64.S" -llog -ldl

#!/usr/bin/env bash
# Build the wasm (with embedded data) and assemble the 2-file jstris deploy bundle in dist/pcbot/.
# If `wasm-opt` (binaryen) is on PATH it is applied (-O3) to shrink the binary further; otherwise
# the raw cargo output ships as-is (functionally identical, ~6% larger).
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown
wasm=target/wasm32-unknown-unknown/release/pcbot_wasm.wasm

out=dist/pcbot
mkdir -p "$out"

if command -v wasm-opt >/dev/null 2>&1; then
  # The deployed wasm uses bulk-memory (memcpy/memmove); the other post-1.70 default features are
  # enabled too so wasm-opt validates the input on every toolchain.
  wasm-opt -O3 \
    --enable-bulk-memory --enable-mutable-globals \
    --enable-nontrapping-float-to-int --enable-sign-ext \
    "$wasm" -o "$out/zxcl_tbp_bg.wasm"
  echo "wasm-opt applied: $(wc -c < "$wasm") -> $(wc -c < "$out/zxcl_tbp_bg.wasm") bytes"
else
  cp "$wasm" "$out/zxcl_tbp_bg.wasm"
  echo "wasm-opt not found; shipping raw cargo output ($(wc -c < "$wasm") bytes)"
fi
cp js/pc_main.js "$out/zxcl_main.js"

# jstris-free browser test harness (same worker + wasm, served from webtest/)
cp "$out/zxcl_main.js" "$out/zxcl_tbp_bg.wasm" webtest/

ls -la "$out"

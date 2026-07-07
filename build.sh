#!/usr/bin/env bash
# Build the wasm (with embedded data) and assemble the 2-file jstris deploy bundle in dist/pcbot/.
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown
wasm=target/wasm32-unknown-unknown/release/pcbot_wasm.wasm

# 2-file jstris deploy bundle
mkdir -p dist/pcbot
cp js/pc_main.js dist/pcbot/zxcl_main.js
cp "$wasm" dist/pcbot/zxcl_tbp_bg.wasm

# jstris-free browser test harness (same worker + wasm, served from webtest/)
cp js/pc_main.js webtest/zxcl_main.js
cp "$wasm" webtest/zxcl_tbp_bg.wasm

ls -la dist/pcbot/

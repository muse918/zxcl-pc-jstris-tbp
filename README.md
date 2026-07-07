# pcbot_wasm — jstris pure-PC (perpetual perfect clear) bot "zxcl"

A WebAssembly TBP worker bot for jstris that plays the perpetual perfect-clear loop. The engine is
a graph-free exact search (bit-parallel SRS+Jstris180 move generation, an 11-projection PC-ability
filter, a boundary value table, exact 840-hidden expectimax) wrapped in the jstris TBP worker
contract (the Tetris Bot Protocol — see github.com/tetris-bot-protocol/tbp-spec).

Deployed as **two files** with no external data (the PC-ability filter and value table are
zlib-compressed and embedded inside the wasm):

```
zxcl_main.js        worker entrypoint (plain JS, no wasm-bindgen)
zxcl_tbp_bg.wasm    engine + embedded data (~0.95 MB; gzips on transport to ~0.82 MB)
```

---

## Deploying on the jstris server

### 1. Serve the two files

Copy `dist/pcbot/` to `public/js/bots/tbp/pcbot/`:

```
public/js/bots/tbp/pcbot/zxcl_main.js
public/js/bots/tbp/pcbot/zxcl_tbp_bg.wasm
```

Serve the `.wasm` with `Content-Type: application/wasm` (and gzip for the ~0.82 MB transfer).
`zxcl_main.js` fetches the wasm relative to itself and calls `tbp_init_embedded()`, which inflates
its own tables — nothing else to host.

### 2. Register the bot (`resources/js/bots/Bots.js`)

Add to `BOT_TYPES`:

```js
{ id: "ZXCL", name: "zxcl", ep: "/js/bots/tbp/pcbot/zxcl_main.js" }
```

### 3. **Required `bot.js` change — feed the bot 7 pieces at the first move**

zxcl needs **7 pieces (hold + 6 previews)** to form a perfect-clear boundary. At the very first
move hold is empty, so it needs the active piece + 6 previews = a 7-long queue. jstris' TBP queue is
`active + g.queue`, and `g.queue` is held at the client's "Next pieces" count — which is often < 6.
When it is, the first move can't be formed and the bot forfeits (`moves: []`).

Fix it at the source, **gated to zxcl only** so no other bot's behavior changes: in `BotPlayer`,
when the active bot is zxcl, top the queue up to 6 (pulling from the deterministic 7-bag randomizer)
at the two places it is read — when building the `start` queue, and after each placement when
computing the `new_piece` list. The randomizer is deterministic, so pulling the next pieces early is
exact; every pulled piece is still delivered to the bot in order (no dup/skip).

Gate on the bot's entrypoint (`this.bot.getEntrypoint()` returns its `ep`, which contains
`pcbot` only for zxcl). In the current minified `bot.js` these two edits are:

```js
// getStringQueue: at the very start of the function body
getStringQueue=function(){
  if(this.bot.getEntrypoint().indexOf("pcbot")>=0) for(;this.g.queue.length<6;)this.g.refillQueue();
  /* ...original body... */
}

// playMove: immediately before the `for(var u=[],f=this.g.queue.length-i; ...)` new-piece loop
if(this.bot.getEntrypoint().indexOf("pcbot")>=0) for(;this.g.queue.length<6;)this.g.refillQueue();
for(var u=[],f=this.g.queue.length-i;f<this.g.queue.length;f++)u.push(...)
```

(Equivalently, gate on `this.bot.botType.id === "ZXCL"`.) With this in place the "Next pieces" UI
setting no longer matters for zxcl, and every other bot is untouched. Without it, only zxcl's first
move of each game forfeits; from the second PC on, hold is occupied and 5 previews suffice.

---

## Behavior

- Plays ONLY the perfect-clear loop. If the board ever leaves it (garbage, any cell above row 3,
  mid-loop resync) the bot forfeits cleanly (`moves: []`).
- Assumes **7-bag** and SRS. `spin` is always `"none"`; hold is expressed implicitly by the piece
  type (the host infers it).
- **See-7 information discipline**: at any decision the bot acts only on hold + 6 previews + one
  reveal per placement already made; pieces the host delivers beyond that window are never read —
  the search averages over them (in-loop fold tables, w4/w2 next-boundary expectation).
- The deal stream is validated against the 7-bag model (every 7-aligned window must be a
  permutation; every reveal must be in the remaining bag). A violation — non-7-bag randomizer or
  a host/stream desync — forfeits with a precise `bag desync` / `not a 7-bag permutation` reason
  in `last_error` instead of misplaying. A genuinely dead reveal sequence (no PC continuation;
  ~1/4000 loops) forfeits as `dead reveal`.
- Move coordinates match **Cold Clear / libtetris exactly** (the reference jstris expects):
  SRS-true-rotation center cell, x from the left, y from the bottom; O canonicalized to north, S/Z/I
  to north/west. Cleared lines are handled (the engine's sunk-row normalization is translated back).
- Suggest is instant EXCEPT the first move of each PC, which runs the full boundary search:
  **~5–15 s in wasm** (single-threaded, f32 value vectors). Use **turn-based (TB)** mode; in PPS mode
  the host logs "Bot delayed". Peak wasm memory ~1.4 GB (desktop-fine; does not shrink).

---

## Build

```
rustup target add wasm32-unknown-unknown
./build.sh                    # -> dist/pcbot/{zxcl_main.js, zxcl_tbp_bg.wasm}
cargo build --release         # native lib + sim + tests
```

`build.sh` compiles the wasm (which `include_bytes!`s the three zlib blobs in `data/`) and copies
the two deploy files into `dist/pcbot/` (and into `webtest/` for the local harness). If
[`wasm-opt`](https://github.com/WebAssembly/binaryen) (binaryen) is on `PATH` it is applied at
`-O3` for a ~2.5% smaller binary; without it the raw cargo output ships and is functionally
identical. rayon (the optional parallel build/solve) is a **native-only** dependency — the wasm
target is single-threaded and never fetches, compiles, or links it.

## Verification

- Native simulator (drives the exact TBP flow with an INDEPENDENT physical-board check of every
  emitted move, hold inference, line clears, 4L+2L PCs):
  `cargo run --release --bin sim -- --proj <proj> --values <values> --pcs N --seed S`
  (add `--projext <proj_ext>` to match the deployed supplementary projection; `--rounds R` for
  jstris-style round restarts; `--corrupt-deal K --expect-giveup` to verify a broken randomizer
  is met with a precise bag-desync forfeit, not a misplay).
- Node smoke on the real wasm: `node js/node_smoke.js dist/pcbot/zxcl_tbp_bg.wasm [pcs] [seed]`.
- Coordinate round-trip unit tests: `cargo test`.
- Local visual harness (no jstris, no network): `cd webtest && python3 serve.py` → open
  `http://localhost:8000/` and click Start to watch the bot play perpetual PCs.

## Layout & data

- `src/lib.rs` — the plain `extern "C"` FFI (`tbp_init_embedded`, `tbp_start/suggest/play/…`).
- `src/bot.rs` — TBP lifecycle, bag/stream tracking, engine-move → TBP coordinates.
- `src/tbpcoord.rs` — Cold-Clear/libtetris-exact coordinate conversion (+ tests).
- `src/engine/` — engine (movegen, value_search, membership filter `proj`, value table `values`,
  `piece`). `src/graph.rs` is a stub: the bot always uses the movegen+ProjFilter edge source.
- `data/proj.zlib` — the 11-projection PC-ability filter (no false negatives); membership floor.
- `data/proj_ext.zlib` — one supplementary 20-bit projection (center columns 4–8) ANDed onto the
  base filter; trims ~17% of in-search false positives for ~5% faster search, at +20 KB.
- `data/values.zlib` — layer0 V* quantized to 12-bit, KEYLESS (values in chain-state-key order;
  keys regenerated at load, so no key table ships). Regenerate the uncompressed form with
  `python3 scripts/pack_values.py <v_pi2_f32.bin> values.bin`, then zlib-compress into `data/`.

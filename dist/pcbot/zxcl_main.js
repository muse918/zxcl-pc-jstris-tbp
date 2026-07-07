// jstris TBP worker entrypoint for the pure-PC bot "zxcl".
// Deploy (2 files): this file (as zxcl_main.js) + zxcl_tbp_bg.wasm (data embedded).
// Put both under public/js/bots/tbp/pcbot/ and register in Bots.BOT_TYPES:
//   { id: "ZXCL", name: "zxcl", ep: "/js/bots/tbp/pcbot/zxcl_main.js" }
//
// Protocol per the Tetris Bot Protocol (tbp-spec). The engine only plays the perpetual
// perfect-clear loop: it needs >= 7 queue pieces at the first move (set the bot
// preview setting to 6+), assumes 7-bag, and forfeits (empty moves) if the board
// ever leaves the PC loop (e.g. garbage).

"use strict";

const PIECES = "IJLOSTZ"; // engine shape indices 0..6
const ORIENT = ["north", "east", "south", "west"];

let wasm = null; // { exports, memory }
let started = false;
const preInitQueue = []; // messages arriving before WASM is ready

function u8(ptr, len) {
    return new Uint8Array(wasm.memory.buffer, ptr, len);
}

async function boot() {
    // Single deployed artifact: the .wasm with the projection filter + value table embedded and
    // zlib-compressed inside it (jstris serves no loose data files). instantiateStreaming both
    // fetches and compiles; tbp_init_embedded inflates the tables in-wasm.
    const base = self.location.href.replace(/[^/]*$/, "");
    const url = base + "zxcl_tbp_bg.wasm";
    let instance;
    try {
        ({ instance } = await WebAssembly.instantiateStreaming(fetch(url), {}));
    } catch (e) {
        // Fallback when the server doesn't send Content-Type: application/wasm.
        const bytes = await (await fetch(url)).arrayBuffer();
        ({ instance } = await WebAssembly.instantiate(bytes, {}));
    }
    wasm = { exports: instance.exports, memory: instance.exports.memory };

    const rc = wasm.exports.tbp_init_embedded();
    if (rc !== 0) throw new Error("tbp_init_embedded failed: " + rc);

    self.postMessage({ type: "info", name: "zxcl", version: "0.1", author: "pcfinder", features: [] });
    for (const m of preInitQueue.splice(0)) handle(m);
}

function lastError() {
    const cap = 256;
    const p = wasm.exports.alloc_bytes(cap) >>> 0;
    const n = wasm.exports.tbp_last_error(p, cap);
    return new TextDecoder().decode(u8(p, n));
}

function handle(m) {
    switch (m.type) {
        case "rules":
            self.postMessage({ type: "ready" });
            break;
        case "start": {
            started = true;
            const hold = m.hold ? PIECES.indexOf(m.hold) : -1;
            const q = (m.queue || []).map(c => PIECES.indexOf(c)).filter(i => i >= 0);
            const qp = wasm.exports.alloc_bytes(Math.max(1, q.length)) >>> 0;
            u8(qp, Math.max(1, q.length)).set(q);
            // board: 40 rows x 10 cells, row 0 = bottom; occupied = anything non-null
            const bp = wasm.exports.alloc_bytes(400) >>> 0;
            const bv = u8(bp, 400);
            bv.fill(0);
            const rows = m.board || [];
            for (let r = 0; r < Math.min(40, rows.length); r++) {
                const row = rows[r] || [];
                for (let c = 0; c < 10; c++) if (row[c]) bv[r * 10 + c] = 1;
            }
            wasm.exports.tbp_start(hold, qp, q.length, bp, m.combo | 0);
            break;
        }
        case "stop":
            started = false;
            wasm.exports.tbp_stop();
            break;
        case "suggest": {
            const op = wasm.exports.alloc_bytes(16) >>> 0;
            const rc = wasm.exports.tbp_suggest(op);
            if (rc !== 1) {
                console.warn("[zxcl] no move (" + rc + "): " + lastError());
                self.postMessage({ type: "suggestion", moves: [] });
                break;
            }
            const o = new Int32Array(wasm.memory.buffer, op, 4);
            self.postMessage({
                type: "suggestion",
                moves: [{
                    location: {
                        type: PIECES[o[0]],
                        orientation: ORIENT[o[1]],
                        x: o[2],
                        y: o[3],
                    },
                    spin: "none",
                }],
            });
            break;
        }
        case "play":
            wasm.exports.tbp_play();
            break;
        case "new_piece": {
            const p = PIECES.indexOf(m.piece);
            if (p >= 0) wasm.exports.tbp_new_piece(p);
            break;
        }
        case "quit":
            self.close && self.close();
            break;
    }
}

self.onmessage = (e) => {
    if (!wasm) { preInitQueue.push(e.data); return; }
    handle(e.data);
};

boot().catch(err => {
    console.error("[zxcl] boot failed:", err);
});

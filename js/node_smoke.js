// Node smoke test: instantiate the real wasm, drive the TBP flow like the jstris host
// (start -> suggest -> play -> new_piece), with a JS-side physical board to verify moves.
// usage: node node_smoke.js <pcbot_wasm.wasm> <pcfield_projection_11.bin> <values.bin> [pcs] [seed]

"use strict";
const fs = require("fs");

const PIECES = "IJLOSTZ";
// bounding-box shapes (bit = dr*10+dc), same tables as the engine
const SHAPES = [
    [0b1111n, 0b1000000000100000000010000000001n, 0b1111n, 0b1000000000100000000010000000001n],
    [0b10000000111n, 0b1100000000010000000001n, 0b1110000000100n, 0b1000000000100000000011n],
    [0b1000000000111n, 0b100000000010000000011n, 0b1110000000001n, 0b1100000000100000000010n],
    [0b110000000011n, 0b110000000011n, 0b110000000011n, 0b110000000011n],
    [0b1100000000011n, 0b100000000110000000010n, 0b1100000000011n, 0b100000000110000000010n],
    [0b100000000111n, 0b100000000110000000001n, 0b1110000000010n, 0b1000000000110000000010n],
    [0b110000000110n, 0b1000000000110000000001n, 0b110000000110n, 0b1000000000110000000001n],
];
const CENTER = [
    [[1,0],[0,2],[2,0],[0,1]],
    [[1,0],[0,1],[1,1],[1,1]],
    [[1,0],[0,1],[1,1],[1,1]],
    [[0,0],[0,1],[1,1],[1,0]],
    [[1,0],[0,1],[1,1],[1,1]],
    [[1,0],[0,1],[1,1],[1,1]],
    [[1,0],[0,1],[1,1],[1,1]],
];

async function main() {
    const [wasmPath, projPath, valPath] = process.argv.slice(2);
    const pcsTarget = parseInt(process.argv[5] || "2");
    let rng = BigInt(parseInt(process.argv[6] || "42"));

    const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasmPath), {});
    const ex = instance.exports;
    const mem = () => new Uint8Array(ex.memory.buffer);

    const put = (bytes) => {
        const p = ex.alloc_bytes(bytes.length) >>> 0;
        mem().set(bytes, p);
        return p;
    };
    let rc;
    if (projPath && valPath) {
        const proj = fs.readFileSync(projPath), val = fs.readFileSync(valPath);
        rc = ex.tbp_init(put(proj), proj.length, put(val), val.length);
        console.log("tbp_init (external) ->", rc);
    } else {
        rc = ex.tbp_init_embedded();
        console.log("tbp_init_embedded ->", rc);
    }
    if (rc !== 0) throw new Error("init " + rc);

    // 7-bag
    let bagLeft = [];
    const next = () => {
        if (!bagLeft.length) bagLeft = [0,1,2,3,4,5,6];
        rng ^= rng >> 12n; rng ^= (rng << 25n) & 0xFFFFFFFFFFFFFFFFn; rng ^= rng >> 27n;
        const i = Number((rng * 0x2545F4914F6CDD1Dn & 0xFFFFFFFFFFFFFFFFn) >> 33n) % bagLeft.length;
        return bagLeft.splice(i, 1)[0];
    };
    const previews = 7;
    let queue = Array.from({ length: previews }, next);
    let hold = null;
    let board = 0n; // 40 bits, bit = r*10+c (screen)

    // start
    {
        const qp = put(new Uint8Array(queue));
        const bp = ex.alloc_bytes(400) >>> 0;
        mem().fill(0, bp, bp + 400);
        ex.tbp_start(-1, qp, queue.length, bp, 0);
    }

    let pcs = 0, placements = 0;
    const t0 = Date.now();
    while (pcs < pcsTarget) {
        const op = ex.alloc_bytes(16) >>> 0;
        const src = ex.tbp_suggest(op);
        if (src !== 1) {
            const ep = ex.alloc_bytes(256) >>> 0;
            const n = ex.tbp_last_error(ep, 256);
            throw new Error("no move rc=" + src + ": " + Buffer.from(ex.memory.buffer, ep, n).toString());
        }
        const o = new Int32Array(ex.memory.buffer, op, 4);
        const [piece, orient, x, y] = o;

        // host-side hold inference + physical placement validation
        const active = queue[0];
        if (piece === active) queue.shift();
        else if (hold === piece) { hold = queue.shift(); }
        else if (hold === null) { hold = queue.shift(); if (queue[0] !== piece) throw new Error("desync"); queue.shift(); }
        else throw new Error("desync2");

        const [dc, dr] = CENTER[piece][orient];
        const col = BigInt(x - dc), row = BigInt(y - dr);
        let cells = [];
        const shape = SHAPES[piece][orient];
        for (let b = 0n; b < 40n; b++) if (shape >> b & 1n) {
            const cr = row + b / 10n, cc = col + (b % 10n);
            if (cr < 0n || cc < 0n || cc > 9n) throw new Error("cell OOB");
            cells.push(cr * 10n + cc);
        }
        for (const c of cells) {
            if (c < 40n && (board >> c & 1n)) throw new Error("overlap");
        }
        for (const c of cells) if (c < 40n) board |= 1n << c;
        // clear rows
        let nb = 0n, dst = 0n;
        for (let r = 0n; r < 4n; r++) {
            const rowBits = (board >> (r * 10n)) & 1023n;
            if (rowBits !== 1023n) { nb |= rowBits << (dst * 10n); dst++; }
        }
        board = nb;

        ex.tbp_play();
        while (queue.length < previews) { const p = next(); queue.push(p); ex.tbp_new_piece(p); }
        placements++;
        if (board === 0n) { pcs++; console.log(`PC ${pcs} at placement ${placements} (${((Date.now()-t0)/1000).toFixed(1)}s, wasm mem ${(ex.memory.buffer.byteLength/1048576).toFixed(0)}MB)`); }
        if (placements > pcsTarget * 12 + 20) throw new Error("not completing PCs");
    }
    console.log(`OK: ${pcs} PCs, ${placements} placements, ${((Date.now() - t0) / 1000).toFixed(1)}s total`);
}

main().catch(e => { console.error("FAIL:", e); process.exit(1); });

// In-browser TBP host that drives the zxcl worker exactly like jstris' BotPlayer would
// (info->rules->ready->start->suggest->play->new_piece), with a self-generated 7-bag stream and
// a physical board so we can render the bot playing perpetual perfect clears. No jstris, no network.
"use strict";

const PIECES = "IJLOSTZ";
const ORIENT = ["north", "east", "south", "west"];
// bounding-box shapes (bit = dr*10+dc) + SRS-true-rotation center offsets, same tables as the engine.
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
// piece colours (IJLOSTZ)
const COLORS = ["#31c7ef", "#5a65ef", "#ef7921", "#f7d308", "#42b642", "#b048b6", "#ef2029"];

const ROWS_SHOWN = 8, COLS = 10, CELL = 26;

let worker = null;
let running = false;
let awaiting = false;      // waiting for a suggestion
let thinkStart = 0;
let delayMs = 250;

// physical state
let grid = new Int8Array(40).fill(-1); // piece type per cell (index r*10+c, r0 bottom), -1 empty
let hold = null;                        // letter or null
let queue = [];                         // letters, front = active
let bag = [];
let rng = 1n;
let pcs = 0, placements = 0, lastPlaced = [];
const PREVIEWS = 7;
let t0 = 0;

const els = {};
function $(id) { return document.getElementById(id); }

function nextPiece() {
    if (!bag.length) bag = [0,1,2,3,4,5,6];
    rng ^= rng >> 12n; rng ^= (rng << 25n) & 0xFFFFFFFFFFFFFFFFn; rng ^= rng >> 27n;
    const i = Number((rng * 0x2545F4914F6CDD1Dn & 0xFFFFFFFFFFFFFFFFn) >> 33n) % bag.length;
    return PIECES[bag.splice(i, 1)[0]];
}

function boardMsg() {
    // 40 rows x 10, row 0 = bottom, occupied = non-null
    const rows = [];
    for (let r = 0; r < 40; r++) {
        const row = [];
        for (let c = 0; c < 10; c++) row.push(grid[r * 10 + c] >= 0 ? "G" : null);
        rows.push(row);
    }
    return rows;
}

function post(m) { worker.postMessage(m); }

function startGame() {
    grid.fill(-1); hold = null; pcs = 0; placements = 0; lastPlaced = [];
    bag = []; rng = BigInt(parseInt(els.seed.value) || 1) & 0xFFFFFFFFFFFFFFFFn; if (rng === 0n) rng = 1n;
    queue = Array.from({ length: PREVIEWS }, nextPiece);
    t0 = performance.now();
    post({ type: "start", hold: null, queue: queue.slice(), combo: 0, back_to_back: false, board: boardMsg() });
    render();
    requestMove();
}

function requestMove() {
    if (!running) return;
    awaiting = true; thinkStart = performance.now();
    post({ type: "suggest" });
    render();
}

function applyMove(mv) {
    const piece = PIECES.indexOf(mv.location.type);
    const orient = ORIENT.indexOf(mv.location.orientation);
    const x = mv.location.x, y = mv.location.y;

    // host-side hold inference (jstris-identical): placed type != active => hold was used
    const active = queue[0];
    if (mv.location.type === active) queue.shift();
    else if (hold === mv.location.type) hold = queue.shift();
    else if (hold === null) { hold = queue.shift(); if (queue[0] !== mv.location.type) throw new Error("desync"); queue.shift(); }
    else throw new Error("desync2");

    // decode center coords -> absolute cells
    const [dc, dr] = CENTER[piece][orient];
    const col = x - dc, row = y - dr;
    const shape = SHAPES[piece][orient];
    const cells = [];
    for (let b = 0n; b < 40n; b++) if ((shape >> b) & 1n) {
        const cr = row + Number(b / 10n), cc = col + Number(b % 10n);
        if (cr < 0 || cc < 0 || cc > 9) throw new Error("cell OOB");
        cells.push(cr * 10 + cc);
    }
    for (const c of cells) { if (grid[c] >= 0) throw new Error("overlap"); grid[c] = piece; }
    lastPlaced = cells.slice();

    // clear full rows (bottom 4 is where the loop lives, but scan all)
    let dst = 0;
    const ng = new Int8Array(40).fill(-1);
    for (let r = 0; r < 4; r++) {
        let full = true;
        for (let c = 0; c < 10; c++) if (grid[r * 10 + c] < 0) { full = false; break; }
        if (!full) { for (let c = 0; c < 10; c++) ng[dst * 10 + c] = grid[r * 10 + c]; dst++; }
    }
    grid = ng;
    placements++;

    post({ type: "play", move: mv });
    while (queue.length < PREVIEWS) { const p = nextPiece(); queue.push(p); post({ type: "new_piece", piece: p }); }

    let empty = true; for (let i = 0; i < 40; i++) if (grid[i] >= 0) { empty = false; break; }
    if (empty) { pcs++; }
}

function onMessage(e) {
    const m = e.data;
    switch (m.type) {
        case "info":
            els.status.textContent = `worker: ${m.name} v${m.version}`;
            post({ type: "rules" });
            break;
        case "ready":
            if (running) startGame();
            break;
        case "suggestion": {
            awaiting = false;
            if (!m.moves || m.moves.length === 0) {
                running = false;
                els.status.textContent = "bot forfeited (moves: []) — out of PC book";
                els.status.className = "err";
                render();
                return;
            }
            try { applyMove(m.moves[0]); }
            catch (err) { running = false; els.status.textContent = "host error: " + err.message; els.status.className = "err"; render(); return; }
            render();
            if (running) setTimeout(requestMove, delayMs);
            break;
        }
    }
}

function boot() {
    // ST default; the MT page sets self.ZXCL_WORKER = {url:"zxcl_mt.js", type:"module"}.
    const spec = self.ZXCL_WORKER || { url: "zxcl_main.js", type: "classic" };
    worker = spec.type === "module" ? new Worker(spec.url, { type: "module" }) : new Worker(spec.url);
    worker.onmessage = onMessage;
    worker.onerror = (e) => { els.status.textContent = "worker error: " + e.message; els.status.className = "err"; };
}

// ---------- rendering ----------
function drawCell(ctx, cx, cy, color, ghost) {
    ctx.fillStyle = color;
    ctx.fillRect(cx + 1, cy + 1, CELL - 2, CELL - 2);
    if (ghost) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.strokeRect(cx + 2, cy + 2, CELL - 4, CELL - 4); }
}
function render() {
    const ctx = els.canvas.getContext("2d");
    ctx.fillStyle = "#0d0f14"; ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
    // grid lines
    ctx.strokeStyle = "#20242c"; ctx.lineWidth = 1;
    for (let r = 0; r <= ROWS_SHOWN; r++) { ctx.beginPath(); ctx.moveTo(0, r * CELL); ctx.lineTo(COLS * CELL, r * CELL); ctx.stroke(); }
    for (let c = 0; c <= COLS; c++) { ctx.beginPath(); ctx.moveTo(c * CELL, 0); ctx.lineTo(c * CELL, ROWS_SHOWN * CELL); ctx.stroke(); }
    const lp = new Set(lastPlaced);
    for (let r = 0; r < ROWS_SHOWN; r++) for (let c = 0; c < COLS; c++) {
        const idx = r * 10 + c;
        if (grid[idx] >= 0) {
            const cy = (ROWS_SHOWN - 1 - r) * CELL, cx = c * CELL;
            drawCell(ctx, cx, cy, COLORS[grid[idx]], lp.has(idx));
        }
    }
    // stats
    els.pcs.textContent = pcs;
    els.pieces.textContent = placements;
    const secs = t0 ? (performance.now() - t0) / 1000 : 0;
    els.time.textContent = secs.toFixed(1) + "s";
    els.pps.textContent = secs > 0 ? (placements / secs).toFixed(2) : "0";
    // hold + queue
    els.hold.textContent = hold || "-";
    els.hold.style.color = hold ? COLORS[PIECES.indexOf(hold)] : "#666";
    els.queue.innerHTML = "";
    for (const q of queue.slice(0, 6)) {
        const s = document.createElement("span");
        s.textContent = q; s.style.color = COLORS[PIECES.indexOf(q)]; s.className = "qp";
        els.queue.appendChild(s);
    }
    // thinking indicator
    if (awaiting) {
        const el = (performance.now() - thinkStart) / 1000;
        els.think.textContent = `searching boundary… ${el.toFixed(1)}s`;
        els.think.style.visibility = "visible";
    } else {
        els.think.style.visibility = "hidden";
    }
}
function tick() { if (awaiting) render(); requestAnimationFrame(tick); }

window.addEventListener("DOMContentLoaded", () => {
    for (const id of ["status","canvas","pcs","pieces","time","pps","hold","queue","think","seed","speed","btnStart","btnStop"]) els[id] = $(id);
    els.canvas.width = COLS * CELL; els.canvas.height = ROWS_SHOWN * CELL;
    els.speed.addEventListener("input", () => { delayMs = parseInt(els.speed.value); els.speedLabel.textContent = delayMs + "ms"; });
    els.speedLabel = $("speedLabel");
    els.btnStart.addEventListener("click", () => {
        if (running) return;
        running = true; els.status.className = ""; els.status.textContent = "starting…";
        if (worker) { worker.terminate(); }
        boot(); // fresh worker -> emits info -> rules -> ready -> startGame
    });
    els.btnStop.addEventListener("click", () => { running = false; els.status.textContent = "stopped"; });
    boot();
    render();
    requestAnimationFrame(tick);
});

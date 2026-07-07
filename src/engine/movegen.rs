//! Move generator for the 4x10 PC box — a faithful port of tetra-tools/srs-4l. Board bit =
//! row*10 + col (row 0 = bottom); this IS the reference PC-graph field hash directly (no
//! mirroring). Reachability = PiecePlacer: left/right/down/cw/ccw (SRS quarter
//! rotations, NO 180). movegen(shape, board) returns the set of child field hashes; it equals
//! graph.edges ∪ {non-PC-able children} (graph prunes to PC-able, movegen doesn't).

use std::sync::atomic::AtomicUsize;

/// Number of 180 kick offsets to try for JLSTZ (pure + kicks). Jstris = 2 (pure + one (0,0)
/// fallback), VERIFIED exact against ALL 15,185,706 reference PC-graph fields: missing=0, extra_pcable=0.
/// The I-piece is pure-only (kicks over-reach vs graph); see half().
pub static HALF_N: AtomicUsize = AtomicUsize::new(2);

const BOARD_MASK: u64 = 0b1111111111_1111111111_1111111111_1111111111; // 40 bits

// PIECE_SHAPES[shape][orientation] as a bitboard at row0,col0. Order IJLOSTZ / NESW.
static PIECE_SHAPES: [[u128; 4]; 7] = [
    [0b1111, 0b1000000000100000000010000000001, 0b1111, 0b1000000000100000000010000000001], // I
    [0b10000000111, 0b1100000000010000000001, 0b1110000000100, 0b1000000000100000000011],   // J
    [0b1000000000111, 0b100000000010000000011, 0b1110000000001, 0b1100000000100000000010],  // L
    [0b110000000011, 0b110000000011, 0b110000000011, 0b110000000011],                        // O
    [0b1100000000011, 0b100000000110000000010, 0b1100000000011, 0b100000000110000000010],    // S
    [0b100000000111, 0b100000000110000000001, 0b1110000000010, 0b1000000000110000000010],    // T
    [0b110000000110, 0b1000000000110000000001, 0b110000000110, 0b1000000000110000000001],    // Z
];
static PIECE_MAX_COLS: [[i32; 4]; 7] = [
    [6, 9, 6, 9], [7, 8, 7, 8], [7, 8, 7, 8], [8, 8, 8, 8], [7, 8, 7, 8], [7, 8, 7, 8], [7, 8, 7, 8],
];
static JLSTZ_KICKS: [[(i32, i32); 5]; 4] = [
    [(1, -1), (0, -1), (0, 0), (1, -3), (0, -3)],
    [(-1, 0), (0, 0), (0, -1), (-1, 2), (0, 2)],
    [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    [(0, 1), (-1, 1), (-1, 0), (0, 3), (-1, 3)],
];
static I_KICKS: [[(i32, i32); 5]; 4] = [
    [(2, -2), (0, -2), (3, -2), (0, -3), (3, 0)],
    [(-2, 1), (-3, 1), (0, 1), (-3, 3), (0, 0)],
    [(1, -1), (3, -1), (0, -1), (3, 0), (0, -3)],
    [(-1, 2), (0, 2), (-3, 2), (0, 0), (-3, 3)],
];
static O_KICKS: [[(i32, i32); 5]; 4] = [[(0, 0); 5]; 4];
// 180 half-rotation kicks indexed by STARTING orientation. Entry 0 = pure rotation (folded =
// sum of the two quarter-kick first entries). These are the standard 180 kicks folded into
// tetra-tools' convention (folded = unfolded_kick + pure_folded). o0 reproduces tetra-tools'
// commented TETRIO_180_KICKS sketch exactly. Jstris uses a prefix of these (SRS + 2 extra).
static HALF_JLSTZ_KICKS: [[(i32, i32); 6]; 4] = [
    [(0,-1),(0,0),(1,0),(-1,0),(1,-1),(-1,-1)],   // o0->o2, pure (0,-1)
    [(-1,0),(0,0),(0,2),(0,1),(-1,2),(-1,1)],     // o1->o3, pure (-1,0)
    [(0,1),(0,0),(-1,0),(1,0),(-1,1),(1,1)],      // o2->o0, pure (0,1)
    [(1,0),(0,0),(0,2),(0,1),(1,2),(1,1)],        // o3->o1, pure (1,0)
];
static HALF_I_KICKS: [[(i32, i32); 6]; 4] = [
    [(0,-1),(-1,-1),(-2,-1),(1,-1),(2,-1),(0,0)], // o0->o2, pure (0,-1)
    [(-1,0),(-1,1),(-1,2),(-1,-1),(-1,-2),(0,0)], // o1->o3, pure (-1,0)
    [(0,1),(1,1),(2,1),(-1,1),(-2,1),(0,0)],      // o2->o0, pure (0,1)
    [(1,0),(1,1),(1,2),(1,-1),(1,-2),(0,0)],      // o3->o1, pure (1,0)
];
static HALF_O_KICKS: [[(i32, i32); 6]; 4] = [[(0,0); 6]; 4];
#[inline]
fn half_kicks(shape: usize, orient: usize) -> &'static [(i32, i32); 6] {
    match shape { 0 => &HALF_I_KICKS[orient], 3 => &HALF_O_KICKS[orient], _ => &HALF_JLSTZ_KICKS[orient] }
}
#[inline]
fn kicks(shape: usize, orient: usize) -> &'static [(i32, i32); 5] {
    match shape { 0 => &I_KICKS[orient], 3 => &O_KICKS[orient], _ => &JLSTZ_KICKS[orient] }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Piece { shape: usize, col: i32, row: i32, orient: usize }

impl Piece {
    #[inline] fn as_bits(self) -> u128 { PIECE_SHAPES[self.shape][self.orient] << (self.row * 10 + self.col) }
    #[inline] fn collides(self, board: u64) -> bool { (self.as_bits() & board as u128) != 0 }
    #[inline] fn in_bounds(self) -> bool {
        let max_col = PIECE_MAX_COLS[self.shape][self.orient];
        self.col >= 0 && self.col <= max_col && self.row >= 0 && self.row <= 5
    }
    #[inline] fn pack(self) -> u16 {
        ((self.orient as u16) << 12) | ((self.shape as u16) << 8) | ((self.col as u16) << 4) | (self.row as u16)
    }
    fn left(self, board: u64) -> Piece { let mut n = self; n.col -= 1; if n.col < 0 || n.collides(board) { self } else { n } }
    fn right(self, board: u64) -> Piece {
        let mut n = self; n.col += 1;
        if n.col > PIECE_MAX_COLS[self.shape][self.orient] || n.collides(board) { self } else { n }
    }
    fn down(self, board: u64) -> Piece { let mut n = self; n.row -= 1; if n.row < 0 || n.collides(board) { self } else { n } }
    fn cw(self, board: u64) -> Piece {
        let orient = (self.orient + 1) % 4;
        for &(kc, kr) in kicks(self.shape, self.orient) {
            let n = Piece { shape: self.shape, col: self.col + kc, row: self.row + kr, orient };
            if n.in_bounds() && !n.collides(board) { return n; }
        }
        self
    }
    fn ccw(self, board: u64) -> Piece {
        let orient = (self.orient + 3) % 4;
        for &(kc, kr) in kicks(self.shape, orient) {
            let n = Piece { shape: self.shape, col: self.col - kc, row: self.row - kr, orient };
            if n.in_bounds() && !n.collides(board) { return n; }
        }
        self
    }
    // 180 half-rotation, deterministic first-fit. Offset list per starting orientation = pure
    // rotation (folded, = composition of the two quarter-kick first entries) followed by kicks,
    // ordered by norm. First offset that fits wins.
    fn half(self, board: u64) -> Piece {
        let orient = (self.orient + 2) % 4;
        // I-piece 180 is pure-only (kicks over-reach vs graph); JLSTZ use pure + kicks.
        let nkick = if self.shape == 0 { 1 } else { HALF_N.load(std::sync::atomic::Ordering::Relaxed) };
        for &(kc, kr) in half_kicks(self.shape, self.orient).iter().take(nkick) {
            let n = Piece { shape: self.shape, col: self.col + kc, row: self.row + kr, orient };
            if n.in_bounds() && !n.collides(board) { return n; }
        }
        self
    }
    fn can_place(self, board: u64) -> bool {
        let bits = self.as_bits();
        (bits & BOARD_MASK as u128) != 0 && (bits >> 40) == 0 && self.down(board) == self
    }
    /// Place into the board and sink full lines to the bottom -> resulting field hash.
    fn place(self, board: u64) -> u64 {
        let mut unordered = board | (self.as_bits() as u64);
        let (mut ordered, mut complete, mut shift) = (0u64, 0u64, 0u32);
        for _ in 0..4 {
            let line = (unordered >> 30) & 0b1111111111;
            unordered <<= 10;
            if line == 0b1111111111 { complete <<= 10; complete |= line; shift += 10; }
            else { ordered <<= 10; ordered |= line; }
        }
        ordered <<= shift; ordered |= complete;
        ordered
    }
}

/// Column-mirror a 40-bit board (bit row*10+c <-> row*10+(9-c)). The reference PC-graph field hash
/// is the column-mirror of tetra-tools' native Board, so we mirror on the way in and out.
#[inline]
pub fn mirror(b: u64) -> u64 {
    let mut m = 0u64;
    for r in 0..4u64 {
        let line = (b >> (r * 10)) & 0b1111111111;
        let mut rev = 0u64;
        for c in 0..10u64 { if (line >> c) & 1 != 0 { rev |= 1 << (9 - c); } }
        m |= rev << (r * 10);
    }
    m
}

/// Sink full rows of `filled` (board|piece, native, ≤40 bits) to the bottom -> field hash.
#[inline]
fn sink_lines(filled: u64) -> u64 {
    let mut unordered = filled;
    let (mut ordered, mut complete, mut shift) = (0u64, 0u64, 0u32);
    for _ in 0..4 {
        let line = (unordered >> 30) & 0b1111111111;
        unordered <<= 10;
        if line == 0b1111111111 { complete <<= 10; complete |= line; shift += 10; }
        else { ordered <<= 10; ordered |= line; }
    }
    ordered <<= shift; ordered |= complete;
    ordered
}

/// Bit-parallel move generation. Positions (col,row∈0..6) map to bit row*10+col of ONE u64 mask
/// per orientation; reachability is a monotone fixpoint of shift-AND steps in which ALL positions
/// advance simultaneously (replacing the per-state BFS). First-fit kick order is preserved per
/// SOURCE position by masking out sources already served by an earlier kick. Board-independent
/// data (cell offsets, bounds masks, ordered kick shifts) is precomputed per shape.
struct BpOrient {
    cells: [u32; 4],  // the 4 cell offsets (dr*10+dc) of PIECE_SHAPES at position 0
    shape_bits: u64,  // PIECE_SHAPES as u64 (max bit 33)
    inbounds: u64,    // positions: rows 0..=5, cols 0..=max_col
    placeable: u64,   // positions where ALL cells land in rows 0..=3 (can_place geometry)
}
struct BpEdge {
    tgt: usize,
    kicks: Vec<(i32, u64)>, // ordered (bit shift = dr*10+dc, source col-guard mask)
}
struct BpShape {
    orients: [BpOrient; 4],
    edges: [[BpEdge; 3]; 4], // per source orient: cw, ccw, half
    spawn: [u64; 4],         // row-4 in-bounds positions
}

/// Position mask of columns c with 0 <= c+dc <= 9 (guards horizontal shifts against row wrap).
fn colguard(dc: i32) -> u64 {
    let (lo, hi) = ((-dc).max(0), (9 - dc).min(9));
    let mut m = 0u64;
    for r in 0..6u32 {
        for c in lo.max(0)..=hi.max(-1) {
            m |= 1u64 << (r * 10 + c as u32);
        }
    }
    m
}

fn build_bp(shape: usize) -> BpShape {
    let nkick_half = if shape == 0 { 1 } else { HALF_N.load(std::sync::atomic::Ordering::Relaxed) };
    let mk_edge = |tgt: usize, list: Vec<(i32, i32)>| -> BpEdge {
        BpEdge { tgt, kicks: list.into_iter().map(|(dc, dr)| (dr * 10 + dc, colguard(dc))).collect() }
    };
    let orients: [BpOrient; 4] = std::array::from_fn(|o| {
        let bits = PIECE_SHAPES[shape][o] as u64;
        let mut cells = [0u32; 4];
        let (mut b, mut n) = (bits, 0);
        while b != 0 { cells[n] = b.trailing_zeros(); b &= b - 1; n += 1; }
        debug_assert_eq!(n, 4);
        let drmax = cells.iter().map(|c| c / 10).max().unwrap();
        let max_col = PIECE_MAX_COLS[shape][o] as u32;
        let (mut inb, mut plc) = (0u64, 0u64);
        for r in 0..6u32 {
            for c in 0..=max_col {
                inb |= 1u64 << (r * 10 + c);
                if r + drmax <= 3 { plc |= 1u64 << (r * 10 + c); }
            }
        }
        BpOrient { cells, shape_bits: bits, inbounds: inb, placeable: plc }
    });
    let edges: [[BpEdge; 3]; 4] = std::array::from_fn(|o| {
        // cw: kicks indexed by SOURCE orient, offsets (+kc,+kr) — matches Piece::cw.
        let cw = mk_edge((o + 1) % 4, kicks(shape, o).iter().map(|&(kc, kr)| (kc, kr)).collect());
        // ccw: kicks indexed by TARGET orient, offsets NEGATED — matches Piece::ccw.
        let tgt = (o + 3) % 4;
        let ccw = mk_edge(tgt, kicks(shape, tgt).iter().map(|&(kc, kr)| (-kc, -kr)).collect());
        // half: first nkick offsets of the source-indexed table — matches Piece::half.
        let half = mk_edge((o + 2) % 4, half_kicks(shape, o).iter().take(nkick_half).map(|&(kc, kr)| (kc, kr)).collect());
        [cw, ccw, half]
    });
    let spawn: [u64; 4] = std::array::from_fn(|o| {
        let mut m = 0u64;
        for c in 0..=(PIECE_MAX_COLS[shape][o] as u32) { m |= 1u64 << (4 * 10 + c); }
        m
    });
    BpShape { orients, edges, spawn }
}

/// Fast, allocation-free move generator for the hot search loop (bit-parallel fixpoint; see
/// BpShape). Emits sorted-deduped child hashes; result is IDENTICAL to `movegen()`
/// (verify-fastmg checks exact set equality against the reference BFS).
pub struct MoveGen {
    rev: Vec<u16>, // rev[x] = bit-reverse of the low 10 bits of x
    bp: Vec<Option<BpShape>>, // per-shape, built on first use
    not_col0: u64,
    not_col9: u64,
}

impl Default for MoveGen {
    fn default() -> Self { Self::new() }
}

impl MoveGen {
    pub fn new() -> Self {
        let mut rev = vec![0u16; 1024];
        for x in 0..1024u16 {
            let mut r = 0u16;
            for c in 0..10 { if (x >> c) & 1 != 0 { r |= 1 << (9 - c); } }
            rev[x as usize] = r;
        }
        let mut col0 = 0u64; let mut col9 = 0u64;
        for r in 0..6u32 { col0 |= 1u64 << (r * 10); col9 |= 1u64 << (r * 10 + 9); }
        MoveGen { rev, bp: (0..7).map(|_| None).collect(), not_col0: !col0, not_col9: !col9 }
    }

    #[inline]
    fn mirror(&self, b: u64) -> u64 {
        let r = &self.rev;
        (r[(b & 1023) as usize] as u64)
            | ((r[((b >> 10) & 1023) as usize] as u64) << 10)
            | ((r[((b >> 20) & 1023) as usize] as u64) << 20)
            | ((r[((b >> 30) & 1023) as usize] as u64) << 30)
    }

    /// Column-mirror a graph-convention hash (public helper for symmetry probes).
    #[inline]
    pub fn mirror_hash(&self, b: u64) -> u64 { self.mirror(b) }

    /// Child field hashes (graph convention) reachable by placing `shape` on `board_hash`.
    /// Writes into `out` (cleared first); result is sorted and deduped.
    pub fn children(&mut self, shape: usize, board_hash: u64, out: &mut Vec<u64>) {
        out.clear();
        let board = self.mirror(board_hash);
        if self.bp[shape].is_none() { self.bp[shape] = Some(build_bp(shape)); }
        let g = self.bp[shape].as_ref().unwrap();

        // fits[o] = in-bounds positions where none of the 4 cells hits the board. `free` as u128
        // keeps cells above the 40-bit box (offsets up to +33 past bit 59) reading as free.
        let free = !(board as u128);
        let fits: [u64; 4] = std::array::from_fn(|o| {
            let oi = &g.orients[o];
            oi.inbounds
                & ((free >> oi.cells[0]) as u64)
                & ((free >> oi.cells[1]) as u64)
                & ((free >> oi.cells[2]) as u64)
                & ((free >> oi.cells[3]) as u64)
        });
        // Spawn rows (4..) are above the box, so spawn ⊆ fits always.
        let mut reach: [u64; 4] = std::array::from_fn(|o| g.spawn[o] & fits[o]);
        let (nc0, nc9) = (self.not_col0, self.not_col9);
        loop {
            let mut changed = false;
            // left/right/down flood to closure within each orientation
            for o in 0..4 {
                let f = fits[o];
                loop {
                    let r = reach[o];
                    let n = r | ((((r & nc0) >> 1) | ((r & nc9) << 1) | (r >> 10)) & f);
                    if n == r { break; }
                    reach[o] = n;
                    changed = true;
                }
            }
            // rotations: first-fit kick per SOURCE position (earlier kick removes the source)
            for o in 0..4 {
                for e in &g.edges[o] {
                    let mut rem = reach[o];
                    for &(s, guard) in &e.kicks {
                        if rem == 0 { break; }
                        let src = rem & guard;
                        let cand = (if s >= 0 { src << s } else { src >> -s }) & fits[e.tgt];
                        let back = if s >= 0 { cand >> s } else { cand << -s };
                        rem &= !back;
                        let new = cand & !reach[e.tgt];
                        if new != 0 { reach[e.tgt] |= new; changed = true; }
                    }
                }
            }
            if !changed { break; }
        }
        // placements: grounded (below-position doesn't fit) AND all cells inside the box
        for o in 0..4 {
            let oi = &g.orients[o];
            let mut m = reach[o] & !(fits[o] << 10) & oi.placeable;
            while m != 0 {
                let p = m.trailing_zeros();
                m &= m - 1;
                out.push(sink_lines(board | (oi.shape_bits << p)));
            }
        }
        for v in out.iter_mut() { *v = self.mirror(*v); }
        out.sort_unstable();
        out.dedup();
    }
}

/// One reachable grounded placement: the resulting child field hash (graph convention) plus the
/// physical state (orientation 0..3 = N/E/S/W, bounding-box anchor col/row in SCREEN coords:
/// col 0 = leftmost, row 0 = bottom). For emitting TBP moves.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub child_hash: u64,
    pub orient: u8,
    pub col: i8,
    pub row: i8,
    /// The 4 occupied cells as (col, row) in SCREEN coords (col 0 = left, row 0 = bottom), in the
    /// ENGINE board (cleared lines still sunk at the bottom). native col == screen col because the
    /// graph hash is the column-mirror of the native board (see mirror()).
    pub cells: [(i8, i8); 4],
}

/// Enumerate ALL reachable grounded placements of `shape` on `board_hash` with their physical
/// states (reference BFS — not the hot path). Same reachability as `movegen`; one entry per
/// reachable STATE, so duplicate-shape orientations (S/Z/I o0==o2, o1==o3) appear multiple times
/// with different `orient` — callers pick a canonical one.
pub fn placements(shape: usize, board_hash: u64) -> Vec<Placement> {
    let board = mirror(board_hash);
    let mut seen = vec![false; 0x4000];
    let mut queue: Vec<Piece> = Vec::new();
    for orient in 0..4usize {
        for col in 0..10i32 {
            let p = Piece { shape, col, row: 4, orient };
            if p.in_bounds() && !seen[p.pack() as usize] {
                seen[p.pack() as usize] = true;
                queue.push(p);
            }
        }
    }
    let mut out = Vec::new();
    while let Some(p) = queue.pop() {
        for np in [p.left(board), p.right(board), p.down(board), p.cw(board), p.ccw(board)] {
            let k = np.pack() as usize;
            if !seen[k] {
                seen[k] = true;
                queue.push(np);
            }
        }
        let np = p.half(board);
        if np != p {
            let k = np.pack() as usize;
            if !seen[k] {
                seen[k] = true;
                queue.push(np);
            }
        }
        if p.can_place(board) {
            // Native col IS the screen col: the graph HASH is the column-mirror of the native
            // board, so mirroring the hash in yields native bit x == screen col x (0 = left).
            let bits = PIECE_SHAPES[shape][p.orient] as u64;
            let mut cells = [(0i8, 0i8); 4];
            let mut n = 0;
            for b in 0..40u32 {
                if bits >> b & 1 != 0 {
                    cells[n] = ((p.col + (b % 10) as i32) as i8, (p.row + (b / 10) as i32) as i8);
                    n += 1;
                }
            }
            out.push(Placement {
                child_hash: mirror(p.place(board)),
                orient: p.orient as u8,
                col: p.col as i8,
                row: p.row as i8,
                cells,
            });
        }
    }
    out
}

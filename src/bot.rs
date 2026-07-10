//! TBP bot core: the graph-free pure-PC engine (movegen + ProjFilter membership + boundary V
//! table + value_search) wrapped in the jstris TBP lifecycle (start / suggest / play /
//! new_piece / stop). Board- and piece-stream tracking, 7-bag phase inference, and
//! engine-move -> physical-placement (TBP center coordinates) conversion live here.
//!
//! Conventions (see the Tetris Bot Protocol spec, tbp-spec 0000-mvp):
//!  - piece letters IJLOSTZ = shape indices 0..6 (crate::piece).
//!  - board: row 0 = bottom, col 0 = leftmost. Engine hash bit = row*10 + (9-col).
//!  - TBP location = SRS-true-rotation CENTER cell, x from left, y from bottom.
//!  - orientation strings north/east/south/west = our movegen orient 0/1/2/3.

use std::cell::RefCell;

use crate::graph::{MAX_HASH, TWO_LINE_HASH};
use crate::movegen::{self, MoveGen};
use crate::piece::{after_reveal, piece_char, Piece, FULL_BAG};
use crate::proj::ProjFilter;
use crate::tbpcoord;
use crate::value_search::{value_search, SearchInput, VsResult};
use crate::values::{ResetEval, ValueTable};

pub const ORIENT_NAMES: [&str; 4] = ["north", "east", "south", "west"];

/// Enumerate every bag-legal completion of the reveal slots [idx, upto) given the fixed prefix
/// `base[0..idx]` and the remaining bag `mask` after that prefix, padding the never-conditioned
/// tail [upto, 4) with a deterministic valid continuation. Used to average a decision over reveals
/// the host hasn't delivered yet.
fn enumerate_reveals(base: [Piece; 4], idx: usize, upto: usize, mask: u8, out: &mut Vec<[Piece; 4]>) {
    if idx >= upto {
        let mut c = base;
        let mut m = mask;
        for slot in c.iter_mut().take(4).skip(upto) {
            let p = (0..7u8).find(|&p| m & (1 << p) != 0).unwrap();
            *slot = p;
            m = after_reveal(m, p);
        }
        out.push(c);
        return;
    }
    for p in 0..7u8 {
        if mask & (1 << p) != 0 {
            let mut b = base;
            b[idx] = p;
            enumerate_reveals(b, idx + 1, upto, after_reveal(mask, p), out);
        }
    }
}

/// A concrete TBP move (already in TBP conventions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TbpMove {
    pub piece: Piece,
    pub orient: u8, // 0..3 = north/east/south/west
    pub x: i8,      // center cell, from left
    pub y: i8,      // center cell, from bottom
}

struct LoopState {
    vs: VsResult,
    path: Vec<usize>,
    /// Deal-count at which this loop's boundary was formed = stream index of h1.
    boundary_dealt: usize,
}

struct Pending {
    edge: usize,
    placed: Piece,
    field_after: u64,
}

pub struct Bot {
    proj: ProjFilter,
    table: ValueTable,
    mg: RefCell<MoveGen>,
    /// All pieces in DEAL order since game start (queue payload order, then new_piece order).
    stream: Vec<Piece>,
    placed_cnt: usize,
    hold: Option<Piece>,
    queue: Vec<Piece>, // front = current active piece
    board_hash: u64,   // engine (graph-convention) hash; 0 = empty
    pub out_of_book: bool,
    loop_state: Option<LoopState>,
    pending: Option<Pending>,
    pub last_error: String,
}

impl Bot {
    pub fn new(proj: ProjFilter, table: ValueTable) -> Self {
        Bot {
            proj,
            table,
            mg: RefCell::new(MoveGen::new()),
            stream: Vec::new(),
            placed_cnt: 0,
            hold: None,
            queue: Vec::new(),
            board_hash: 0,
            out_of_book: false,
            loop_state: None,
            pending: None,
            last_error: String::new(),
        }
    }

    /// `board_cells`: 400 bytes, cell (row r 0=bottom, col c 0=left) at index r*10+c, nonzero =
    /// occupied. Returns false (out-of-book) if any cell above row 3 is filled.
    pub fn start(&mut self, hold: Option<Piece>, queue: &[Piece], board_cells: &[u8], _combo: i32) {
        self.pending = None;
        self.loop_state = None;
        self.out_of_book = false;
        self.last_error.clear();

        // Board -> engine hash (bit = r*10 + (9-c)).
        let mut hash = 0u64;
        let mut overflow = false;
        for (i, &cell) in board_cells.iter().enumerate().take(400) {
            if cell != 0 {
                let (r, c) = (i / 10, i % 10);
                if r >= 4 {
                    overflow = true;
                } else {
                    hash |= 1u64 << (r as u64 * 10 + (9 - c as u64));
                }
            }
        }

        // Fresh-loop detection: an EMPTY board. The host sends `start` at a game start or a board
        // reset (garbage -> non-empty board), never mid-loop, so an empty board is always a fresh
        // loop and resets the bag stream (no previous game leaks in). CRUCIAL: when the host gives a
        // hold at the reset (games can start with a piece already held), it is the deal immediately
        // BEFORE the queue and MUST take stream slot 0 — boundary_dealt (= placed_cnt + 1 + 6)
        // reserves that "+1" slot for the hold. Omitting it shifts every reveal read by one
        // (stream[bd+i]), so the engine places the wrong piece and the host reports a queue desync.
        let fresh = hash == 0 && !overflow;
        if fresh || self.stream.is_empty() {
            self.stream = match hold {
                Some(h) => {
                    let mut s = Vec::with_capacity(queue.len() + 1);
                    s.push(h);
                    s.extend_from_slice(queue);
                    s
                }
                None => queue.to_vec(),
            };
            self.placed_cnt = 0;
        } else {
            // Resync: counters/stream continue; the payload queue must be our stream's tail.
            let dealt = self.placed_cnt + self.hold.iter().count() + queue.len();
            if dealt > self.stream.len() || self.stream[dealt - queue.len()..dealt] != *queue {
                // Host knows something we don't (shouldn't happen without garbage modes):
                // rebuild the stream tail from the payload; bag phase stays count-based.
                self.last_error = "resync queue mismatch; stream tail rebuilt".into();
                let keep = self.stream.len().min(dealt - queue.len());
                self.stream.truncate(keep);
                self.stream.extend_from_slice(queue);
            }
        }
        self.hold = hold;
        self.queue = queue.to_vec();
        self.board_hash = hash;
        if overflow {
            self.out_of_book = true;
            self.last_error = "board has cells above row 3".into();
        }
    }

    pub fn new_piece(&mut self, p: Piece) {
        self.stream.push(p);
        self.queue.push(p);
    }

    pub fn stop(&mut self) {
        self.pending = None;
        self.loop_state = None;
    }

    /// Commit the pending suggestion (host echoes it back via `play`).
    pub fn play(&mut self) {
        let Some(p) = self.pending.take() else { return };
        // Physical queue/hold bookkeeping (host infers hold from the placed piece type).
        if !self.queue.is_empty() && p.placed == self.queue[0] {
            self.queue.remove(0);
        } else {
            match self.hold {
                Some(h) if h == p.placed => {
                    // swap: active goes to hold, old hold is placed
                    let active = self.queue.remove(0);
                    self.hold = Some(active);
                }
                None => {
                    // first hold: active -> hold, next piece placed
                    let active = self.queue.remove(0);
                    self.hold = Some(active);
                    if !self.queue.is_empty() && self.queue[0] == p.placed {
                        self.queue.remove(0);
                    } else {
                        self.out_of_book = true;
                        self.last_error = "play: piece not available (queue desync)".into();
                        return;
                    }
                }
                _ => {
                    self.out_of_book = true;
                    self.last_error = "play: piece matches neither active nor hold".into();
                    return;
                }
            }
        }
        self.placed_cnt += 1;
        self.board_hash = p.field_after;

        if let Some(ls) = self.loop_state.as_mut() {
            ls.path.push(p.edge);
            let depth = ls.path.len(); // depth AFTER this placement
            let done_4l = depth == 10 && p.field_after == MAX_HASH;
            let done_2l = depth == 5 && p.field_after == TWO_LINE_HASH;
            if done_4l || done_2l {
                self.board_hash = 0; // lines clear; physical board is empty
                self.loop_state = None;
            }
        }
    }

    /// Produce the next move, running a fresh boundary search if the board is empty and no loop
    /// is active. None = no move (host should treat as forfeit).
    pub fn suggest(&mut self) -> Option<TbpMove> {
        if self.out_of_book {
            return None;
        }
        if self.loop_state.is_none() {
            if self.board_hash != 0 {
                self.out_of_book = true;
                // Board rows 0..3 (bottom-up), '#'=filled '.'=empty, so a garbage board (host sent a
                // start/board-update the PC bot can't play) is distinguishable from a stray residual.
                let mut rows = String::new();
                for r in 0..4 {
                    if r > 0 { rows.push('/'); }
                    for c in 0..10 {
                        rows.push(if self.board_hash >> (r * 10 + (9 - c)) & 1 != 0 { '#' } else { '.' });
                    }
                }
                self.last_error = format!(
                    "non-empty board without an active loop (host sent a non-empty board, or a PC \
                     completion desynced): board[r0..r3]={} placed={}",
                    rows, self.placed_cnt
                );
                return None;
            }
            if !self.form_boundary() {
                return None;
            }
        }

        // Reveal information available for this decision (DAG depth = path_len):
        //  * The delivered reveals h1..h_known are FIXED. `known` is capped by see-7 (never read
        //    more than `path_len` reveals — don't peek ahead of what `path_len` placements expose,
        //    even if the host deals the stream further) and by 4 (the search consumes only h1..h4).
        //  * The current decision conditions on reveals 0..`upto` (= min(path_len,4)). Any of those
        //    in [known, upto) that HASN'T arrived yet is UNKNOWN, so we AVERAGE analyze's per-edge
        //    score over every bag-legal completion of them — the correct decision under partial
        //    information. When reveals arrive on time (known == upto) this is a single evaluation.
        //  * jstris reveals lazily (e.g. 0 reveals after the 1st placement); that is normal — a
        //    reveal is only physically REQUIRED once it is placed (depth 6+ places h1..h4), so
        //    `needed = max(0, path_len-5)` fewer delivered than that is a genuine underrun/desync.
        let (bd, path_len) = {
            let ls = self.loop_state.as_ref().unwrap();
            (ls.boundary_dealt, ls.path.len())
        };
        let delivered = self.stream.len().saturating_sub(bd);
        let needed = path_len.saturating_sub(5);
        if delivered < needed {
            self.out_of_book = true;
            self.last_error = format!(
                "reveal underrun: {} placements consume {} reveal(s) but only {} delivered",
                path_len, needed, delivered
            );
            return None;
        }
        let known = delivered.min(path_len).min(4);
        let upto = path_len.min(4);

        // Fixed delivered prefix h1..h_known, validated against the running bag.
        let mut real = [0u8; 4];
        let mut m_known = self.loop_mask(bd);
        for i in 0..known {
            let h = self.stream[bd + i];
            if m_known & (1 << h) == 0 {
                // A real reveal the bag model forbids: randomizer not 7-bag, or stream/count desync.
                self.out_of_book = true;
                self.last_error = format!(
                    "bag desync: reveal h{}={} not in remaining bag {:07b} (boundary_dealt={})",
                    i + 1, piece_char(h), m_known, bd
                );
                return None;
            }
            real[i] = h;
            m_known = after_reveal(m_known, h);
        }

        // Enumerate the not-yet-delivered conditioned reveals [known, upto) and average.
        let mut completions: Vec<[Piece; 4]> = Vec::new();
        enumerate_reveals(real, known, upto, m_known, &mut completions);

        let ls = self.loop_state.as_ref().unwrap();
        // edge -> (summed score, count, placed piece, child field); placed/field are reveal-independent.
        let mut agg: hashbrown::HashMap<usize, (f64, u32, Piece, u64)> = hashbrown::HashMap::new();
        let mut any_live = false;
        let mut field_seen = self.board_hash;
        for hid in &completions {
            let node = ls.vs.analyze(&ls.path, *hid);
            field_seen = node.field;
            if node.terminal != 0 || node.cands.is_empty() || node.best_score <= 0.0 {
                continue; // this reveal-completion is dead; a different completion may be live
            }
            any_live = true;
            for c in &node.cands {
                let e = agg.entry(c.edge).or_insert((0.0, 0, c.placed, c.field_after));
                e.0 += c.score;
                e.1 += 1;
            }
        }
        if field_seen != self.board_hash {
            self.out_of_book = true;
            self.last_error = format!(
                "internal desync: engine field {:#012x} != tracked board {:#012x}",
                field_seen, self.board_hash
            );
            return None;
        }
        if !any_live {
            self.out_of_book = true;
            let revealed: String = (0..known).map(|i| piece_char(self.stream[bd + i])).collect();
            self.last_error = format!(
                "dead reveal: no PC continuation after [{}] at depth {} (over {} reveal completion(s))",
                revealed, path_len, completions.len()
            );
            return None;
        }
        // Best move = edge with the highest MEAN score across completions (at depth < 6 every
        // completion shares the edge set; at depth >= 6 the placed reveal is in the fixed prefix,
        // so the edge set and counts match across completions).
        let (best_edge, best_placed, best_field_after) = agg
            .iter()
            .map(|(&e, &(sum, cnt, placed, fa))| (sum / cnt as f64, e, placed, fa))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, e, placed, fa)| (e, placed, fa))
            .unwrap();

        // Engine transition -> physical placement. Any reachable state producing this child field
        // works; the libtetris-exact encoder canonicalizes S/Z/I/O so the orientation matches
        // what Cold Clear (hence jstris) would emit.
        let mut chosen: Option<movegen::Placement> = None;
        for pl in movegen::placements(best_placed as usize, self.board_hash) {
            if pl.child_hash == best_field_after {
                chosen = Some(pl);
                break;
            }
        }
        let Some(pl) = chosen else {
            self.out_of_book = true;
            self.last_error = "no physical placement reproduces the engine transition".into();
            return None;
        };
        // The engine field keeps cleared lines SUNK at the bottom (solver normalization); the
        // physical board has them removed. Shift the placement down by the sunk-row count, then
        // encode in the exact Cold Clear / libtetris coordinate convention.
        let sunk = Self::full_bottom_rows(self.board_hash) as i8;
        let phys: [(i8, i8); 4] = pl.cells.map(|(c, r)| (c, r - sunk));
        let Some((orient, x, y)) = tbpcoord::encode(best_placed, &phys) else {
            self.out_of_book = true;
            self.last_error = "placement cells do not encode to a canonical orientation".into();
            return None;
        };
        self.pending = Some(Pending { edge: best_edge, placed: best_placed, field_after: best_field_after });
        Some(TbpMove { piece: best_placed, orient, x, y })
    }

    /// Number of full rows sunk at the bottom of an engine field (= lines physically cleared so
/// far within the current loop; the solver keeps them, the real board doesn't).
fn full_bottom_rows(h: u64) -> u32 {
    let mut k = 0;
    while k < 4 && (h >> (10 * k)) & 0b1111111111 == 0b1111111111 {
        k += 1;
    }
    k
}

/// Bag mask from which the reveal at deal index `bd` is drawn: `FULL_BAG` minus the pieces of the
    /// current partial bag already dealt. The deal stream is 7-bag aligned (stream[0] is a bag
    /// boundary — set at each fresh start), so the partial bag is exactly the `bd % 7` deals since
    /// the last boundary.
    fn loop_mask(&self, bd: usize) -> u8 {
        let phase = bd % 7;
        if phase == 0 {
            return FULL_BAG;
        }
        let mut used = 0u8;
        for &p in &self.stream[bd - phase..bd] {
            used |= 1 << p;
        }
        FULL_BAG & !used
    }

    /// Build the boundary (engine hold + 6 visible + bag mask) from tracked state and run the
    /// search. Engine hold = physical hold, or the active piece when hold is empty.
    fn form_boundary(&mut self) -> bool {
        let (eh, vis_src): (Piece, &[Piece]) = match self.hold {
            Some(h) => (h, &self.queue[..]),
            None => {
                if self.queue.is_empty() {
                    self.last_error = "empty queue at boundary".into();
                    return false;
                }
                (self.queue[0], &self.queue[1..])
            }
        };
        if vis_src.len() < 6 {
            self.last_error = format!(
                "need 6 previews beyond the engine hold at a PC boundary (have {}); set Next pieces to 6",
                vis_src.len()
            );
            self.out_of_book = true;
            return false;
        }
        let visible: [Piece; 6] = vis_src[..6].try_into().unwrap();
        let boundary_dealt = self.placed_cnt + 1 + 6; // engine hold always occupies one deal slot
        let mask = self.loop_mask(boundary_dealt);

        let term_a = MAX_HASH;
        let term_b = TWO_LINE_HASH;
        let proj = &self.proj;
        let memo = RefCell::new(hashbrown::HashMap::<u64, Vec<u64>>::new());
        let mg = &self.mg;
        let edge = |h: u64, piece: u8, out: &mut Vec<u64>| {
            let key = (h << 3) | piece as u64;
            if let Some(v) = memo.borrow().get(&key) {
                out.extend_from_slice(v);
                return;
            }
            let mut tmp = Vec::new();
            mg.borrow_mut().children(piece as usize, h, &mut tmp);
            tmp.retain(|&c| c == term_a || c == term_b || proj.maybe(c));
            out.extend_from_slice(&tmp);
            memo.borrow_mut().insert(key, tmp);
        };
        let vs = value_search(SearchInput {
            graph: None,
            hold: eh,
            visible,
            mask,
            reset: ResetEval::new(Some(&self.table)),
            edge_ids: Some(&edge),
            par_edge: None,
        });
        if vs.root_value <= 0.0 {
            self.last_error = "boundary unsolvable (root value 0)".into();
            self.out_of_book = true;
            return false;
        }
        self.loop_state = Some(LoopState { vs, path: Vec::new(), boundary_dealt });
        true
    }
}

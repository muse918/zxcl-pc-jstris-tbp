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
use crate::piece::{after_reveal, Piece, FULL_BAG};
use crate::proj::ProjFilter;
use crate::tbpcoord;
use crate::value_search::{value_search, SearchInput, VsResult};
use crate::values::{ResetEval, ValueTable};

pub const ORIENT_NAMES: [&str; 4] = ["north", "east", "south", "west"];

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

        // Fresh-game detection: empty board + empty hold (mid-game this combination requires a
        // just-completed PC before the first hold, which our engine's play makes near-impossible;
        // a fresh worker per bot instance covers the rest).
        let fresh = hash == 0 && hold.is_none() && !overflow;
        if fresh || self.stream.is_empty() {
            self.stream = queue.to_vec();
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
                self.last_error = "non-empty board without an active loop (resync mid-loop?)".into();
                return None;
            }
            if !self.form_boundary() {
                return None;
            }
        }

        // Reveals consumed inside this loop: h1.. = stream[boundary_dealt..]; pad the unknown
        // tail with a valid bag continuation (never read by decision-time scoring).
        let ls = self.loop_state.as_ref().unwrap();
        let bd = ls.boundary_dealt;
        let known = self.stream.len().saturating_sub(bd).min(4);
        let mut hidden = [0u8; 4];
        let mut m = self.loop_mask(bd);
        for i in 0..4 {
            let h = if i < known {
                self.stream[bd + i]
            } else {
                (0..7u8).find(|&p| m & (1 << p) != 0).unwrap()
            };
            hidden[i] = h;
            m = after_reveal(m, h);
        }

        let node = ls.vs.analyze(&ls.path, hidden);
        if node.terminal != 0 || node.cands.is_empty() || node.best_score <= 0.0 {
            self.out_of_book = true;
            self.last_error = format!("dead position (terminal={} cands={})", node.terminal, node.cands.len());
            return None;
        }
        let best = &node.cands[0];
        debug_assert_eq!(node.field, self.board_hash);

        // Engine transition -> physical placement. Any reachable state producing this child field
        // works; the libtetris-exact encoder canonicalizes S/Z/I/O so the orientation matches
        // what Cold Clear (hence jstris) would emit.
        let mut chosen: Option<movegen::Placement> = None;
        for pl in movegen::placements(best.placed as usize, self.board_hash) {
            if pl.child_hash == best.field_after {
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
        let Some((orient, x, y)) = tbpcoord::encode(best.placed, &phys) else {
            self.out_of_book = true;
            self.last_error = "placement cells do not encode to a canonical orientation".into();
            return None;
        };
        self.pending = Some(Pending { edge: best.edge, placed: best.placed, field_after: best.field_after });
        Some(TbpMove { piece: best.placed, orient, x, y })
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

/// Bag mask from which this loop's h1 is drawn: remaining bag after `bd` total deals.
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
                "need 6 previews beyond the engine hold at a PC boundary (have {}); raise the bot preview setting",
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

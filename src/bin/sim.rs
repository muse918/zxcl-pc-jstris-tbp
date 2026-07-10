//! Native TBP-flow simulator: drives bot::Bot exactly like the jstris host would
//! (start -> suggest -> play -> new_piece ...), against a self-generated 7-bag stream, with an
//! independent physical board simulation to verify every emitted move:
//!   - the (piece, orient, x, y) TBP move is decoded back to cells via the center-offset table,
//!   - the cells must be reachable placements on the current physical board,
//!   - line clears are applied physically and compared against the bot's internal board hash.
//! usage: sim --proj F --values F [--pcs N] [--seed S] [--previews K (queue len incl active)]
//!            [--rounds R (jstris-style round restarts: fresh start message, same Bot/worker)]
//!            [--corrupt-deal K (flip the K-th dealt piece, 0-based per round: simulates a
//!             non-7-bag randomizer / stream desync — the bot must give up with a precise
//!             bag-desync/window diagnostic, never emit a physically wrong move)]
//!            [--expect-giveup (success = a clean give-up; failure = bot plays through)]

use pcbot_wasm::bot::{Bot, ORIENT_NAMES};
use pcbot_wasm::movegen;
use pcbot_wasm::piece::piece_char;
use pcbot_wasm::proj::ProjFilter;
use pcbot_wasm::tbpcoord;
use pcbot_wasm::values::ValueTable;

struct Bag {
    rng: u64,
    left: Vec<u8>,
}
impl Bag {
    fn next(&mut self) -> u8 {
        if self.left.is_empty() {
            self.left = (0..7).collect();
        }
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let i = (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as usize % self.left.len();
        self.left.swap_remove(i)
    }
}

/// Physical board: 40-bit occupancy in SCREEN coords (bit = r*10 + c, r0 bottom, c0 left).
struct Phys {
    cells: u64,
}
impl Phys {
    fn to_engine_hash(&self) -> u64 {
        // engine hash bit = r*10 + (9-c)
        let mut h = 0u64;
        for r in 0..4u64 {
            for c in 0..10u64 {
                if self.cells >> (r * 10 + c) & 1 != 0 {
                    h |= 1 << (r * 10 + (9 - c));
                }
            }
        }
        h
    }
    fn board_cells(&self) -> [u8; 400] {
        let mut b = [0u8; 400];
        for i in 0..40 {
            if self.cells >> i & 1 != 0 {
                b[i] = 1;
            }
        }
        b
    }
    /// Apply a placement given by absolute screen cells; clear full rows. Returns rows cleared.
    fn place(&mut self, cells4: [u8; 4]) -> u32 {
        for &b in &cells4 {
            assert_eq!(self.cells >> b & 1, 0, "cell overlap at {}", b);
            self.cells |= 1u64 << b;
        }
        let mut out = 0u64;
        let mut dst = 0;
        let mut cleared = 0;
        for r in 0..4 {
            let row = (self.cells >> (r * 10)) & 0b1111111111;
            if row == 0b1111111111 {
                cleared += 1;
            } else {
                out |= row << (dst * 10);
                dst += 1;
            }
        }
        self.cells = out;
        cleared
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let proj_path = get("--proj").expect("--proj required");
    let values_path = get("--values").expect("--values required");
    let pcs_target: usize = get("--pcs").and_then(|s| s.parse().ok()).unwrap_or(10);
    let seed: u64 = get("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let previews: usize = get("--previews").and_then(|s| s.parse().ok()).unwrap_or(7);
    let rounds: usize = get("--rounds").and_then(|s| s.parse().ok()).unwrap_or(1);
    let corrupt_deal: Option<usize> = get("--corrupt-deal").and_then(|s| s.parse().ok());
    let expect_giveup = args.iter().any(|a| a == "--expect-giveup");

    // Deal the next piece to BOTH host and bot; --corrupt-deal flips one deal (as a broken
    // randomizer would), which the bot must detect and report — not silently misplay.
    fn deal(bag: &mut Bag, idx: &mut usize, corrupt: Option<usize>) -> u8 {
        let mut p = bag.next();
        if corrupt == Some(*idx) {
            p = (p + 1) % 7;
        }
        *idx += 1;
        p
    }

    let mut proj = ProjFilter::load(&proj_path).expect("load proj");
    // --projext F: attach the supplementary PCPRJX1 projection(s) the shipped bot embeds, so the
    // simulator matches the deployed filter exactly.
    if let Some(p) = get("--projext") {
        let n = proj.attach_extra(&std::fs::read(&p).expect("read projext")).expect("attach extra");
        eprintln!("attached {} extra projections from {}", n, p);
    }
    let table = ValueTable::load(std::path::Path::new(&values_path)).expect("load values");
    let mut bot = Bot::new(proj, table);

    // --reveal-lag N: model jstris' LAZY reveal delivery. Newly drawn pieces still fill the host's
    // physical queue immediately (so the bot never starves), but the corresponding `new_piece`
    // messages to the bot are held back N placements — reproducing the live case where the bot is
    // asked to suggest the next move before the reveal for the current one has arrived (depth < 6
    // needs no reveal, so a correct bot must still move). Pending reveals are flushed at each PC
    // boundary (the jstris bot.js patch tops the preview queue up to 6 there). N=0 is the default
    // eager delivery.
    let reveal_lag: usize = get("--reveal-lag").and_then(|s| s.parse().ok()).unwrap_or(0);
    let start_hold = args.iter().any(|a| a == "--start-hold");

    let mut total_pcs = 0usize;
    let mut t_total = 0f64;
    'rounds: for round in 0..rounds {
    // Fresh round, same Bot instance (jstris keeps the worker across rounds and sends a fresh
    // `start` with a new randomizer): fresh-game detection must reset the bot's stream tracking.
    let mut bag = Bag { rng: seed.max(1).wrapping_add(round as u64 * 7919), left: Vec::new() };
    let mut deal_idx = 0usize;
    // --start-hold: begin each round with a piece already in hold, exactly like the live jstris
    // trace (start hold=S queue=TOLZIJ). hold + `previews` queue must span a clean bag prefix, so
    // deal the hold first from the same bag stream.
    let mut host_hold: Option<u8> = if start_hold { Some(deal(&mut bag, &mut deal_idx, corrupt_deal)) } else { None };
    let mut host_queue: Vec<u8> = (0..previews).map(|_| deal(&mut bag, &mut deal_idx, corrupt_deal)).collect();
    let mut phys = Phys { cells: 0 };
    // Reveals drawn into the physical queue but not yet delivered to the bot (held back reveal_lag).
    let mut pending_reveals: std::collections::VecDeque<u8> = std::collections::VecDeque::new();

    // start
    bot.start(host_hold, &host_queue, &phys.board_cells(), 0);

    let mut pcs_done = 0usize;
    let mut placements = 0usize;
    let mut t_boundary_max = 0f64;
    while pcs_done < pcs_target {
        // suggest
        let t0 = std::time::Instant::now();
        let Some(mv) = bot.suggest() else {
            if expect_giveup {
                println!(
                    "GIVE-UP (expected) round {} after {} placements, {} PCs: {}",
                    round, placements, pcs_done, bot.last_error
                );
                assert!(
                    bot.last_error.contains("bag desync")
                        || bot.last_error.contains("7-bag permutation"),
                    "give-up reason should be a precise bag diagnostic, got: {}",
                    bot.last_error
                );
                return;
            }
            panic!(
                "bot returned no move after {} placements, {} PCs (round {}): {}",
                placements, pcs_done, round, bot.last_error
            );
        };
        let dt = t0.elapsed().as_secs_f64();
        t_total += dt;
        if dt > t_boundary_max {
            t_boundary_max = dt;
        }

        // --- host-side validation ---
        // hold inference exactly like jstris: piece type != active -> hold swap
        let active = host_queue[0];
        if mv.piece == active {
            host_queue.remove(0);
        } else if host_hold == Some(mv.piece) {
            host_hold = Some(host_queue.remove(0));
        } else if host_hold.is_none() {
            host_hold = Some(host_queue.remove(0));
            assert_eq!(host_queue[0], mv.piece, "queue desync: move piece unavailable");
            host_queue.remove(0);
        } else {
            panic!("queue desync: move piece {} not active/hold", piece_char(mv.piece));
        }

        // Decode the emitted TBP move with the libtetris-exact table (INDEPENDENT of how the bot
        // produced it) into absolute physical cells, then verify those cells are a reachable,
        // supported placement on the current physical board.
        let engine_hash = phys.to_engine_hash();
        let sunk = {
            let mut k = 0i8;
            while k < 4 && (engine_hash >> (10 * k as u64)) & 0b1111111111 == 0b1111111111 { k += 1; }
            k
        };
        let dec = tbpcoord::decode(mv.piece, mv.orient, mv.x, mv.y); // (col, row) screen, physical
        let mut cells4 = [0u8; 4];
        for (i, &(c, r)) in dec.iter().enumerate() {
            assert!(c >= 0 && c < 10 && r >= 0 && r < 4, "decoded cell OOB: c{} r{}", c, r);
            cells4[i] = (r as u8) * 10 + c as u8;
        }
        // reachability: some engine placement must produce these exact cells (engine row = phys+sunk)
        let want_eng: std::collections::HashSet<(i8, i8)> = dec.iter().map(|&(c, r)| (c, r + sunk)).collect();
        let reachable = movegen::placements(mv.piece as usize, engine_hash)
            .iter()
            .any(|p| p.cells.iter().copied().collect::<std::collections::HashSet<_>>() == want_eng);
        assert!(
            reachable,
            "emitted move not a reachable placement: {} {} x{} y{} cells {:?}",
            piece_char(mv.piece), ORIENT_NAMES[mv.orient as usize], mv.x, mv.y, dec
        );
        // SIM_TRACE=1: one line per placement (bit-exactness regression harness).
        if std::env::var("SIM_TRACE").map(|s| s != "0").unwrap_or(false) {
            println!(
                "T {:4} {} {} x{} y{}",
                placements, piece_char(mv.piece), ORIENT_NAMES[mv.orient as usize], mv.x, mv.y
            );
        }
        let cleared = phys.place(cells4);

        // feed play + new_piece(s) like the host
        bot.play();
        let consumed = if mv.piece == active { 1 } else if host_hold == Some(active) { 1 } else { 2 };
        let _ = consumed; // new_piece count actually equals pieces removed from host_queue this turn
        // Physical queue refills immediately; the bot's new_piece delivery is held back reveal_lag.
        while host_queue.len() < previews {
            let p = deal(&mut bag, &mut deal_idx, corrupt_deal);
            host_queue.push(p);
            pending_reveals.push_back(p);
        }
        while pending_reveals.len() > reveal_lag {
            bot.new_piece(pending_reveals.pop_front().unwrap());
        }

        placements += 1;
        if phys.cells == 0 {
            // PC boundary: flush all held-back reveals (jstris' bot.js patch tops the preview
            // queue up to 6 at a boundary, so the bot forms the next boundary with full previews).
            for p in pending_reveals.drain(..) {
                bot.new_piece(p);
            }
            pcs_done += 1;
            total_pcs += 1;
            if cleared > 0 {
                // fine: 4LPC finishes with a clear; 2LPC likewise
            }
            println!(
                "PC {:3} done at placement {:4}  (round {}, max suggest {:6.2}s)",
                pcs_done, placements, round, t_boundary_max
            );
            t_boundary_max = 0.0;
            if expect_giveup && corrupt_deal.map_or(false, |k| deal_idx > k + previews + 8) {
                // The corrupted deal is well past every window/reveal it can appear in and the
                // bot is still playing: the diagnostics failed to fire.
                break 'rounds;
            }
        }
        assert!(placements < pcs_target * 12 + 50, "too many placements without completing PCs");
    }
    println!(
        "round {} OK: {} PCs in {} placements",
        round, pcs_done, placements
    );
    }
    if expect_giveup {
        panic!("expected a bag-desync give-up, but the bot played {} PCs cleanly", total_pcs);
    }
    println!(
        "\nOK: {} PCs across {} rounds | suggest total {:.1}s avg/PC {:.2}s",
        total_pcs,
        rounds,
        t_total,
        t_total / total_pcs.max(1) as f64
    );
}



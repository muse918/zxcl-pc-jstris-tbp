//! Exact reproduction probe for the live 3rd-boundary forfeit (jez report):
//! boundary hold=L, visible=[O,J,L,I,S,T], bag mask={Z} (phase-6 singleton, h1 forced=Z),
//! observed reveals h2=T, h3=L. Enumerates ALL 210 hidden completions and greedy-walks each
//! branch exactly like bot::suggest (known = one reveal per placement, pad = averaged tail),
//! under (a) base projection filter and (b) base+ext — a branch alive in exact play but dead
//! here means the edge filter dropped the winning line (false negative).
//!
//! usage: probe --proj F --values F [--projext F] [--source field_source.u40.bin]
//! --source additionally scans EVERY PC-able member field through both filters (exhaustive
//! false-negative check; the base filter's build-time claim, re-verified, and the ext's first).

use std::cell::RefCell;

use hashbrown::{HashMap, HashSet};
use pcbot_wasm::graph::{MAX_HASH, TWO_LINE_HASH};
use pcbot_wasm::movegen::MoveGen;
use pcbot_wasm::piece::{after_reveal, piece_char, Piece};
use pcbot_wasm::proj::ProjFilter;
use pcbot_wasm::value_search::{value_search, SearchInput, VsResult};
use pcbot_wasm::values::{ResetEval, ValueTable};

const HOLD: Piece = 2; // L
const VISIBLE: [Piece; 6] = [3, 1, 2, 0, 4, 5]; // O J L I S T
const MASK: u8 = 1 << 6; // {Z}

fn pieces_of(mask: u8) -> Vec<Piece> {
    (0..7u8).filter(|&p| mask & (1 << p) != 0).collect()
}

fn build(proj: &ProjFilter, table: &ValueTable) -> VsResult {
    let memo = RefCell::new(HashMap::<u64, Vec<u64>>::new());
    let mg = RefCell::new(MoveGen::new());
    let edge = |h: u64, piece: u8, out: &mut Vec<u64>| {
        let key = (h << 3) | piece as u64;
        if let Some(v) = memo.borrow().get(&key) {
            out.extend_from_slice(v);
            return;
        }
        let mut tmp = Vec::new();
        mg.borrow_mut().children(piece as usize, h, &mut tmp);
        tmp.retain(|&c| c == MAX_HASH || c == TWO_LINE_HASH || proj.maybe(c));
        out.extend_from_slice(&tmp);
        memo.borrow_mut().insert(key, tmp);
    };
    value_search(SearchInput {
        graph: None,
        hold: HOLD,
        visible: VISIBLE,
        mask: MASK,
        reset: ResetEval::new(Some(table)),
        edge_ids: Some(&edge),
        par_edge: None,
    })
}

/// Greedy-walk one full hidden branch exactly like bot::suggest. Returns Ok(terminal 1|2) or
/// Err((depth, field, hold, active, best_score, cands)) at the death node.
#[allow(clippy::type_complexity)]
fn walk(vs: &VsResult, hidden: [Piece; 4]) -> Result<u8, (u8, u64, Piece, Piece, f64, usize)> {
    let mut path: Vec<usize> = Vec::new();
    loop {
        let known = path.len().min(4);
        let mut hid = [0u8; 4];
        let mut m = MASK;
        for i in 0..4 {
            let h = if i < known {
                hidden[i]
            } else {
                (0..7u8).find(|&p| m & (1 << p) != 0).unwrap()
            };
            hid[i] = h;
            m = after_reveal(m, h);
        }
        let node = vs.analyze(&path, hid);
        if node.terminal == 1 || node.terminal == 2 {
            return Ok(node.terminal);
        }
        if node.terminal != 0 || node.cands.is_empty() || node.best_score <= 0.0 {
            return Err((node.depth, node.field, node.hold, node.active, node.best_score, node.cands.len()));
        }
        path.push(node.cands[0].edge);
        assert!(path.len() <= 10, "walk overran the loop");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let proj_path = get("--proj").expect("--proj required");
    let values_path = get("--values").expect("--values required");
    let ext_path = get("--projext");
    let source_path = get("--source");

    let table = ValueTable::load(std::path::Path::new(&values_path)).expect("load values");

    // --source: exhaustive member scan through both filters (definitive FN check).
    let members: Option<HashSet<u64>> = source_path.as_ref().map(|p| {
        let bytes = std::fs::read(p).expect("read source");
        // PCFLDSR1 header: 8-byte magic + u64 LE count + count x u40 LE.
        assert_eq!(&bytes[..8], b"PCFLDSR1");
        let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let bytes = &bytes[16..16 + count * 5];
        let mut s = HashSet::with_capacity(count);
        for ch in bytes.chunks_exact(5) {
            let mut v = 0u64;
            for (i, &b) in ch.iter().enumerate() {
                v |= (b as u64) << (8 * i);
            }
            s.insert(v);
        }
        s
    });
    if let (Some(members), Some(ext_path)) = (&members, &ext_path) {
        let base = ProjFilter::load(&proj_path).expect("load proj");
        let mut ext = ProjFilter::load(&proj_path).expect("load proj");
        ext.attach_extra(&std::fs::read(ext_path).expect("read ext")).expect("attach");
        let (mut base_fn, mut ext_fn) = (0u64, 0u64);
        let mut ext_fn_samples: Vec<u64> = Vec::new();
        for &f in members.iter() {
            if !base.maybe(f) {
                base_fn += 1;
            } else if !ext.maybe(f) {
                ext_fn += 1;
                if ext_fn_samples.len() < 8 {
                    ext_fn_samples.push(f);
                }
            }
        }
        println!(
            "member scan: {} fields | base FN = {} | ext-only FN = {}",
            members.len(),
            base_fn,
            ext_fn
        );
        for f in &ext_fn_samples {
            println!("  ext FN field {:#012x}", f);
        }
    }

    for cfg in ["base", "base+ext"] {
        let mut proj = ProjFilter::load(&proj_path).expect("load proj");
        if cfg == "base+ext" {
            let Some(p) = &ext_path else { continue };
            proj.attach_extra(&std::fs::read(p).expect("read ext")).expect("attach");
        }
        let vs = build(&proj, &table);
        println!("\n[{}] root_value={:.6} missing_keys={}", cfg, vs.root_value, vs.missing_keys);
        {
            // Root decision (depth 0): what the bot actually plays first at this boundary.
            let mut hid = [0u8; 4];
            let mut m = MASK;
            for i in 0..4 {
                let h = (0..7u8).find(|&p| m & (1 << p) != 0).unwrap();
                hid[i] = h;
                m = after_reveal(m, h);
            }
            let node = vs.analyze(&[], hid);
            for c in node.cands.iter().take(3) {
                println!(
                    "  root cand: place {} -> field {:#012x} hold_after {} score {:.4}{}",
                    piece_char(c.placed), c.field_after, piece_char(c.hold_after), c.score,
                    if c.best { "  <= chosen" } else { "" }
                );
            }
        }

        let (mut done4, mut done2, mut dead) = (0u32, 0u32, 0u32);
        for h1 in pieces_of(MASK) {
            let m1 = after_reveal(MASK, h1);
            for h2 in pieces_of(m1) {
                let m2 = after_reveal(m1, h2);
                for h3 in pieces_of(m2) {
                    let m3 = after_reveal(m2, h3);
                    for h4 in pieces_of(m3) {
                        match walk(&vs, [h1, h2, h3, h4]) {
                            Ok(1) => done4 += 1,
                            Ok(_) => done2 += 1,
                            Err((d, field, hold, active, best, ncand)) => {
                                dead += 1;
                                let member = members
                                    .as_ref()
                                    .map(|s| if s.contains(&field) { "MEMBER" } else { "non-member" })
                                    .unwrap_or("?");
                                println!(
                                    "  DEAD [{}{}{}{}] at depth {} field {:#012x} ({}) hold {} active {} best {:.3} cands {}",
                                    piece_char(h1), piece_char(h2), piece_char(h3), piece_char(h4),
                                    d, field, member, piece_char(hold), piece_char(active), best, ncand
                                );
                            }
                        }
                    }
                }
            }
        }
        println!("[{}] branches: 4L {} | 2L {} | dead {}", cfg, done4, done2, dead);
    }

    // H1 family — live-failure hypothesis: exactly ONE delivered piece was lost somewhere in
    // deals 21..27 (any earlier loss desyncs the bot's actives from the physical game and the
    // HOST would have errored during loops 1-2; any later loss leaves loop 3's inputs intact).
    // Lost deal p removes piece L_p from the bot's window: its loop-3 boundary becomes
    // (hold L | survivors of [O,J,L,I,S,T,Z] minus L_p | mask {L_p}), and the reveals it reads
    // are shifted: h1 = true h2 = T, h2 = true h3 = L. The old code fed these unvalidated; every
    // fold pack is L_p-prefixed, so (unless L_p == T) every candidate scores 0 at depth 1 ->
    // the live log's fake "dead position (terminal=3)". Match the variant by its cands count.
    println!("\n[H1] one-lost-deal variants (live log: DEAD at shallow depth, cands=48):");
    let mut proj = ProjFilter::load(&proj_path).expect("load proj");
    if let Some(p) = &ext_path {
        proj.attach_extra(&std::fs::read(p).expect("read ext")).expect("attach");
    }
    let window: [Piece; 7] = [3, 1, 2, 0, 4, 5, 6]; // true deals 21..27 = O J L I S T Z
    for lost_idx in 0..7 {
        let lost = window[lost_idx];
        let mut vis = Vec::new();
        for (i, &p) in window.iter().enumerate() {
            if i != lost_idx {
                vis.push(p);
            }
        }
        let visible: [Piece; 6] = vis[..6].try_into().unwrap();
        let mask = 1u8 << lost;
        let memo = RefCell::new(HashMap::<u64, Vec<u64>>::new());
        let mg = RefCell::new(MoveGen::new());
        let edge = |h: u64, piece: u8, out: &mut Vec<u64>| {
            let key = (h << 3) | piece as u64;
            if let Some(v) = memo.borrow().get(&key) {
                out.extend_from_slice(v);
                return;
            }
            let mut tmp = Vec::new();
            mg.borrow_mut().children(piece as usize, h, &mut tmp);
            tmp.retain(|&c| c == MAX_HASH || c == TWO_LINE_HASH || proj.maybe(c));
            out.extend_from_slice(&tmp);
            memo.borrow_mut().insert(key, tmp);
        };
        let vs = value_search(SearchInput {
            graph: None,
            hold: HOLD,
            visible,
            mask,
            reset: ResetEval::new(Some(&table)),
            edge_ids: Some(&edge),
            par_edge: None,
        });
        // Fed reveals (shifted by the loss): T, L, then unknown — pad legally per this mask.
        let mut hid = [0u8; 4];
        let mut m = mask;
        let fed = [5u8, 2];
        for i in 0..4 {
            let h = if i < 2 { fed[i] } else { (0..7u8).find(|&p| m & (1 << p) != 0).unwrap() };
            hid[i] = h;
            m = after_reveal(m, h);
        }
        let vstr: String = visible.iter().map(|&p| piece_char(p)).collect();
        match walk(&vs, hid) {
            Ok(t) => println!(
                "  lost {} @deal{} (boundary L|{}|{{{}}} root {:.1}): completed terminal={} — reveals stayed legal",
                piece_char(lost), 21 + lost_idx, vstr, piece_char(lost), vs.root_value, t
            ),
            Err((d, field, _hold, active, best, ncand)) => println!(
                "  lost {} @deal{} (boundary L|{}|{{{}}} root {:.1}): DEAD depth={} field={:#012x} active={} best={:.3} cands={}",
                piece_char(lost), 21 + lost_idx, vstr, piece_char(lost), vs.root_value, d, field, piece_char(active), best, ncand
            ),
        }
    }
}

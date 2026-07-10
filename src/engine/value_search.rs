//! Value-seeded one-loop search (fork of fh_search).
//!
//! fh_search computes the one-loop PC PROBABILITY with a 0/1 bitset trick (terminal = 1,
//! max over choices = bitwise OR). Here terminals carry REAL values from a boundary value
//! table: a completed PC is worth 1 + V(reset)/5040, where V is the exported layer0
//! one-loop u16 (so the root value is the exact 2-PC-horizon optimum). The bitset OR
//! generalizes to elementwise max over per-hidden-sequence value vectors.
//!
//! Additions over fh_search:
//!   - 2LPC terminal: a placement into TWO_LINE_HASH at depth 5 completes a 2-line PC and
//!     resets; its value is E_h5[1 + V(reset)/5040] (q5 and h1..h4 are known there).
//!   - hidden-sequence RANGE storage: full_hidden is built in DFS order, so every pack
//!     prefix owns a contiguous leaf range; value vectors are stored per-range.
//!   - retained tables for policy walks (sample play) and best-first-move reports.

use hashbrown::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Collect `slice.par_iter().<m>(body)` when `par` (native only), else `slice.iter().<m>(body)`.
/// On wasm the parallel arm is cfg'd out entirely — the deployed bot never enables `par_edge`, so
/// rayon is neither referenced nor linked into the wasm binary.
macro_rules! par_or_serial {
    ($par:expr, $slice:expr, $m:ident, $body:expr) => {{
        #[cfg(not(target_arch = "wasm32"))]
        {
            if $par {
                $slice.par_iter().$m($body).collect()
            } else {
                $slice.iter().$m($body).collect()
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = $par;
            $slice.iter().$m($body).collect()
        }
    }};
}

// std::time::Instant panics on wasm32-unknown-unknown; the timings are diagnostics only.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct Instant;
#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self { Instant }
    fn elapsed(&self) -> std::time::Duration { std::time::Duration::ZERO }
}

use crate::graph::{HydraGraph, MAX_HASH, TWO_LINE_HASH};
use crate::piece::{after_reveal, pieces, Piece, PIECE_COUNT};
use crate::values::ResetEval;

const FULL_MASK: u8 = 0b111_1111;

/// The DAG is keyed by the 40-bit field HASH directly (no graph FieldId index), so the search is
/// graph-free: the empty box is hash 0 (root), a completed 4LPC is the full box `MAX_HASH`
/// (place() sinks 4 full lines -> all-ones), and a 2LPC is `TWO_LINE_HASH`.
const TERMINAL_HASH: u64 = MAX_HASH;
const ROOT_HASH: u64 = 0;

/// Node ids stay well under 2^32 (a few million per boundary), so u32 halves every id-carrying
/// structure: CSR edges, the build index, fold keys, and the shallow reverse index.
type NodeId = u32;

/// Per-node value vectors (depths 4..10) dominate memory — they hold one value per hidden-reveal
/// leaf (up to 840) for hundreds of thousands of live nodes. Storing them as u16 instead of f32
/// halves that (the single biggest wasm-memory consumer). Values are "expected consecutive PCs",
/// bounded by 1 + max boundary V*; QUANT_MAX gives comfortable headroom so nothing clips.
///
/// The backup is elementwise MAX, and quantization q(v)=round(v/QUANT_MAX*65535) is monotone, so
/// max(q(a),q(b)) == q(max(a,b)) — the whole backup runs in u16 with NO error beyond quantizing the
/// terminal seeds (~QUANT_MAX/65535 ≈ 0.067). The fold averages (depths 0..3) dequantize to f64, so
/// only they carry that one-step error — same regime as the offline V* table (12-bit: move-exact).
type Quant = u16;
const QUANT_MAX: f64 = 4400.0; // > 1 + max V* (~4356); no live value reaches it, so no clipping
const QZERO: Quant = 0; // quant(0.0); a dead leaf

#[inline]
fn quant(v: f64) -> Quant {
    let x = (v / QUANT_MAX * 65535.0).round();
    if x <= 0.0 {
        0
    } else if x >= 65535.0 {
        65535
    } else {
        x as Quant
    }
}

#[inline]
fn dequant(q: Quant) -> f64 {
    q as f64 * (QUANT_MAX / 65535.0)
}

type FoldTable = HashMap<u16, HashMap<NodeId, f64>>;

/// One depth's value vectors as a single arena: all vectors concatenated in layer order, with a
/// per-layer-slot (offset, len) index. Node ids are assigned layer-contiguously (build discovers
/// depth by depth and prune preserves id order), so `id - start` is the slot directly — lookups
/// are two array reads instead of a hash probe, and the per-node Vec headers, allocator overhead
/// and hash-table slack (~70MB per boundary) disappear.
#[derive(Default)]
struct ValLayer {
    start: NodeId,  // first node id of this depth's layer
    off: Vec<u32>,  // per slot: offset into data; u32::MAX = no value (dead node)
    len: Vec<u32>,  // per slot: vector length (undefined when off == MAX)
    data: Vec<Quant>,
}

impl ValLayer {
    #[inline]
    fn get(&self, id: NodeId) -> Option<&[Quant]> {
        let slot = id.checked_sub(self.start)? as usize;
        let o = *self.off.get(slot)?;
        if o == u32::MAX {
            return None;
        }
        Some(&self.data[o as usize..o as usize + self.len[slot] as usize])
    }

    #[inline]
    fn contains(&self, id: NodeId) -> bool {
        id.checked_sub(self.start)
            .and_then(|slot| self.off.get(slot as usize))
            .map_or(false, |&o| o != u32::MAX)
    }

    fn live(&self) -> usize {
        self.off.iter().filter(|&&o| o != u32::MAX).count()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NodeKey {
    depth: u8,  // 0..10
    field: u64, // 40-bit field hash (graph convention); 0=empty box, MAX=4LPC done
    hold: Piece,
    /// Remaining-bag mask (canonical, nonzero). Depths 0..=6: the initial mask (no reveals
    /// consumed). Depths 7..10: the bag after the reveals consumed so far. Keying by MASK instead
    /// of the reveal SEQUENCE merges transpositions: the subtree below (field, hold) depends only
    /// on the DOMAIN of the remaining reveals, so prefixes with equal multiset share one node
    /// whose value vector is indexed by suffix-rank (see SuffixTables).
    mask: u8,
}

/// CSR node storage: keys plus one flat edge array. Nodes are created in id order and their edge
/// lists are also FILLED in id order (the build walks each layer's frontier ascending, and every
/// child id it creates is larger), so edges append contiguously — node id's edges live at
/// edge_data[edge_start[id]..edge_start[id+1]]. Replaces a Vec<NodeId> per node (24B header +
/// allocator overhead x millions of nodes).
#[derive(Clone, Debug, Default)]
struct Dag {
    keys: Vec<NodeKey>,
    edge_start: Vec<u32>, // len keys.len()+1 once sealed
    edge_data: Vec<NodeId>,
    layers: Vec<Vec<NodeId>>, // 0..10
    // Keyed by pack_key(NodeKey): field(40b) | depth(4b) | hold(3b) | mask(7b) = 54 bits.
    // A single-word key hashes much faster than the 12-byte struct (~10^8 lookups per boundary).
    index: HashMap<u64, NodeId>,
    root: NodeId,
}

impl Dag {
    #[inline]
    fn key(&self, id: NodeId) -> &NodeKey {
        &self.keys[id as usize]
    }

    #[inline]
    fn edges(&self, id: NodeId) -> &[NodeId] {
        &self.edge_data[self.edge_start[id as usize] as usize..self.edge_start[id as usize + 1] as usize]
    }

    /// Begin node `id`'s edge list (must be called in ascending id order; children then append
    /// via push_edge). Terminal-layer nodes are covered by seal().
    #[inline]
    fn begin_edges(&mut self, id: NodeId) {
        debug_assert!(self.edge_start.len() == id as usize);
        self.edge_start.push(self.edge_data.len() as u32);
    }

    /// Pad edge_start so every node (incl. never-processed terminal-layer ones) has a range.
    fn seal(&mut self) {
        while self.edge_start.len() <= self.keys.len() {
            self.edge_start.push(self.edge_data.len() as u32);
        }
    }
}

#[inline]
fn pack_key(k: &NodeKey) -> u64 {
    k.field | ((k.depth as u64) << 40) | ((k.hold as u64) << 44) | ((k.mask as u64) << 47)
}

/// Per-(mask, remaining-length) suffix-enumeration sizes and per-reveal block offsets, matching
/// build_ranges' DFS order exactly (ascending piece index; after_reveal auto-refills the bag).
/// A merged node's value vector is indexed by suffix-rank over (its mask, 10-depth); the child
/// block for reveal q inside a parent's vector starts at off[parent_mask][parent_len][q].
struct SuffixTables {
    cnt: [[u32; 5]; 128],      // cnt[m][len] = #reveal sequences of `len` from canonical mask m
    off: [[[u32; 7]; 5]; 128], // off[m][len][q] = block offset of reveal q within (m, len)
}

fn build_suffix_tables() -> Box<SuffixTables> {
    let mut t = Box::new(SuffixTables { cnt: [[0; 5]; 128], off: [[[0; 7]; 5]; 128] });
    for m in 0..128 {
        t.cnt[m][0] = 1;
    }
    for len in 1..=4usize {
        for m in 1..=127usize {
            let mut acc = 0u32;
            for q in 0..7usize {
                if m & (1 << q) != 0 {
                    t.off[m][len][q] = acc;
                    acc += t.cnt[after_reveal(m as u8, q as u8) as usize][len - 1];
                }
            }
            t.cnt[m][len] = acc;
        }
    }
    t
}

#[derive(Clone, Copy, Debug)]
pub struct SeqRange {
    pub start: u32,
    pub len: u32,
}

pub struct VsResult {
    pub root_value: f64,
    pub missing_keys: u64,
    pub nodes_total: usize,
    pub nodes_pruned: usize,
    pub leaf_count: usize,
    retained: Retained,
}

struct Retained {
    dag: Dag,
    ranges: HashMap<(u8, u16), SeqRange>, // (prefix_len 0..=4, pack) -> leaf range
    vals: Vec<ValLayer>,                  // depth 4..=10 at index depth
    folds: Vec<FoldTable>,                // index 1..=3: value_d keyed by len-d prefix pack
    initial_mask: u8,
    visible: [Piece; 6],
    two_line_field: u64,
}

pub struct SearchInput<'a> {
    /// Only used by the reference edge path (`edge_ids: None`). A graph-free run passes `None`
    /// here and supplies `edge_ids`, so the whole search touches no precomputed graph.
    pub graph: Option<&'a HydraGraph>,
    pub hold: Piece,
    pub visible: [Piece; 6],
    pub mask: u8,
    pub reset: ResetEval<'a>,
    /// Edge source keyed by field HASH (the WASM bot's movegen+ProjFilter): given
    /// (field_hash, piece), APPEND the child field HASHES into the provided buffer (which the
    /// caller clears first — filling a reused buffer avoids per-edge allocation). None = graph.edges.
    pub edge_ids: Option<&'a (dyn Fn(u64, u8, &mut Vec<u64>) + 'a)>,
    /// Optional SYNC edge source for the PARALLEL build (rayon fans the per-node movegen across
    /// threads). When Some, the build runs in parallel and this replaces `edge_ids`/`graph`; the
    /// result is bit-identical to the serial build. Thread count = the ambient rayon pool.
    pub par_edge: Option<&'a (dyn Fn(u64, u8, &mut Vec<u64>) + Sync + 'a)>,
}

pub fn value_search(mut input: SearchInput<'_>) -> VsResult {
    let verbose = std::env::var("VS_VERBOSE").map(|s| s != "0").unwrap_or(false);
    let initial_mask = canonical_mask(input.mask);
    let two_line_field = TWO_LINE_HASH;

    // Hidden-sequence leaf ranges (DFS order, matching build_hidden_prefixes).
    let mut ranges: HashMap<(u8, u16), SeqRange> = HashMap::new();
    let mut next_leaf = 0u32;
    build_ranges(initial_mask, 0, 0, &mut next_leaf, &mut ranges);
    let leaf_count = next_leaf as usize;

    let mut full_hidden_packs = vec![0u16; leaf_count];
    for (&(len, pack), &r) in &ranges {
        if len == 4 {
            full_hidden_packs[r.start as usize] = pack;
        }
    }

    // Suffix-rank bookkeeping for transposition-merged nodes (depths 7..10).
    let suffix = build_suffix_tables();
    debug_assert_eq!(suffix.cnt[initial_mask as usize][4] as usize, leaf_count);

    let _t = Instant::now();
    let (dag, nodes_total, t_build) = {
        // Scope the UNPRUNED dag so it's freed right after pruning (shadowing alone would keep
        // both DAGs alive through the whole solve), and drop its build-only node index first.
        let mut full = build_dag(input.graph, input.edge_ids, input.par_edge, input.hold, input.visible, initial_mask, two_line_field);
        let nodes_total = full.keys.len();
        let t_build = _t.elapsed();
        full.index = HashMap::new();
        (prune_to_terminal_reachable(&full, two_line_field, input.par_edge.is_some()), nodes_total, t_build)
    };
    let nodes_pruned = dag.keys.len();
    let t_prune = _t.elapsed() - t_build;
    if verbose { eprintln!(
        "value-search: nodes {} -> {} (terminal-reachable), hidden_leaves={} | build {:.2}ms prune {:.2}ms",
        nodes_total, nodes_pruned, leaf_count, t_build.as_secs_f64() * 1e3, t_prune.as_secs_f64() * 1e3
    );
        let per_layer: Vec<usize> = dag.layers.iter().map(|l| l.len()).collect();
        eprintln!(
            "value-search: edges_total={} ({}MB u32) node_keys={}MB layers={:?}",
            dag.edge_data.len(), dag.edge_data.len() * 4 / 1_000_000,
            dag.keys.len() * std::mem::size_of::<NodeKey>() / 1_000_000, per_layer
        );
    }
    let _t = Instant::now();
    // Parallel solve rides the same switch as the parallel build.
    let par = input.par_edge.is_some();

    let vec_len = |key: &NodeKey| -> usize {
        if key.depth <= 6 {
            leaf_count
        } else {
            suffix.cnt[key.mask as usize][(10 - key.depth) as usize] as usize
        }
    };

    // ---- seed depth 10 (4LPC terminals) ----
    let mut vals: Vec<ValLayer> = Vec::with_capacity(11);
    for _ in 0..11 {
        vals.push(ValLayer::default());
    }
    {
        let layer = &dag.layers[10];
        let mut l = ValLayer {
            start: layer.first().copied().unwrap_or(0),
            off: vec![u32::MAX; layer.len()],
            len: vec![0u32; layer.len()],
            data: Vec::new(),
        };
        for (slot, &id) in layer.iter().enumerate() {
            let key = dag.key(id);
            if key.field != TERMINAL_HASH {
                continue;
            }
            // key.mask at depth 10 IS the bag after all four reveals (mask4).
            let v = input.reset.w4(key.hold, key.mask);
            if v > 0.0 {
                l.off[slot] = l.data.len() as u32;
                l.len[slot] = 1;
                l.data.push(quant(v)); // suffix-len 0 -> vector of 1
            }
        }
        if verbose { eprintln!("value-search: depth10 live={}", l.live()); }
        vals[10] = l;
    }

    // ---- elementwise-max backup depths 9..4, with 2LPC injection at depth 5 ----
    // FORWARD-edge, per-parent, in TWO passes over the layer: pass 1 (cheap) marks live parents
    // (any live child; at depth 5 also the childless 2LPC terminals) and lays out one arena;
    // pass 2 max-folds every parent directly into its disjoint arena slice. No per-parent Vec is
    // ever allocated, so the old transient (a Vec per node before table insert) is gone.
    // u16 max is exact and order-independent (quant is monotone), so serial/parallel/any-order
    // produce IDENTICAL values.
    for depth in (4..10u8).rev() {
        let prev = &vals[(depth + 1) as usize];
        let layer = &dag.layers[depth as usize];

        // Pass 1: liveness. A parent is live iff any child carries a vector; depth-5 two-line
        // terminals are live unconditionally (they get their reset value injected below).
        let live_of = |&id: &NodeId| -> bool {
            (depth == 5 && dag.key(id).field == two_line_field)
                || dag.edges(id).iter().any(|&c| prev.contains(c))
        };
        let alive: Vec<bool> = par_or_serial!(par, layer, map, live_of);

        let mut next = ValLayer {
            start: layer.first().copied().unwrap_or(0),
            off: vec![u32::MAX; layer.len()],
            len: vec![0u32; layer.len()],
            data: Vec::new(),
        };
        let mut total = 0usize;
        for (slot, (&id, &a)) in layer.iter().zip(&alive).enumerate() {
            if a {
                let l = vec_len(dag.key(id)) as u32;
                next.off[slot] = total as u32;
                next.len[slot] = l;
                total += l as usize;
            }
        }
        next.data = vec![QZERO; total];

        // Pass 2: carve the arena into per-parent slices (disjoint by construction) and fold.
        let mut jobs: Vec<(NodeId, &mut [Quant])> = Vec::new();
        {
            let mut rest: &mut [Quant] = &mut next.data;
            for (slot, &id) in layer.iter().enumerate() {
                if next.off[slot] == u32::MAX {
                    continue;
                }
                let (dst, tail) = std::mem::take(&mut rest).split_at_mut(next.len[slot] as usize);
                rest = tail;
                jobs.push((id, dst));
            }
        }
        let fold = |(parent_id, dst): &mut (NodeId, &mut [Quant])| {
            let parent_key = dag.key(*parent_id);
            for &child_id in dag.edges(*parent_id) {
                let Some(child_vec) = prev.get(child_id) else { continue };
                // Reveal-consuming transitions (child depth >= 7): the child's block sits at the
                // parent's suffix offset of the revealed piece q. q is unique from the mask diff
                // (pm\cm = {q}; empty diff means the singleton bag refilled, so q = that piece).
                let offset = if depth >= 6 {
                    let pm = parent_key.mask;
                    let d = pm & !dag.key(child_id).mask;
                    let q = if d != 0 { d.trailing_zeros() } else { pm.trailing_zeros() };
                    suffix.off[pm as usize][(10 - depth) as usize][q as usize] as usize
                } else {
                    0
                };
                // branchless elementwise max over the aligned slice -> auto-vectorizes (AVX).
                let d = &mut dst[offset..offset + child_vec.len()];
                for (dd, &cv) in d.iter_mut().zip(child_vec.iter()) {
                    *dd = if cv > *dd { cv } else { *dd };
                }
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            if par {
                jobs.par_iter_mut().for_each(fold);
            } else {
                jobs.iter_mut().for_each(fold);
            }
        }
        #[cfg(target_arch = "wasm32")]
        jobs.iter_mut().for_each(fold);
        drop(jobs);

        if depth == 5 {
            // 2LPC terminals: childless two-line nodes get their reset value directly.
            let q5 = input.visible[5];
            let mut memo: HashMap<(Piece, u16), f64> = HashMap::new();
            for (slot, &id) in dag.layers[5].iter().enumerate() {
                let key = *dag.key(id);
                if key.field != two_line_field {
                    continue;
                }
                let o = next.off[slot] as usize; // live by construction (pass 1)
                let vec = &mut next.data[o..o + leaf_count];
                for leaf in 0..leaf_count {
                    let pack = full_hidden_packs[leaf];
                    let v = *memo.entry((key.hold, pack)).or_insert_with(|| {
                        let h = [
                            get_hidden(pack, 0),
                            get_hidden(pack, 1),
                            get_hidden(pack, 2),
                            get_hidden(pack, 3),
                        ];
                        let mask4 = mask_after_hidden_prefix(initial_mask, pack, 4);
                        input.reset.w2(key.hold, q5, h, mask4)
                    });
                    let vq = quant(v);
                    if vq > vec[leaf] {
                        vec[leaf] = vq;
                    }
                }
            }
        }
        if verbose {
            eprintln!(
                "value-search: depth{} live={} elems={} u16_MB={:.1}",
                depth, next.live(), next.data.len(), next.data.len() as f64 * 2.0 / 1e6
            );
        }
        vals[depth as usize] = next;
    }

    // ---- folds: average h4..h1 at depths 3..0 (see7 information timing) ----
    // value_d[prefix_pack(h1..h_d)][node] = max_child avg_{h_{d+1}} value_{d+1}[..][child]
    let mut folds: Vec<FoldTable> = vec![FoldTable::new(); 4];

    // depth 3: consume depth-4 range vectors, average h4. FORWARD per-parent (parallel-safe):
    // per (pack3, child) the sum runs over the child's vector in ascending leaf order — the same
    // accumulation order as the old reverse-index version, so values are identical.
    {
        // Per-leaf depth-3 prefix pack and per-pack h4 branch counts, precomputed once.
        let mut pack3_of = vec![0u16; leaf_count];
        for (i, &fp) in full_hidden_packs.iter().enumerate() {
            pack3_of[i] = prefix_pack(fp, 3);
        }
        let mut branches3: HashMap<u16, f64> = HashMap::new();
        for &p3 in &pack3_of {
            branches3.entry(p3).or_insert_with(|| {
                pieces_in_mask(mask_after_hidden_prefix(initial_mask, p3, 3)).len() as f64
            });
        }
        let vals4 = &vals[4];
        let compute = |&parent_id: &NodeId| -> (NodeId, Vec<(u16, f64)>) {
            let mut per: HashMap<(u16, NodeId), f64> = HashMap::new();
            for &child_id in dag.edges(parent_id) {
                if let Some(child_vec) = vals4.get(child_id) {
                    for (i, &cv) in child_vec.iter().enumerate() {
                        if cv == QZERO {
                            continue;
                        }
                        *per.entry((pack3_of[i], child_id)).or_insert(0.0) += dequant(cv);
                    }
                }
            }
            // average over h4, then max over children (exact; order-free).
            let mut best: HashMap<u16, f64> = HashMap::new();
            for ((p3, _child), sum) in per {
                let avg = sum / branches3[&p3];
                match best.get_mut(&p3) {
                    Some(b) => {
                        if avg > *b {
                            *b = avg;
                        }
                    }
                    None => {
                        best.insert(p3, avg);
                    }
                }
            }
            (parent_id, best.into_iter().collect())
        };
        let layer = &dag.layers[3];
        let results: Vec<(NodeId, Vec<(u16, f64)>)> = par_or_serial!(par, layer, map, compute);
        let out = &mut folds[3];
        for (pid, list) in results {
            for (p3, val) in list {
                out.entry(p3).or_default().insert(pid, val);
            }
        }
        if verbose { eprintln!("value-search: fold depth3 tables={}", folds[3].len()); }
    }

    // Mini reverse index for the shallow folds (children at depths 1..=3, parents at 0..=2);
    // within a layer parents are id-ascending, matching the old full reverse index's order.
    // Keyed by child id (only ~10^4 shallow children exist — a dense per-node Vec wasted ~20MB
    // of empty Vec headers across the whole DAG).
    let mut mini_rev: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for d in 0..=2usize {
        for &pid in &dag.layers[d] {
            for &c in dag.edges(pid) {
                mini_rev.entry(c).or_default().push(pid);
            }
        }
    }

    // depths 2..0. Child tables are iterated in SORTED pack order so the f64 accumulation order
    // is canonical (deterministic regardless of hash-map insertion history).
    for depth in (0..3u8).rev() {
        let child_tables = std::mem::take(&mut folds[(depth + 1) as usize]);
        let mut sums: HashMap<(u16, NodeId, NodeId), f64> = HashMap::new();
        let mut packs: Vec<u16> = child_tables.keys().copied().collect();
        packs.sort_unstable();
        for &child_pack in &packs {
            let table = &child_tables[&child_pack];
            let parent_pack = prefix_pack(child_pack, depth);
            for (&child_id, &cv) in table {
                if cv <= 0.0 {
                    continue;
                }
                let Some(parents) = mini_rev.get(&child_id) else { continue };
                for &parent_id in parents {
                    if dag.key(parent_id).depth != depth {
                        continue;
                    }
                    *sums.entry((parent_pack, parent_id, child_id)).or_insert(0.0) += cv;
                }
            }
        }
        if verbose {
            eprintln!(
                "value-search: fold depth{} sums_entries={} (~{}MB in map)",
                depth, sums.len(), sums.len() * 28 / 1_000_000
            );
        }
        let out = fold_from_sums(sums, initial_mask, depth);
        if verbose { eprintln!("value-search: fold depth{} tables={}", depth, out.len()); }
        if depth + 1 <= 3 {
            folds[(depth + 1) as usize] = child_tables;
        }
        folds[depth as usize] = out;
    }

    let root_value = folds[0]
        .get(&0u16)
        .and_then(|t| t.get(&dag.root))
        .copied()
        .unwrap_or(0.0);

    if verbose {
        for d in 0..=3usize {
            let entries: usize = folds[d].values().map(|t| t.len()).sum();
            eprintln!("value-search: fold depth{} tables={} entries={}", d, folds[d].len(), entries);
        }
        eprintln!("value-search: solve(rev+backup+fold) {:.2}ms", _t.elapsed().as_secs_f64() * 1e3);
    }

    let missing_keys = input.reset.missing_keys;
    VsResult {
        root_value,
        missing_keys,
        nodes_total,
        nodes_pruned,
        leaf_count,
        retained: Retained {
            dag,
            ranges,
            vals,
            folds,
            initial_mask,
            visible: input.visible,
            two_line_field,
        },
    }
}

fn fold_from_sums(
    sums: HashMap<(u16, NodeId, NodeId), f64>,
    initial_mask: u8,
    depth: u8,
) -> FoldTable {
    // Denominator: number of h_{depth+1} choices after the known prefix = mask size at
    // that point (deterministic per level given the initial mask).
    let mut out: FoldTable = FoldTable::new();
    for ((parent_pack, parent_id, _child_id), sum) in sums {
        let mask = mask_after_hidden_prefix(initial_mask, parent_pack, depth);
        let branches = pieces_in_mask(mask).len() as f64;
        let avg = sum / branches;
        let table = out.entry(parent_pack).or_default();
        match table.get_mut(&parent_id) {
            Some(existing) => {
                if avg > *existing {
                    *existing = avg;
                }
            }
            None => {
                table.insert(parent_id, avg);
            }
        }
    }
    out
}

/// One candidate placement (a DAG edge) at an analysis node, with its expected value.
#[derive(Clone, Debug)]
pub struct AnalysisCand {
    pub edge: usize,        // index into the parent's edge list (stable selector for navigation)
    pub placed: Piece,      // the tetromino this placement drops
    pub hold_after: Piece,  // hold after the move
    pub field_before: u64,
    pub field_after: u64,
    pub score: f64,         // expected consecutive PCs from here (0 = dead line for this reveal)
    pub best: bool,         // the policy's argmax move
}

/// A navigable analysis position: the current node plus its ranked candidate moves, the line
/// taken to reach it, and the valid reveal options (for reveal what-if).
pub struct AnalysisNode {
    pub depth: u8,
    pub field: u64,
    pub hold: Piece,
    pub active: Piece,      // piece placed if NOT swapping hold, at this depth
    pub terminal: u8,       // 0=in-progress 1=4LPC 2=2LPC 3=dead (no positive line for this reveal)
    pub best_score: f64,
    pub root_value: f64,
    pub path_steps: Vec<AnalysisCand>, // the chosen line root->here (board reconstruction/breadcrumb)
    pub cands: Vec<AnalysisCand>,      // candidates at the current node, sorted by score desc
    pub reveal_options: [Vec<Piece>; 4], // valid pieces for h1..h4 (bag process)
    pub visible: [Piece; 6],
}

impl VsResult {
    pub fn two_line_field(&self) -> u64 {
        self.retained.two_line_field
    }

    /// Expected value of a depth-d child under the information available before its
    /// placement (h1..h_{d-1} revealed; averages over the rest).
    fn child_score(&self, child_id: NodeId, child_depth: u8, hidden: &[Piece; 4], leaf: usize) -> f64 {
        let r = &self.retained;
        match child_depth {
            1..=3 => {
                // folded tables keyed by len-child_depth prefix; the last prefix piece
                // h_{child_depth} is not yet revealed at decision time -> average it.
                let known = child_depth - 1;
                let mut prefix = 0u16;
                for i in 0..known {
                    prefix = set_hidden(prefix, i, hidden[i as usize]);
                }
                let mut mask = r.initial_mask;
                for i in 0..known {
                    mask = after_reveal(mask, hidden[i as usize]);
                }
                let mut sum = 0.0;
                let mut n = 0.0;
                for h in pieces(mask) {
                    let pack = set_hidden(prefix, known, h);
                    if let Some(v) = r.folds[child_depth as usize]
                        .get(&pack)
                        .and_then(|t| t.get(&child_id))
                    {
                        sum += v;
                    }
                    n += 1.0;
                }
                if n == 0.0 { 0.0 } else { sum / n }
            }
            4..=10 => {
                let key = *r.dag.key(child_id);
                if key.depth <= 6 {
                    return r.vals[key.depth as usize]
                        .get(child_id)
                        .map(|v| dequant(v[leaf]))
                        .unwrap_or(0.0);
                }
                // Merged node: its vector is indexed by suffix-rank. Rebuild the reveal prefix
                // from `hidden`; a mask mismatch means this child is on a different reveal branch.
                let plen = key.depth - 6;
                let mut pack = 0u16;
                let mut m = r.initial_mask;
                for i in 0..plen {
                    pack = set_hidden(pack, i, hidden[i as usize]);
                    m = after_reveal(m, hidden[i as usize]);
                }
                if m != key.mask {
                    return 0.0;
                }
                let block = r.ranges[&(plen, pack)];
                r.vals[key.depth as usize]
                    .get(child_id)
                    .map(|v| dequant(v[leaf - block.start as usize]))
                    .unwrap_or(0.0)
            }
            _ => 0.0,
        }
    }

    /// Decision-time expected value of a child, respecting reveal timing. For a depth-3 node
    /// (child_depth 4) the piece h4 opens only AS A RESULT of that placement, so it is NOT known
    /// when the move is chosen — average over it (like the folded tables do for h1..h3), rather
    /// than scoring against the one realized h4. Otherwise (depths 0..2 fold h1..h3; depths 4+
    /// have h4 already revealed) use the ordinary child_score.
    fn decision_score(&self, child: NodeId, child_depth: u8, hidden: &[Piece; 4]) -> f64 {
        let r = &self.retained;
        if child_depth == 4 {
            let mut prefix = 0u16;
            for i in 0..3 {
                prefix = set_hidden(prefix, i, hidden[i as usize]);
            }
            let mask = mask_after_hidden_prefix(r.initial_mask, prefix, 3);
            let mut sum = 0.0;
            let mut n = 0.0;
            for h4 in pieces(mask) {
                let mut hid = *hidden;
                hid[3] = h4;
                if let Some(lf) = self.full_leaf(hid) {
                    sum += self.child_score(child, 4, &hid, lf);
                    n += 1.0;
                }
            }
            if n > 0.0 { sum / n } else { 0.0 }
        } else {
            let leaf = self.full_leaf(*hidden).unwrap_or(0);
            self.child_score(child, child_depth, hidden, leaf)
        }
    }

    /// Leaf index of the full reveal sequence `hidden` (None if not a valid bag sequence).
    pub fn full_leaf(&self, hidden: [Piece; 4]) -> Option<usize> {
        let mut pack = 0u16;
        for (i, &h) in hidden.iter().enumerate() {
            pack = set_hidden(pack, i as u8, h);
        }
        self.retained.ranges.get(&(4, pack)).map(|r| r.start as usize)
    }

    /// Navigate the loop DAG: replay `path` (edge indices from the root) under the reveal
    /// sequence `hidden`, and report the resulting node with its ranked candidate moves. This
    /// is the analysis primitive — every alternative placement is a candidate, and changing
    /// `hidden` is the reveal what-if. All scores are the decision-time expected value.
    pub fn analyze(&self, path: &[usize], hidden: [Piece; 4]) -> AnalysisNode {
        let r = &self.retained;
        let leaf = self.full_leaf(hidden).unwrap_or(0);

        let _ = leaf;
        let make_cand = |edge: usize, parent: NodeId, child: NodeId, child_depth: u8| -> AnalysisCand {
            let pk = *r.dag.key(parent);
            let ck = *r.dag.key(child);
            let active = active_piece(&r.visible, &hidden, pk.depth);
            let placed = if ck.hold == pk.hold { active } else { pk.hold };
            let score = self.decision_score(child, child_depth, &hidden);
            AnalysisCand {
                edge, placed, hold_after: ck.hold,
                field_before: pk.field, field_after: ck.field, score, best: false,
            }
        };

        // Replay the chosen line, recording each step.
        let mut node = r.dag.root;
        let mut path_steps = Vec::new();
        for &e in path {
            let edges = r.dag.edges(node);
            if e >= edges.len() {
                break;
            }
            let child = edges[e];
            let cd = r.dag.key(node).depth + 1;
            path_steps.push(make_cand(e, node, child, cd));
            node = child;
        }

        let key = *r.dag.key(node);
        let depth = key.depth;
        let active = active_piece(&r.visible, &hidden, depth);

        let mut cands = Vec::new();
        for (ei, &child) in r.dag.edges(node).iter().enumerate() {
            // At reveal depths (>=6) the DAG branches over EVERY piece that COULD be revealed here;
            // once the reveal is known (the passed `hidden`), the other branches are impossible, so
            // prune every edge whose reveal disagrees with the actual hidden. Under transposition
            // merging the reveal is identified by the child's mask (unique per reveal from a node).
            if depth >= 6 {
                let idx = (depth - 6) as usize;
                if r.dag.key(child).mask != after_reveal(key.mask, hidden[idx]) {
                    continue;
                }
            }
            cands.push(make_cand(ei, node, child, depth + 1));
        }
        let mut best_pos = usize::MAX;
        let mut best_s = 0.0f64;
        for (i, c) in cands.iter().enumerate() {
            if c.score > best_s {
                best_s = c.score;
                best_pos = i;
            }
        }
        if best_pos != usize::MAX {
            cands[best_pos].best = true;
        }
        cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let terminal = if key.field == r.two_line_field && depth == 5 {
            2
        } else if depth == 10 {
            1
        } else if cands.is_empty() || best_s <= 0.0 {
            3
        } else {
            0
        };

        let opts = |k: u8| -> Vec<Piece> {
            let mut p = 0u16;
            for i in 0..k {
                p = set_hidden(p, i, hidden[i as usize]);
            }
            pieces_in_mask(mask_after_hidden_prefix(r.initial_mask, p, k))
        };
        let reveal_options = [opts(0), opts(1), opts(2), opts(3)];

        AnalysisNode {
            depth, field: key.field, hold: key.hold, active, terminal,
            best_score: best_s, root_value: self.root_value,
            path_steps, cands, reveal_options, visible: r.visible,
        }
    }
}

fn active_piece(visible: &[Piece; 6], hidden: &[Piece; 4], depth: u8) -> Piece {
    if depth < 6 {
        visible[depth as usize]
    } else {
        // depth 6..9 place h1..h4; clamp so a terminal (depth 10) node can't index out of bounds.
        hidden[((depth - 6) as usize).min(3)]
    }
}

/* -------------------------------------------------------------------------- */
/* Build / prune (adapted from fh_search)                                      */
/* -------------------------------------------------------------------------- */

fn build_dag(
    graph: Option<&HydraGraph>,
    edge_ids: Option<&(dyn Fn(u64, u8, &mut Vec<u64>) + '_)>,
    par_edge: Option<&(dyn Fn(u64, u8, &mut Vec<u64>) + Sync + '_)>,
    hold: Piece,
    visible: [Piece; 6],
    initial_mask: u8,
    two_line_field: u64,
) -> Dag {
    #[cfg(target_arch = "wasm32")]
    let _ = &par_edge; // parallel build is native-only; keep the signature stable across targets
    let mut dag = Dag {
        keys: Vec::new(),
        edge_start: Vec::new(),
        edge_data: Vec::new(),
        layers: vec![Vec::new(); 11],
        index: HashMap::new(),
        root: 0,
    };
    let root = get_or_add_node(
        &mut dag,
        NodeKey { depth: 0, field: ROOT_HASH, hold, mask: initial_mask },
    );
    dag.root = root;

    // Serial fetch (graph-free closure OR graph.edges); parallel uses the Sync `par_edge` instead.
    let serial_fetch = |field: u64, piece: u8, buf: &mut Vec<u64>| {
        match edge_ids {
            Some(f) => f(field, piece, buf),
            None => {
                let g = graph.expect("graph required when edge_ids is None");
                let fid = g.hash_lookup(field).expect("node field hash not in graph");
                for &c in g.edges(fid, piece) { buf.push(g.hash(c)); }
            }
        }
    };

    // Reused scratch (cleared per use); no edge sort/dedup (duplicates structurally impossible).
    let mut kids: Vec<u64> = Vec::new();
    let mut hold_kids: Vec<u64> = Vec::new();
    let mut triples: Vec<(u64, Piece, u8)> = Vec::new();
    for depth in 0..10u8 {
        let child_depth = depth + 1;
        let frontier = dag.layers[depth as usize].clone();
        // PARALLEL BUILD (native only): Phase A (rayon) computes each frontier node's child triples
        // via the Sync edge source (movegen is the cost, read-only); Phase B inserts them SERIALLY
        // in the identical frontier/child order, so the DAG is bit-identical to the serial build.
        // On wasm par_edge is always None and rayon is not linked, so the branch is cfg'd out.
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(pe) = par_edge {
            let all: Vec<Vec<(u64, Piece, u8)>> = frontier
                .par_iter()
                .map(|&id| {
                    let key = *dag.key(id);
                    let mut k = Vec::new();
                    let mut h = Vec::new();
                    let mut o = Vec::new();
                    node_child_triples(key, &visible, two_line_field, &|f, p, b| pe(f, p, b), &mut k, &mut h, &mut o);
                    o
                })
                .collect();
            for (&id, tr) in frontier.iter().zip(all.iter()) {
                dag.begin_edges(id);
                for &(nf, nh, nm) in tr {
                    let c = get_or_add_node(&mut dag, NodeKey { depth: child_depth, field: nf, hold: nh, mask: nm });
                    dag.edge_data.push(c);
                }
            }
            continue;
        }
        // Serial build (the only path on wasm).
        {
            for id in frontier {
                let key = *dag.key(id);
                node_child_triples(key, &visible, two_line_field, &serial_fetch, &mut kids, &mut hold_kids, &mut triples);
                dag.begin_edges(id);
                for &(nf, nh, nm) in &triples {
                    let c = get_or_add_node(&mut dag, NodeKey { depth: child_depth, field: nf, hold: nh, mask: nm });
                    dag.edge_data.push(c);
                }
            }
        }
    }
    dag.seal();
    dag
}

/// Compute a node's ordered child triples (child field hash, new hold, child remaining-mask) by
/// placing the active/hold pieces (depth<6) or every reveal from the remaining bag (depth>=6).
/// `fetch(field, piece, buf)` APPENDS child field hashes into `buf` (caller-cleared). Order is
/// fixed and matches the serial build so parallel and serial produce identical DAGs.
fn node_child_triples(
    key: NodeKey,
    visible: &[Piece; 6],
    two_line_field: u64,
    fetch: &dyn Fn(u64, u8, &mut Vec<u64>),
    kids: &mut Vec<u64>,
    hold_kids: &mut Vec<u64>,
    out: &mut Vec<(u64, Piece, u8)>,
) {
    out.clear();
    let depth = key.depth;
    if depth == 5 && key.field == two_line_field {
        return; // 2LPC terminal: no children.
    }
    if depth < 6 {
        let active = visible[depth as usize];
        kids.clear();
        fetch(key.field, active as u8, kids);
        for &nf in kids.iter() { out.push((nf, key.hold, key.mask)); }
        if active != key.hold {
            hold_kids.clear();
            fetch(key.field, key.hold as u8, hold_kids);
            for &nf in hold_kids.iter() { out.push((nf, active, key.mask)); }
        }
    } else {
        // Hoist the hold-placement children: (field,hold) is IDENTICAL across this node's reveal
        // branches — fetch once, rewire per branch (only the mask differs).
        hold_kids.clear();
        fetch(key.field, key.hold as u8, hold_kids);
        for p in pieces_in_mask(key.mask) {
            let child_mask = after_reveal(key.mask, p);
            kids.clear();
            fetch(key.field, p as u8, kids);
            for &nf in kids.iter() { out.push((nf, key.hold, child_mask)); }
            if p != key.hold {
                for &nf in hold_kids.iter() { out.push((nf, p, child_mask)); }
            }
        }
    }
}

fn get_or_add_node(dag: &mut Dag, key: NodeKey) -> NodeId {
    let pk = pack_key(&key);
    if let Some(&id) = dag.index.get(&pk) {
        return id;
    }
    let id = dag.keys.len() as NodeId;
    dag.keys.push(key);
    dag.index.insert(pk, id);
    dag.layers[key.depth as usize].push(id);
    id
}

fn prune_to_terminal_reachable(dag: &Dag, two_line_field: u64, par: bool) -> Dag {
    // Layered backward sweep (edges go depth -> depth+1, so children are finalized before their
    // parents): a node survives iff it IS a terminal or ANY child survives. Same set as the old
    // reverse-index BFS, but with no reverse index to build, and each layer is parallel-safe.
    let mut marked = vec![false; dag.keys.len()];
    for depth in (0..=10usize).rev() {
        let layer = &dag.layers[depth];
        let mark_of = |id: NodeId, marked: &[bool]| -> bool {
            let key = dag.key(id);
            if depth == 10 {
                key.field == TERMINAL_HASH
            } else if depth == 5 && key.field == two_line_field {
                true
            } else {
                dag.edges(id).iter().any(|&c| marked[c as usize])
            }
        };
        let res: Vec<bool> = par_or_serial!(par, layer, map, |&id| mark_of(id, &marked));
        for (&id, m) in layer.iter().zip(res) {
            marked[id as usize] = m;
        }
    }

    let mut old_to_new = vec![NodeId::MAX; dag.keys.len()];
    let mut keys = Vec::new();
    let mut layers = vec![Vec::new(); 11];
    for old_id in 0..dag.keys.len() {
        if !marked[old_id] {
            continue;
        }
        let new_id = keys.len() as NodeId;
        old_to_new[old_id] = new_id;
        let key = dag.keys[old_id];
        keys.push(key);
        layers[key.depth as usize].push(new_id);
    }
    // Remap edges into the new CSR in one ascending pass (old ids ascend -> new ids ascend, so
    // layer contiguity is preserved); dup-free, original order.
    let mut edge_start = Vec::with_capacity(keys.len() + 1);
    let mut edge_data = Vec::new();
    for old_id in 0..dag.keys.len() {
        if !marked[old_id] {
            continue;
        }
        edge_start.push(edge_data.len() as u32);
        for &c in dag.edges(old_id as NodeId) {
            if marked[c as usize] {
                edge_data.push(old_to_new[c as usize]);
            }
        }
    }
    edge_start.push(edge_data.len() as u32);
    let root = old_to_new[dag.root as usize];
    // NOTE: the pruned DAG's `index` is never read again (get_or_add_node runs only during build),
    // so we skip rebuilding it — saves ~1.1M hashmap inserts per boundary.
    Dag { keys, edge_start, edge_data, layers, index: HashMap::new(), root }
}

/* -------------------------------------------------------------------------- */
/* Hidden-sequence utilities (same semantics as fh_search)                     */
/* -------------------------------------------------------------------------- */

fn build_ranges(
    mask: u8,
    idx: u8,
    pack: u16,
    next_leaf: &mut u32,
    out: &mut HashMap<(u8, u16), SeqRange>,
) -> u32 {
    if idx == 4 {
        out.insert((4, pack), SeqRange { start: *next_leaf, len: 1 });
        *next_leaf += 1;
        return 1;
    }
    let start = *next_leaf;
    let mut total = 0u32;
    for p in pieces_in_mask(canonical_mask(mask)) {
        total += build_ranges(after_reveal(mask, p), idx + 1, set_hidden(pack, idx, p), next_leaf, out);
    }
    out.insert((idx, pack), SeqRange { start, len: total });
    total
}

fn canonical_mask(mask: u8) -> u8 {
    let m = mask & FULL_MASK;
    if m == 0 { FULL_MASK } else { m }
}

fn mask_after_hidden_prefix(initial_mask: u8, pack: u16, len: u8) -> u8 {
    let mut mask = canonical_mask(initial_mask);
    for i in 0..len {
        mask = after_reveal(mask, get_hidden(pack, i));
    }
    mask
}

fn pieces_in_mask(mask: u8) -> Vec<Piece> {
    let m = canonical_mask(mask);
    let mut out = Vec::new();
    for p in 0..PIECE_COUNT as u8 {
        if (m & (1u8 << p)) != 0 {
            out.push(p);
        }
    }
    out
}

fn set_hidden(pack: u16, idx: u8, p: Piece) -> u16 {
    debug_assert!(idx < 4);
    let shift = (idx as u16) * 3;
    (pack & !(0b111u16 << shift)) | ((p as u16) << shift)
}

fn get_hidden(pack: u16, idx: u8) -> Piece {
    let shift = (idx as u16) * 3;
    ((pack >> shift) & 0b111) as Piece
}

fn prefix_pack(pack: u16, len: u8) -> u16 {
    if len == 0 {
        return 0;
    }
    pack & ((1u16 << (3 * len)) - 1)
}

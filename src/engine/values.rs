//! Reset-boundary value table: layer0 one-loop u16 values keyed by ORDERED chain-state
//! key (hold<<24 | bag_mask<<17 | queue6), exported by the MDP value pipeline's
//! `export-l0-ordered-u16`. A reset state's value contribution is 1 + v/5040
//! ("this PC" + probability of the next PC from the reset state).

use anyhow::{bail, Context, Result};
use hashbrown::HashMap;
use std::fs;
use std::path::Path;

use crate::piece::{after_reveal, pieces, Piece, FULL_BAG};

pub const VALUE_SCALE: f64 = 5040.0;

/* ---- layer0 chain-state enumeration (matches the value-table generator's ordering) --------- *
 * The keyless quantized value file (L0Q12V1) stores values in ASCENDING state_key order with no
 * keys; the loader regenerates the identical sorted key set here, so index i in the file is the
 * value for keys[i]. state_key = hold<<24 | bag_mask<<17 | queue6 (queue6 little-endian base-7). */

fn gen_prefixes(len: usize, cur: &mut Vec<u8>, used: u8, out: &mut Vec<Vec<u8>>) {
    if cur.len() == len {
        out.push(cur.clone());
        return;
    }
    for p in 0u8..7 {
        let bit = 1u8 << p;
        if used & bit != 0 {
            continue;
        }
        cur.push(p);
        gen_prefixes(len, cur, used | bit, out);
        cur.pop();
    }
}

fn gen_suffixes_from_mask(mask: u8, cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
    if cur.len() == mask.count_ones() as usize {
        out.push(cur.clone());
        return;
    }
    for p in pieces(mask) {
        if cur.contains(&p) {
            continue;
        }
        cur.push(p);
        gen_suffixes_from_mask(mask, cur, out);
        cur.pop();
    }
}

/// All 1,120,140 valid layer0 chain-state keys, ascending (matches the keyed export's sort order).
fn enumerate_sorted_keys() -> Vec<u32> {
    let mut prefix_cache: Vec<Vec<Vec<u8>>> = vec![Vec::new(); 7];
    for len in 0..=6usize {
        gen_prefixes(len, &mut Vec::new(), 0, &mut prefix_cache[len]);
    }
    let mut suffix_cache: Vec<Vec<Vec<u8>>> = vec![Vec::new(); 128];
    for mask in 0u8..128 {
        gen_suffixes_from_mask(mask, &mut Vec::new(), &mut suffix_cache[mask as usize]);
    }
    let mut keys: Vec<u32> = Vec::with_capacity(1_120_140);
    for bag_mask in 1u8..128 {
        let k = bag_mask.count_ones() as usize;
        let prefix_len = k - 1;
        let suffix_mask = FULL_BAG ^ bag_mask;
        for prefix in &prefix_cache[prefix_len] {
            for suffix in &suffix_cache[suffix_mask as usize] {
                let mut q = [0u8; 6];
                for (i, &p) in prefix.iter().enumerate() {
                    q[i] = p;
                }
                for (j, &p) in suffix.iter().enumerate() {
                    q[prefix_len + j] = p;
                }
                let mut queue6 = 0u32;
                let mut mul = 1u32;
                for &p in &q {
                    queue6 += p as u32 * mul;
                    mul *= 7;
                }
                for hold in 0u8..7 {
                    keys.push(((hold as u32) << 24) | ((bag_mask as u32) << 17) | queue6);
                }
            }
        }
    }
    keys.sort_unstable();
    keys
}

/// Chain-state key matching the value-table generator's `state_key(encode_queue6(..))`.
/// NOTE: the generator's encode_queue6 is little-endian in the queue (q[0] is the least
/// significant base-7 digit), so the reset key is built explicitly here to match the file order.
#[inline]
fn boundary_key(hold: Piece, queue6: [Piece; 6], bag_mask: u8) -> u32 {
    let mut q = 0u32;
    let mut mul = 1u32;
    for &p in &queue6 {
        q += p as u32 * mul;
        mul *= 7;
    }
    ((hold as u32) << 24) | ((bag_mask as u32) << 17) | q
}

/// Boundary reset value table keyed by ordered chain-state key. Two source encodings:
///   U16 (b"L0U16V1\0"): one-loop PC counts /5040 in [0,1]; reset contributes 1 + v/5040.
///   F32 (b"L0F32V1\0"): a full boundary V (e.g. iter0); reset contributes 1 + v.
/// Both yield "this PC (=1) + the value of the state we reset into".
pub struct ValueTable {
    keys: Vec<u32>,
    /// Value already normalized to the reset CONTRIBUTION BEYOND the +1: v/5040 for u16,
    /// v for f32. reset_value adds the +1.
    contrib: Vec<f32>,
}

impl ValueTable {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("in {}", path.display()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            bail!("value table too short");
        }
        let magic = &bytes[0..8];
        // For keyed formats bytes[8..16] is the record count; for L0Q12V1 it's part of the
        // min/max header, so defer the count read into that branch.
        let count = if magic == b"L0Q12V1\0" { 0 } else { u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize };
        let mut keys = Vec::with_capacity(count);
        let mut contrib = Vec::with_capacity(count);
        if magic == b"L0U16V1\0" {
            if bytes.len() != 16 + count * 6 {
                bail!("u16 value table size mismatch: {} bytes for {} records", bytes.len(), count);
            }
            for i in 0..count {
                let off = 16 + i * 6;
                keys.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                let raw = u16::from_le_bytes(bytes[off + 4..off + 6].try_into().unwrap());
                contrib.push(raw as f32 / VALUE_SCALE as f32);
            }
        } else if magic == b"L0F32V1\0" {
            if bytes.len() != 16 + count * 8 {
                bail!("f32 value table size mismatch: {} bytes for {} records", bytes.len(), count);
            }
            for i in 0..count {
                let off = 16 + i * 8;
                keys.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                contrib.push(f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()));
            }
        } else if magic == b"L0Q12V1\0" {
            // Keyless 12-bit-quantized V*: magic + f32 min + f32 max + u64 count + packed 12-bit
            // values (2 per 3 bytes) in ascending-key order. Keys are regenerated, not stored.
            let mn = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
            let mx = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
            let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
            let packed = (count + 1) / 2 * 3;
            if bytes.len() != 24 + packed {
                bail!("q12 size mismatch: {} bytes for {} values", bytes.len(), count);
            }
            let span = (mx - mn) / 4095.0;
            keys = enumerate_sorted_keys();
            if keys.len() != count {
                bail!("q12 count {} != enumerated key count {}", count, keys.len());
            }
            contrib = Vec::with_capacity(count);
            let body = &bytes[24..];
            for i in 0..count {
                let (byte, lo) = (i / 2 * 3, i % 2);
                let q = if lo == 0 {
                    (body[byte] as u16) | (((body[byte + 1] & 0x0F) as u16) << 8)
                } else {
                    (((body[byte + 1] >> 4) & 0x0F) as u16) | ((body[byte + 2] as u16) << 4)
                };
                contrib.push(mn + q as f32 * span);
            }
        } else {
            bail!("unknown value table magic");
        }
        debug_assert!(keys.windows(2).all(|w| w[0] < w[1]));
        Ok(ValueTable { keys, contrib })
    }

    #[inline]
    pub fn get(&self, key: u32) -> Option<f32> {
        self.keys.binary_search(&key).ok().map(|idx| self.contrib[idx])
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Terminal evaluator over reset boundary states. `table = None` is FLAT mode: every
/// completed PC is worth exactly 1.0 (the search then reproduces the plain one-loop PC
/// probability bit-for-bit in expectation — the fh_search compatibility check).
pub struct ResetEval<'a> {
    pub table: Option<&'a ValueTable>,
    memo_w4: HashMap<(Piece, u8), f64>,
    pub missing_keys: u64,
}

impl<'a> ResetEval<'a> {
    pub fn new(table: Option<&'a ValueTable>) -> Self {
        ResetEval {
            table,
            memo_w4: HashMap::new(),
            missing_keys: 0,
        }
    }

    #[inline]
    fn reset_value(&mut self, hold: Piece, queue6: [Piece; 6], bag_mask: u8) -> f64 {
        let Some(table) = self.table else {
            return 1.0;
        };
        let key = boundary_key(hold, queue6, bag_mask);
        match table.get(key) {
            Some(v) => 1.0 + v as f64,
            None => {
                // Should be unreachable for reveal-consistent reset states; count and treat
                // the reset as dead so the error is visible, never silently optimistic.
                self.missing_keys += 1;
                0.0
            }
        }
    }

    /// 4LPC terminal (depth 10): the reset queue h5..h10 is entirely unrevealed at loop
    /// end, so average the reset value over every reveal completion from `mask4` (the bag
    /// state after the four in-loop reveals). Memoized on (hold, mask4): at most 7*127
    /// entries, each <= 5040 sequences.
    pub fn w4(&mut self, hold: Piece, mask4: u8) -> f64 {
        if self.table.is_none() {
            return 1.0;
        }
        if let Some(&v) = self.memo_w4.get(&(hold, mask4)) {
            return v;
        }
        let mut queue = [0u8; 6];
        let (sum, count) = self.w4_rec(hold, mask4, 0, &mut queue);
        let v = if count == 0 { 0.0 } else { sum / count as f64 };
        self.memo_w4.insert((hold, mask4), v);
        v
    }

    fn w4_rec(&mut self, hold: Piece, mask: u8, depth: usize, queue: &mut [u8; 6]) -> (f64, u64) {
        if depth == 6 {
            return (self.reset_value(hold, *queue, mask), 1);
        }
        let mut sum = 0.0;
        let mut count = 0u64;
        for p in pieces(mask) {
            queue[depth] = p;
            let (s, c) = self.w4_rec(hold, after_reveal(mask, p), depth + 1, queue);
            sum += s;
            count += c;
        }
        (sum, count)
    }

    /// 2LPC terminal (depth 5): the reset queue is [q5, h1..h5] where q5 and the in-loop
    /// reveals h1..h4 are known at the node and only h5 is unrevealed; average over h5.
    /// `mask4` = bag state after h1..h4.
    pub fn w2(&mut self, hold: Piece, q5: Piece, h: [Piece; 4], mask4: u8) -> f64 {
        if self.table.is_none() {
            return 1.0;
        }
        let mut sum = 0.0;
        let mut count = 0u64;
        for h5 in pieces(mask4) {
            let queue = [q5, h[0], h[1], h[2], h[3], h5];
            sum += self.reset_value(hold, queue, after_reveal(mask4, h5));
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

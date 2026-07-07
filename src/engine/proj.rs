//! Projection-bitset PC-ability filter (the bot's membership oracle).
//!
//! Base filter `PCPRJ11`: 11 structured 20-bit projections of the 40-bit field, 1.44 MB.
//! `maybe(x)` passes iff popcount%4==0 AND every projection bit is set — NO false negatives
//! (validated against the 10.7M-field PC-able source at build time), few false positives.
//!
//! Optional `PCPRJX1` extras: additional 20-bit projections ANDed after the base filter to trim
//! more false positives (the shipped bot attaches one — center columns — for ~5% faster search).

pub struct ProjFilter {
    pub data: Vec<u8>,
    /// EXTRA 20-bit projections (PCPRJX1), ANDed after the base filter. Each = (per-row 10-bit
    /// column masks [m0..m3] totalling 20 set bits, 2^20-bit set).
    extra: Vec<([u16; 4], Vec<u8>)>,
}

pub const PROJ_BYTES: usize = (1 << 20) / 8;
pub const PROJ_FILE_LEN: usize = 16 + 11 * PROJ_BYTES;

impl ProjFilter {
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        if data.len() != PROJ_FILE_LEN || &data[0..8] != b"PCPRJ11\0" {
            return Err(format!("bad proj filter: len {} magic {:?}", data.len(), &data[..8.min(data.len())]));
        }
        Ok(ProjFilter { data, extra: Vec::new() })
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::from_bytes(std::fs::read(path)?)?)
    }

    /// Attach EXTRA 20-bit projections from a `PCPRJX1` blob (magic + u32 n + u32 pad +
    /// n*[u16;4] descriptors + n*128KB bitsets). ANDed after the base filter in `maybe`.
    pub fn attach_extra(&mut self, data: &[u8]) -> Result<usize, String> {
        if data.len() < 16 || &data[0..8] != b"PCPRJX1\0" {
            return Err("bad PCPRJX1 magic".into());
        }
        let n = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let desc_off = 16;
        let bits_off = desc_off + n * 8;
        if data.len() != bits_off + n * PROJ_BYTES {
            return Err(format!("bad PCPRJX1 len {} (n={})", data.len(), n));
        }
        for i in 0..n {
            let d = &data[desc_off + i * 8..desc_off + i * 8 + 8];
            let masks = [
                u16::from_le_bytes([d[0], d[1]]),
                u16::from_le_bytes([d[2], d[3]]),
                u16::from_le_bytes([d[4], d[5]]),
                u16::from_le_bytes([d[6], d[7]]),
            ];
            let b = &data[bits_off + i * PROJ_BYTES..bits_off + (i + 1) * PROJ_BYTES];
            self.extra.push((masks, b.to_vec()));
        }
        Ok(n)
    }

    #[inline]
    fn bitget(&self, proj: usize, v: u32) -> bool {
        self.data[16 + proj * PROJ_BYTES + (v as usize >> 3)] & (1u8 << (v & 7)) != 0
    }

    #[inline]
    fn extra_pass(&self, r0: u32, r1: u32, r2: u32, r3: u32) -> bool {
        for (m, bits) in &self.extra {
            let compact = |r: u32, mask: u16| -> u32 {
                let mut out = 0u32;
                let mut j = 0u32;
                let mut b = 0u32;
                while b < 10 {
                    if (mask >> b) & 1 != 0 {
                        out |= ((r >> b) & 1) << j;
                        j += 1;
                    }
                    b += 1;
                }
                out
            };
            let mut v = compact(r0, m[0]);
            let mut sh = m[0].count_ones();
            v |= compact(r1, m[1]) << sh;
            sh += m[1].count_ones();
            v |= compact(r2, m[2]) << sh;
            sh += m[2].count_ones();
            v |= compact(r3, m[3]) << sh;
            if bits[v as usize >> 3] & (1u8 << (v & 7)) == 0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn maybe(&self, x: u64) -> bool {
        if x >= (1u64 << 40) || (x.count_ones() & 3) != 0 {
            return false;
        }
        let (r0, r1, r2, r3) = ((x & 1023) as u32, ((x >> 10) & 1023) as u32, ((x >> 20) & 1023) as u32, ((x >> 30) & 1023) as u32);
        let c5 = |r: u32, b: [u32; 5]| ((r >> b[0]) & 1) | (((r >> b[1]) & 1) << 1) | (((r >> b[2]) & 1) << 2) | (((r >> b[3]) & 1) << 3) | (((r >> b[4]) & 1) << 4);
        let cp = |b: [u32; 5]| c5(r0, b) | (c5(r1, b) << 5) | (c5(r2, b) << 10) | (c5(r3, b) << 15);
        // projection order matches the filter file (density-ascending)
        self.bitget(0, cp([5, 6, 7, 8, 9]))
            && self.bitget(1, cp([0, 1, 2, 3, 4]))
            && self.bitget(2, cp([2, 3, 4, 5, 6]))
            && self.bitget(3, r2 | (r3 << 10))
            && self.bitget(4, cp([0, 3, 4, 5, 9]))
            && self.bitget(5, r0 | (r3 << 10))
            && self.bitget(6, r0 | (r1 << 10))
            && self.bitget(7, r1 | (r3 << 10))
            && self.bitget(8, cp([0, 2, 4, 6, 8]))
            && self.bitget(9, r1 | (r2 << 10))
            && self.bitget(10, r0 | (r2 << 10))
            && (self.extra.is_empty() || self.extra_pass(r0, r1, r2, r3))
    }
}

//! TBP piece-coordinate conversion, EXACT to Cold Clear / libtetris (the reference jstris expects).
//!
//! Cold Clear's TBP frontend sends `libtetris::FallingPiece { x, y }` verbatim, where (x,y) is the
//! position of the cell with offset (0,0) in libtetris' cell table, and the orientation is
//! CANONICALIZED (O -> north only; S/Z/I -> north or west only; T/J/L keep all four). We reproduce
//! that exactly by matching a placement's 4 SCREEN cells against the canonical orientations.
//!
//! Orientation index here: 0=north, 1=east, 2=south, 3=west (matches ORIENT_NAMES / the TBP string).
//! Piece index: IJLOSTZ = 0..6 (crate::piece).

use crate::piece::Piece;

// libtetris North offsets (dx, dy; dy up), REORDERED to IJLOSTZ. Source (libtetris/src/piece.rs):
//   I:(-1,0)(0,0)(1,0)(2,0)  O:(0,0)(1,0)(0,1)(1,1)  T:(-1,0)(0,0)(1,0)(0,1)
//   L:(-1,0)(0,0)(1,0)(1,1)  J:(-1,0)(0,0)(1,0)(-1,1) S:(-1,0)(0,0)(0,1)(1,1)
//   Z:(-1,1)(0,1)(0,0)(1,0)
const NORTH: [[(i8, i8); 4]; 7] = [
    [(-1, 0), (0, 0), (1, 0), (2, 0)],  // I
    [(-1, 0), (0, 0), (1, 0), (-1, 1)], // J
    [(-1, 0), (0, 0), (1, 0), (1, 1)],  // L
    [(0, 0), (1, 0), (0, 1), (1, 1)],   // O
    [(-1, 0), (0, 0), (0, 1), (1, 1)],  // S
    [(-1, 0), (0, 0), (1, 0), (0, 1)],  // T
    [(-1, 1), (0, 1), (0, 0), (1, 0)],  // Z
];

/// libtetris rotation of a North offset: north=(x,y) east=(y,-x) south=(-x,-y) west=(-y,x).
#[inline]
fn rot((x, y): (i8, i8), orient: u8) -> (i8, i8) {
    match orient {
        0 => (x, y),
        1 => (y, -x),
        2 => (-x, -y),
        _ => (-y, x),
    }
}

/// Which orientations Cold Clear ever emits for a piece (its canonical set).
#[inline]
fn canonical_orients(piece: Piece) -> &'static [u8] {
    match piece {
        3 => &[0],          // O: north only
        0 | 4 | 6 => &[0, 3], // I, S, Z: north or west
        _ => &[0, 1, 2, 3], // T, J, L: all four
    }
}

/// Encode 4 SCREEN cells (col from left, row from bottom, PHYSICAL board) of a `piece` placement
/// into (orientation 0..3, x, y) in the exact Cold Clear / libtetris convention. Returns None if
/// the cells don't form a valid orientation of the piece (shouldn't happen for a real placement).
pub fn encode(piece: Piece, cells: &[(i8, i8); 4]) -> Option<(u8, i8, i8)> {
    let mut want = *cells;
    want.sort_unstable();
    for &o in canonical_orients(piece) {
        let offs = NORTH[piece as usize].map(|off| rot(off, o));
        // the (0,0) offset lands on the anchor (x,y); try each cell as that anchor.
        for &(ax, ay) in cells {
            let mut got = offs.map(|(dx, dy)| (ax + dx, ay + dy));
            got.sort_unstable();
            if got == want {
                return Some((o, ax, ay));
            }
        }
    }
    None
}

/// Decode (orientation, x, y) back to 4 SCREEN cells — inverse of `encode`, for validation.
pub fn decode(piece: Piece, orient: u8, x: i8, y: i8) -> [(i8, i8); 4] {
    NORTH[piece as usize].map(|off| {
        let (dx, dy) = rot(off, orient);
        (x + dx, y + dy)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every canonical (piece, orientation) at a range of anchors must round-trip
    // decode -> encode -> same (orientation, anchor). Catches offset/rotation typos.
    #[test]
    fn roundtrip_canonical() {
        for piece in 0u8..7 {
            for &o in canonical_orients(piece) {
                for x in 2..8i8 {
                    for y in 2..8i8 {
                        let cells = decode(piece, o, x, y);
                        let got = encode(piece, &cells).expect("must encode");
                        assert_eq!(got, (o, x, y), "piece {} orient {} anchor ({},{})", piece, o, x, y);
                    }
                }
            }
        }
    }

    // A NON-canonical orientation's cells must encode to its canonical twin, never to itself.
    // (S/Z/I: east->west, south->north; O: everything->north.)
    #[test]
    fn noncanonical_maps_to_canonical() {
        for piece in [0u8, 3, 4, 6] {
            let canon: std::collections::HashSet<u8> = canonical_orients(piece).iter().copied().collect();
            for o in 0u8..4 {
                if canon.contains(&o) {
                    continue;
                }
                let cells = decode(piece, o, 5, 5);
                let (eo, _, _) = encode(piece, &cells).expect("must encode");
                assert!(canon.contains(&eo), "piece {} orient {} encoded to non-canonical {}", piece, o, eo);
            }
        }
    }
}

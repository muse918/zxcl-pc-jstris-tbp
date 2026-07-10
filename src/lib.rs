//! Graph-free pure-PC bot as a WASM library implementing the jstris TBP worker contract.
//!
//! The FFI is plain extern "C" over linear memory (no wasm-bindgen): `tbp_init_embedded` inflates
//! the projection filter + value table baked into the wasm, then the JS worker drives
//! start/suggest/play/new_piece/stop with small integers. Pointer args are validated by the JS
//! caller (it owns the linear-memory buffers), so the FFI boundary intentionally allows raw-pointer
//! deref in safe `extern "C"` exports.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

// Engine sources (movegen, value_search, membership filter, value table) under src/engine/.
// graph.rs is a stub — the bot always uses the movegen+ProjFilter edge source, so value_search's
// optional precomputed-graph edge path is never reached.
#[path = "engine/piece.rs"]
pub mod piece;
#[path = "engine/movegen.rs"]
pub mod movegen;
pub mod graph; // stub (constants only)
#[path = "engine/values.rs"]
pub mod values;
#[path = "engine/value_search.rs"]
pub mod value_search;
#[path = "engine/proj.rs"]
pub mod proj;
pub mod tbpcoord;
pub mod bot;

use std::cell::RefCell;

use bot::{Bot, TbpMove};
use proj::ProjFilter;
use values::ValueTable;

thread_local! {
    static BOT: RefCell<Option<Bot>> = RefCell::new(None);
}

/// Allocate a buffer the JS side can fill (returns pointer into wasm linear memory).
/// Pair every call with free_bytes(ptr, len) — the old mem::forget version leaked each buffer
/// (KB-scale per TBP message, but a perpetual bot should not leak at all).
#[no_mangle]
pub extern "C" fn alloc_bytes(len: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(len.max(1), 1).unwrap();
    unsafe { std::alloc::alloc(layout) }
}

/// Free a buffer from alloc_bytes; `len` must be the same value passed to alloc_bytes.
#[no_mangle]
pub unsafe extern "C" fn free_bytes(p: *mut u8, len: usize) {
    if p.is_null() {
        return;
    }
    let layout = std::alloc::Layout::from_size_align(len.max(1), 1).unwrap();
    std::alloc::dealloc(p, layout);
}

// The projection filter + quantized V* table, zlib-compressed and embedded straight into the wasm
// data segment so the ONLY deployed files are this .wasm and pc_main.js (jstris serves no loose
// data blobs). Decompressed once at init with a pure-Rust inflate.
static PROJ_ZLIB: &[u8] = include_bytes!("../data/proj.zlib");
static VALUES_ZLIB: &[u8] = include_bytes!("../data/values.zlib");
// One supplementary 20-bit projection (center columns 4-8, all rows) ANDed onto the base filter.
// Kills ~17% of the base filter's in-search false positives for a measured ~5% faster boundary
// search (zero false negatives — built over the full PC-able source set). 20 KB.
static PROJ_EXT_ZLIB: &[u8] = include_bytes!("../data/proj_ext.zlib");

fn init_from(proj_bytes: Vec<u8>, val_bytes: &[u8]) -> i32 {
    let Ok(mut proj) = ProjFilter::from_bytes(proj_bytes) else { return -1 };
    if let Ok(ext) = miniz_oxide::inflate::decompress_to_vec_zlib(PROJ_EXT_ZLIB) {
        let _ = proj.attach_extra(&ext);
    }
    let Ok(table) = ValueTable::from_bytes(val_bytes) else { return -2 };
    BOT.with(|b| *b.borrow_mut() = Some(Bot::new(proj, table)));
    0
}

/// Initialize from the wasm-EMBEDDED data (no JS-supplied blobs). Returns 0 on success.
#[no_mangle]
pub extern "C" fn tbp_init_embedded() -> i32 {
    let Ok(proj) = miniz_oxide::inflate::decompress_to_vec_zlib(PROJ_ZLIB) else { return -3 };
    let Ok(vals) = miniz_oxide::inflate::decompress_to_vec_zlib(VALUES_ZLIB) else { return -4 };
    init_from(proj, &vals)
}

/// Initialize from two EXTERNAL data blobs (projection filter + value table) copied into wasm
/// memory by the caller. Used by the native/node harnesses; jstris uses tbp_init_embedded.
#[no_mangle]
pub extern "C" fn tbp_init(proj_ptr: *const u8, proj_len: usize, val_ptr: *const u8, val_len: usize) -> i32 {
    let proj_bytes = unsafe { std::slice::from_raw_parts(proj_ptr, proj_len) }.to_vec();
    let val_bytes = unsafe { std::slice::from_raw_parts(val_ptr, val_len) };
    init_from(proj_bytes, val_bytes)
}

/// `hold`: piece index 0..6 or -1 for empty. `queue`: piece indices. `board`: 400 bytes,
/// index r*10+c with r0 = bottom, c0 = left, nonzero = occupied.
#[no_mangle]
pub extern "C" fn tbp_start(hold: i32, queue_ptr: *const u8, queue_len: usize, board_ptr: *const u8, combo: i32) -> i32 {
    let queue = unsafe { std::slice::from_raw_parts(queue_ptr, queue_len) };
    let board = unsafe { std::slice::from_raw_parts(board_ptr, 400) };
    let h = if (0..7).contains(&hold) { Some(hold as u8) } else { None };
    BOT.with(|b| match b.borrow_mut().as_mut() {
        Some(bot) => {
            bot.start(h, queue, board, combo);
            0
        }
        None => -1,
    })
}

#[no_mangle]
pub extern "C" fn tbp_new_piece(p: i32) {
    if !(0..7).contains(&p) {
        return;
    }
    BOT.with(|b| {
        if let Some(bot) = b.borrow_mut().as_mut() {
            bot.new_piece(p as u8);
        }
    });
}

#[no_mangle]
pub extern "C" fn tbp_play() {
    BOT.with(|b| {
        if let Some(bot) = b.borrow_mut().as_mut() {
            bot.play();
        }
    });
}

#[no_mangle]
pub extern "C" fn tbp_stop() {
    BOT.with(|b| {
        if let Some(bot) = b.borrow_mut().as_mut() {
            bot.stop();
        }
    });
}

/// Compute the next move. `out` receives 4 i32s: [piece 0..6, orient 0..3, x, y].
/// Returns 1 if a move was written, 0 for "no move" (forfeit), negative on error.
#[no_mangle]
pub extern "C" fn tbp_suggest(out: *mut i32) -> i32 {
    BOT.with(|b| match b.borrow_mut().as_mut() {
        Some(bot) => match bot.suggest() {
            Some(TbpMove { piece, orient, x, y }) => {
                let o = unsafe { std::slice::from_raw_parts_mut(out, 4) };
                o[0] = piece as i32;
                o[1] = orient as i32;
                o[2] = x as i32;
                o[3] = y as i32;
                1
            }
            None => 0,
        },
        None => -1,
    })
}

/// Copy the last error message into `out` (up to `cap` bytes); returns its length.
#[no_mangle]
pub extern "C" fn tbp_last_error(out: *mut u8, cap: usize) -> usize {
    BOT.with(|b| match b.borrow().as_ref() {
        Some(bot) => {
            let msg = bot.last_error.as_bytes();
            let n = msg.len().min(cap);
            unsafe { std::ptr::copy_nonoverlapping(msg.as_ptr(), out, n) };
            n
        }
        None => 0,
    })
}

//! Stub graph module for the graph-free WASM bot. value_search's REFERENCE edge path needs a
//! HydraGraph, but the bot always supplies `edge_ids` (movegen + ProjFilter), so these methods
//! are never called. Only the constants are real.

pub type FieldId = u32;
pub const MAX_HASH: u64 = 0xFF_FF_FF_FF_FF;
pub const TWO_LINE_HASH: u64 = 0xF_FF_FF;

pub struct HydraGraph;

impl HydraGraph {
    pub fn edges(&self, _field: FieldId, _piece: u8) -> &[FieldId] {
        unimplemented!("graph-free bot: reference edge path unavailable")
    }
    pub fn hash(&self, _field: FieldId) -> u64 {
        unimplemented!("graph-free bot")
    }
    pub fn hash_lookup(&self, _h: u64) -> Option<FieldId> {
        unimplemented!("graph-free bot")
    }
}

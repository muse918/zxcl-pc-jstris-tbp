#!/usr/bin/env python3
# Pack the keyed layer0 V* (L0F32V1: (u32 key, f32 value) sorted by key) into the keyless
# 12-bit-quantized L0Q12V1 format the bot loads (values only, in ascending-key order; keys are
# regenerated at load by values.rs::enumerate_sorted_keys). Then gzip for the dist.
#   usage: python3 scripts/pack_values.py <v_pi2_f32.bin> <out values.bin>
import struct, sys, gzip

src, out = sys.argv[1], sys.argv[2]
data = open(src, "rb").read()
assert data[:8] == b"L0F32V1\0", "expected keyed L0F32V1 input"
n = struct.unpack("<Q", data[8:16])[0]
keys, vals = [], []
for i in range(n):
    o = 16 + i * 8
    keys.append(struct.unpack("<I", data[o:o+4])[0])
    vals.append(struct.unpack("<f", data[o+4:o+8])[0])
assert keys == sorted(keys), "keyed file must be sorted by key"
mn, mx = min(vals), max(vals)
span = mx - mn
q = [min(4095, max(0, round((v - mn) / span * 4095))) for v in vals]
err = max(abs((mn + qi / 4095 * span) - v) for qi, v in zip(q, vals))
print(f"{n} values, range [{mn:.3f},{mx:.3f}], max 12-bit err {err:.4f} ({100*err/span:.3f}% span)")

buf = bytearray(b"L0Q12V1\0") + struct.pack("<ffQ", mn, mx, n)
i = 0
while i + 1 < n:
    a, b = q[i], q[i+1]
    buf += bytes([a & 0xFF, ((a >> 8) & 0xF) | ((b & 0xF) << 4), (b >> 4) & 0xFF]); i += 2
if i < n:
    a = q[i]; buf += bytes([a & 0xFF, (a >> 8) & 0xF, 0])
open(out, "wb").write(buf)
open(out + ".gz", "wb").write(gzip.compress(bytes(buf), 9))
print(f"wrote {out} ({len(buf)} B) and {out}.gz ({len(gzip.compress(bytes(buf),9))} B)")

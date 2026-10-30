// Textures the GPU makes from guest memory as it is (block-linear bytes in a storage buffer, one invocation per block):
// - ASTC: a compute shader unswizzles and decodes every block of a mip level straight into the RGBA8 texture. The CPU
//   decoder took 6 s of thread time (0.7-1.2 s of the frame that first drew it) for TOTK's 1024x1024 x121 terrain
//   material array; the GPU takes milliseconds. RX_ASTC_CPU=1: the CPU decoder as before.
// - Other block-linear textures of 4-16 byte blocks (BC, RGBA8, RGBA16F...): a compute shader unswizzles them into a
//   linear buffer, copied to the texture. The CPU unswizzle of TOTK's 3686x3686 and 7373x3686 BC maps was ~150 ms of
//   the boot frame. RX_UNSWIZZLE_CPU=1: the CPU as before.
//
// The shader is a WGSL port of astc_decoder.comp from yuzu/suyu/Eden (GPL-3.0-or-later, like the rest of the suyu
// code in a64-kit: gpu/third_party/suyu), itself based on the FasTC decoder (Apache-2.0).
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "rx_internal.h"

namespace rx {
namespace {

const char* const kAstcWgsl = R"WGSL(
struct Params {
    block_dims: vec2<u32>,
    layer_stride: u32,
    block_size: u32,
    x_shift: u32,
    block_height: u32,
    block_height_mask: u32,
    base_offset: u32,
};
@group(0) @binding(0) var<storage, read> astc_data: array<vec4<u32>>;
@group(0) @binding(1) var dest_image: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

const JUST_BITS: u32 = 0u;
const QUINT: u32 = 1u;
const TRIT: u32 = 2u;

var<private> encoding_values: array<u32, 22> = array<u32, 22>(
    0u, (1u << 8u), 2u, (2u << 8u),
    1u, 2u | (1u << 8u), (3u << 8u), 1u | (1u << 8u),
    2u | (2u << 8u), (4u << 8u), 1u | (2u << 8u), 2u | (3u << 8u),
    (5u << 8u), 1u | (3u << 8u), 2u | (4u << 8u), (6u << 8u),
    1u | (4u << 8u), 2u | (5u << 8u), (7u << 8u), 1u | (5u << 8u),
    2u | (6u << 8u), (8u << 8u));

var<private> total_bitsread: u32;
var<private> local_buff: vec4<u32>;
var<private> color_endpoint_data: vec4<u32>;
var<private> color_bitsread: u32;
var<private> result_vector: array<u32, 288>;
var<private> result_index: u32;
var<private> result_vector_max_index: u32;
var<private> result_limit_reached: bool;
var<private> color_values: array<u32, 34>;
var<private> colvals_index: u32;
var<private> endpoints0: array<vec4<u32>, 4>;
var<private> endpoints1: array<vec4<u32>, 4>;

fn Encoding(v: u32) -> u32 { return v & 0xFFu; }
fn NumBits(v: u32) -> u32 { return (v >> 8u) & 0xFFu; }
fn BitValue(v: u32) -> u32 { return (v >> 16u) & 0xFFu; }
fn QuintTritValue(v: u32) -> u32 { return (v >> 24u) & 0xFFu; }
fn CreateEncodingData(e: u32, n: u32, b: u32, q: u32) -> u32 {
    return (e & 0xFFu) | ((n & 0xFFu) << 8u) | ((b & 0xFFu) << 16u) | ((q & 0xFFu) << 24u);
}

fn ResultEmplaceBack(val: u32) {
    if (result_index >= result_vector_max_index) {
        result_limit_reached = true;
        return;
    }
    result_vector[result_index] = val;
    result_index = result_index + 1u;
}

fn ReplicateBits(value: u32, num_bits: u32, to_bit: u32) -> u32 {
    if (value == 0u || num_bits == 0u) { return 0u; }
    if (num_bits >= to_bit) { return value; }
    let v = value & ((1u << num_bits) - 1u);
    var res = v;
    var reslen = num_bits;
    loop {
        if (reslen >= to_bit) { break; }
        let up = min(num_bits, to_bit - reslen);
        let down = num_bits - up;
        res = (res << up) | (v >> down);
        reslen = reslen + num_bits;
    }
    return res;
}

fn Div3Floor(v: u32) -> u32 { return (v * 0x5556u) >> 16u; }
fn Div3Ceil(v: u32) -> u32 { return Div3Floor(v + 2u); }
fn Div5Floor(v: u32) -> u32 { return (v * 0x3334u) >> 16u; }
fn Div5Ceil(v: u32) -> u32 { return Div5Floor(v + 4u); }

fn Hash52(p_in: u32) -> u32 {
    var p = p_in;
    p = p ^ (p >> 15u);
    p = p - (p << 17u);
    p = p + (p << 7u);
    p = p + (p << 4u);
    p = p ^ (p >> 5u);
    p = p + (p << 16u);
    p = p ^ (p >> 7u);
    p = p ^ (p >> 3u);
    p = p ^ (p << 6u);
    p = p ^ (p >> 17u);
    return p;
}

fn Select2DPartition(seed_in: u32, x_in: u32, y_in: u32, partition_count: u32) -> u32 {
    var x = x_in;
    var y = y_in;
    if ((params.block_dims.y * params.block_dims.x) < 32u) {
        x = x << 1u;
        y = y << 1u;
    }
    let seed = seed_in + (partition_count - 1u) * 1024u;
    let rnum = Hash52(seed);
    var seed1 = rnum & 0xFu;
    var seed2 = (rnum >> 4u) & 0xFu;
    var seed3 = (rnum >> 8u) & 0xFu;
    var seed4 = (rnum >> 12u) & 0xFu;
    var seed5 = (rnum >> 16u) & 0xFu;
    var seed6 = (rnum >> 20u) & 0xFu;
    var seed7 = (rnum >> 24u) & 0xFu;
    var seed8 = (rnum >> 28u) & 0xFu;
    seed1 = seed1 * seed1;
    seed2 = seed2 * seed2;
    seed3 = seed3 * seed3;
    seed4 = seed4 * seed4;
    seed5 = seed5 * seed5;
    seed6 = seed6 * seed6;
    seed7 = seed7 * seed7;
    seed8 = seed8 * seed8;
    var sh1: u32;
    var sh2: u32;
    if ((seed & 1u) != 0u) {
        sh1 = select(5u, 4u, (seed & 2u) != 0u);
        sh2 = select(5u, 6u, partition_count == 3u);
    } else {
        sh1 = select(5u, 6u, partition_count == 3u);
        sh2 = select(5u, 4u, (seed & 2u) != 0u);
    }
    seed1 = seed1 >> sh1;
    seed2 = seed2 >> sh2;
    seed3 = seed3 >> sh1;
    seed4 = seed4 >> sh2;
    seed5 = seed5 >> sh1;
    seed6 = seed6 >> sh2;
    seed7 = seed7 >> sh1;
    seed8 = seed8 >> sh2;
    var a = (seed1 * x + seed2 * y + (rnum >> 14u)) & 0x3Fu;
    var b = (seed3 * x + seed4 * y + (rnum >> 10u)) & 0x3Fu;
    var c = (seed5 * x + seed6 * y + (rnum >> 6u)) & 0x3Fu;
    var d = (seed7 * x + seed8 * y + (rnum >> 2u)) & 0x3Fu;
    if (partition_count < 4u) { d = 0u; }
    if (partition_count < 3u) { c = 0u; }
    if (a >= b && a >= c && a >= d) { return 0u; }
    if (b >= c && b >= d) { return 1u; }
    if (c >= d) { return 2u; }
    return 3u;
}

fn ExtractBits(payload: vec4<u32>, offset: u32, bits: u32) -> u32 {
    if (bits == 0u || bits > 32u) { return 0u; }
    let si = offset >> 5u;
    if (si > 3u) { return 0u; }
    let last = offset + bits - 1u;
    if ((last >> 5u) == si) {
        return extractBits(payload[si], offset & 31u, bits);
    }
    let first_bits = 32u - (offset & 31u);
    let r1 = extractBits(payload[si], offset & 31u, first_bits);
    var r2 = 0u;
    if (si + 1u <= 3u) {
        r2 = extractBits(payload[si + 1u], 0u, bits - first_bits);
    }
    return r1 | (r2 << first_bits);
}

fn StreamBits(num_bits: u32) -> u32 {
    let ret = ExtractBits(local_buff, total_bitsread, num_bits);
    total_bitsread = total_bitsread + num_bits;
    return ret;
}

fn SkipBits(num_bits: u32) {
    total_bitsread = total_bitsread + num_bits;
}

fn StreamColorBits(num_bits: u32) -> u32 {
    let ret = ExtractBits(color_endpoint_data, color_bitsread, num_bits);
    color_bitsread = color_bitsread + num_bits;
    return ret;
}

fn GetBitLength(n_vals: u32, encoding_index: u32) -> u32 {
    let ev = encoding_values[encoding_index];
    let enc = Encoding(ev);
    var total_bits = NumBits(ev) * n_vals;
    if (enc == TRIT) {
        total_bits = total_bits + Div5Ceil(n_vals * 8u);
    } else if (enc == QUINT) {
        total_bits = total_bits + Div3Ceil(n_vals * 7u);
    }
    return total_bits;
}

fn GetNumWeightValues(size: vec2<u32>, dual_plane: bool) -> u32 {
    var n_vals = size.x * size.y;
    if (dual_plane) { n_vals = n_vals * 2u; }
    return n_vals;
}

fn GetPackedBitSize(size: vec2<u32>, dual_plane: bool, max_weight: u32) -> u32 {
    return GetBitLength(GetNumWeightValues(size, dual_plane), max_weight);
}

fn BitsBracket(bits: u32, pos: u32) -> u32 { return (bits >> pos) & 1u; }
fn BitsOp(bits: u32, start: u32, end: u32) -> u32 {
    let mask = (1u << (end - start + 1u)) - 1u;
    return (bits >> start) & mask;
}

fn DecodeQuintBlock(num_bits: u32) {
    var m: array<u32, 3>;
    var q: array<u32, 3>;
    m[0] = StreamColorBits(num_bits);
    var Q = StreamColorBits(3u);
    m[1] = StreamColorBits(num_bits);
    Q = Q | (StreamColorBits(2u) << 3u);
    m[2] = StreamColorBits(num_bits);
    Q = Q | (StreamColorBits(2u) << 5u);
    if (BitsOp(Q, 1u, 2u) == 3u && BitsOp(Q, 5u, 6u) == 0u) {
        q[0] = 4u;
        q[1] = 4u;
        q[2] = (BitsBracket(Q, 0u) << 2u) | ((BitsBracket(Q, 4u) & ~BitsBracket(Q, 0u)) << 1u) |
               (BitsBracket(Q, 3u) & ~BitsBracket(Q, 0u));
    } else {
        var C = 0u;
        if (BitsOp(Q, 1u, 2u) == 3u) {
            q[2] = 4u;
            C = (BitsOp(Q, 3u, 4u) << 3u) | ((~BitsOp(Q, 5u, 6u) & 3u) << 1u) | BitsBracket(Q, 0u);
        } else {
            q[2] = BitsOp(Q, 5u, 6u);
            C = BitsOp(Q, 0u, 4u);
        }
        if (BitsOp(C, 0u, 2u) == 5u) {
            q[1] = 4u;
            q[0] = BitsOp(C, 3u, 4u);
        } else {
            q[1] = BitsOp(C, 3u, 4u);
            q[0] = BitsOp(C, 0u, 2u);
        }
    }
    for (var i = 0u; i < 3u; i = i + 1u) {
        ResultEmplaceBack(CreateEncodingData(QUINT, num_bits, m[i], q[i]));
    }
}

fn DecodeTritBlock(num_bits: u32) {
    var m: array<u32, 5>;
    var t: array<u32, 5>;
    m[0] = StreamColorBits(num_bits);
    var T = StreamColorBits(2u);
    m[1] = StreamColorBits(num_bits);
    T = T | (StreamColorBits(2u) << 2u);
    m[2] = StreamColorBits(num_bits);
    T = T | (StreamColorBits(1u) << 4u);
    m[3] = StreamColorBits(num_bits);
    T = T | (StreamColorBits(2u) << 5u);
    m[4] = StreamColorBits(num_bits);
    T = T | (StreamColorBits(1u) << 7u);
    var C = 0u;
    if (BitsOp(T, 2u, 4u) == 7u) {
        C = (BitsOp(T, 5u, 7u) << 2u) | BitsOp(T, 0u, 1u);
        t[4] = 2u;
        t[3] = 2u;
    } else {
        C = BitsOp(T, 0u, 4u);
        if (BitsOp(T, 5u, 6u) == 3u) {
            t[4] = 2u;
            t[3] = BitsBracket(T, 7u);
        } else {
            t[4] = BitsBracket(T, 7u);
            t[3] = BitsOp(T, 5u, 6u);
        }
    }
    if (BitsOp(C, 0u, 1u) == 3u) {
        t[2] = 2u;
        t[1] = BitsBracket(C, 4u);
        t[0] = (BitsBracket(C, 3u) << 1u) | (BitsBracket(C, 2u) & ~BitsBracket(C, 3u));
    } else if (BitsOp(C, 2u, 3u) == 3u) {
        t[2] = 2u;
        t[1] = 2u;
        t[0] = BitsOp(C, 0u, 1u);
    } else {
        t[2] = BitsBracket(C, 4u);
        t[1] = BitsOp(C, 2u, 3u);
        t[0] = (BitsBracket(C, 1u) << 1u) | (BitsBracket(C, 0u) & ~BitsBracket(C, 1u));
    }
    for (var i = 0u; i < 5u; i = i + 1u) {
        ResultEmplaceBack(CreateEncodingData(TRIT, num_bits, m[i], t[i]));
    }
}

fn DecodeIntegerSequence(max_range: u32, num_values: u32) {
    let val = encoding_values[max_range];
    let enc = Encoding(val);
    let num_bits = NumBits(val);
    var vals_decoded = 0u;
    loop {
        if (vals_decoded >= num_values || result_limit_reached) { break; }
        if (enc == QUINT) {
            DecodeQuintBlock(num_bits);
            vals_decoded = vals_decoded + 3u;
        } else if (enc == TRIT) {
            DecodeTritBlock(num_bits);
            vals_decoded = vals_decoded + 5u;
        } else {
            ResultEmplaceBack(CreateEncodingData(JUST_BITS, num_bits, StreamColorBits(num_bits), 0u));
            vals_decoded = vals_decoded + 1u;
        }
    }
}

fn DecodeColorValues(modes: vec4<u32>, num_partitions: u32, color_data_bits: u32) {
    var num_values = 0u;
    for (var i = 0u; i < num_partitions; i = i + 1u) {
        num_values = num_values + (((modes[i] >> 2u) + 1u) << 1u);
    }
    var range = 0u;
    loop {
        range = range + 1u;
        if (range >= 22u) { break; }
        if (GetBitLength(num_values, range) > color_data_bits) { break; }
    }
    DecodeIntegerSequence(range - 1u, num_values);
    var out_index = 0u;
    for (var itr = 0u; itr < result_index; itr = itr + 1u) {
        if (out_index >= num_values) { break; }
        let val = result_vector[itr];
        let enc = Encoding(val);
        let bitlen = NumBits(val);
        let bitval = BitValue(val);
        let A = (bitval & 1u) * 511u;
        var B = 0u;
        var C = 0u;
        var D = 0u;
        if (enc == JUST_BITS) {
            out_index = out_index + 1u;
            color_values[out_index] = ReplicateBits(bitval, bitlen, 8u);
            continue;
        }
        D = QuintTritValue(val);
        if (enc == TRIT) {
            switch (bitlen) {
                case 1u: { C = 204u; }
                case 2u: {
                    C = 93u;
                    let b = (bitval >> 1u) & 1u;
                    B = (b << 8u) | (b << 4u) | (b << 2u) | (b << 1u);
                }
                case 3u: {
                    C = 44u;
                    let cb = (bitval >> 1u) & 3u;
                    B = (cb << 7u) | (cb << 2u) | cb;
                }
                case 4u: {
                    C = 22u;
                    let dcb = (bitval >> 1u) & 7u;
                    B = (dcb << 6u) | dcb;
                }
                case 5u: {
                    C = 11u;
                    let edcb = (bitval >> 1u) & 0xFu;
                    B = (edcb << 5u) | (edcb >> 2u);
                }
                case 6u: {
                    C = 5u;
                    let fedcb = (bitval >> 1u) & 0x1Fu;
                    B = (fedcb << 4u) | (fedcb >> 4u);
                }
                default: {}
            }
        } else {
            switch (bitlen) {
                case 1u: { C = 113u; }
                case 2u: {
                    C = 54u;
                    let b = (bitval >> 1u) & 1u;
                    B = (b << 8u) | (b << 3u) | (b << 2u);
                }
                case 3u: {
                    C = 26u;
                    let cb = (bitval >> 1u) & 3u;
                    B = (cb << 7u) | (cb << 1u) | (cb >> 1u);
                }
                case 4u: {
                    C = 13u;
                    let dcb = (bitval >> 1u) & 7u;
                    B = (dcb << 6u) | (dcb >> 1u);
                }
                case 5u: {
                    C = 6u;
                    let edcb = (bitval >> 1u) & 0xFu;
                    B = (edcb << 5u) | (edcb >> 3u);
                }
                default: {}
            }
        }
        var T = (D * C) + B;
        T = T ^ A;
        T = (A & 0x80u) | (T >> 2u);
        out_index = out_index + 1u;
        color_values[out_index] = T;
    }
}

fn BitTransferSigned(a: i32, b: i32) -> vec2<i32> {
    var y = b >> 1u;
    y = y | (a & 0x80);
    var x = a >> 1u;
    x = x & 0x3F;
    if ((x & 0x20) != 0) { x = x - 0x40; }
    return vec2<i32>(x, y);
}

fn ClampByte(c: vec4<i32>) -> vec4<u32> {
    return vec4<u32>(clamp(c, vec4<i32>(0), vec4<i32>(255)));
}

fn BlueContract(a: i32, r: i32, g: i32, b: i32) -> vec4<i32> {
    return vec4<i32>(a, (r + b) >> 1u, (g + b) >> 1u, b);
}

fn IsHDRColorEndpointMode(cem: u32) -> bool {
    return cem == 2u || cem == 3u || cem == 7u || cem == 11u || cem == 14u || cem == 15u;
}

fn SignExtend(value: i32, nbits: u32) -> i32 {
    let sign_bit = 1 << (nbits - 1u);
    return (value ^ sign_bit) - sign_bit;
}

struct Endpoints3 {
    e0: vec3<i32>,
    e1: vec3<i32>,
};

fn DecodeHDREndpointMode7(v0: u32, v1: u32, v2: u32, v3: u32) -> Endpoints3 {
    let modeval = ((v0 & 0xC0u) >> 6u) | ((v1 & 0x80u) >> 5u) | ((v2 & 0x80u) >> 4u);
    var majcomp: u32;
    var mode: u32;
    if ((modeval & 0xCu) != 0xCu) {
        majcomp = modeval >> 2u;
        mode = modeval & 3u;
    } else if (modeval != 0xFu) {
        majcomp = modeval & 3u;
        mode = 4u;
    } else {
        majcomp = 0u;
        mode = 5u;
    }
    var red = i32(v0 & 0x3Fu);
    var green = i32(v1 & 0x1Fu);
    var blue = i32(v2 & 0x1Fu);
    var scale = i32(v3 & 0x1Fu);
    let x0 = (v1 >> 6u) & 1u;
    let x1 = (v1 >> 5u) & 1u;
    let x2 = (v2 >> 6u) & 1u;
    let x3 = (v2 >> 5u) & 1u;
    let x4 = (v3 >> 7u) & 1u;
    let x5 = (v3 >> 6u) & 1u;
    let x6 = (v3 >> 5u) & 1u;
    let ohm = 1u << mode;
    if ((ohm & 0x30u) != 0u) { green = green | i32(x0 << 6u); }
    if ((ohm & 0x3Au) != 0u) { green = green | i32(x1 << 5u); }
    if ((ohm & 0x30u) != 0u) { blue = blue | i32(x2 << 6u); }
    if ((ohm & 0x3Au) != 0u) { blue = blue | i32(x3 << 5u); }
    if ((ohm & 0x3Du) != 0u) { scale = scale | i32(x6 << 5u); }
    if ((ohm & 0x2Du) != 0u) { scale = scale | i32(x5 << 6u); }
    if ((ohm & 0x04u) != 0u) { scale = scale | i32(x4 << 7u); }
    if ((ohm & 0x3Bu) != 0u) { red = red | i32(x4 << 6u); }
    if ((ohm & 0x04u) != 0u) { red = red | i32(x3 << 6u); }
    if ((ohm & 0x10u) != 0u) { red = red | i32(x5 << 7u); }
    if ((ohm & 0x0Fu) != 0u) { red = red | i32(x2 << 7u); }
    if ((ohm & 0x05u) != 0u) { red = red | i32(x1 << 8u); }
    if ((ohm & 0x0Au) != 0u) { red = red | i32(x0 << 8u); }
    if ((ohm & 0x05u) != 0u) { red = red | i32(x0 << 9u); }
    if ((ohm & 0x02u) != 0u) { red = red | i32(x6 << 9u); }
    if ((ohm & 0x01u) != 0u) { red = red | i32(x3 << 10u); }
    if ((ohm & 0x02u) != 0u) { red = red | i32(x5 << 10u); }
    var shamts = array<u32, 6>(1u, 1u, 2u, 3u, 4u, 5u);
    let shamt = shamts[mode];
    red = red << shamt;
    green = green << shamt;
    blue = blue << shamt;
    scale = scale << shamt;
    if (mode != 5u) {
        green = red - green;
        blue = red - blue;
    }
    if (majcomp == 1u) {
        let t = red;
        red = green;
        green = t;
    }
    if (majcomp == 2u) {
        let t = red;
        red = blue;
        blue = t;
    }
    var r: Endpoints3;
    r.e1 = vec3<i32>(clamp(red, 0, 0xFFF), clamp(green, 0, 0xFFF), clamp(blue, 0, 0xFFF));
    r.e0 = vec3<i32>(clamp(red - scale, 0, 0xFFF), clamp(green - scale, 0, 0xFFF), clamp(blue - scale, 0, 0xFFF));
    return r;
}

fn DecodeHDREndpointMode11(v0: u32, v1: u32, v2: u32, v3: u32, v4: u32, v5: u32) -> Endpoints3 {
    var r: Endpoints3;
    let majcomp = ((v4 & 0x80u) >> 7u) | ((v5 & 0x80u) >> 6u);
    if (majcomp == 3u) {
        r.e0 = vec3<i32>(i32(v0 << 4u), i32(v2 << 4u), i32((v4 & 0x7Fu) << 5u));
        r.e1 = vec3<i32>(i32(v1 << 4u), i32(v3 << 4u), i32((v5 & 0x7Fu) << 5u));
        return r;
    }
    let mode = ((v1 & 0x80u) >> 7u) | ((v2 & 0x80u) >> 6u) | ((v3 & 0x80u) >> 5u);
    var va = i32(v0 | ((v1 & 0x40u) << 2u));
    var vb0 = i32(v2 & 0x3Fu);
    var vb1 = i32(v3 & 0x3Fu);
    var vc = i32(v1 & 0x3Fu);
    var vd0 = i32(v4 & 0x7Fu);
    var vd1 = i32(v5 & 0x7Fu);
    var dbitstab = array<u32, 8>(7u, 6u, 7u, 6u, 5u, 6u, 5u, 6u);
    vd0 = SignExtend(vd0, dbitstab[mode]);
    vd1 = SignExtend(vd1, dbitstab[mode]);
    let x0 = (v2 >> 6u) & 1u;
    let x1 = (v3 >> 6u) & 1u;
    let x2 = (v4 >> 6u) & 1u;
    let x3 = (v5 >> 6u) & 1u;
    let x4 = (v4 >> 5u) & 1u;
    let x5 = (v5 >> 5u) & 1u;
    let ohm = 1u << mode;
    if ((ohm & 0xA4u) != 0u) { va = va | i32(x0 << 9u); }
    if ((ohm & 0x08u) != 0u) { va = va | i32(x2 << 9u); }
    if ((ohm & 0x50u) != 0u) { va = va | i32(x4 << 9u); }
    if ((ohm & 0x50u) != 0u) { va = va | i32(x5 << 10u); }
    if ((ohm & 0xA0u) != 0u) { va = va | i32(x1 << 10u); }
    if ((ohm & 0xC0u) != 0u) { va = va | i32(x2 << 11u); }
    if ((ohm & 0x04u) != 0u) { vc = vc | i32(x1 << 6u); }
    if ((ohm & 0xE8u) != 0u) { vc = vc | i32(x3 << 6u); }
    if ((ohm & 0x20u) != 0u) { vc = vc | i32(x2 << 7u); }
    if ((ohm & 0x5Bu) != 0u) { vb0 = vb0 | i32(x0 << 6u); }
    if ((ohm & 0x5Bu) != 0u) { vb1 = vb1 | i32(x1 << 6u); }
    if ((ohm & 0x12u) != 0u) { vb0 = vb0 | i32(x2 << 7u); }
    if ((ohm & 0x12u) != 0u) { vb1 = vb1 | i32(x3 << 7u); }
    let shamt = (mode >> 1u) ^ 3u;
    va = va << shamt;
    vb0 = vb0 << shamt;
    vb1 = vb1 << shamt;
    vc = vc << shamt;
    vd0 = vd0 << shamt;
    vd1 = vd1 << shamt;
    var r1 = clamp(va, 0, 0xFFF);
    var g1 = clamp(va - vb0, 0, 0xFFF);
    var b1 = clamp(va - vb1, 0, 0xFFF);
    var r0 = clamp(va - vc, 0, 0xFFF);
    var g0 = clamp(va - vb0 - vc - vd0, 0, 0xFFF);
    var b0 = clamp(va - vb1 - vc - vd1, 0, 0xFFF);
    if (majcomp == 1u) {
        var t = r0; r0 = g0; g0 = t;
        t = r1; r1 = g1; g1 = t;
    } else if (majcomp == 2u) {
        var t = r0; r0 = b0; b0 = t;
        t = r1; r1 = b1; b1 = t;
    }
    r.e0 = vec3<i32>(r0, g0, b0);
    r.e1 = vec3<i32>(r1, g1, b1);
    return r;
}

fn NextColorValue() -> u32 {
    colvals_index = colvals_index + 1u;
    return color_values[colvals_index];
}

fn ComputeEndpoints(p: u32, cem: u32) {
    var V: array<u32, 8>;
    var n = 0u;
    switch (cem) {
        case 0u, 1u, 2u, 3u: { n = 2u; }
        case 4u, 5u, 6u, 7u: { n = 4u; }
        case 8u, 9u, 10u, 11u: { n = 6u; }
        default: { n = 8u; }
    }
    for (var i = 0u; i < n; i = i + 1u) { V[i] = NextColorValue(); }
    var ep1 = vec4<u32>(0xFFu, 0xFFu, 0u, 0u);
    var ep2 = vec4<u32>(0xFFu, 0xFFu, 0u, 0u);
    switch (cem) {
        case 0u: {
            ep1 = vec4<u32>(0xFFu, V[0], V[0], V[0]);
            ep2 = vec4<u32>(0xFFu, V[1], V[1], V[1]);
        }
        case 1u: {
            let L0 = (V[0] >> 2u) | (V[1] & 0xC0u);
            let L1 = min(L0 + (V[1] & 0x3Fu), 0xFFu);
            ep1 = vec4<u32>(0xFFu, L0, L0, L0);
            ep2 = vec4<u32>(0xFFu, L1, L1, L1);
        }
        case 4u: {
            ep1 = vec4<u32>(V[2], V[0], V[0], V[0]);
            ep2 = vec4<u32>(V[3], V[1], V[1], V[1]);
        }
        case 5u: {
            var I: array<i32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) { I[i] = i32(V[i]); }
            var tr = BitTransferSigned(I[1], I[0]);
            I[1] = tr.x;
            I[0] = tr.y;
            tr = BitTransferSigned(I[3], I[2]);
            I[3] = tr.x;
            I[2] = tr.y;
            ep1 = ClampByte(vec4<i32>(I[2], I[0], I[0], I[0]));
            ep2 = ClampByte(vec4<i32>(I[2] + I[3], I[0] + I[1], I[0] + I[1], I[0] + I[1]));
        }
        case 6u: {
            ep1 = vec4<u32>(0xFFu, (V[0] * V[3]) >> 8u, (V[1] * V[3]) >> 8u, (V[2] * V[3]) >> 8u);
            ep2 = vec4<u32>(0xFFu, V[0], V[1], V[2]);
        }
        case 8u: {
            if ((V[1] + V[3] + V[5]) >= (V[0] + V[2] + V[4])) {
                ep1 = vec4<u32>(0xFFu, V[0], V[2], V[4]);
                ep2 = vec4<u32>(0xFFu, V[1], V[3], V[5]);
            } else {
                ep1 = vec4<u32>(BlueContract(0xFF, i32(V[1]), i32(V[3]), i32(V[5])));
                ep2 = vec4<u32>(BlueContract(0xFF, i32(V[0]), i32(V[2]), i32(V[4])));
            }
        }
        case 9u: {
            var I: array<i32, 6>;
            for (var i = 0u; i < 6u; i = i + 1u) { I[i] = i32(V[i]); }
            var tr = BitTransferSigned(I[1], I[0]);
            I[1] = tr.x;
            I[0] = tr.y;
            tr = BitTransferSigned(I[3], I[2]);
            I[3] = tr.x;
            I[2] = tr.y;
            tr = BitTransferSigned(I[5], I[4]);
            I[5] = tr.x;
            I[4] = tr.y;
            if ((I[1] + I[3] + I[5]) >= 0) {
                ep1 = ClampByte(vec4<i32>(0xFF, I[0], I[2], I[4]));
                ep2 = ClampByte(vec4<i32>(0xFF, I[0] + I[1], I[2] + I[3], I[4] + I[5]));
            } else {
                ep1 = ClampByte(BlueContract(0xFF, I[0] + I[1], I[2] + I[3], I[4] + I[5]));
                ep2 = ClampByte(BlueContract(0xFF, I[0], I[2], I[4]));
            }
        }
        case 10u: {
            ep1 = vec4<u32>(V[4], (V[0] * V[3]) >> 8u, (V[1] * V[3]) >> 8u, (V[2] * V[3]) >> 8u);
            ep2 = vec4<u32>(V[5], V[0], V[1], V[2]);
        }
        case 12u: {
            if ((V[1] + V[3] + V[5]) >= (V[0] + V[2] + V[4])) {
                ep1 = vec4<u32>(V[6], V[0], V[2], V[4]);
                ep2 = vec4<u32>(V[7], V[1], V[3], V[5]);
            } else {
                ep1 = vec4<u32>(BlueContract(i32(V[7]), i32(V[1]), i32(V[3]), i32(V[5])));
                ep2 = vec4<u32>(BlueContract(i32(V[6]), i32(V[0]), i32(V[2]), i32(V[4])));
            }
        }
        case 13u: {
            var I: array<i32, 8>;
            for (var i = 0u; i < 8u; i = i + 1u) { I[i] = i32(V[i]); }
            var tr = BitTransferSigned(I[1], I[0]);
            I[1] = tr.x;
            I[0] = tr.y;
            tr = BitTransferSigned(I[3], I[2]);
            I[3] = tr.x;
            I[2] = tr.y;
            tr = BitTransferSigned(I[5], I[4]);
            I[5] = tr.x;
            I[4] = tr.y;
            tr = BitTransferSigned(I[7], I[6]);
            I[7] = tr.x;
            I[6] = tr.y;
            if ((I[1] + I[3] + I[5]) >= 0) {
                ep1 = ClampByte(vec4<i32>(I[6], I[0], I[2], I[4]));
                ep2 = ClampByte(vec4<i32>(I[7] + I[6], I[0] + I[1], I[2] + I[3], I[4] + I[5]));
            } else {
                ep1 = ClampByte(BlueContract(I[6] + I[7], I[0] + I[1], I[2] + I[3], I[4] + I[5]));
                ep2 = ClampByte(BlueContract(I[6], I[0], I[2], I[4]));
            }
        }
        case 2u: {
            var y0: u32;
            var y1: u32;
            if (V[1] >= V[0]) {
                y0 = V[0] << 4u;
                y1 = V[1] << 4u;
            } else {
                y0 = (V[1] << 4u) + 8u;
                y1 = (V[0] << 4u) - 8u;
            }
            ep1 = vec4<u32>(0x780u, y0, y0, y0);
            ep2 = vec4<u32>(0x780u, y1, y1, y1);
        }
        case 3u: {
            var y0: u32;
            var d: u32;
            if ((V[0] & 0x80u) != 0u) {
                y0 = ((V[1] & 0xE0u) << 4u) | ((V[0] & 0x7Fu) << 2u);
                d = (V[1] & 0x1Fu) << 2u;
            } else {
                y0 = ((V[1] & 0xF0u) << 4u) | ((V[0] & 0x7Fu) << 1u);
                d = (V[1] & 0x0Fu) << 1u;
            }
            let y1 = min(y0 + d, 0xFFFu);
            ep1 = vec4<u32>(0x780u, y0, y0, y0);
            ep2 = vec4<u32>(0x780u, y1, y1, y1);
        }
        case 7u: {
            let e = DecodeHDREndpointMode7(V[0], V[1], V[2], V[3]);
            ep1 = vec4<u32>(0x780u, u32(e.e0.x), u32(e.e0.y), u32(e.e0.z));
            ep2 = vec4<u32>(0x780u, u32(e.e1.x), u32(e.e1.y), u32(e.e1.z));
        }
        case 11u: {
            let e = DecodeHDREndpointMode11(V[0], V[1], V[2], V[3], V[4], V[5]);
            ep1 = vec4<u32>(0x780u, u32(e.e0.x), u32(e.e0.y), u32(e.e0.z));
            ep2 = vec4<u32>(0x780u, u32(e.e1.x), u32(e.e1.y), u32(e.e1.z));
        }
        case 14u: {
            let e = DecodeHDREndpointMode11(V[0], V[1], V[2], V[3], V[4], V[5]);
            ep1 = vec4<u32>(V[6], u32(e.e0.x), u32(e.e0.y), u32(e.e0.z));
            ep2 = vec4<u32>(V[7], u32(e.e1.x), u32(e.e1.y), u32(e.e1.z));
        }
        case 15u: {
            let e = DecodeHDREndpointMode11(V[0], V[1], V[2], V[3], V[4], V[5]);
            let mode = ((V[6] >> 7u) & 1u) | ((V[7] >> 6u) & 2u);
            var a6 = i32(V[6] & 0x7Fu);
            var a7 = i32(V[7] & 0x7Fu);
            var alpha0: i32;
            var alpha1: i32;
            if (mode == 3u) {
                alpha0 = a6 << 5u;
                alpha1 = a7 << 5u;
            } else {
                a6 = a6 | ((a7 << (mode + 1u)) & 0x780);
                a7 = a7 & i32(0x3Fu >> mode);
                a7 = a7 ^ i32(0x20u >> mode);
                a7 = a7 - i32(0x20u >> mode);
                a6 = a6 << (4u - mode);
                a7 = a7 << (4u - mode);
                a7 = a7 + a6;
                alpha0 = a6;
                alpha1 = clamp(a7, 0, 0xFFF);
            }
            ep1 = vec4<u32>(u32(alpha0), u32(e.e0.x), u32(e.e0.y), u32(e.e0.z));
            ep2 = vec4<u32>(u32(alpha1), u32(e.e1.x), u32(e.e1.y), u32(e.e1.z));
        }
        default: {}
    }
    endpoints0[p] = ep1;
    endpoints1[p] = ep2;
}

fn UnquantizeTexelWeight(val: u32) -> u32 {
    let enc = Encoding(val);
    let bitlen = NumBits(val);
    let bitval = BitValue(val);
    if (enc == JUST_BITS) {
        var r = ReplicateBits(bitval, bitlen, 6u);
        if (r > 32u) { r = r + 1u; }
        return r;
    }
    var B = 0u;
    var C = 0u;
    let D = QuintTritValue(val);
    let b_mask = (0x3100u >> (bitlen * 4u)) & 0xFu;
    let b = (bitval >> 1u) & b_mask;
    if (enc == TRIT) {
        switch (bitlen) {
            case 0u: { return D * 32u; }
            case 1u: { C = 50u; }
            case 2u: { C = 23u; B = (b << 6u) | (b << 2u) | b; }
            case 3u: { C = 11u; B = (b << 5u) | b; }
            default: {}
        }
    } else {
        switch (bitlen) {
            case 0u: { return D * 16u; }
            case 1u: { C = 28u; }
            case 2u: { C = 13u; B = (b << 6u) | (b << 1u); }
            default: {}
        }
    }
    let A = (bitval & 1u) * 127u;
    let res = (A & 0x20u) | ((((D * C) + B) ^ A) >> 2u);
    return res + select(0u, 1u, res > 32u);
}

fn UnquantizeTexelWeights(size: vec2<u32>, dual_plane: bool) {
    let num_planes = select(1u, 2u, dual_plane);
    let area = size.x * size.y;
    let loop_count = min(result_index, area * num_planes);
    for (var itr = 0u; itr < loop_count; itr = itr + 1u) {
        result_vector[itr] = UnquantizeTexelWeight(result_vector[itr]);
    }
}

fn GetUnquantizedTexelWeight(offset_base: u32, plane: u32, dual_plane: bool) -> u32 {
    let offset = select(offset_base, 2u * offset_base + plane, dual_plane);
    return result_vector[min(offset, 287u)];
}

fn GetUnquantizedWeightVector(t: u32, s: u32, size: vec2<u32>, plane_index: u32, dual_plane: bool) -> vec4<u32> {
    let bw = params.block_dims.x;
    let bh = params.block_dims.y;
    let Ds = (1024u + (bw / 2u)) / (bw - 1u);
    let Dt = (1024u + (bh / 2u)) / (bh - 1u);
    let area = size.x * size.y;
    let cs = Ds * s;
    let ct = Dt * t;
    let gs = (cs * (size.x - 1u) + 32u) >> 6u;
    let gt = (ct * (size.y - 1u) + 32u) >> 6u;
    let js = gs >> 4u;
    let fs = gs & 0xFu;
    let jt = gt >> 4u;
    let ft = gt & 0x0Fu;
    let w11 = (fs * ft + 8u) >> 4u;
    let w10 = ft - w11;
    let w01 = fs - w11;
    let w00 = 16u - fs - ft + w11;
    let w = vec4<u32>(w00, w01, w10, w11);
    let v0 = jt * size.x + js;
    var p0 = vec4<u32>(0u);
    var p1 = vec4<u32>(0u);
    if (v0 < area) {
        p0.x = GetUnquantizedTexelWeight(v0, 0u, dual_plane);
        p1.x = GetUnquantizedTexelWeight(v0, 1u, dual_plane);
    }
    if ((v0 + 1u) < area) {
        p0.y = GetUnquantizedTexelWeight(v0 + 1u, 0u, dual_plane);
        p1.y = GetUnquantizedTexelWeight(v0 + 1u, 1u, dual_plane);
    }
    if ((v0 + size.x) < area) {
        p0.z = GetUnquantizedTexelWeight(v0 + size.x, 0u, dual_plane);
        p1.z = GetUnquantizedTexelWeight(v0 + size.x, 1u, dual_plane);
    }
    if ((v0 + size.x + 1u) < area) {
        p0.w = GetUnquantizedTexelWeight(v0 + size.x + 1u, 0u, dual_plane);
        p1.w = GetUnquantizedTexelWeight(v0 + size.x + 1u, 1u, dual_plane);
    }
    let primary = (dot(p0, w) + 8u) >> 4u;
    var weight_vec = vec4<u32>(primary);
    if (dual_plane) {
        let secondary = (dot(p1, w) + 8u) >> 4u;
        for (var c = 0u; c < 4u; c = c + 1u) {
            weight_vec[c] = select(primary, secondary, ((plane_index + 1u) & 3u) == c);
        }
    }
    return weight_vec;
}

fn FindLayout(mode: u32) -> u32 {
    if ((mode & 3u) != 0u) {
        if ((mode & 8u) != 0u) {
            if ((mode & 4u) != 0u) {
                if ((mode & 0x100u) != 0u) { return 4u; }
                return 3u;
            }
            return 2u;
        }
        if ((mode & 4u) != 0u) { return 1u; }
        return 0u;
    }
    if ((mode & 0x100u) != 0u) {
        if ((mode & 0x80u) != 0u) {
            if ((mode & 0x20u) != 0u) { return 8u; }
            return 7u;
        }
        return 9u;
    }
    if ((mode & 0x80u) != 0u) { return 6u; }
    return 5u;
}

fn Store(coord: vec2<u32>, layer: u32, c: vec4<f32>) {
    let dims = textureDimensions(dest_image);
    if (coord.x < dims.x && coord.y < dims.y) {
        textureStore(dest_image, vec2<i32>(coord), i32(layer), c);
    }
}

fn FillError(coord: vec2<u32>, layer: u32) {
    for (var j = 0u; j < params.block_dims.y; j = j + 1u) {
        for (var i = 0u; i < params.block_dims.x; i = i + 1u) {
            Store(coord + vec2<u32>(i, j), layer, vec4<f32>(0.0));
        }
    }
}

fn FillVoidExtentLDR(coord: vec2<u32>, layer: u32) {
    SkipBits(52u);
    let r_u = StreamBits(16u);
    let g_u = StreamBits(16u);
    let b_u = StreamBits(16u);
    let a_u = StreamBits(16u);
    let c = vec4<f32>(f32(r_u), f32(g_u), f32(b_u), f32(a_u)) / 65535.0;
    for (var j = 0u; j < params.block_dims.y; j = j + 1u) {
        for (var i = 0u; i < params.block_dims.x; i = i + 1u) {
            Store(coord + vec2<u32>(i, j), layer, c);
        }
    }
}

fn IsError(mode: u32) -> bool {
    if ((mode & 0x1FFu) == 0x1FCu) {
        if ((mode & 0x200u) != 0u) { return true; }
        if ((mode & 0x400u) == 0u || StreamBits(1u) == 0u) { return true; }
        return false;
    }
    if ((mode & 0xFu) == 0u) { return true; }
    if ((mode & 3u) == 0u && (mode & 0x1C0u) == 0x1C0u) { return true; }
    return false;
}

fn DecodeBlockSize(mode: u32) -> vec2<u32> {
    let A = (mode >> 5u) & 0x3u;
    switch (FindLayout(mode)) {
        case 0u: { return vec2<u32>(((mode >> 7u) & 0x3u) + 4u, A + 2u); }
        case 1u: { return vec2<u32>(((mode >> 7u) & 0x3u) + 8u, A + 2u); }
        case 2u: { return vec2<u32>(A + 2u, ((mode >> 7u) & 0x3u) + 8u); }
        case 3u: { return vec2<u32>(A + 2u, ((mode >> 7u) & 0x1u) + 6u); }
        case 4u: { return vec2<u32>(((mode >> 7u) & 0x1u) + 2u, A + 2u); }
        case 5u: { return vec2<u32>(12u, A + 2u); }
        case 6u: { return vec2<u32>(A + 2u, 12u); }
        case 7u: { return vec2<u32>(6u, 10u); }
        case 8u: { return vec2<u32>(10u, 6u); }
        case 9u: { return vec2<u32>(A + 6u, ((mode >> 9u) & 0x3u) + 6u); }
        default: { return vec2<u32>(0u); }
    }
}

fn DecodeMaxWeight(mode: u32) -> u32 {
    let mode_layout = FindLayout(mode);
    var weight_index = select(0u, 1u, (mode & 0x10u) != 0u);
    if (mode_layout < 5u) {
        weight_index = weight_index | ((mode & 0x3u) << 1u);
    } else {
        weight_index = weight_index | ((mode & 0xCu) >> 1u);
    }
    weight_index = weight_index - 2u;
    if ((mode_layout != 9u) && ((mode & 0x200u) != 0u)) {
        weight_index = weight_index + 6u;
    }
    return weight_index + 1u;
}

fn DecompressBlock(coord: vec2<u32>, layer: u32) {
    let mode = StreamBits(11u);
    if (IsError(mode)) {
        FillError(coord, layer);
        return;
    }
    if ((mode & 0x1FFu) == 0x1FCu) {
        FillVoidExtentLDR(coord, layer);
        return;
    }
    let size_params = DecodeBlockSize(mode);
    if ((size_params.x > params.block_dims.x) || (size_params.y > params.block_dims.y)) {
        FillError(coord, layer);
        return;
    }
    let num_partitions = StreamBits(2u) + 1u;
    let mode_layout = FindLayout(mode);
    let dual_plane = (mode_layout != 9u) && ((mode & 0x400u) != 0u);
    if (num_partitions > 4u || (num_partitions == 4u && dual_plane)) {
        FillError(coord, layer);
        return;
    }
    var partition_index = 1u;
    var color_endpoint_mode = vec4<u32>(0u);
    var ced_pointer = 0u;
    var base_cem = 0u;
    if (num_partitions == 1u) {
        color_endpoint_mode.x = StreamBits(4u);
        partition_index = 0u;
    } else {
        partition_index = StreamBits(10u);
        base_cem = StreamBits(6u);
    }
    let base_mode = base_cem & 3u;
    let max_weight = DecodeMaxWeight(mode);
    let weight_bits = GetPackedBitSize(size_params, dual_plane, max_weight);
    var remaining_bits = 128u - weight_bits - total_bitsread;
    var extra_cem_bits = 0u;
    if (base_mode > 0u) {
        switch (num_partitions) {
            case 2u: { extra_cem_bits = 2u; }
            case 3u: { extra_cem_bits = 5u; }
            case 4u: { extra_cem_bits = 8u; }
            default: { return; }
        }
    }
    remaining_bits = remaining_bits - extra_cem_bits;
    let plane_selector_bits = select(0u, 2u, dual_plane);
    remaining_bits = remaining_bits - plane_selector_bits;
    if (remaining_bits > 128u) {
        return;
    }
    let color_data_bits = remaining_bits;
    loop {
        if (remaining_bits == 0u || ced_pointer > 3u) { break; }
        let nb = min(remaining_bits, 32u);
        let b = StreamBits(nb);
        color_endpoint_data[ced_pointer] = extractBits(b, 0u, nb);
        ced_pointer = ced_pointer + 1u;
        remaining_bits = remaining_bits - nb;
    }
    let plane_index = StreamBits(plane_selector_bits);
    if (base_mode > 0u) {
        let extra_cem = StreamBits(extra_cem_bits);
        var cem = (extra_cem << 6u) | base_cem;
        cem = cem >> 2u;
        var C = vec4<u32>(0u);
        for (var i = 0u; i < num_partitions; i = i + 1u) {
            C[i] = cem & 1u;
            cem = cem >> 1u;
        }
        var M = vec4<u32>(0u);
        for (var i = 0u; i < num_partitions; i = i + 1u) {
            M[i] = cem & 3u;
            cem = cem >> 2u;
        }
        for (var i = 0u; i < num_partitions; i = i + 1u) {
            var m = base_mode;
            if (C[i] == 0u) { m = m - 1u; }
            color_endpoint_mode[i] = (m << 2u) | M[i];
        }
    } else if (num_partitions > 1u) {
        let cem = base_cem >> 2u;
        for (var i = 0u; i < num_partitions; i = i + 1u) {
            color_endpoint_mode[i] = cem;
        }
    }

    result_vector_max_index = 32u;
    colvals_index = 0u;
    DecodeColorValues(color_endpoint_mode, num_partitions, color_data_bits);
    for (var i = 0u; i < num_partitions; i = i + 1u) {
        ComputeEndpoints(i, color_endpoint_mode[i]);
    }

    let rev = reverseBits(local_buff);
    color_endpoint_data = vec4<u32>(rev.w, rev.z, rev.y, rev.x);
    let clear_byte_start = (weight_bits >> 3u) + 1u;
    let byte_insert = ExtractBits(color_endpoint_data, (clear_byte_start - 1u) * 8u, 8u) &
                      ((1u << (weight_bits % 8u)) - 1u);
    let vec_index = (clear_byte_start - 1u) >> 2u;
    color_endpoint_data[vec_index] = insertBits(color_endpoint_data[vec_index], byte_insert,
                                                ((clear_byte_start - 1u) % 4u) * 8u, 8u);
    for (var i = clear_byte_start; i < 16u; i = i + 1u) {
        let idx = i >> 2u;
        color_endpoint_data[idx] = insertBits(color_endpoint_data[idx], 0u, (i % 4u) * 8u, 8u);
    }

    result_index = 0u;
    color_bitsread = 0u;
    result_limit_reached = false;
    result_vector_max_index = size_params.x * size_params.y;
    if (dual_plane) {
        result_vector_max_index = result_vector_max_index * 2u;
    }
    DecodeIntegerSequence(max_weight, GetNumWeightValues(size_params, dual_plane));
    UnquantizeTexelWeights(size_params, dual_plane);

    for (var j = 0u; j < params.block_dims.y; j = j + 1u) {
        for (var i = 0u; i < params.block_dims.x; i = i + 1u) {
            var local_partition = 0u;
            if (num_partitions > 1u) {
                local_partition = Select2DPartition(partition_index, i, j, num_partitions);
            }
            let local_cem = color_endpoint_mode[local_partition];
            let weight_vec = GetUnquantizedWeightVector(j, i, size_params, plane_index, dual_plane);
            var p: vec4<f32>;
            if (IsHDRColorEndpointMode(local_cem)) {
                let C0 = endpoints0[local_partition] << vec4<u32>(4u);
                let C1 = endpoints1[local_partition] << vec4<u32>(4u);
                let Cv = (C0 * (vec4<u32>(64u) - weight_vec) + C1 * weight_vec + vec4<u32>(32u)) / 64u;
                let E = (Cv & vec4<u32>(0xF800u)) >> vec4<u32>(11u);
                let M = Cv & vec4<u32>(0x7FFu);
                let Mt_lo = 3u * M;
                let Mt_mid = 4u * M - vec4<u32>(512u);
                let Mt_hi = 5u * M - vec4<u32>(2048u);
                let Mt = select(Mt_lo, select(Mt_mid, Mt_hi, M >= vec4<u32>(1536u)), M >= vec4<u32>(512u));
                let Cf = (E << vec4<u32>(10u)) + (Mt >> vec4<u32>(3u));
                let half_bits = select(Cf, vec4<u32>(0x7BFFu), Cf >= vec4<u32>(0x7C00u));
                p = vec4<f32>(unpack2x16float(half_bits.x).x, unpack2x16float(half_bits.y).x,
                              unpack2x16float(half_bits.z).x, unpack2x16float(half_bits.w).x);
                if (local_cem == 14u) {
                    let a0 = endpoints0[local_partition].x * 0x101u;
                    let a1 = endpoints1[local_partition].x * 0x101u;
                    let Ca = (a0 * (64u - weight_vec.x) + a1 * weight_vec.x + 32u) / 64u;
                    p.x = f32(Ca) / 65535.0;
                }
            } else {
                let C0 = endpoints0[local_partition] * 0x101u;
                let C1 = endpoints1[local_partition] * 0x101u;
                let Cv = (C0 * (vec4<u32>(64u) - weight_vec) + C1 * weight_vec + vec4<u32>(32u)) / 64u;
                p = vec4<f32>(Cv) / 65535.0;
            }
            Store(coord + vec2<u32>(i, j), layer, clamp(vec4<f32>(p.y, p.z, p.w, p.x), vec4<f32>(0.0), vec4<f32>(1.0)));
        }
    }
}

fn SwizzleOffset(pos: vec2<u32>) -> u32 {
    return ((pos.x & 32u) << 3u) | ((pos.y & 6u) << 5u) | ((pos.x & 16u) << 1u) | ((pos.y & 1u) << 4u) | (pos.x & 15u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = gid.xy * params.block_dims;
    let dims = textureDimensions(dest_image);
    // (no textureNumLayers: AMD loses the device on it; the dispatch is exactly as deep as the layers)
    if (coord.x >= dims.x || coord.y >= dims.y) {
        return;
    }
    let px = gid.x << 4u;
    let swizzle = SwizzleOffset(vec2<u32>(px, gid.y));
    let block_y = gid.y >> 3u;
    var offset = params.base_offset;
    offset = offset + gid.z * params.layer_stride;
    offset = offset + (block_y >> params.block_height) * params.block_size;
    offset = offset + ((block_y & params.block_height_mask) << 9u);
    offset = offset + ((px >> 6u) << params.x_shift);
    offset = offset + swizzle;
    local_buff = astc_data[offset / 16u];
    DecompressBlock(coord, gid.z);
}
)WGSL";

const char* const kUnswizzleWgsl = R"WGSL(
struct Params {
    tiles: vec2<u32>,
    bpp_log2: u32,
    layer_stride: u32,
    block_size: u32,
    x_shift: u32,
    block_height: u32,
    block_height_mask: u32,
    base_offset: u32,
    out_offset: u32,
    out_row: u32,
    out_layer: u32,
};
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.tiles.x || gid.y >= params.tiles.y) {
        return;
    }
    let px = gid.x << params.bpp_log2;   // (bytes into the row)
    let swizzle = ((px & 32u) << 3u) | ((gid.y & 6u) << 5u) | ((px & 16u) << 1u) | ((gid.y & 1u) << 4u) | (px & 15u);
    let block_y = gid.y >> 3u;
    var offset = params.base_offset;
    offset = offset + gid.z * params.layer_stride;
    offset = offset + (block_y >> params.block_height) * params.block_size;
    offset = offset + ((block_y & params.block_height_mask) << 9u);
    offset = offset + ((px >> 6u) << params.x_shift);
    offset = offset + swizzle;
    let out = params.out_offset + gid.z * params.out_layer + gid.y * params.out_row + px;
    let words = 1u << (params.bpp_log2 - 2u);
    for (var k = 0u; k < words; k = k + 1u) {
        dst[(out >> 2u) + k] = src[(offset >> 2u) + k];
    }
}
)WGSL";

struct Pass {
    WGPUComputePipeline pipeline = nullptr;
    WGPUBindGroupLayout bgl = nullptr;
    bool tried = false;
};
struct Gpu {
    std::mutex mu;
    Pass astc, unswizzle;
} gpu;

bool make_pass(Pass& p, const char* wgsl, bool texture_out, const char* what) {
    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = WGPUStringView{wgsl, std::strlen(wgsl)};
    WGPUShaderModuleDescriptor md{};
    md.nextInChain = &src.chain;
    scope_push();
    WGPUShaderModule module = program_module(&md);   // (RX_SHADER_CHECKS applies)
    WGPUBindGroupLayoutEntry e[3]{};
    e[0].binding = 0;
    e[0].visibility = WGPUShaderStage_Compute;
    e[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    e[1].binding = 1;
    e[1].visibility = WGPUShaderStage_Compute;
    if (texture_out) {
        e[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        e[1].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
        e[1].storageTexture.viewDimension = WGPUTextureViewDimension_2DArray;
    } else {
        e[1].buffer.type = WGPUBufferBindingType_Storage;
    }
    e[2].binding = 2;
    e[2].visibility = WGPUShaderStage_Compute;
    e[2].buffer.type = WGPUBufferBindingType_Uniform;
    WGPUBindGroupLayoutDescriptor ld{};
    ld.entryCount = 3;
    ld.entries = e;
    p.bgl = wgpuDeviceCreateBindGroupLayout(dev.device, &ld);
    WGPUPipelineLayoutDescriptor pld{};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &p.bgl;
    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(dev.device, &pld);
    WGPUComputePipelineDescriptor cd{};
    cd.layout = layout;
    cd.compute.module = module;
    cd.compute.entryPoint = WGPUStringView{"main", 4};
    p.pipeline = wgpuDeviceCreateComputePipeline(dev.device, &cd);
    const std::string err = scope_pop();
    wgpuPipelineLayoutRelease(layout);
    wgpuShaderModuleRelease(module);
    if (err.empty() && p.pipeline && p.bgl) return true;
    hos_log("rx: the GPU %s did not compile (the CPU does it): %s", what, err.c_str());
    p.pipeline = nullptr;
    return false;
}

/// Each pass made the first time it is asked for, and only if it is on (a pass that is off never reaches the driver)
bool pass_on(Pass& p, bool allowed, const char* wgsl, bool texture_out, const char* what) {
    if (RX_WEB) return false;   // (the browser: WebGPU calls only on the render thread; decodes run on a worker)
    if (!allowed) return false;
    std::lock_guard lk(gpu.mu);
    if (!p.tried) {
        p.tried = true;
        if (make_pass(p, wgsl, texture_out, what)) hos_log("rx: the %s runs on the GPU", what);
    }
    return p.pipeline != nullptr;
}

/// The guest bytes in a new storage buffer (written in whole words)
WGPUBuffer upload_storage(const uint8_t* src, uint64_t total, uint64_t* size) {
    *size = (total + 15) & ~15ull;
    WGPUBufferDescriptor bd{};
    bd.size = *size;
    bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    WGPUBuffer data = wgpuDeviceCreateBuffer(dev.device, &bd);
    if (!data) return nullptr;
    wgpuQueueWriteBuffer(dev.queue, data, 0, src, total & ~3ull);
    if (total & 3) {
        uint8_t tail[4] = {};
        std::memcpy(tail, src + (total & ~3ull), total & 3);
        wgpuQueueWriteBuffer(dev.queue, data, total & ~3ull, tail, 4);
    }
    return data;
}

/// One uniform slot per level (256 bytes apart: the offset alignment)
template <typename P> WGPUBuffer uniforms(const std::vector<P>& per_level) {
    std::vector<uint8_t> bytes(per_level.size() * 256, 0);
    for (size_t l = 0; l < per_level.size(); l++) std::memcpy(bytes.data() + l * 256, &per_level[l], sizeof(P));
    WGPUBufferDescriptor bd{};
    bd.size = bytes.size();
    bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    WGPUBuffer u = wgpuDeviceCreateBuffer(dev.device, &bd);
    wgpuQueueWriteBuffer(dev.queue, u, 0, bytes.data(), bytes.size());
    return u;
}

WGPUBindGroup bind(const Pass& p, WGPUBuffer in, uint64_t in_size, WGPUBuffer out, uint64_t out_size,
                   WGPUTextureView out_view, WGPUBuffer ubuf, uint32_t level, uint64_t params_size) {
    WGPUBindGroupEntry ge[3]{};
    ge[0].binding = 0;
    ge[0].buffer = in;
    ge[0].size = in_size;
    ge[1].binding = 1;
    if (out_view) {
        ge[1].textureView = out_view;
    } else {
        ge[1].buffer = out;
        ge[1].size = out_size;
    }
    ge[2].binding = 2;
    ge[2].buffer = ubuf;
    ge[2].offset = 256ull * level;
    ge[2].size = params_size;
    WGPUBindGroupDescriptor gd{};
    gd.layout = p.bgl;
    gd.entryCount = 3;
    gd.entries = ge;
    return wgpuDeviceCreateBindGroup(dev.device, &gd);
}

void submit(WGPUCommandEncoder enc) {
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
}

} // namespace

bool astc_gpu_on() { return pass_on(gpu.astc, !env_flag("RX_ASTC_CPU"), kAstcWgsl, true, "ASTC decoder"); }
bool unswizzle_gpu_on() { return pass_on(gpu.unswizzle, !env_flag("RX_UNSWIZZLE_CPU"), kUnswizzleWgsl, false, "unswizzle"); }

bool astc_gpu_decode(WGPUTexture tex, const uint8_t* src, uint64_t total, const BlockLinearLevel* levels, int nlevels,
                     uint32_t layers, uint32_t layer_stride, uint32_t block_w, uint32_t block_h) {
    if (!astc_gpu_on() || !tex || !src || !total || total > dev.limits.maxStorageBufferBindingSize) return false;
    uint64_t size;
    WGPUBuffer data = upload_storage(src, total, &size);
    if (!data) return false;
    struct Params {
        uint32_t block_w, block_h, layer_stride, block_size, x_shift, block_height, block_height_mask, base_offset;
    };
    std::vector<Params> params;
    for (int l = 0; l < nlevels; l++)
        params.push_back({block_w, block_h, layer_stride, levels[l].block_size, levels[l].x_shift, levels[l].block_height,
                          (1u << levels[l].block_height) - 1, levels[l].offset});
    WGPUBuffer ubuf = uniforms(params);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, gpu.astc.pipeline);
    std::vector<WGPUTextureView> views;
    std::vector<WGPUBindGroup> groups;
    for (int l = 0; l < nlevels; l++) {
        WGPUTextureViewDescriptor vd{};
        vd.format = WGPUTextureFormat_RGBA8Unorm;
        vd.dimension = WGPUTextureViewDimension_2DArray;
        vd.baseMipLevel = (uint32_t)l;
        vd.mipLevelCount = 1;
        vd.baseArrayLayer = 0;
        vd.arrayLayerCount = layers;
        vd.aspect = WGPUTextureAspect_All;
        vd.usage = WGPUTextureUsage_StorageBinding;
        WGPUTextureView v = wgpuTextureCreateView(tex, &vd);
        WGPUBindGroup g = bind(gpu.astc, data, size, nullptr, 0, v, ubuf, (uint32_t)l, sizeof(Params));
        wgpuComputePassEncoderSetBindGroup(pass, 0, g, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, (levels[l].tiles_w + 7) / 8, (levels[l].tiles_h + 7) / 8, layers);
        views.push_back(v);
        groups.push_back(g);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    submit(enc);
    for (WGPUBindGroup g : groups) wgpuBindGroupRelease(g);
    for (WGPUTextureView v : views) wgpuTextureViewRelease(v);
    wgpuBufferRelease(ubuf);
    wgpuBufferRelease(data);
    count("tex_astc_gpu");
    return true;
}

bool unswizzle_gpu(WGPUTexture tex, const uint8_t* src, uint64_t total, const BlockLinearLevel* levels, int nlevels,
                   uint32_t layers, uint32_t layer_stride, uint32_t bpp_log2, uint32_t block_w, uint32_t block_h) {
    if (!unswizzle_gpu_on() || !tex || !src || !total || bpp_log2 < 2 || bpp_log2 > 4 ||
        total > dev.limits.maxStorageBufferBindingSize)
        return false;
    struct Params {
        uint32_t tiles_w, tiles_h, bpp_log2, layer_stride, block_size, x_shift, block_height, block_height_mask,
            base_offset, out_offset, out_row, out_layer;
    };
    std::vector<Params> params;
    uint64_t out_size = 0;
    for (int l = 0; l < nlevels; l++) {
        const BlockLinearLevel& v = levels[l];
        const uint32_t row = ((v.tiles_w << bpp_log2) + 255) & ~255u;
        params.push_back({v.tiles_w, v.tiles_h, bpp_log2, layer_stride, v.block_size, v.x_shift, v.block_height,
                          (1u << v.block_height) - 1, v.offset, (uint32_t)out_size, row, row * v.tiles_h});
        out_size += (uint64_t)row * v.tiles_h * layers;
    }
    if (out_size > dev.limits.maxStorageBufferBindingSize || out_size >= (4ull << 30)) return false;
    uint64_t size;
    WGPUBuffer data = upload_storage(src, total, &size);
    if (!data) return false;
    WGPUBufferDescriptor bd{};
    bd.size = out_size;
    bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    WGPUBuffer out = wgpuDeviceCreateBuffer(dev.device, &bd);
    WGPUBuffer ubuf = uniforms(params);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, gpu.unswizzle.pipeline);
    std::vector<WGPUBindGroup> groups;
    for (int l = 0; l < nlevels; l++) {
        WGPUBindGroup g = bind(gpu.unswizzle, data, size, out, out_size, nullptr, ubuf, (uint32_t)l, sizeof(Params));
        wgpuComputePassEncoderSetBindGroup(pass, 0, g, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, (levels[l].tiles_w + 7) / 8, (levels[l].tiles_h + 7) / 8, layers);
        groups.push_back(g);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    for (int l = 0; l < nlevels; l++) {
        WGPUTexelCopyBufferInfo from{};
        from.buffer = out;
        from.layout.offset = params[l].out_offset;
        from.layout.bytesPerRow = params[l].out_row;
        from.layout.rowsPerImage = levels[l].tiles_h;
        WGPUTexelCopyTextureInfo to{};
        to.texture = tex;
        to.mipLevel = (uint32_t)l;
        to.aspect = WGPUTextureAspect_All;
        const WGPUExtent3D extent{levels[l].tiles_w * block_w, levels[l].tiles_h * block_h, layers};
        wgpuCommandEncoderCopyBufferToTexture(enc, &from, &to, &extent);
    }
    submit(enc);
    for (WGPUBindGroup g : groups) wgpuBindGroupRelease(g);
    wgpuBufferRelease(ubuf);
    wgpuBufferRelease(out);
    wgpuBufferRelease(data);
    count("tex_unswizzle_gpu");
    return true;
}

std::vector<uint8_t> read_level(WGPUTexture tex, uint32_t level, uint32_t layer, uint32_t width, uint32_t height,
                                uint32_t row_bytes, uint32_t rows) {
    const size_t stride = ((size_t)row_bytes + 255) & ~size_t(255), size = stride * rows;
    WGPUBufferDescriptor bd{};
    bd.size = size;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(dev.device, &bd);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    WGPUTexelCopyTextureInfo src{};
    src.texture = tex;
    src.mipLevel = level;
    src.origin = {0, 0, layer};
    src.aspect = WGPUTextureAspect_All;
    WGPUTexelCopyBufferInfo dst{};
    dst.buffer = buffer;
    dst.layout.bytesPerRow = (uint32_t)stride;
    dst.layout.rowsPerImage = rows;
    const WGPUExtent3D extent{width, height, 1};
    wgpuCommandEncoderCopyTextureToBuffer(e, &src, &dst, &extent);
    submit(e);
    struct Done {
        std::atomic<bool> complete{false};
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
    } done;
    WGPUBufferMapCallbackInfo mc{};
    mc.mode = WGPUCallbackMode_AllowSpontaneous;
    mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        auto* d = static_cast<Done*>(u1);
        d->status = status;
        d->complete = true;
    };
    mc.userdata1 = &done;
    wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, size, mc);
    while (!done.complete) wgpuDevicePoll(dev.device, true, nullptr);
    std::vector<uint8_t> bytes;
    if (done.status == WGPUMapAsyncStatus_Success) {
        const auto* px = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(buffer, 0, size));
        bytes.resize((size_t)row_bytes * rows);
        for (uint32_t y = 0; y < rows; y++) std::memcpy(&bytes[(size_t)y * row_bytes], px + y * stride, row_bytes);
        wgpuBufferUnmap(buffer);
    }
    wgpuBufferRelease(buffer);
    return bytes;
}

} // namespace rx

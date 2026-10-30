// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// WGSL backend (Rexcomp): texture and image instructions, rewritten for WGSL's separate texture/sampler model.

#include <cstdlib>
#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/profile.h"

namespace Shader::Backend::WGSL {
namespace {
const TextureImageDefinition& TextureDef(EmitContext& ctx, const IR::TextureInstInfo& info) {
    if (info.type == TextureType::Buffer) {
        throw NotImplementedException("WGSL texture buffers");
    }
    return ctx.textures.at(info.descriptor_index);
}

/// "texN,sampN": the texture and its sampler
std::string TexSamp(EmitContext& ctx, const IR::TextureInstInfo& info) {
    const auto& def{TextureDef(ctx, info)};
    return fmt::format("tex{},samp{}", def.binding, def.textures_sampler_binding());
}

std::string Tex(EmitContext& ctx, const IR::TextureInstInfo& info) {
    return fmt::format("tex{}", TextureDef(ctx, info).binding);
}

std::string Swizzled(EmitContext& ctx, const IR::TextureInstInfo& info, const std::string& expression) {
    return fmt::format("swizzle{}({})", TextureDef(ctx, info).binding, expression);
}

/// The normalized coordinates a border test looks at (xyz; axes the texture type has not are inside), or "" when
/// the type has no border (cubes, unnormalized coordinates).
std::string BorderCoords(const IR::TextureInstInfo& info, std::string_view c) {
    switch (info.type) {
    case TextureType::Color1D: return fmt::format("vec3<f32>(f32({}),0.5,0.5)", c);
    case TextureType::Color2D:
    case TextureType::ColorArray2D: return fmt::format("vec3<f32>({}.xy,0.5)", c);
    case TextureType::Color3D: return fmt::format("vec3<f32>({}.xyz)", c);
    default: return "";
    }
}

/// A sampled color with the sampler's border color outside [0, 1] on its border axes.
std::string Bordered(EmitContext& ctx, const IR::TextureInstInfo& info, std::string_view coords,
                     const std::string& expression) {
    const auto bc{BorderCoords(info, coords)};
    if (bc.empty()) {
        return expression;
    }
    return fmt::format("border{}({},{})", TextureDef(ctx, info).binding, bc, expression);
}

/// A depth comparison's reference, clamped to [0, 1] as the console compares it. WebGPU (Vulkan) clamps it only for a
/// UNORM depth texture: against a float one a point just past a shadow cascade's far plane compared 1.0x <= 1.0 and
/// fell into shadow. TOTK's moonlight cascades end within view: dark bands across the ground at night, moving with the
/// camera (gone with the clamp, 2026-10-28). RX_NO_DREF_CLAMP=1: the reference as computed (debugging).
std::string Dref(std::string_view dref) {
    static const bool no_clamp{std::getenv("RX_NO_DREF_CLAMP") != nullptr};
    return no_clamp ? std::string{dref} : fmt::format("clamp({},0.0,1.0)", dref);
}

/// The same for a depth comparison: outside, the reference is compared with the border depth.
std::string BorderedCompare(EmitContext& ctx, const IR::TextureInstInfo& info, std::string_view coords,
                            std::string_view dref, const std::string& expression) {
    const auto bc{BorderCoords(info, coords)};
    if (bc.empty()) {
        return expression;
    }
    return fmt::format("bordercmp{}({},{},{})", TextureDef(ctx, info).binding, bc, dref, expression);
}

std::string Image(EmitContext& ctx, const IR::TextureInstInfo& info) {
    if (info.type == TextureType::Buffer) {
        throw NotImplementedException("WGSL image buffers");
    }
    return fmt::format("img{}", ctx.images.at(info.descriptor_index).binding);
}

bool IsArray(const IR::TextureInstInfo& info) {
    return info.type == TextureType::ColorArray2D || info.type == TextureType::ColorArrayCube;
}

/// Sampling coordinates, with the array layer as WGSL's separate argument.
std::string SampleCoords(const IR::TextureInstInfo& info, std::string_view c) {
    switch (info.type) {
    case TextureType::Color1D:
    case TextureType::Color2D:
    case TextureType::Color2DRect:
    case TextureType::Color3D:
    case TextureType::ColorCube:
        return std::string{c};
    case TextureType::ColorArray2D:
        return fmt::format("{}.xy,i32({}.z)", c, c);
    case TextureType::ColorArrayCube:
        return fmt::format("{}.xyz,i32({}.w)", c, c);
    default:
        throw NotImplementedException("WGSL sampling of texture type {}", info.type.Value());
    }
}

/// Integer coordinates for textureLoad/textureStore (u32 inputs), with the array layer separate; `offset` (an
/// i32 vector expression or empty) is added to the texel coordinates.
std::string LoadCoords(const IR::TextureInstInfo& info, std::string_view c, std::string_view offset = {}) {
    const auto add{offset.empty() ? std::string{} : fmt::format("+{}", offset)};
    switch (info.type) {
    case TextureType::Color1D:
        return fmt::format("i32({}){}", c, add);
    case TextureType::Color2D:
    case TextureType::Color2DRect:
        return fmt::format("vec2<i32>({}){}", c, add);
    case TextureType::ColorArray2D:
        return fmt::format("vec2<i32>({}.xy){},i32({}.z)", c, add, c);
    case TextureType::Color3D:
        return fmt::format("vec3<i32>({}){}", c, add);
    default:
        throw NotImplementedException("WGSL texel loads of texture type {}", info.type.Value());
    }
}

bool OffsetAllowed(const IR::TextureInstInfo& info) {
    switch (info.type) {
    case TextureType::Color2D:
    case TextureType::Color2DRect:
    case TextureType::ColorArray2D:
    case TextureType::Color3D:
        return true;
    default:
        return false;
    }
}

/// A texel offset as a WGSL constant expression (WGSL requires one), or "" when there is none or it is not constant.
std::string ConstOffset(const IR::Value& offset) {
    if (offset.IsEmpty()) {
        return "";
    }
    if (offset.IsImmediate()) {
        return fmt::format("{}", static_cast<s32>(offset.U32()));
    }
    IR::Inst* const inst{offset.InstRecursive()};
    if (inst->AreAllArgsImmediates()) {
        switch (inst->GetOpcode()) {
        case IR::Opcode::CompositeConstructU32x2:
            return fmt::format("vec2<i32>({},{})", static_cast<s32>(inst->Arg(0).U32()),
                               static_cast<s32>(inst->Arg(1).U32()));
        case IR::Opcode::CompositeConstructU32x3:
            return fmt::format("vec3<i32>({},{},{})", static_cast<s32>(inst->Arg(0).U32()),
                               static_cast<s32>(inst->Arg(1).U32()), static_cast<s32>(inst->Arg(2).U32()));
        default:
            break;
        }
    }
    LOG_WARNING(Shader_WGSL, "Variable texture offset dropped (WGSL offsets are constants)");
    return "";
}

/// ",offset" for a sampling call, when the texture type takes one
std::string SampleOffsetArg(const IR::TextureInstInfo& info, const IR::Value& offset) {
    const auto o{ConstOffset(offset)};
    if (o.empty() && !offset.IsEmpty()) return "/*VAROFFSET*/";
    return o.empty() || !OffsetAllowed(info) ? "" : "," + o;
}

/// A texel offset as an i32 vector expression for textureLoad (variable offsets are fine here).
std::string LoadOffset(EmitContext& ctx, const IR::TextureInstInfo& info, const IR::Value& offset) {
    if (offset.IsEmpty()) {
        return "";
    }
    const auto o{ConstOffset(offset)};
    if (!o.empty()) {
        return o;
    }
    const auto v{ctx.var_alloc.Consume(offset)};
    switch (info.type) {
    case TextureType::Color1D:
        return fmt::format("i32({})", v);
    case TextureType::Color3D:
        return fmt::format("vec3<i32>({})", v);
    default:
        return fmt::format("vec2<i32>({})", v);
    }
}

IR::Inst* PrepareSparse(EmitContext& ctx, IR::Inst& inst) {
    IR::Inst* const sparse_inst{inst.GetAssociatedPseudoOperation(IR::Opcode::GetSparseFromOp)};
    if (sparse_inst) {
        // WebGPU has no sparse residency: every texel is resident
        ctx.AddU1("{}=true;", *sparse_inst);
        sparse_inst->Invalidate();
    }
    return sparse_inst;
}
} // Anonymous namespace

void EmitImageSampleImplicitLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                                std::string_view coords, std::string_view bias_lc,
                                const IR::Value& offset) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (info.has_lod_clamp) {
        throw NotImplementedException("EmitImageSampleImplicitLod Lod clamp samples");
    }
    (void)index;
    PrepareSparse(ctx, inst);
    const auto ts{TexSamp(ctx, info)};
    const auto c{SampleCoords(info, coords)};
    const auto o{SampleOffsetArg(info, offset)};
    // the sampler's LOD bias (lodbiasN, from the uniform block) is added: WebGPU samplers have none
    const auto lb{fmt::format("lodbias{}()", TextureDef(ctx, info).binding)};
    if (ctx.stage != Stage::Fragment) {
        ctx.AddF32x4("{}={};", inst, Bordered(ctx, info, coords, Swizzled(ctx, info, fmt::format("textureSampleLevel({},{},{}{})", ts, c, lb, o))));
    } else if (info.has_bias) {
        ctx.AddF32x4("{}={};", inst, Bordered(ctx, info, coords, Swizzled(ctx, info, fmt::format("textureSampleBias({},{},lodok_({}+{}){})", ts, c, bias_lc, lb, o))));
    } else if (info.type == TextureType::Color1D) {
        ctx.AddF32x4("{}={};", inst, Bordered(ctx, info, coords, Swizzled(ctx, info, fmt::format("textureSample({},{}{})", ts, c, o))));
    } else {
        ctx.AddF32x4("{}={};", inst, Bordered(ctx, info, coords, Swizzled(ctx, info, fmt::format("textureSampleBias({},{},{}{})", ts, c, lb, o))));
    }
}

void EmitImageSampleExplicitLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                                std::string_view coords, std::string_view lod_lc,
                                const IR::Value& offset) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (info.has_bias) {
        throw NotImplementedException("EmitImageSampleExplicitLod Bias texture samples");
    }
    if (info.has_lod_clamp) {
        throw NotImplementedException("EmitImageSampleExplicitLod Lod clamp samples");
    }
    (void)index;
    PrepareSparse(ctx, inst);
    ctx.AddF32x4("{}={};", inst, Bordered(ctx, info, coords, Swizzled(ctx, info, fmt::format("textureSampleLevel({},{},lodok_({}+lodbias{}()){})",
                 TexSamp(ctx, info), SampleCoords(info, coords), lod_lc, TextureDef(ctx, info).binding,
                 SampleOffsetArg(info, offset)))));
}

void EmitImageSampleDrefImplicitLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                                    std::string_view coords, std::string_view dref,
                                    std::string_view bias_lc, const IR::Value& offset) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (info.has_lod_clamp) {
        throw NotImplementedException("EmitImageSampleDrefImplicitLod Lod clamp samples");
    }
    (void)index;
    (void)bias_lc;
    PrepareSparse(ctx, inst);
    const auto fn{ctx.stage == Stage::Fragment ? "textureSampleCompare" : "textureSampleCompareLevel"};
    const auto d{Dref(dref)};
    ctx.AddF32("{}={};", inst, BorderedCompare(ctx, info, coords, d, fmt::format("{}({},{},{}{})", fn,
               TexSamp(ctx, info), SampleCoords(info, coords), d, SampleOffsetArg(info, offset))));
}

void EmitImageSampleDrefExplicitLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                                    std::string_view coords, std::string_view dref,
                                    std::string_view lod_lc, const IR::Value& offset) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (info.has_lod_clamp) {
        throw NotImplementedException("EmitImageSampleDrefExplicitLod Lod clamp samples");
    }
    (void)index;
    (void)lod_lc;   // WGSL compares at level 0 only
    PrepareSparse(ctx, inst);
    const auto d{Dref(dref)};
    ctx.AddF32("{}={};", inst, BorderedCompare(ctx, info, coords, d, fmt::format("textureSampleCompareLevel({},{},{}{})",
               TexSamp(ctx, info), SampleCoords(info, coords), d, SampleOffsetArg(info, offset))));
}

void EmitImageGather(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                     std::string_view coords, const IR::Value& offset, const IR::Value& offset2) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    (void)index;
    if (!offset2.IsEmpty()) ctx.Add("/*PTP*/");   // per-texel offsets (PTP): WGSL has none; the first offset is used
    PrepareSparse(ctx, inst);
    const bool depth{ctx.info.texture_descriptors.at(info.descriptor_index).is_depth};
    const auto component{depth ? std::string{} : fmt::format("{},", info.gather_component)};
    // A gather at a texel's center sits on the edge between two footprints: the game uses .w of such a gather as a
    // point fetch (TOTK's half-resolution lighting upsample), so 1/256 of a texel decides the footprint the way
    // Maxwell's rounding does instead of the interpolation error of the coordinate
    std::string c{coords};
    static const bool no_nudge{std::getenv("RX_NO_GATHER_NUDGE") != nullptr};   // (debugging)
    if (no_nudge) {
    } else if (info.type == TextureType::Color2D || info.type == TextureType::Color2DRect) {
        c = fmt::format("({}+vec2<f32>(0.00390625)/vec2<f32>(textureDimensions({},0)))", coords, Tex(ctx, info));
    } else if (info.type == TextureType::ColorArray2D) {
        c = fmt::format("vec3<f32>({0}.xy+vec2<f32>(0.00390625)/vec2<f32>(textureDimensions({1},0)),{0}.z)", coords,
                        Tex(ctx, info));
    }
    ctx.AddF32x4("{}=textureGather({}{},{}{});", inst, component, TexSamp(ctx, info), SampleCoords(info, c),
                 SampleOffsetArg(info, offset));
}

void EmitImageGatherDref(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                         std::string_view coords, const IR::Value& offset, const IR::Value& offset2,
                         std::string_view dref) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    (void)index;
    (void)offset2;
    PrepareSparse(ctx, inst);
    ctx.AddF32x4("{}=textureGatherCompare({},{},{}{});", inst, TexSamp(ctx, info), SampleCoords(info, coords),
                 Dref(dref), SampleOffsetArg(info, offset));
}

void EmitImageFetch(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                    std::string_view coords, const IR::Value& offset, std::string_view lod,
                    std::string_view ms) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (info.has_bias) {
        throw NotImplementedException("EmitImageFetch Bias texture samples");
    }
    if (info.has_lod_clamp) {
        throw NotImplementedException("EmitImageFetch Lod clamp samples");
    }
    (void)index;
    PrepareSparse(ctx, inst);
    const auto o{LoadOffset(ctx, info, offset)};
    const auto c{LoadCoords(info, coords, o)};
    const bool depth{ctx.info.texture_descriptors.at(info.descriptor_index).is_depth};
    const auto level{!ms.empty() ? fmt::format("i32({})", ms) : fmt::format("i32({})", lod)};
    if (depth) {
        ctx.AddF32x4("{}={};", inst, Swizzled(ctx, info, fmt::format("vec4<f32>(textureLoad({},{},{}),0.0f,0.0f,1.0f)", Tex(ctx, info), c, level)));
    } else {
        ctx.AddF32x4("{}={};", inst, Swizzled(ctx, info, fmt::format("textureLoad({},{},{})", Tex(ctx, info), c, level)));
    }
}

void EmitImageQueryDimensions(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                              std::string_view lod, const IR::Value& skip_mips_val) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    (void)index;
    const auto t{Tex(ctx, info)};
    const bool is_msaa{ctx.info.texture_descriptors.at(info.descriptor_index).is_multisample};
    const auto mips{skip_mips_val.U1() || is_msaa ? std::string{"0u"} : fmt::format("textureNumLevels({})", t)};
    const auto dims{is_msaa ? fmt::format("textureDimensions({})", t)
                            : fmt::format("textureDimensions({},u32({}))", t, lod)};
    switch (info.type) {
    case TextureType::Color1D:
        return ctx.AddU32x4("{}=vec4<u32>({},0u,0u,{});", inst, dims, mips);
    case TextureType::Color2D:
    case TextureType::Color2DRect:
    case TextureType::ColorCube:
        return ctx.AddU32x4("{}=vec4<u32>({},0u,{});", inst, dims, mips);
    case TextureType::ColorArray2D:
    case TextureType::ColorArrayCube:
        return ctx.AddU32x4("{}=vec4<u32>({},textureNumLayers({}),{});", inst, dims, t, mips);
    case TextureType::Color3D:
        return ctx.AddU32x4("{}=vec4<u32>({},{});", inst, dims, mips);
    default:
        throw NotImplementedException("WGSL dimension query of texture type {}", info.type.Value());
    }
}

void EmitImageQueryLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                       std::string_view coords) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    (void)index;
    if (ctx.stage != Stage::Fragment || (info.type != TextureType::Color2D && info.type != TextureType::ColorArray2D)) {
        ctx.AddF32x4("{}=vec4<f32>(0.0f);", inst);
        return;
    }
    // WGSL has no textureQueryLod: the level from the screen-space derivatives of the texel coordinates
    const auto t{Tex(ctx, info)};
    ctx.AddF32x4("{}=vec4<f32>(vec2<f32>(max(0.5f*log2(max(dot(dpdx({}.xy*vec2<f32>(textureDimensions({},0u))),"
                 "dpdx({}.xy*vec2<f32>(textureDimensions({},0u)))),dot(dpdy({}.xy*vec2<f32>(textureDimensions({},0u))),"
                 "dpdy({}.xy*vec2<f32>(textureDimensions({},0u)))))),0.0f)),0.0f,0.0f);",
                 inst, coords, t, coords, t, coords, t, coords, t);
}

void EmitImageGradient(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                       std::string_view coords, const IR::Value& derivatives,
                       const IR::Value& offset, [[maybe_unused]] const IR::Value& lod_clamp) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (info.has_lod_clamp) {
        throw NotImplementedException("EmitImageGradient Lod clamp samples");
    }
    (void)index;
    PrepareSparse(ctx, inst);
    const auto d{ctx.var_alloc.Consume(derivatives)};
    const auto ts{TexSamp(ctx, info)};
    const auto c{SampleCoords(info, coords)};
    if (info.num_derivatives >= 3) {
        const auto o{ctx.var_alloc.Consume(offset)};
        ctx.AddF32x4("{}={};", inst, Swizzled(ctx, info, fmt::format("textureSampleGrad({},{},vec3<f32>({}.xz,{}.x),vec3<f32>({}.yw,{}.y))", ts, c, d, o, d, o)));
        return;
    }
    if (info.num_derivatives == 2) {
        // (the LOD bias as scaled derivatives)
        ctx.AddF32x4("{}={};", inst, Bordered(ctx, info, coords, Swizzled(ctx, info, fmt::format("textureSampleGrad({},{},{}.xz*exp2(lodbias{}()),{}.yw*exp2(lodbias{}()){})", ts, c, d,
                     TextureDef(ctx, info).binding, d, TextureDef(ctx, info).binding, SampleOffsetArg(info, offset)))));
        return;
    }
    throw NotImplementedException("WGSL 1D gradient samples");
}

void EmitImageRead(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                   std::string_view coords) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    (void)index;
    PrepareSparse(ctx, inst);
    ctx.AddU32x4("{}=textureLoad({},{});", inst, Image(ctx, info), LoadCoords(info, coords));
}

void EmitImageWrite(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                    std::string_view coords, std::string_view color) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    (void)index;
    ctx.Add("textureStore({},{},{});", Image(ctx, info), LoadCoords(info, coords), color);
}

#define IMAGE_ATOMIC(name)                                                                         \
    void EmitImageAtomic##name(EmitContext&, IR::Inst&, const IR::Value&, std::string_view,       \
                               std::string_view) {                                                 \
        throw NotImplementedException("WGSL image atomics");                                      \
    }
IMAGE_ATOMIC(IAdd32)
IMAGE_ATOMIC(SMin32)
IMAGE_ATOMIC(UMin32)
IMAGE_ATOMIC(SMax32)
IMAGE_ATOMIC(UMax32)
IMAGE_ATOMIC(Inc32)
IMAGE_ATOMIC(Dec32)
IMAGE_ATOMIC(And32)
IMAGE_ATOMIC(Or32)
IMAGE_ATOMIC(Xor32)
IMAGE_ATOMIC(Exchange32)
#undef IMAGE_ATOMIC

void EmitIsTextureScaled(EmitContext& ctx, IR::Inst& inst, const IR::Value& index) {
    if (!index.IsImmediate()) {
        throw NotImplementedException("Non-constant texture rescaling");
    }
    ctx.AddU1("{}=(bitcast<u32>(scaling().x)&{}u)!=0u;", inst, 1u << index.U32());
}

void EmitIsImageScaled(EmitContext& ctx, IR::Inst& inst, const IR::Value& index) {
    if (!index.IsImmediate()) {
        throw NotImplementedException("Non-constant texture rescaling");
    }
    ctx.AddU1("{}=(bitcast<u32>(scaling().y)&{}u)!=0u;", inst, 1u << index.U32());
}

void EmitBindlessImageSampleImplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageSampleExplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageSampleDrefImplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageSampleDrefExplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageGather(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageGatherDref(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageFetch(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageQueryDimensions(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageQueryLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageGradient(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageRead(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageWrite(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageSampleImplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageSampleExplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageSampleDrefImplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageSampleDrefExplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageGather(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageGatherDref(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageFetch(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageQueryDimensions(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageQueryLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageGradient(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageRead(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageWrite(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicIAdd32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicSMin32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicUMin32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicSMax32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicUMax32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicInc32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicDec32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicAnd32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicOr32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicXor32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicExchange32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicIAdd32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicSMin32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicUMin32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicSMax32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicUMax32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicInc32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicDec32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicAnd32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicOr32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicXor32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicExchange32(EmitContext&) {
    NotImplemented();
}

} // namespace Shader::Backend::WGSL

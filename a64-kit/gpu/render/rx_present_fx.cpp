// What the window's present does with the game's frame: keeps its aspect (black bars where the window is wider or
// taller; RX_ASPECT=stretch fills it) and, when that is larger than the frame, upscales it with AMD FidelityFX Super
// Resolution 1 (EASU, the edge-adaptive upscale, then RCAS, the sharpening) instead of a bilinear stretch.
//   RX_FSR=0: bilinear      RX_FSR_SHARPNESS=<stops> (0 the sharpest, 2 soft; default 0.25)
// The player changes them while playing (rx_present_hotkey: F8 FSR on/off, F9 sharpness, F10 fit/stretch; F7 frame
// interpolation), kept in <user>/present.txt.
// Both passes run on native (wgpu) and the web (WebGPU) alike: fragment shaders on a full-window triangle.
//
// EASU and RCAS below are a WGSL port of ffx_fsr1.h (FidelityFX Super Resolution 1.0):
//   Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
//   to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above
//   copyright notice and this permission notice shall be included in all copies or substantial portions of the
//   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
//   LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
//   SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
//   OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE.
#include "rx_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <string>

extern "C" const char* hos_user_dir;
extern "C" float hos_view_aspect;   // (hos/totk_ultrawide.c: the game's cameras' aspect; 0 their 16:9)
extern "C" void hos_ultrawide_apply(void);

namespace rx {
namespace {

// ---- the settings: the environment's, then the player's (present.txt), changed by hotkeys ----
const float kSharpness[] = {0.0f, 0.25f, 0.5f, 1.0f, 2.0f};
struct Settings {
    bool fsr = true, stretch = false, framegen = false, ultrawide = false;
    float sharpness = 0.25f;
};
std::mutex settings_mu;
Settings settings;
bool settings_read;
std::string settings_path() {
    return std::string(hos_user_dir ? hos_user_dir : "user") + "/present.txt";
}
Settings current_settings() {
    std::lock_guard lk(settings_mu);
    if (!settings_read) {
        settings_read = true;
        if (const char* v = getenv("RX_FSR")) settings.fsr = *v != '0';
        if (const char* v = getenv("RX_FSR_SHARPNESS")) settings.sharpness = std::clamp((float)atof(v), 0.0f, 2.0f);
        if (const char* v = getenv("RX_ASPECT")) settings.stretch = !strcmp(v, "stretch");
        if (const char* v = getenv("RX_FRAMEGEN")) settings.framegen = *v != '0';
        if (const char* v = getenv("RX_ULTRAWIDE")) settings.ultrawide = *v != '0';
        if (FILE* f = fopen(settings_path().c_str(), "r")) {   // (the player's choices win over the defaults)
            char key[32];
            float val;
            while (fscanf(f, "%31s %f", key, &val) == 2) {
                if (!strcmp(key, "fsr")) settings.fsr = val != 0;
                else if (!strcmp(key, "sharpness")) settings.sharpness = std::clamp(val, 0.0f, 2.0f);
                else if (!strcmp(key, "stretch")) settings.stretch = val != 0;
                else if (!strcmp(key, "framegen")) settings.framegen = val != 0;
                else if (!strcmp(key, "ultrawide")) settings.ultrawide = val != 0;
            }
            fclose(f);
        }
    }
    return settings;
}
void save_settings(const Settings& s) {
    if (FILE* f = fopen(settings_path().c_str(), "w")) {
        fprintf(f, "fsr %d\nsharpness %.2f\nstretch %d\nframegen %d\nultrawide %d\n", s.fsr, s.sharpness, s.stretch,
                s.framegen, s.ultrawide);
        fclose(f);
    }
}

const char* kPresentWgsl = R"(
struct P { con0: vec4<f32>, src: vec4<f32>, rcas: vec4<f32>, rect: vec4<f32> };
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> p: P;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

// the source texel q, clamped to the image (p.src.xy: its size)
fn ld(q: vec2<i32>) -> vec3<f32> {
  return textureLoad(tex, clamp(q, vec2<i32>(0), vec2<i32>(p.src.xy) - vec2<i32>(1)), 0).rgb;
}
// luma times 2 (FSR's cheap approximation)
fn luma(c: vec3<f32>) -> f32 { return c.g + 0.5 * (c.r + c.b); }

// ---- EASU: edge-adaptive spatial upsampling (12 taps; output pixel -> a direction and an anisotropic lobe) ----
fn easu_set(dir: ptr<function, vec2<f32>>, len: ptr<function, f32>, w: f32, lA: f32, lB: f32, lC: f32, lD: f32, lE: f32) {
  let dirX = lD - lB;
  let dX = max(abs(lD - lC), abs(lC - lB));
  var lenX = select(0.0, clamp(abs(dirX) / dX, 0.0, 1.0), dX > 0.0);
  lenX = lenX * lenX;
  let dirY = lE - lA;
  let dY = max(abs(lE - lC), abs(lC - lA));
  var lenY = select(0.0, clamp(abs(dirY) / dY, 0.0, 1.0), dY > 0.0);
  lenY = lenY * lenY;
  *dir = *dir + vec2<f32>(dirX, dirY) * w;
  *len = *len + (lenX + lenY) * w;
}
fn easu_tap(aC: ptr<function, vec3<f32>>, aW: ptr<function, f32>, off: vec2<f32>, dir: vec2<f32>, len: vec2<f32>,
            lob: f32, clp: f32, c: vec3<f32>) {
  let v = vec2<f32>(dot(off, dir), dot(off, vec2<f32>(-dir.y, dir.x))) * len;
  let d2 = min(dot(v, v), clp);
  var wB = 0.4 * d2 - 1.0;
  var wA = lob * d2 - 1.0;
  wB = wB * wB;
  wA = wA * wA;
  wB = 1.5625 * wB - 0.5625;
  let w = wB * wA;
  *aC = *aC + c * w;
  *aW = *aW + w;
}
@fragment fn easu(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  // the output pixel's position in the source (texel centers at integers)
  var pp = floor(pos.xy) * p.con0.xy + p.con0.zw;
  let fp = floor(pp);
  pp = pp - fp;
  let f0 = vec2<i32>(fp);
  //    b c
  //  e f g h
  //  i j k l
  //    n o
  let bC = ld(f0 + vec2<i32>(0, -1)); let cC = ld(f0 + vec2<i32>(1, -1));
  let eC = ld(f0 + vec2<i32>(-1, 0)); let fC = ld(f0);                    let gC = ld(f0 + vec2<i32>(1, 0)); let hC = ld(f0 + vec2<i32>(2, 0));
  let iC = ld(f0 + vec2<i32>(-1, 1)); let jC = ld(f0 + vec2<i32>(0, 1)); let kC = ld(f0 + vec2<i32>(1, 1)); let lC = ld(f0 + vec2<i32>(2, 1));
  let nC = ld(f0 + vec2<i32>(0, 2)); let oC = ld(f0 + vec2<i32>(1, 2));
  let bL = luma(bC); let cL = luma(cC); let eL = luma(eC); let fL = luma(fC); let gL = luma(gC); let hL = luma(hC);
  let iL = luma(iC); let jL = luma(jC); let kL = luma(kC); let lL = luma(lC); let nL = luma(nC); let oL = luma(oC);
  // the direction and edge length, bilinearly weighted over the 4 nearest texels' neighbourhoods
  var dir = vec2<f32>(0.0);
  var len = 0.0;
  easu_set(&dir, &len, (1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL);
  easu_set(&dir, &len, pp.x * (1.0 - pp.y), cL, fL, gL, hL, kL);
  easu_set(&dir, &len, (1.0 - pp.x) * pp.y, fL, iL, jL, kL, nL);
  easu_set(&dir, &len, pp.x * pp.y, gL, jL, kL, lL, oL);
  let dir2 = dir * dir;
  var dirR = dir2.x + dir2.y;
  let zro = dirR < (1.0 / 32768.0);
  dirR = select(inverseSqrt(dirR), 1.0, zro);
  dir = select(dir, vec2<f32>(1.0, dir.y), zro) * dirR;
  len = len * 0.5;
  len = len * len;
  let stretch = dot(dir, dir) / max(abs(dir.x), abs(dir.y));
  let len2 = vec2<f32>(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
  let lob = 0.5 - 0.29 * len;
  let clp = 1.0 / lob;
  let min4 = min(min(fC, gC), min(jC, kC));
  let max4 = max(max(fC, gC), max(jC, kC));
  var aC = vec3<f32>(0.0);
  var aW = 0.0;
  easu_tap(&aC, &aW, vec2<f32>(0.0, -1.0) - pp, dir, len2, lob, clp, bC);
  easu_tap(&aC, &aW, vec2<f32>(1.0, -1.0) - pp, dir, len2, lob, clp, cC);
  easu_tap(&aC, &aW, vec2<f32>(-1.0, 1.0) - pp, dir, len2, lob, clp, iC);
  easu_tap(&aC, &aW, vec2<f32>(0.0, 1.0) - pp, dir, len2, lob, clp, jC);
  easu_tap(&aC, &aW, vec2<f32>(0.0, 0.0) - pp, dir, len2, lob, clp, fC);
  easu_tap(&aC, &aW, vec2<f32>(-1.0, 0.0) - pp, dir, len2, lob, clp, eC);
  easu_tap(&aC, &aW, vec2<f32>(1.0, 1.0) - pp, dir, len2, lob, clp, kC);
  easu_tap(&aC, &aW, vec2<f32>(2.0, 1.0) - pp, dir, len2, lob, clp, lC);
  easu_tap(&aC, &aW, vec2<f32>(2.0, 0.0) - pp, dir, len2, lob, clp, hC);
  easu_tap(&aC, &aW, vec2<f32>(1.0, 0.0) - pp, dir, len2, lob, clp, gC);
  easu_tap(&aC, &aW, vec2<f32>(1.0, 2.0) - pp, dir, len2, lob, clp, oC);
  easu_tap(&aC, &aW, vec2<f32>(0.0, 2.0) - pp, dir, len2, lob, clp, nC);
  // normalized, and clamped to the 4 nearest texels (no ringing)
  let c = select(fC, aC / aW, abs(aW) > 1e-6);
  return vec4<f32>(min(max4, max(min4, c)), 1.0);
}

// ---- RCAS: robust contrast-adaptive sharpening (5 taps; the lobe limited so it never clips) ----
@fragment fn rcas(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let ip = vec2<i32>(floor(pos.xy - p.rect.xy));
  let b = ld(ip + vec2<i32>(0, -1));
  let d = ld(ip + vec2<i32>(-1, 0));
  let e = ld(ip);
  let f = ld(ip + vec2<i32>(1, 0));
  let h = ld(ip + vec2<i32>(0, 1));
  let mn4 = min(min(b, d), min(f, h));
  let mx4 = max(max(b, d), max(f, h));
  let hitMin = mn4 / max(4.0 * mx4, vec3<f32>(1e-5));
  let hitMax = (1.0 - mx4) / min(4.0 * mn4 - 4.0, vec3<f32>(-1e-5));
  let lobeRGB = max(-hitMin, hitMax);
  let lobe = max(-0.1875, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * p.rcas.x;
  return vec4<f32>((lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0), 1.0);
}

// ---- bilinear (RX_FSR=0, or no upscale): the source stretched over the rectangle ----
@fragment fn bilinear(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let s = (pos.xy - p.rect.xy) * p.src.xy / p.rect.zw - 0.5;
  let i = vec2<i32>(floor(s));
  let t = s - floor(s);
  let c = mix(mix(ld(i), ld(i + vec2<i32>(1, 0)), t.x), mix(ld(i + vec2<i32>(0, 1)), ld(i + vec2<i32>(1, 1)), t.x), t.y);
  return vec4<f32>(c, 1.0);
}
)";

struct Params {
    float con0[4], src[4], rcas[4], rect[4];
};

struct State {
    WGPUTextureFormat target_format = WGPUTextureFormat_Undefined;
    WGPUBindGroupLayout layout = nullptr;
    WGPURenderPipeline easu = nullptr, rcas = nullptr, bilinear = nullptr;
    WGPUBuffer params[2] = {};   // (the EASU pass's and the window pass's)
    WGPUTexture mid = nullptr;   // EASU's output, the rectangle's size
    WGPUTextureView mid_view = nullptr;
    uint32_t mid_w = 0, mid_h = 0;
} st;

WGPURenderPipeline make_pipeline(WGPUShaderModule mod, WGPUPipelineLayout pl, const char* fs, WGPUTextureFormat format) {
    WGPUColorTargetState target{};
    target.format = format;
    target.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState f{};
    f.module = mod;
    f.entryPoint = sv(fs);
    f.targetCount = 1;
    f.targets = &target;
    WGPURenderPipelineDescriptor rd{};
    rd.layout = pl;
    rd.vertex.module = mod;
    rd.vertex.entryPoint = sv("vs");
    rd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    rd.primitive.cullMode = WGPUCullMode_None;
    rd.primitive.frontFace = WGPUFrontFace_CCW;
    rd.multisample.count = 1;
    rd.multisample.mask = 0xFFFFFFFF;
    rd.fragment = &f;
    return wgpuDeviceCreateRenderPipeline(dev.device, &rd);
}

void init(WGPUTextureFormat target_format) {
    if (st.layout && st.target_format == target_format) return;
    st.target_format = target_format;
    if (!st.layout) {
        WGPUBindGroupLayoutEntry e[2]{};
        e[0].binding = 0;
        e[0].visibility = WGPUShaderStage_Fragment;
        e[0].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        e[0].texture.viewDimension = WGPUTextureViewDimension_2D;
        e[1].binding = 1;
        e[1].visibility = WGPUShaderStage_Fragment;
        e[1].buffer.type = WGPUBufferBindingType_Uniform;
        e[1].buffer.minBindingSize = sizeof(Params);
        WGPUBindGroupLayoutDescriptor bd{};
        bd.entryCount = 2;
        bd.entries = e;
        st.layout = wgpuDeviceCreateBindGroupLayout(dev.device, &bd);
        for (WGPUBuffer& b : st.params) {
            WGPUBufferDescriptor d{};
            d.size = sizeof(Params);
            d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
            b = wgpuDeviceCreateBuffer(dev.device, &d);
        }
    }
    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = sv(kPresentWgsl);
    WGPUShaderModuleDescriptor md{};
    md.nextInChain = &src.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(dev.device, &md);
    WGPUPipelineLayoutDescriptor pd{};
    pd.bindGroupLayoutCount = 1;
    pd.bindGroupLayouts = &st.layout;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(dev.device, &pd);
    for (WGPURenderPipeline* p : {&st.easu, &st.rcas, &st.bilinear})
        if (*p) wgpuRenderPipelineRelease(*p);
    st.easu = make_pipeline(mod, pl, "easu", WGPUTextureFormat_RGBA8Unorm);
    st.rcas = make_pipeline(mod, pl, "rcas", target_format);
    st.bilinear = make_pipeline(mod, pl, "bilinear", target_format);
    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(mod);
}

void ensure_mid(uint32_t w, uint32_t h) {
    if (st.mid && st.mid_w == w && st.mid_h == h) return;
    if (st.mid_view) wgpuTextureViewRelease(st.mid_view);
    if (st.mid) wgpuTextureRelease(st.mid);
    WGPUTextureDescriptor d{};
    d.label = sv("rx FSR upscale");
    d.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
    d.dimension = WGPUTextureDimension_2D;
    d.size = {w, h, 1};
    d.format = WGPUTextureFormat_RGBA8Unorm;
    d.mipLevelCount = 1;
    d.sampleCount = 1;
    st.mid = wgpuDeviceCreateTexture(dev.device, &d);
    st.mid_view = wgpuTextureCreateView(st.mid, nullptr);
    st.mid_w = w;
    st.mid_h = h;
}

/// A pass drawing one full-window triangle with `pipe` over `rect` of `target` (cleared to `clear` first)
void draw_pass(WGPUCommandEncoder e, WGPUTextureView target, const WGPUColor& clear, WGPURenderPipeline pipe,
               WGPUTextureView src, WGPUBuffer params, const float rect[4]) {
    WGPURenderPassColorAttachment ca{};
    ca.view = target;
    ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    ca.loadOp = WGPULoadOp_Clear;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = clear;
    WGPURenderPassDescriptor pd{};
    pd.colorAttachmentCount = 1;
    pd.colorAttachments = &ca;
    WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(e, &pd);
    WGPUBindGroupEntry be[2]{};
    be[0].binding = 0;
    be[0].textureView = src;
    be[1].binding = 1;
    be[1].buffer = params;
    be[1].size = sizeof(Params);
    WGPUBindGroupDescriptor bd{};
    bd.layout = st.layout;
    bd.entryCount = 2;
    bd.entries = be;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev.device, &bd);
    wgpuRenderPassEncoderSetViewport(p, rect[0], rect[1], rect[2], rect[3], 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(p, (uint32_t)rect[0], (uint32_t)rect[1], (uint32_t)rect[2], (uint32_t)rect[3]);
    wgpuRenderPassEncoderSetPipeline(p, pipe);
    wgpuRenderPassEncoderSetBindGroup(p, 0, bg, 0, nullptr);
    wgpuRenderPassEncoderDraw(p, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(p);
    wgpuRenderPassEncoderRelease(p);
    wgpuBindGroupRelease(bg);
}
}  // namespace

PresentRect present_rect(uint32_t sw, uint32_t sh, uint32_t tw, uint32_t th) {
    const Settings set = current_settings();
    // (ultrawide: the frame holds the window's field of view, so it is stretched over the window)
    const bool stretch = set.stretch || (set.ultrawide && hos_view_aspect > 0);
    PresentRect r{0, 0, tw, th};
    if (stretch || !sw || !sh || !tw || !th) return r;
    // the largest rectangle of the frame's aspect that fits, centred (pixels: at most one off the exact ratio)
    if ((uint64_t)tw * sh > (uint64_t)th * sw) {
        r.w = std::max<uint32_t>(1, (uint32_t)((uint64_t)th * sw / sh));
        r.x = (tw - r.w) / 2;
    } else {
        r.h = std::max<uint32_t>(1, (uint32_t)((uint64_t)tw * sh / sw));
        r.y = (th - r.h) / 2;
    }
    return r;
}

void present_fx_draw(WGPUCommandEncoder e, WGPUTextureView src, uint32_t sw, uint32_t sh, WGPUTextureView target,
                     WGPUTextureFormat target_format, uint32_t tw, uint32_t th, const WGPUColor& clear) {
    init(target_format);
    const Settings set = current_settings();
    const float sharpness = set.sharpness;
    static std::string logged;
    const PresentRect r = present_rect(sw, sh, tw, th);
    const float rect[4] = {(float)r.x, (float)r.y, (float)r.w, (float)r.h};
    const bool upscale = set.fsr && (r.w > sw || r.h > sh);
    char what[96];
    std::snprintf(what, sizeof what, "%ux%u %ux%u %ux%u %d", sw, sh, r.w, r.h, tw, th, upscale);
    if (logged != what) {   // (each new size or mode)
        logged = what;
        hos_log("rx: present: %ux%u frames into %ux%u at %u,%u of a %ux%u window, %s", sw, sh, r.w, r.h, r.x, r.y, tw, th,
                upscale ? "FSR 1 (EASU + RCAS)" : "bilinear");
    }
    Params pm{};
    pm.src[0] = (float)sw;
    pm.src[1] = (float)sh;
    pm.rect[0] = rect[0];
    pm.rect[1] = rect[1];
    pm.rect[2] = rect[2];
    pm.rect[3] = rect[3];
    if (!upscale) {
        wgpuQueueWriteBuffer(dev.queue, st.params[1], 0, &pm, sizeof pm);
        draw_pass(e, target, clear, st.bilinear, src, st.params[1], rect);
        return;
    }
    ensure_mid(r.w, r.h);
    Params easu = pm;
    easu.con0[0] = (float)sw / (float)r.w;
    easu.con0[1] = (float)sh / (float)r.h;
    easu.con0[2] = 0.5f * easu.con0[0] - 0.5f;
    easu.con0[3] = 0.5f * easu.con0[1] - 0.5f;
    const float whole[4] = {0, 0, (float)r.w, (float)r.h};
    std::memcpy(easu.rect, whole, sizeof whole);
    wgpuQueueWriteBuffer(dev.queue, st.params[0], 0, &easu, sizeof easu);
    draw_pass(e, st.mid_view, WGPUColor{0, 0, 0, 1}, st.easu, src, st.params[0], whole);
    Params sharpen = pm;
    sharpen.src[0] = (float)r.w;   // (RCAS reads EASU's output)
    sharpen.src[1] = (float)r.h;
    sharpen.rcas[0] = std::exp2(-sharpness);
    wgpuQueueWriteBuffer(dev.queue, st.params[1], 0, &sharpen, sizeof sharpen);
    draw_pass(e, target, clear, st.rcas, st.mid_view, st.params[1], rect);
}
bool present_framegen() { return current_settings().framegen; }

/// The window's size (rx_resize): with ultrawide on, a window wider than 16:9 gives the game's cameras its aspect
void present_window_size(int32_t w, int32_t h) {
    const float a = w > 0 && h > 0 ? (float)w / (float)h : 0.0f;
    hos_view_aspect = current_settings().ultrawide && a > 16.0f / 9.0f + 0.01f ? std::min(a, 4.0f) : 0.0f;
    hos_ultrawide_apply();
}
}  // namespace rx

/* A hotkey (7..10: F7 frame interpolation, F8 FSR, F9 sharpness, F10 fit/stretch; 0: none): the settings after it,
   as a line for the window's title */
extern "C" const char* rx_present_hotkey(int key) {
    using namespace rx;
    Settings s = current_settings();
    if (key) {
        if (key == 7) s.framegen = !s.framegen;
        if (key == 8) s.fsr = !s.fsr;
        if (key == 9) {
            int i = 0;
            while (i < 5 && kSharpness[i] < s.sharpness - 0.01f) i++;
            s.sharpness = kSharpness[(i + 1) % 5];
        }
        if (key == 10) s.stretch = !s.stretch;
        if (key == 6) s.ultrawide = !s.ultrawide;
        {
            std::lock_guard lk(settings_mu);
            settings = s;
        }
        save_settings(s);
    }
    static thread_local char line[160];
    std::snprintf(line, sizeof line,
                  "FSR %s (F8), sharpness %.2f (F9), %s (F10), frame interpolation %s (F7), ultrawide %s (F6)",
                  s.fsr ? "on" : "off", s.sharpness, s.stretch ? "stretched" : "aspect kept", s.framegen ? "on" : "off",
                  s.ultrawide ? (hos_view_aspect > 0 ? "on" : "on (the window is 16:9)") : "off");
    return line;
}

extern "C" void rx_present_get_settings(int* fsr, int* sharpness_idx, int* framegen, int* ultrawide, int* stretch) {
    using namespace rx;
    Settings s = current_settings();
    if (fsr) *fsr = s.fsr ? 1 : 0;
    if (framegen) *framegen = s.framegen ? 1 : 0;
    if (ultrawide) *ultrawide = s.ultrawide ? 1 : 0;
    if (stretch) *stretch = s.stretch ? 1 : 0;
    if (sharpness_idx) {
        int i = 0;
        while (i < 5 && kSharpness[i] < s.sharpness - 0.01f) i++;
        *sharpness_idx = (i < 5) ? i : 4;
    }
}

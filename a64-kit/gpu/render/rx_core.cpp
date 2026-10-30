// The renderer's core: the WebGPU device and the window's surface, batches (one command encoder and its upload
// memory per GPU submission), render targets, clears, the 3D engine's draw methods (suyu's DrawManager), present.
#include <intrin.h>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdarg>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <set>
#include <functional>
#include <mutex>
#include <thread>
#include <cstdlib>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "rx_internal.h"
#include "video_core/textures/decoders.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/threading.h>
extern "C" void rx_web_canvas_init(int w, int h);   // hos/web/library.js
extern "C" void rx_web_present(void);
namespace rx { extern Device dev; }
static thread_local bool on_render_thread;   // (the thread that makes every WebGPU call)
// Back to the worker's event loop, where WebGPU's callbacks run and the canvas is shown. Not emscripten_sleep: a
// timer, which the browser holds to 4 ms once timers nest, and every present, readback wait and idle turn paid it.
// (one channel a worker, kept: making and closing one at each yield was ~2.5% of the render worker at the castle)
EM_ASYNC_JS(void, rx_web_yield, (), {
    if (!globalThis.rxYield) {
        const c = new MessageChannel(), waiting = [];
        c.port1.onmessage = () => { const resolve = waiting.shift(); if (resolve) resolve(); };
        globalThis.rxYield = (resolve) => { waiting.push(resolve); c.port2.postMessage(0); };
    }
    await new Promise(globalThis.rxYield);
});
// Until the word at addr is no longer value (Atomics.notify: emscripten_futex_wake), or ms pass.
EM_ASYNC_JS(void, rx_web_wait_word, (uint32_t addr, uint32_t value, double ms), {
    const r = Atomics.waitAsync(HEAP32, addr >>> 2, value | 0, ms);
    if (r.async) await r.value;
});
bool wgpuDevicePoll(WGPUDevice device, bool wait, const void* submission) {
    (void)device;
    (void)submission;
    wgpuInstanceProcessEvents(rx::dev.instance);
    if (wait) rx_web_yield();   // (the callbacks run on this worker's event loop)
    return true;
}
#endif

extern "C" const char* hos_user_dir;
namespace rx {
bool draw_must_run = false;
bool rt_history_on() { static const bool on = getenv("RX_RT_HISTORY") != nullptr; return on; }

/// A short wait for WebGPU's callbacks: on the web the worker's event loop must run for them (JSPI suspends us).
void wait_step() {
#ifdef __EMSCRIPTEN__
    emscripten_sleep(1);
#else
    Sleep(1);
#endif
}

Device dev;
RecursiveLock mu;
void RecursiveLock::lock() {
    const unsigned long me = GetCurrentThreadId();
    if (owner_.load(std::memory_order_relaxed) == me) {
        depth_++;
        return;
    }
    AcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(&srw_));
    owner_.store(me, std::memory_order_relaxed);
    depth_ = 1;
}
void RecursiveLock::unlock() {
    if (--depth_) return;
    owner_.store(0, std::memory_order_relaxed);
    ReleaseSRWLockExclusive(reinterpret_cast<PSRWLOCK>(&srw_));
}
int64_t frame_number;
int64_t st_draws, st_skipped, st_pipelines, st_clears;
Stream stream_ro, stream_rw;

namespace {
bool started;
WGPUSurface surface;
WGPUTextureFormat surface_format = WGPUTextureFormat_BGRA8Unorm;
WGPUPresentMode present_mode = WGPUPresentMode_Fifo;
uint32_t surface_w, surface_h;
volatile LONG want_w, want_h;
WGPURenderPipeline blit_pipeline;
WGPUBindGroupLayout blit_layout;
WGPUSampler blit_sampler;

WGPUCommandEncoder enc;
WGPURenderPassEncoder pass;
Attachments pass_at;

std::unordered_map<uint64_t, std::unique_ptr<Image>> images;   // by guest address
// Surfaces the game drew at an address before the one there now (it reuses memory for different transient targets
// within a frame): kept for when that surface comes back, instead of a texture created and destroyed every frame.
std::unordered_multimap<uint64_t, std::unique_ptr<Image>> spare_images;
uint64_t image_versions;   // render target versions come from one counter: no two images share one
std::atomic<int64_t> count_parts, st_finish_ns, st_write_ns, st_submit_ns, st_after_ns, st_wait_ns;   // the submit thread's
std::atomic<int64_t> st_writeback_words;
uint64_t images_changed = 1;   // bumped whenever which image is at an address changes (see images_generation)
struct ImageChange {
    uint64_t generation, guest;
};
ImageChange image_changes[256];   // the last changes, by generation (images_changed_in)
void note_image_change(uint64_t guest) {
    images_changed++;
    image_changes[images_changed & 255] = {images_changed, guest};
}
// (by the key's address: the keys are literals; a string key was built and hashed per call, every draw for some)
std::unordered_map<const char*, int> log_counts;

void on_error(WGPUDevice const*, WGPUErrorType type, WGPUStringView message, void*, void*) {
    static int n;
    if (n++ < 60) {
        std::string m = str(message);
        if (m.size() > 1500) m.resize(1500);
        hos_log("rx: WebGPU error (type %d): %s", (int)type, m.c_str());
    }
}
void on_lost(WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView message, void*, void*) {
    hos_log("rx: WebGPU device lost (%d): %s", (int)reason, str(message).c_str());
}

const char* backend_name(WGPUBackendType t) {
    switch (t) {
    case WGPUBackendType_Vulkan: return "Vulkan";
    case WGPUBackendType_D3D12: return "D3D12";
    case WGPUBackendType_D3D11: return "D3D11";
    case WGPUBackendType_OpenGL: return "OpenGL";
    case WGPUBackendType_Metal: return "Metal";
    case WGPUBackendType_WebGPU: return "WebGPU";
    default: return "?";
    }
}

void configure_surface(uint32_t w, uint32_t h, bool force = false) {
    if (!surface || !w || !h) return;
    if (!force && w == surface_w && h == surface_h) return; // avoid redundant surface reconfigurations on Intel Arc
    WGPUSurfaceConfiguration c{};
    c.device = dev.device;
    c.format = surface_format;
    c.usage = WGPUTextureUsage_RenderAttachment;
    c.width = w;
    c.height = h;
    c.alphaMode = WGPUCompositeAlphaMode_Auto;
    c.presentMode = present_mode;
    wgpuSurfaceConfigure(surface, &c);
    surface_w = w;
    surface_h = h;
}

const char* kBlitWgsl = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VO { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var o: VO;
  let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  o.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  o.uv = uv;
  return o;
}
@fragment fn fs(v: VO) -> @location(0) vec4<f32> {
  return vec4<f32>(textureSample(src, samp, v.uv).rgb, 1.0);
}
)";

void create_blit() {
    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = sv(kBlitWgsl);
    WGPUShaderModuleDescriptor md{};
    md.nextInChain = &src.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(dev.device, &md);

    WGPUBindGroupLayoutEntry e[2]{};
    e[0].binding = 0;
    e[0].visibility = WGPUShaderStage_Fragment;
    e[0].texture.sampleType = WGPUTextureSampleType_Float;
    e[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    e[1].binding = 1;
    e[1].visibility = WGPUShaderStage_Fragment;
    e[1].sampler.type = WGPUSamplerBindingType_Filtering;
    WGPUBindGroupLayoutDescriptor bd{};
    bd.entryCount = 2;
    bd.entries = e;
    blit_layout = wgpuDeviceCreateBindGroupLayout(dev.device, &bd);
    WGPUPipelineLayoutDescriptor pd{};
    pd.bindGroupLayoutCount = 1;
    pd.bindGroupLayouts = &blit_layout;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(dev.device, &pd);

    WGPUColorTargetState target{};
    target.format = surface_format;
    target.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fs{};
    fs.module = mod;
    fs.entryPoint = sv("fs");
    fs.targetCount = 1;
    fs.targets = &target;
    WGPURenderPipelineDescriptor rd{};
    rd.layout = pl;
    rd.vertex.module = mod;
    rd.vertex.entryPoint = sv("vs");
    rd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    rd.primitive.cullMode = WGPUCullMode_None;
    rd.primitive.frontFace = WGPUFrontFace_CCW;
    rd.multisample.count = 1;
    rd.multisample.mask = 0xFFFFFFFF;
    rd.fragment = &fs;
    blit_pipeline = wgpuDeviceCreateRenderPipeline(dev.device, &rd);
    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(mod);

    WGPUSamplerDescriptor sd{};
    sd.addressModeU = sd.addressModeV = sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter = sd.minFilter = WGPUFilterMode_Linear;
    sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    sd.lodMaxClamp = 32.0f;
    sd.maxAnisotropy = 1;
    blit_sampler = wgpuDeviceCreateSampler(dev.device, &sd);
}

/// CPU memory for one batch's stream contents: recycled (pages are touched only as far as batches fill them).
std::mutex stream_pool_mu;
struct PooledStream {
    uint64_t capacity;
    uint8_t* memory;
    int64_t since;
    bool reset;
};
std::vector<PooledStream> stream_pool;
uint8_t* take_stream_memory(uint64_t size) {
    {
        std::lock_guard lk(stream_pool_mu);
        static const bool oldest = env_flag("RX_STREAM_POOL_FIFO");
        for (size_t k = 0; k < stream_pool.size(); k++) {
            const size_t i = oldest ? k : stream_pool.size() - 1 - k;
            if (stream_pool[i].capacity == size) {
                uint8_t* m = stream_pool[i].memory;
                stream_pool.erase(stream_pool.begin() + (ptrdiff_t)i);
                return m;
            }
        }
    }
    return static_cast<uint8_t*>(VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
}
void give_stream_memory(uint64_t size, uint8_t* m, uint64_t used) {
#if !RX_WEB
    constexpr uint64_t keep = 16ull << 20;
    if (used > keep && size > keep) {
        const uint64_t end = std::min<uint64_t>(size, (used + 0xFFFF) & ~0xFFFFull);
        if (VirtualFree(m + keep, end - keep, MEM_DECOMMIT) &&
            !VirtualAlloc(m + keep, end - keep, MEM_COMMIT, PAGE_READWRITE)) {
            VirtualFree(m, 0, MEM_RELEASE);
            return;
        }
    }
#endif
    std::lock_guard lk(stream_pool_mu);
#if !RX_WEB
    int same = 0;
    for (const auto& e : stream_pool) same += e.capacity == size;
    if (same >= 10) {
        VirtualFree(m, 0, MEM_RELEASE);
        return;
    }
#endif
    stream_pool.push_back({size, m, hos_now_ns(), false});
}

void reset_idle_stream_memory() {
#if !RX_WEB
    static const bool off = env_flag("RX_STREAM_POOL_FIFO");
    if (off) return;
    constexpr uint64_t keep = 16ull << 20;
    const int64_t now = hos_now_ns();
    std::lock_guard lk(stream_pool_mu);
    int n = 0;
    for (size_t i = 0; i < stream_pool.size();) {
        PooledStream& e = stream_pool[i];
        if (e.reset || now - e.since < 10000000000LL) {
            i++;
            continue;
        }
        const uint64_t len = std::min<uint64_t>(e.capacity, keep);
        if (VirtualFree(e.memory, len, MEM_DECOMMIT) && !VirtualAlloc(e.memory, len, MEM_COMMIT, PAGE_READWRITE)) {
            VirtualFree(e.memory, 0, MEM_RELEASE);
            stream_pool.erase(stream_pool.begin() + (ptrdiff_t)i);
            continue;
        }
        e.reset = true;
        n++;
        i++;
    }
    if (n) count("stream_blocks_reset", n);
#endif
}

void create_stream(Stream& s, const char* name, uint64_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor d{};
    d.label = sv(name);
    d.usage = usage | WGPUBufferUsage_CopyDst;
    d.size = size;
    s.buffer = wgpuDeviceCreateBuffer(dev.device, &d);
    s.cpu = take_stream_memory(size);
    s.capacity = size;
    s.used = 0;
    s.usage = usage;
    s.name = name;
}

bool init_device(HWND hwnd) {
#ifdef __EMSCRIPTEN__
    (void)hwnd;
    dev.instance = wgpuCreateInstance(nullptr);
    if (!dev.instance) {
        hos_log("rx: no WebGPU in this browser");
        return false;
    }
    {   // (the page's canvas, transferred to this thread: hos/gpu.c)
        WGPUEmscriptenSurfaceSourceCanvasHTMLSelector src{};
        src.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
        src.selector = sv("#screen");
        WGPUSurfaceDescriptor sd{};
        sd.nextInChain = &src.chain;
        surface = wgpuInstanceCreateSurface(dev.instance, &sd);
    }
#else
    WGPUInstanceExtras extras{};
    extras.chain.sType = (WGPUSType)WGPUSType_InstanceExtras;
    std::string be_file;
    const char* be = getenv("RX_BACKEND");
    if (!be) {
        std::string bpath = std::string(hos_user_dir ? hos_user_dir : "user") + "/backend.txt";
        if (FILE* f = fopen(bpath.c_str(), "r")) {
            char buf[32] = "";
            if (fscanf(f, "%31s", buf) == 1) {
                be_file = buf;
                be = be_file.c_str();
            }
            fclose(f);
        }
    }
    if (be && !_stricmp(be, "dx12")) {
        hos_log("rx: Direct3D 12 requested, but this build of wgpu-native was compiled only for Vulkan. Falling back to Vulkan.");
        be = "vulkan";
    }
    extras.backends = !be                           ? WGPUInstanceBackend_Primary
                      : !_stricmp(be, "vulkan")     ? WGPUInstanceBackend_Vulkan
                      : !_stricmp(be, "gl")         ? WGPUInstanceBackend_GL
                                                    : WGPUInstanceBackend_Primary;
    if (getenv("RX_VALIDATION")) extras.flags = WGPUInstanceFlag_Validation | WGPUInstanceFlag_Debug;
    // wgpu's own log (why a device was lost, driver warnings) into ours: warnings and errors, 400 different ones;
    // RX_WGPU_LOG=info|debug|trace for more, off for none
    {
        const char* lv = getenv("RX_WGPU_LOG");
        const WGPULogLevel level = !lv                        ? WGPULogLevel_Warn
                                   : !_stricmp(lv, "off")     ? WGPULogLevel_Off
                                   : !_stricmp(lv, "error")   ? WGPULogLevel_Error
                                   : !_stricmp(lv, "info")    ? WGPULogLevel_Info
                                   : !_stricmp(lv, "debug")   ? WGPULogLevel_Debug
                                   : !_stricmp(lv, "trace")   ? WGPULogLevel_Trace
                                                              : WGPULogLevel_Warn;
        wgpuSetLogCallback(
            [](WGPULogLevel l, WGPUStringView message, void*) {
                static std::mutex m;   // (each message once: some warnings come for every pipeline)
                static std::set<std::string> seen;
                std::string t = str(message);
                {
                    std::lock_guard g(m);
                    if (seen.size() >= 400 || !seen.insert(t).second) return;
                }
                hos_log("wgpu %s: %s", l <= WGPULogLevel_Error ? "error" : l == WGPULogLevel_Warn ? "warning" : "log", t.c_str());
            },
            nullptr);
        wgpuSetLogLevel(level);
    }
    WGPUInstanceDescriptor id{};
    id.nextInChain = &extras.chain;
    dev.instance = wgpuCreateInstance(&id);
    if (!dev.instance) {
        hos_log("rx: no WebGPU instance");
        return false;
    }
    if (hwnd) {
        WGPUSurfaceSourceWindowsHWND src{};
        src.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
        src.hinstance = GetModuleHandleA(nullptr);
        src.hwnd = hwnd;
        WGPUSurfaceDescriptor sd{};
        sd.nextInChain = &src.chain;
        surface = wgpuInstanceCreateSurface(dev.instance, &sd);
    }
#endif

#if !RX_WEB
    // RX_ADAPTER=<part of a GPU's name> (e.g. "AMD", "Radeon"): that GPU instead of the high-performance one (a PC
    // with two GPUs; testing on another vendor's driver). The log lists the GPUs when it is set.
    if (const char* want = getenv("RX_ADAPTER")) {
        WGPUInstanceEnumerateAdapterOptions eo{};
        eo.backends = extras.backends;
        std::vector<WGPUAdapter> all(wgpuInstanceEnumerateAdapters(dev.instance, &eo, nullptr));
        wgpuInstanceEnumerateAdapters(dev.instance, &eo, all.data());
        std::string w = want;
        for (char& c : w) c = (char)tolower((unsigned char)c);
        for (WGPUAdapter a : all) {
            WGPUAdapterInfo ai{};
            wgpuAdapterGetInfo(a, &ai);
            std::string name = str(ai.device), lower = name;
            for (char& c : lower) c = (char)tolower((unsigned char)c);
            const bool pick = !dev.adapter && lower.find(w) != std::string::npos;
            hos_log("rx: GPU %s (%s)%s", name.c_str(), backend_name(ai.backendType), pick ? ": chosen (RX_ADAPTER)" : "");
            wgpuAdapterInfoFreeMembers(ai);
            if (pick) dev.adapter = a;
            else wgpuAdapterRelease(a);
        }
        if (!dev.adapter) hos_log("rx: no GPU's name contains \"%s\" (RX_ADAPTER): the default one", want);
    }
#endif
    WGPURequestAdapterOptions ao{};
    ao.featureLevel = WGPUFeatureLevel_Core;
    ao.powerPreference = WGPUPowerPreference_HighPerformance;
    ao.compatibleSurface = surface;
    WGPURequestAdapterCallbackInfo acb{};
    acb.mode = WGPUCallbackMode_AllowSpontaneous;
    acb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* u1, void*) {
        if (status == WGPURequestAdapterStatus_Success) *(WGPUAdapter*)u1 = adapter;
        else hos_log("rx: no adapter: %s", str(message).c_str());
    };
    acb.userdata1 = &dev.adapter;
    if (!dev.adapter) wgpuInstanceRequestAdapter(dev.instance, &ao, acb);
    for (int i = 0; i < 5000 && !dev.adapter; i++) {
        wgpuInstanceProcessEvents(dev.instance);
        wait_step();
    }
    if (!dev.adapter) return false;

    WGPUAdapterInfo info{};
    wgpuAdapterGetInfo(dev.adapter, &info);
    char desc[256];
    std::snprintf(desc, sizeof desc, "%s (%s)", str(info.device).c_str(), backend_name(info.backendType));
    dev.description = desc;

    std::string dev_name = str(info.device);
    std::string vendor_name = str(info.vendor);
    auto to_lower = [](std::string s) {
        for (char& c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    std::string dev_lower = to_lower(dev_name);
    std::string vendor_lower = to_lower(vendor_name);

    dev.is_intel = (info.vendorID == 0x8086) || (vendor_lower.find("intel") != std::string::npos) ||
                   (dev_lower.find("intel") != std::string::npos);
    dev.is_intel_arc = dev.is_intel && (dev_lower.find("arc") != std::string::npos ||
                                        (info.deviceID >= 0x5690 && info.deviceID <= 0x56cf));

    if (dev.is_intel) {
        hos_log("rx: Intel GPU detected: %s (vendorID %#x, deviceID %#x)", dev_name.c_str(), info.vendorID, info.deviceID);
        if (dev.is_intel_arc) {
            hos_log("rx: Intel Arc GPU detected: enabling specialized driver mitigations for Windows 11");
        }
    }
    wgpuAdapterInfoFreeMembers(info);

    // what the adapter has of what we use
    std::vector<WGPUFeatureName> want = {
        WGPUFeatureName_TextureCompressionBC,  WGPUFeatureName_Float32Filterable,
        WGPUFeatureName_Depth32FloatStencil8,  WGPUFeatureName_RG11B10UfloatRenderable,
        WGPUFeatureName_DepthClipControl,      WGPUFeatureName_IndirectFirstInstance,
        WGPUFeatureName_Float32Blendable,      WGPUFeatureName_TextureFormatsTier1,
        WGPUFeatureName_ClipDistances,
    };
    // DualSourceBlending: avoid requesting on Intel Arc by default due to driver/Naga layout compiler issues,
    // unless explicitly forced via RX_INTEL_DUAL_SOURCE=1 (dual-source is already approximated in rx_draw)
    if (!dev.is_intel_arc || env_flag("RX_INTEL_DUAL_SOURCE")) {
        want.push_back(WGPUFeatureName_DualSourceBlending);
    }
    std::vector<WGPUFeatureName> have;
    for (auto f : want)
        if (wgpuAdapterHasFeature(dev.adapter, f)) have.push_back(f);
    // native (wgpu) only; the web gets the same formats from texture-formats-tier1
    const auto norm16 = (WGPUFeatureName)WGPUNativeFeature_TextureFormat16bitNorm;
    if (wgpuAdapterHasFeature(dev.adapter, norm16)) have.push_back(norm16);
#if !RX_WEB
    // (@early_depth_test(force): pixel programs whose stores the depth test must stop)
    const auto early_z = (WGPUFeatureName)WGPUNativeFeature_ShaderEarlyDepthTest;
    dev.early_depth_test = wgpuAdapterHasFeature(dev.adapter, early_z);
    if (dev.early_depth_test) have.push_back(early_z);
#endif
    dev.norm16 = wgpuAdapterHasFeature(dev.adapter, norm16) ||
                 // (the web: r16unorm, rg16unorm, rgba16unorm and the snorm ones come with texture-formats-tier1)
                 (RX_WEB && wgpuAdapterHasFeature(dev.adapter, WGPUFeatureName_TextureFormatsTier1));
    dev.bc = wgpuAdapterHasFeature(dev.adapter, WGPUFeatureName_TextureCompressionBC);
    dev.depth_clip_control = wgpuAdapterHasFeature(dev.adapter, WGPUFeatureName_DepthClipControl);
    dev.float32_filterable = wgpuAdapterHasFeature(dev.adapter, WGPUFeatureName_Float32Filterable);
    dev.depth32_stencil8 = wgpuAdapterHasFeature(dev.adapter, WGPUFeatureName_Depth32FloatStencil8);
    dev.timestamps = env_flag("RX_GPU_TIMING") && wgpuAdapterHasFeature(dev.adapter, WGPUFeatureName_TimestampQuery);
    if (dev.timestamps) have.push_back(WGPUFeatureName_TimestampQuery);
    WGPULimits limits{};
    wgpuAdapterGetLimits(dev.adapter, &limits);

    WGPUDeviceDescriptor dd{};
    dd.label = sv("rexcomp");
    dd.requiredFeatureCount = have.size();
    dd.requiredFeatures = have.data();
    dd.requiredLimits = &limits;
    dd.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    dd.deviceLostCallbackInfo.callback = on_lost;
    dd.uncapturedErrorCallbackInfo.callback = on_error;
    WGPURequestDeviceCallbackInfo dcb{};
    dcb.mode = WGPUCallbackMode_AllowSpontaneous;
    dcb.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* u1, void*) {
        if (status == WGPURequestDeviceStatus_Success) *(WGPUDevice*)u1 = device;
        else hos_log("rx: no device: %s", str(message).c_str());
    };
    dcb.userdata1 = &dev.device;
    wgpuAdapterRequestDevice(dev.adapter, &dd, dcb);
    for (int i = 0; i < 5000 && !dev.device; i++) {
        wgpuInstanceProcessEvents(dev.instance);
        wait_step();
    }
    if (!dev.device) return false;
    dev.queue = wgpuDeviceGetQueue(dev.device);
    wgpuDeviceGetLimits(dev.device, &dev.limits);
    hos_log("rx: limits: per stage %u storage, %u uniform buffers; dynamic %u storage, %u uniform; %u bindings a group; "
            "offset alignment storage %u uniform %u", dev.limits.maxStorageBuffersPerShaderStage,
            dev.limits.maxUniformBuffersPerShaderStage, dev.limits.maxDynamicStorageBuffersPerPipelineLayout,
            dev.limits.maxDynamicUniformBuffersPerPipelineLayout, dev.limits.maxBindingsPerBindGroup,
            dev.limits.minStorageBufferOffsetAlignment, dev.limits.minUniformBufferOffsetAlignment);
    // RX_SSBO_ALIGN=<bytes>: a coarser storage offset alignment (the browser's 256, to test its paths natively)
    if (const char* a = getenv("RX_SSBO_ALIGN"); a && atoi(a) > (int)dev.limits.minStorageBufferOffsetAlignment)
        dev.limits.minStorageBufferOffsetAlignment = (uint32_t)atoi(a);
    if (dev.is_intel_arc && dev.limits.minStorageBufferOffsetAlignment < 64)
        dev.limits.minStorageBufferOffsetAlignment = 64; // align with Xe cachelines
    hos_log("rx: storage buffer offset alignment %u, uniform %u, vertex buffers %u", dev.limits.minStorageBufferOffsetAlignment,
            dev.limits.minUniformBufferOffsetAlignment, dev.limits.maxVertexBuffers);

    if (!surface) {   // headless
        create_blit();
        create_stream(stream_ro, "rx stream", (RX_WEB ? 32ull : 96ull) << 20,
                      WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex |
                          WGPUBufferUsage_Index);
        create_stream(stream_rw, "rx storage", (RX_WEB ? 16ull : 32ull) << 20, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
        astc_gpu_on();
        unswizzle_gpu_on();
        hos_log("rx: WebGPU on %s; %zu optional features; no window (offscreen)", desc, have.size());
        return true;
    }
    WGPUSurfaceCapabilities caps{};
    wgpuSurfaceGetCapabilities(surface, dev.adapter, &caps);
    surface_format = caps.formatCount ? caps.formats[0] : WGPUTextureFormat_BGRA8Unorm;
    for (size_t i = 0; i < caps.formatCount; i++)
        if (caps.formats[i] == WGPUTextureFormat_BGRA8Unorm || caps.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
            surface_format = caps.formats[i];   // the window buffer holds encoded values: no sRGB conversion
            break;
        }
    bool mailbox = false, immediate = false;
    for (size_t i = 0; i < caps.presentModeCount; i++) {
        mailbox |= caps.presentModes[i] == WGPUPresentMode_Mailbox;
        immediate |= caps.presentModes[i] == WGPUPresentMode_Immediate;
    }
    // the game paces itself (nn_vi's buffer queue): presenting must not wait for a vblank
    present_mode = mailbox ? WGPUPresentMode_Mailbox : immediate ? WGPUPresentMode_Immediate : WGPUPresentMode_Fifo;
    if (const char* want = getenv("RX_PRESENT")) {
        if (!std::strcmp(want, "fifo")) present_mode = WGPUPresentMode_Fifo;
        else if (!std::strcmp(want, "mailbox") && mailbox) present_mode = WGPUPresentMode_Mailbox;
        else if (!std::strcmp(want, "immediate") && immediate) present_mode = WGPUPresentMode_Immediate;
        hos_log("rx: present mode %s (RX_PRESENT=%s)",
                present_mode == WGPUPresentMode_Fifo ? "fifo" : present_mode == WGPUPresentMode_Mailbox ? "mailbox" : "immediate",
                want);
    }
    wgpuSurfaceCapabilitiesFreeMembers(caps);
#ifdef __EMSCRIPTEN__
    present_mode = WGPUPresentMode_Fifo;
    configure_surface(1280, 720);
#else
    RECT rc;
    GetClientRect(hwnd, &rc);
    configure_surface((uint32_t)std::max<LONG>(rc.right, 1), (uint32_t)std::max<LONG>(rc.bottom, 1));
    present_window_size(rc.right, rc.bottom);   // (ultrawide's aspect before the game makes its cameras)
#endif

    create_blit();
    create_stream(stream_ro, "rx stream", (RX_WEB ? 32ull : 96ull) << 20,
                  WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex | WGPUBufferUsage_Index);
    create_stream(stream_rw, "rx storage", (RX_WEB ? 16ull : 32ull) << 20, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    pipecache_start();   // (the browser: the pipelines of earlier sessions, made again in the background)
    native_cache_start();   // (natively: the same, rxcache/pipelines.bin next to the exe)
    astc_gpu_on();   // (the compute passes that are on made here: error scopes belong to the device, not to a thread)
    unswizzle_gpu_on();
    hos_log("rx: WebGPU on %s; %zu optional features; surface %ux%u format %#x present mode %d", desc, have.size(),
            surface_w, surface_h, (unsigned)surface_format, (int)present_mode);
    return true;
}

Image* rt_image(const Regs& r, uint32_t i) {
    const auto& rt = r.rt[i];
    const uint64_t va = rt.Address();
    if (!va || rt.format == Tegra::RenderTargetFormat::NONE) return nullptr;
    const PixelFormat pf = VideoCore::Surface::PixelFormatFromRenderTargetFormat(rt.format);
    uint32_t w = rt.width;
    if (rt.tile_mode.is_pitch_linear) w = rt.width / std::max(1u, VideoCore::Surface::BytesPerBlock(pf));
    // Maxwell permits padded backing strides; the surface clip is the logical render area. WebGPU requires
    // equal attachment extents, so remove only the row padding, never enlarge an undersized target.
    const uint32_t clip_width = r.surface_clip.x + r.surface_clip.width;
    const uint32_t clip_height = r.surface_clip.y + r.surface_clip.height;
    if (clip_width && clip_width < w && (rt.tile_mode.is_pitch_linear || w - clip_width < 64)) w = clip_width;
    const uint32_t h = clip_height && clip_height < rt.height && rt.height - clip_height < 8 ? clip_height : rt.height;
    Image* im = find_render_target(va, w, h, pf);
    if (im) {
        im->pitch_linear = rt.tile_mode.is_pitch_linear;
        im->bl_height = rt.tile_mode.block_height;
        im->bl_depth = rt.tile_mode.block_depth;
        im->reg_width = rt.width;   // (pitch linear: the row pitch in bytes)
    }
    return im;
}

Image* zeta_image(const Regs& r) {
    if (!r.zeta_enable) return nullptr;
    const uint64_t va = r.zeta.Address();
    if (!va) return nullptr;
    const PixelFormat pf = VideoCore::Surface::PixelFormatFromDepthFormat(r.zeta.format);
    uint32_t w = r.zeta_size.width, h = r.zeta_size.height;
    const uint32_t clip_width = r.surface_clip.x + r.surface_clip.width;
    const uint32_t clip_height = r.surface_clip.y + r.surface_clip.height;
    if (clip_width && clip_width < w && w - clip_width < 64) w = clip_width;
    if (clip_height && clip_height < h && h - clip_height < 8) h = clip_height;
    return find_render_target(va, w, h, pf);
}

uint64_t passes_begun;

struct PendingClear {
    Image* im;
    bool color, depth, stencil;
    WGPUColor value;
    float depth_value;
    uint32_t stencil_value;
};
std::vector<PendingClear> pending_clears;
void flush_pending_clears();

void begin_pass(const Attachments& at, const WGPURenderPassColorAttachment* colors,
                const WGPURenderPassDepthStencilAttachment* depth) {
    end_render_pass("pe_begin");
    std::array<WGPURenderPassColorAttachment, 8> folded_colors;
    WGPURenderPassDepthStencilAttachment folded_depth;
    if (!pending_clears.empty()) {
        for (auto it = pending_clears.begin(); it != pending_clears.end();) {
            int slot = -1;
            if (it->color && colors)
                for (int i = 0; i < 8; i++)
                    if (at.color[i] == it->im && colors[i].loadOp == WGPULoadOp_Load) slot = i;
            const bool depth_here = !it->color && depth && at.depth == it->im &&
                                    (!it->depth || depth->depthLoadOp == WGPULoadOp_Load) &&
                                    (!it->stencil || depth->stencilLoadOp == WGPULoadOp_Load);
            if (slot >= 0) {
                if (colors != folded_colors.data()) {
                    std::copy(colors, colors + 8, folded_colors.begin());
                    colors = folded_colors.data();
                }
                folded_colors[slot].loadOp = WGPULoadOp_Clear;
                folded_colors[slot].clearValue = it->value;
            } else if (depth_here) {
                if (depth != &folded_depth) {
                    folded_depth = *depth;
                    depth = &folded_depth;
                }
                if (it->depth) {
                    folded_depth.depthLoadOp = WGPULoadOp_Clear;
                    folded_depth.depthClearValue = it->depth_value;
                }
                if (it->stencil) {
                    folded_depth.stencilLoadOp = WGPULoadOp_Clear;
                    folded_depth.stencilClearValue = it->stencil_value;
                }
            } else {
                ++it;
                continue;
            }
            count("clears_folded");
            it = pending_clears.erase(it);
        }
        flush_pending_clears();
    }
    passes_begun++;
    count("passes");
    static const bool old_versions = env_flag("RX_OLD_VERSIONS");   // (debugging: per-image counters)
    for (Image* im : at.color)
        if (im) im->version = old_versions ? im->version + 1 : ++image_versions;
    if (at.depth) at.depth->version = old_versions ? at.depth->version + 1 : ++image_versions;
    uint32_t n = 0;
    for (uint32_t i = 0; i < 8; i++)
        if (at.color[i]) n = i + 1;
    WGPURenderPassDescriptor pd{};
    pd.colorAttachmentCount = n;
    pd.colorAttachments = colors;
    pd.depthStencilAttachment = at.depth ? depth : nullptr;
    if (gpu_timing()) {
        const Image* first = nullptr;
        for (uint32_t i = 0; !first && i < n; i++) first = at.color[i];
        if (!first) first = at.depth;
        const uint32_t cf = at.color[0] ? static_cast<uint32_t>(at.color[0]->format) : 0;
        const uint32_t df = at.depth ? static_cast<uint32_t>(at.depth->format) : 0;
        ce_pass_tag(first ? first->gpu_va : 0, uint64_t{at.width & 0x7FFF} << 48 | uint64_t{at.height & 0xFFFF} << 32 |
                                                   uint64_t{n} << 24 | uint64_t{cf & 0xFFFF} << 8 | (df & 0xFF));
    }
    pass = ce_begin_render_pass(encoder(), &pd);
    pass_at = at;
}

WGPURenderPassColorAttachment color_attachment(Image* im, WGPULoadOp load, WGPUColor clear) {
    WGPURenderPassColorAttachment c{};
    c.view = im ? im->attachment : nullptr;
    c.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    c.loadOp = im ? load : WGPULoadOp_Undefined;
    c.storeOp = im ? WGPUStoreOp_Store : WGPUStoreOp_Undefined;
    c.clearValue = clear;
    return c;
}

/// A depth clear value; on a 16-bit target, on the step draws put that depth on (d16q_ in the WGSL backend), so that
/// draws testing EQUAL against cleared pixels match them.
float depth_clear_value(const Image* im, float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    static const bool no_d16q = env_flag("RX_NO_D16Q");
    if (!im || im->format != WGPUTextureFormat_Depth16Unorm || no_d16q) return v;
    return std::min((std::floor(v * 65535.0f + 0.5f) + 0.25f) / 65535.0f, 1.0f);
}

WGPURenderPassDepthStencilAttachment depth_attachment(Image* im, WGPULoadOp depth_load, WGPULoadOp stencil_load,
                                                      float depth_clear, uint32_t stencil_clear) {
    WGPURenderPassDepthStencilAttachment d{};
    d.view = im->attachment;
    const bool has_depth = im->format != WGPUTextureFormat_Stencil8;
    d.depthLoadOp = has_depth ? depth_load : WGPULoadOp_Undefined;
    d.depthStoreOp = has_depth ? WGPUStoreOp_Store : WGPUStoreOp_Undefined;
    d.depthClearValue = depth_clear;
    d.stencilLoadOp = has_stencil(im->format) ? stencil_load : WGPULoadOp_Undefined;
    d.stencilStoreOp = has_stencil(im->format) ? WGPUStoreOp_Store : WGPUStoreOp_Undefined;
    d.stencilClearValue = stencil_clear;
    return d;
}

uint32_t rt_byte_cost(WGPUTextureFormat f) {
    switch (f) {
    case WGPUTextureFormat_R8Unorm: case WGPUTextureFormat_R8Snorm: case WGPUTextureFormat_R8Uint:
    case WGPUTextureFormat_R8Sint:
        return 1;
    case WGPUTextureFormat_RG8Unorm: case WGPUTextureFormat_RG8Snorm: case WGPUTextureFormat_RG8Uint:
    case WGPUTextureFormat_RG8Sint: case WGPUTextureFormat_R16Float: case WGPUTextureFormat_R16Uint:
    case WGPUTextureFormat_R16Sint: case WGPUTextureFormat_R16Unorm:
        return 2;
    case WGPUTextureFormat_RG16Float: case WGPUTextureFormat_RG16Uint: case WGPUTextureFormat_RG16Sint:
    case WGPUTextureFormat_R32Float: case WGPUTextureFormat_R32Uint: case WGPUTextureFormat_R32Sint:
        return 4;
    case WGPUTextureFormat_RGBA32Float: case WGPUTextureFormat_RGBA32Uint: case WGPUTextureFormat_RGBA32Sint:
        return 16;
    default:
        return 8;
    }
}

void flush_pending_clears() {
    if (pending_clears.empty()) return;
    static std::vector<PendingClear> list;
    list.clear();
    list.swap(pending_clears);
    Attachments at;
    std::array<WGPURenderPassColorAttachment, 8> c{};
    WGPURenderPassDepthStencilAttachment d{};
    uint32_t n = 0;
    for (const PendingClear& p : list) {
        at.width = p.im->width;
        at.height = p.im->height;
        if (p.color) {
            at.color[n] = p.im;
            c[n++] = color_attachment(p.im, WGPULoadOp_Clear, p.value);
        } else {
            at.depth = p.im;
            d = depth_attachment(p.im, p.depth ? WGPULoadOp_Clear : WGPULoadOp_Load,
                                 p.stencil ? WGPULoadOp_Clear : WGPULoadOp_Load, p.depth_value, p.stencil_value);
        }
    }
    begin_pass(at, n ? c.data() : nullptr, at.depth ? &d : nullptr);
    end_render_pass("pe_clear");
}

bool clears_deferred() {
    static const bool off = env_flag("RX_NO_DEFERRED_CLEARS");
    return !off && !frame_dump_active();
}

void add_pending_clear(const PendingClear& p) {
    end_render_pass("pe_begin");
    p.im->version = ++image_versions;
    for (PendingClear& q : pending_clears)
        if (q.im == p.im) {
            if (p.color) q.color = true, q.value = p.value;
            if (p.depth) q.depth = true, q.depth_value = p.depth_value;
            if (p.stencil) q.stencil = true, q.stencil_value = p.stencil_value;
            return;
        }
    uint32_t colors = 0, bytes = 0;
    bool has_depth = false;
    for (const PendingClear& q : pending_clears) {
        colors += q.color;
        if (q.color) bytes += rt_byte_cost(q.im->format);
        has_depth |= !q.color;
    }
    const bool fits = pending_clears.empty() ||
                      (pending_clears[0].im->width == p.im->width && pending_clears[0].im->height == p.im->height &&
                       (p.color ? colors < 8 && bytes + rt_byte_cost(p.im->format) <= dev.limits.maxColorAttachmentBytesPerSample
                                : !has_depth));
    if (!fits) flush_pending_clears();
    pending_clears.push_back(p);
}

// ---- draws: suyu's Maxwell3D::DrawManager ----
using Topology = Regs::PrimitiveTopology;

/// Whether none of the draw's targets was drawn in the last 64 frames (each target's last two frames drawn, by GPU
/// address; the answer is kept while consecutive draws name the same targets in a frame)
bool one_off_targets(const Regs& r) {
    static const bool off = env_flag("RX_SKIP_ONE_OFF");
    if (off) return false;
    uint64_t addr[9];
    int n = 0;
    for (uint32_t i = 0; i < 8; i++)
        if (r.rt[i].format != Tegra::RenderTargetFormat::NONE && r.rt[i].Address()) addr[n++] = r.rt[i].Address();
    if (r.zeta_enable && r.zeta.Address()) addr[n++] = r.zeta.Address();
    if (!n) return false;
    const uint64_t sig = hash_bytes(addr, n * sizeof addr[0]);
    static uint64_t last_sig;
    static int64_t last_frame = -1;
    static bool last_answer;
    if (sig == last_sig && last_frame == frame_number) return last_answer;
    struct Seen { int64_t last = -100, prev = -100; };
    static std::unordered_map<uint64_t, Seen> seen;
    bool recent = false;
    for (int i = 0; i < n; i++) {
        Seen& s = seen[addr[i]];
        if (s.last != frame_number) s.prev = s.last, s.last = frame_number;
        recent |= s.prev >= frame_number - 64;   // (some targets are drawn every 9 frames)
    }
    last_sig = sig;
    last_frame = frame_number;
    last_answer = !recent;
    return last_answer;
}

void process_draw(Channel& ch, const Regs& r, bool indexed, uint32_t instance_count, uint32_t base_instance,
                  int32_t base_vertex, uint32_t first, uint32_t count) {
    DrawParams p;
    p.indexed = indexed;
    p.topology = ch.topology;
    if (r.primitive_topology_control == Regs::PrimitiveTopologyControl::UseSeparateState) {
        switch (r.topology_override) {
        case Regs::PrimitiveTopologyOverride::None: break;
        case Regs::PrimitiveTopologyOverride::Points: p.topology = (uint32_t)Topology::Points; break;
        case Regs::PrimitiveTopologyOverride::Lines: p.topology = (uint32_t)Topology::Lines; break;
        case Regs::PrimitiveTopologyOverride::LineStrip: p.topology = (uint32_t)Topology::LineStrip; break;
        default: p.topology = (uint32_t)r.topology_override; break;
        }
    }
    p.first = first;
    p.count = count;
    p.instance_count = std::max(1u, instance_count);
    p.base_instance = base_instance;
    p.base_vertex = base_vertex;
    if (indexed) {
        if (ch.draw_mode == Channel::DrawMode::InlineIndex) {
            p.inline_index = &ch.inline_index;
            p.index_size = 4;
        } else {
            p.index_va = r.index_buffer.StartAddress();
            p.index_size = r.index_buffer.format == Regs::IndexFormat::UnsignedByte    ? 1
                           : r.index_buffer.format == Regs::IndexFormat::UnsignedShort ? 2
                                                                                       : 4;
        }
    }
    if (count && r.zpass_pixel_count_enable) ch.zpass += (uint64_t)count * p.instance_count * 64;   // (estimate)
    if (count) {
        draw_must_run = one_off_targets(r);
        draw(ch, r, p);
        draw_must_run = false;
    } else {   // (nothing to draw: counted, and logged in a frame dump, where indirect arguments came out empty)
        rx::count("draws_empty");
        if (frame_dump_active())
            hos_log("rx-frame:   draw with no vertices: vs %llx fs %llx x%u %s",
                    (unsigned long long)(r.program_region.Address() + r.pipelines[1].offset),
                    (unsigned long long)(r.program_region.Address() + r.pipelines[5].offset), instance_count,
                    indexed ? "indexed" : "");
    }
}

void draw_end(Channel& ch, const Regs& r, uint32_t instance_count, bool force) {
    switch (ch.draw_mode) {
    case Channel::DrawMode::Instance:
        if (!force) break;
        [[fallthrough]];
    case Channel::DrawMode::General:
        if (ch.draw_indexed)
            process_draw(ch, r, true, instance_count, r.global_base_instance_index,
                         (int32_t)r.global_base_vertex_index, r.index_buffer.first, r.index_buffer.count);
        else
            process_draw(ch, r, false, instance_count, r.global_base_instance_index, 0, r.vertex_buffer.first,
                         r.vertex_buffer.count);
        ch.draw_indexed = false;
        break;
    case Channel::DrawMode::InlineIndex:
        process_draw(ch, r, true, instance_count, r.global_base_instance_index, (int32_t)r.global_base_vertex_index,
                     0, (uint32_t)(ch.inline_index.size() / 4));
        ch.inline_index.clear();
        break;
    }
}

void draw_deferred(Channel& ch, const Regs& r) {
    if (ch.draw_mode != Channel::DrawMode::Instance || ch.instance_count == 0) return;
    draw_end(ch, r, ch.instance_count + 1, true);
    ch.instance_count = 0;
}

// RX_RT_HISTORY=1 (debugging): every color target address a draw or clear named, with its last frame (what
// rt_history_find answers for a frame dump's texinfo)
struct RtSeen { int64_t frame; uint32_t width, height, format; bool on; };
std::unordered_map<uint64_t, RtSeen> rt_history;
void rt_history_note(const Regs& r, bool on) {
    for (uint32_t i = 0; i < 8; i++)
        if (r.rt[i].format != Tegra::RenderTargetFormat::NONE && r.rt[i].Address()) {
            // (RX_RT_HISTORY_LOG=<width>: targets that wide logged when first named)
            static const uint32_t log_w = getenv("RX_RT_HISTORY_LOG") ? atoi(getenv("RX_RT_HISTORY_LOG")) : 0;
            // (RX_RT_HISTORY_RANGE=<lo>-<hi> GPU hex: targets there logged when first named)
            static uint64_t range_lo = 1, range_hi;
            if (range_lo == 1) {
                range_lo = 0;
                if (const char* v = getenv("RX_RT_HISTORY_RANGE")) {
                    char* e;
                    range_lo = strtoull(v, &e, 16);
                    range_hi = *e == '-' ? strtoull(e + 1, nullptr, 16) : 0;
                }
            }
            const bool in_range = (range_hi && r.rt[i].Address() >= range_lo && r.rt[i].Address() < range_hi) ||
                                  guest_watch_hit(r.rt[i].Address(), 1);
            if (((log_w && r.rt[i].width == log_w) || in_range) && !rt_history.count(r.rt[i].Address()))
                hos_log("rx: target %#llx %ux%u format %#x first drawn, frame %lld", (unsigned long long)r.rt[i].Address(),
                        r.rt[i].width, r.rt[i].height, (unsigned)r.rt[i].format, (long long)frame_number);
            rt_history[r.rt[i].Address()] = {frame_number, r.rt[i].width, r.rt[i].height, (uint32_t)r.rt[i].format, on};
        }
}

void draw_begin(Channel& ch, const Regs& r) {
    if (r.draw.instance_id == Regs::Draw::InstanceId::First) {
        draw_deferred(ch, r);
        ch.instance_count = 0;
        ch.draw_mode = Channel::DrawMode::General;
    } else if (r.draw.instance_id == Regs::Draw::InstanceId::Subsequent) {
        ch.instance_count++;
        ch.draw_mode = Channel::DrawMode::Instance;
    }
    ch.topology = (uint32_t)r.draw.topology.Value();
}

void push_inline_index(Channel& ch, uint32_t index) {
    ch.inline_index.insert(ch.inline_index.end(), {uint8_t(index), uint8_t(index >> 8), uint8_t(index >> 16),
                                                   uint8_t(index >> 24)});
    ch.draw_mode = Channel::DrawMode::InlineIndex;
}

void draw_array_instanced(Channel& ch, const Regs& r, const Regs::VertexArray& va, bool subsequent) {
    ch.topology = (uint32_t)va.topology.Value();
    if (!subsequent) ch.instance_count = 1;
    const uint32_t base_instance = ch.instance_count - 1;
    ch.draw_mode = Channel::DrawMode::Instance;
    ch.instance_count++;
    process_draw(ch, r, false, 1, base_instance, 0, va.start, va.count);
}

void draw_index_small(Channel& ch, const Regs& r, uint32_t arg) {
    Regs::IndexBufferSmall s{};
    s.raw = arg;
    ch.topology = (uint32_t)s.topology.Value();
    const auto mode = ch.draw_mode;
    ch.draw_mode = Channel::DrawMode::General;   // the index buffer, not inline indices
    process_draw(ch, r, true, 1, r.global_base_instance_index, (int32_t)r.global_base_vertex_index, s.first,
                 s.count);
    ch.draw_mode = mode;
}

std::array<uint8_t, 0x1000> method_table() {
    std::array<uint8_t, 0x1000> t{};
    for (uint32_t m : {RX_REG(clear_surface), RX_REG(draw.begin), RX_REG(draw.end), RX_REG(index_buffer.count),
                       RX_REG(index_buffer32_first), RX_REG(index_buffer16_first), RX_REG(index_buffer8_first),
                       RX_REG(index_buffer32_subsequent), RX_REG(index_buffer16_subsequent),
                       RX_REG(index_buffer8_subsequent), RX_REG(draw_inline_index), RX_REG(inline_index_2x16.even),
                       RX_REG(inline_index_4x8.index0), RX_REG(vertex_array_instance_first),
                       RX_REG(vertex_array_instance_subsequent), RX_REG(draw_texture.src_y0),
                       RX_REG(invalidate_texture_data_cache), RX_REG(tiled_cache_barrier),
                       RX_REG(invalidate_texture_cache_no_wfi), RX_REG(invalidate_texture_data_cache_lines),
                       RX_REG(render_enable.mode), RX_REG(render_enable_override), RX_REG(clear_report_value)})
        t[m] = 1;
    for (uint32_t s = 0; s < 5; s++) t[RX_REG(bind_groups) + s * 8 + 4] = 1;
    return t;
}
} // namespace

WGPUShaderModule program_module(const WGPUShaderModuleDescriptor* d) {
#if !RX_WEB
    static const char* checks = getenv("RX_SHADER_CHECKS");
    if (checks)
        return wgpuDeviceCreateShaderModuleTrusted(dev.device, d, (WGPUShaderRuntimeChecks)std::strtoul(checks, 0, 0));
#endif
    return wgpuDeviceCreateShaderModule(dev.device, d);
}

/// The render targets a draw writes: rt_control maps each fragment output to a physical color target.
/// WebGPU receives one attachment per fragment output, restricted to one size per pass.
Attachments draw_attachments_uncached(const Regs& r);

/// Draws of a pass have the same attachments: the last answer holds while the target registers, the frame and the
/// set of images are the same (it was ~4.5% of the GPU thread, an image lookup per target per draw).
Attachments draw_attachments(const Regs& r) {
    struct Key {
        decltype(r.rt) rt;
        decltype(r.rt_control) rt_control;
        decltype(r.zeta) zeta;
        decltype(r.zeta_size) zeta_size;
        decltype(r.surface_clip) surface_clip;
        uint32_t zeta_enable;
    };
    static Key last_key;
    static Attachments last;
    static int64_t last_frame = -1;
    static uint64_t last_images = 0;
    Key k;
    std::memset(&k, 0, sizeof k);
    std::memcpy(&k.rt, &r.rt, sizeof k.rt);
    std::memcpy(&k.rt_control, &r.rt_control, sizeof k.rt_control);
    std::memcpy(&k.zeta, &r.zeta, sizeof k.zeta);
    std::memcpy(&k.zeta_size, &r.zeta_size, sizeof k.zeta_size);
    std::memcpy(&k.surface_clip, &r.surface_clip, sizeof k.surface_clip);
    k.zeta_enable = r.zeta_enable;
    if (last_frame == frame_number && last_images == images_changed && !std::memcmp(&k, &last_key, sizeof k)) {
        count("attachments_memo_hits");
        return last;
    }
    last = draw_attachments_uncached(r);
    last_key = k;
    last_frame = frame_number;
    last_images = images_changed;   // (the lookup itself can make or replace images)
    return last;
}

Attachments draw_attachments_uncached(const Regs& r) {
    Attachments at;
    const uint32_t n = std::min<uint32_t>(r.rt_control.count, 8);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t m = r.rt_control.Map(i);
        if (m >= 8) {
            log_limited("rt_map", 4, "rx: render target output %u has invalid mapping %u", i, m);
            continue;
        }
        at.color[i] = rt_image(r, m);
    }
    at.depth = zeta_image(r);
    for (Image* im : at.color)
        if (im && !at.width) {
            at.width = im->width;
            at.height = im->height;
        }
    if (!at.width && at.depth) {
        at.width = at.depth->width;
        at.height = at.depth->height;
    }
    for (auto& im : at.color)
        if (im && (im->width != at.width || im->height != at.height)) {
            log_limited("rt_size", 8, "rx: color target %ux%u dropped from a %ux%u pass", im->width, im->height,
                        at.width, at.height);
            static int attachment_trace_count;
            if (static const bool trace_att = env_flag("RX_TRACE_ATTACHMENTS"); trace_att && attachment_trace_count++ < 16) {
                hos_log("rx: rt routing count %u maps %u/%u/%u/%u/%u/%u/%u/%u",
                        static_cast<uint32_t>(r.rt_control.count), r.rt_control.Map(0), r.rt_control.Map(1),
                        r.rt_control.Map(2), r.rt_control.Map(3), r.rt_control.Map(4), r.rt_control.Map(5),
                        r.rt_control.Map(6), r.rt_control.Map(7));
                for (uint32_t i = 0; i < n; i++) {
                    const uint32_t target = r.rt_control.Map(i);
                    const auto& rt = r.rt[target];
                    hos_log("rx: rt output %u -> %u VA %#llx %ux%u format %u", i, target,
                            (unsigned long long)rt.Address(), rt.width, rt.height, (unsigned)rt.format);
                }
            }
            im = nullptr;
        }
    if (at.depth && (at.depth->width != at.width || at.depth->height != at.height)) {
        log_limited("zt_size", 8, "rx: depth target %ux%u dropped from a %ux%u pass", at.depth->width,
                    at.depth->height, at.width, at.height);
        at.depth = nullptr;
    }
    return at;
}

// ---- shared helpers ----
void scope_push() {
    if (RX_WEB) return;   // (the browser reports errors in its console, and an invalid submit does not end the page)
    wgpuDevicePushErrorScope(dev.device, WGPUErrorFilter_Validation);
}

std::string scope_pop() {
    if (RX_WEB) return {};
    struct ScopeResult {
        std::mutex mutex;
        std::condition_variable ready;
        bool complete = false;
        std::string error;
    } result;
    WGPUPopErrorScopeCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message, void* u1, void*) {
        auto& result = *static_cast<ScopeResult*>(u1);
        {
            std::lock_guard lock(result.mutex);
            if (type != WGPUErrorType_NoError) result.error = str(message);
            result.complete = true;
        }
        result.ready.notify_one();
    };
    cb.userdata1 = &result;
    wgpuDevicePopErrorScope(dev.device, cb);
    {
        std::unique_lock lock(result.mutex);
        result.ready.wait(lock, [&result] { return result.complete; });
    }
    if (result.error.size() > 1500) result.error.resize(1500);
    return result.error;
}

/// Conditional rendering (suyu's Maxwell3D::ProcessQueryCondition): draws and clears run only while the render
/// enable condition holds (Channel::render_on, evaluated when the mode or the override is written); its modes compare
/// 64-bit values the GPU released, at the address and 16 bytes on. TOTK
/// skips the later clears of its half-resolution depth target this way, keeping the scene's depth in it for the
/// cloud dome (done, they let the clouds cover the terrain above the horizon). RX_NO_CONDITIONAL=1: always render.
bool render_enabled(const Regs& r) {
    using RE = Regs::RenderEnable;
    static const bool off = env_flag("RX_NO_CONDITIONAL");
    if (off) return true;
    if (r.render_enable_override == RE::Override::AlwaysRender) return true;
    if (r.render_enable_override == RE::Override::NeverRender) return false;
    switch (r.render_enable.mode) {
    case RE::Mode::False: return false;
    case RE::Mode::True: return true;
    default: break;
    }
    uint64_t v[3]{};
    gpu_read(r.render_enable.Address(), v, sizeof v);
    switch (r.render_enable.mode) {
    case RE::Mode::Conditional: return v[0] != 0;
    case RE::Mode::IfEqual: return v[0] == v[2];
    case RE::Mode::IfNotEqual: return v[0] != v[2];
    default: return true;
    }
}

extern "C" volatile long maps_generation;   // hos/nvdrv.c: bumped by every mapping change

// (the counters and the timers only with RX_STATS: ~10 timestamps and as many counts a draw were ~3% of the render
// thread)
// (the web's environment comes after static initialization: read once the renderer runs, in rx_start)
bool stats_on = false;

int64_t clock_ns() {
    if (!stats_on) return 0;
    static const double ns_per_tick = [] {
        const int64_t t0 = hos_now_ns();
        const uint64_t c0 = __rdtsc();
        Sleep(20);
        const int64_t t1 = hos_now_ns();
        const uint64_t c1 = __rdtsc();
        return c1 > c0 ? double(t1 - t0) / double(c1 - c0) : 0.0;
    }();
    return (int64_t)(double(__rdtsc()) * ns_per_tick);
}

bool gpu_read(uint64_t va, void* dst, uint64_t n) {
    auto* d = static_cast<uint8_t*>(dst);
    // the last mapping this thread read from (most reads are small and land in the one before: TIC/TSC entries,
    // constant buffers; hos_gpu_ptr, out of line, was ~7% of the GPU thread)
    thread_local struct {
        uint64_t lo = 1, hi = 0;
        const uint8_t* host = nullptr;
        long generation = 0;
    } last;
    const long generation = maps_generation;
    if (va >= last.lo && va + n <= last.hi && last.generation == generation && n) {
        std::memcpy(d, last.host + (va - last.lo), n);
        return true;
    }
    {
        uint64_t avail = 0;
        if (const uint8_t* p = hos_gpu_ptr(va, &avail); p && avail >= n) {
            last.lo = va;
            last.hi = va + avail;
            last.host = p;
            last.generation = generation;
            if (n) std::memcpy(d, p, n);
            return true;
        }
    }
    bool ok = true;
    while (n) {
        uint64_t avail = 0;
        const uint8_t* p = hos_gpu_ptr(va, &avail);
        if (!p) {
            const uint64_t step = std::min<uint64_t>(n, 0x1000 - (va & 0xFFF));
            std::memset(d, 0, step);
            d += step;
            va += step;
            n -= step;
            ok = false;
            continue;
        }
        const uint64_t k = std::min(n, avail);
        std::memcpy(d, p, k);
        d += k;
        va += k;
        n -= k;
    }
    return ok;
}

const uint8_t* gpu_span(uint64_t va, uint64_t n) {
    uint64_t avail = 0;
    const uint8_t* p = hos_gpu_ptr(va, &avail);
    return p && avail >= n ? p : nullptr;
}

/// A PNG of RGBA8 rows (stored deflate blocks: no compression library needed).
void write_png(const char* path, const uint8_t* rgba, uint32_t w, uint32_t h, size_t stride) {
    std::vector<uint8_t> raw;
    raw.reserve((size_t)(w * 4 + 1) * h);
    for (uint32_t y = 0; y < h; y++) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba + y * stride, rgba + y * stride + (size_t)w * 4);
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    for (size_t i = 0; i < raw.size(); i += 65535) {
        const size_t n = std::min<size_t>(65535, raw.size() - i);
        z.push_back(i + n == raw.size());
        z.insert(z.end(), {uint8_t(n), uint8_t(n >> 8), uint8_t(~n), uint8_t(~n >> 8)});
        z.insert(z.end(), raw.begin() + i, raw.begin() + i + n);
    }
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    const uint32_t adler = b << 16 | a;
    z.insert(z.end(), {uint8_t(adler >> 24), uint8_t(adler >> 16), uint8_t(adler >> 8), uint8_t(adler)});
    static uint32_t table[256];
    if (!table[1])
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
        const uint32_t n = (uint32_t)data.size();
        const uint8_t len[4] = {uint8_t(n >> 24), uint8_t(n >> 16), uint8_t(n >> 8), uint8_t(n)};
        std::fwrite(len, 1, 4, f);
        uint32_t crc = 0xFFFFFFFFu;
        for (int i = 0; i < 4; i++) crc = table[(crc ^ (uint8_t)type[i]) & 0xFF] ^ (crc >> 8);
        for (uint8_t c : data) crc = table[(crc ^ c) & 0xFF] ^ (crc >> 8);
        crc ^= 0xFFFFFFFFu;
        std::fwrite(type, 1, 4, f);
        std::fwrite(data.data(), 1, data.size(), f);
        const uint8_t c4[4] = {uint8_t(crc >> 24), uint8_t(crc >> 16), uint8_t(crc >> 8), uint8_t(crc)};
        std::fwrite(c4, 1, 4, f);
    };
    static const uint8_t signature[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    std::fwrite(signature, 1, 8, f);
    chunk("IHDR", {uint8_t(w >> 24), uint8_t(w >> 16), uint8_t(w >> 8), uint8_t(w), uint8_t(h >> 24), uint8_t(h >> 16),
                   uint8_t(h >> 8), uint8_t(h), 8, 6, 0, 0, 0});
    chunk("IDAT", z);
    chunk("IEND", {});
    std::fclose(f);
}

void memory_report() {
    uint64_t rt_bytes = 0, spare_bytes = 0, copy_bytes = 0;
    for (const auto& [g, im] : images)
        if (im) {
            rt_bytes += image_bytes(*im);
            if (im->sampled_copy) copy_bytes += image_bytes(*im->sampled_copy);
        }
    for (const auto& [g, im] : spare_images)
        if (im) spare_bytes += image_bytes(*im);
    size_t ntex = 0, ntext = 0;
    uint64_t tex_bytes = 0, text_bytes = 0, priv = 0, ws = 0;
    decoded_texture_stats(&ntex, &tex_bytes);
    wgsl_text_stats(&ntext, &text_bytes);
    hos_process_memory(&priv, &ws);
    hos_log("mem: process %llu MB private, %llu MB working set | GPU: decoded textures %llu MB (%zu), render targets "
            "%llu MB (%zu) + %llu MB spare (%zu) + %llu MB feedback copies, guest-memory buffers %zu x 16 MB | "
            "%zu pipelines, %zu WGSL texts %llu MB",
            (unsigned long long)(priv >> 20), (unsigned long long)(ws >> 20), (unsigned long long)(tex_bytes >> 20),
            ntex, (unsigned long long)(rt_bytes >> 20), images.size(), (unsigned long long)(spare_bytes >> 20),
            spare_images.size(), (unsigned long long)(copy_bytes >> 20), buffer_segments(), pipeline_count(), ntext,
            (unsigned long long)(text_bytes >> 20));
    hos_memory_map_log();
}

void background_thread() {
#if !RX_WEB && defined(_WIN32)
    static const bool off = getenv("RX_NO_BACKGROUND_PRIORITY") != nullptr;
    if (!off) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

#if !RX_WEB
extern "C" int hos_ab(void);
#endif
bool ab_named(const char* name) {
    const char* sw = getenv("HOS_AB_SWITCH");
    return !RX_WEB && sw && !std::strcmp(sw, name);
}

bool ab_flipped() {
#if RX_WEB
    return false;
#else
    return hos_ab() != 0;
#endif
}

bool env_flag(const char* name) {
    // keyed by the literal's address (callers pass literals): the hot paths ask several times per draw, and a
    // locked string map was ~3% of the GPU thread
    struct Entry {
        std::atomic<const char*> key{nullptr};
        bool value = false;
    };
    static Entry table[64];
    for (Entry& e : table) {
        const char* k = e.key.load(std::memory_order_acquire);
        if (k == name) return e.value;
        if (!k) break;
    }
    static std::mutex m;
    std::lock_guard lk(m);
    for (Entry& e : table) {
        const char* k = e.key.load(std::memory_order_relaxed);
        if (k == name) return e.value;
        if (!k) {
            e.value = getenv(name) != nullptr;
            e.key.store(name, std::memory_order_release);
            return e.value;
        }
    }
    return getenv(name) != nullptr;
}

namespace {
std::unordered_map<const char*, int64_t> counters;   // this period's, by key (string literals)
uint64_t counters_generation;   // bumped when the counters are cleared (count's slot cache)
int64_t period_start, period_frames;

/// Every 5 s with RX_STATS: where the renderer's time went, and why draws were skipped.
void report_period() {
    const int64_t now = hos_now_ns();
    if (!period_start) period_start = now;
    period_frames++;
    if (now - period_start < 5000000000LL) return;
    if (env_flag("RX_STATS")) {
        std::unordered_map<std::string, int64_t> merged;
        for (auto& [k, v] : counters) merged[k] += v;
        const double frames = (double)std::max<int64_t>(period_frames, 1);
        auto ms = [&](const char* k) { return merged.count(k) ? merged[k] / 1e6 / frames : 0.0; };
        auto per = [&](const char* k) { return merged.count(k) ? merged[k] / frames : 0.0; };
        hos_log("rx: %.1f fps; per frame: %.0f draws %.1f ms, %.0f skipped, flush %.1f ms, present %.1f ms, "
                "translate %.1f ms, decode %.1f ms, pipelines %.1f ms (%.1f new), bind groups %.0f, upload %.1f MB",
                period_frames * 1e9 / (now - period_start), per("draws"), ms("t_draw"), per("skipped"),
                ms("t_flush"), ms("t_present"), ms("t_translate"), ms("t_decode"), ms("t_pipeline"),
                per("pipelines"), per("bind_groups"), per("upload_bytes") / 1048576.0);
        std::string reasons;
        for (auto& [k, v] : merged)
            if (k.rfind("t_", 0) && k != "draws" && k != "skipped" && k != "pipelines" && k != "bind_groups" &&
                k != "upload_bytes") {
                char b[96];
                std::snprintf(b, sizeof b, "%s%s %.0f", reasons.empty() ? "" : ", ", k.c_str(), v / frames);
                reasons += b;
            }
        if (!reasons.empty()) hos_log("rx:   per frame: %s", reasons.c_str());
        std::string timers;
        for (auto& [k, v] : merged)
            if (!k.rfind("t_", 0)) {
                char b[96];
                std::snprintf(b, sizeof b, "%s%s %.2f ms", timers.empty() ? "" : ", ", k.c_str() + 2, v / 1e6 / frames);
                timers += b;
            }
        if (!timers.empty()) hos_log("rx:   time per frame: %s", timers.c_str());
        hos_log("rx:   submit thread per frame: %.1f encoders, replay+finish %.2f ms, writes %.2f ms, submit %.2f ms, present %.2f ms; the "
                "GPU thread waited for it %.2f ms",
                (double)count_parts.exchange(0) / frames, st_finish_ns.exchange(0) / 1e6 / frames,
                st_write_ns.exchange(0) / 1e6 / frames,
                st_submit_ns.exchange(0) / 1e6 / frames, st_after_ns.exchange(0) / 1e6 / frames,
                st_wait_ns.exchange(0) / 1e6 / frames);
        if (const int64_t w = st_writeback_words.exchange(0))
            hos_log("rx:   shader writes copied back to guest memory: %.0f words per frame", w / frames);
    }
    counters.clear();
    counters_generation++;
    period_start = now;
    period_frames = 0;
}
/// RX_SCREENSHOT=<dir>: the presented image every 5 s as <dir>/frame_<n>.png (synchronous readback: debugging only).
std::atomic<int64_t> screenshot_interval{-1};   // (rx_set_screenshot_ms at the end of this file)
void screenshot(Image* im) {
    static const char* dir = getenv("RX_SCREENSHOT");
    static int64_t last;
    if (!dir || !im || (im->format != WGPUTextureFormat_RGBA8Unorm && im->format != WGPUTextureFormat_RGBA8UnormSrgb))
        return;
    // RX_SCREENSHOT_MS=<ms>: another interval; RX_SCREENSHOT_AFTER=<s>: none before that
    if (screenshot_interval < 0)
        screenshot_interval = getenv("RX_SCREENSHOT_MS") ? atoll(getenv("RX_SCREENSHOT_MS")) * 1000000LL : 5000000000LL;
    static const int64_t after = getenv("RX_SCREENSHOT_AFTER") ? atoll(getenv("RX_SCREENSHOT_AFTER")) * 1000000000LL : 0;
    static const int64_t start = hos_now_ns();
    const int64_t now = hos_now_ns();
    if (now - start < after) return;
    if (last && now - last < screenshot_interval) return;
    last = now;
    wait_submitted();   // (the frame's batches are on the submit thread)
    const size_t stride = ((size_t)im->width * 4 + 255) & ~size_t(255), size = stride * im->height;
    WGPUBufferDescriptor bd{};
    bd.size = size;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(dev.device, &bd);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    WGPUTexelCopyTextureInfo src{};
    src.texture = im->tex;
    WGPUTexelCopyBufferInfo dst{};
    dst.buffer = buffer;
    dst.layout.bytesPerRow = (uint32_t)stride;
    dst.layout.rowsPerImage = im->height;
    const WGPUExtent3D extent{im->width, im->height, 1};
    wgpuCommandEncoderCopyTextureToBuffer(e, &src, &dst, &extent);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(e);
    struct Done {
        bool complete = false;
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
    if (done.status == WGPUMapAsyncStatus_Success) {
        const auto* px = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(buffer, 0, size));
        static int n;
        char path[512];
        std::snprintf(path, sizeof path, "%s/frame_%03d.png", dir, n++);
        // the image is drawn with alpha that the display ignores: store it opaque
        std::vector<uint8_t> copy(px, px + size);
        for (size_t i = 3; i < copy.size(); i += 4) copy[i] = 255;
        write_png(path, copy.data(), im->width, im->height, stride);
        hos_log("rx: screenshot %s (frame %lld)", path, (long long)frame_number);
        wgpuBufferUnmap(buffer);
    }
    wgpuBufferRelease(buffer);
}
} // namespace

thread_local bool rx_side_thread;

void count(const char* key, int64_t n) {
    if (!stats_on || rx_side_thread) return;
    // (a slot cache by the literal's address: a map lookup per call was ~2% of the GPU thread)
    struct Slot {
        const char* key;
        int64_t* value;
        uint64_t generation;
    };
    static Slot cache[256];
    Slot& s = cache[((uintptr_t)key >> 3) & 255];
    if (s.key != key || s.generation != counters_generation) s = {key, &counters[key], counters_generation};
    *s.value += n;
}

void log_limited(const char* key, int limit, const char* fmt, ...) {
    if (rx_side_thread) return;
    int& n = log_counts[key];
    if (n >= limit) return;
    n++;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    hos_log("%s", buf);
}

uint64_t hash_bytes(const void* p, size_t n, uint64_t h) {
    // the seed is mixed first: used as is, (seed ^ first word) made hash(A, 0) == hash(B, 1) for data differing in the
    // low bit of its first word: sampler_for's mode seeds then handed a filtering sampler to an unfilterable texture
    // (and could hand a comparison sampler to a colour one)
    h = (h + 0x9e3779b97f4a7c15ull) * 0xbf58476d1ce4e5b9ull;
    h ^= h >> 31;
    const auto* b = static_cast<const uint8_t*>(p);
    size_t i = 0;
    if (n >= 64) {   // four independent lanes (the multiply's latency, not its throughput, bounded the one-lane loop)
        uint64_t l[4] = {h, h ^ 0x9e3779b97f4a7c15ull, h ^ 0xc2b2ae3d27d4eb4full, h ^ 0x165667b19e3779f9ull};
        for (; i + 32 <= n; i += 32)
            for (int k = 0; k < 4; k++) {
                uint64_t v;
                std::memcpy(&v, b + i + 8 * k, 8);
                l[k] = (l[k] ^ v) * 0x100000001b3ull;
                l[k] ^= l[k] >> 29;
            }
        h = l[0];
        for (int k = 1; k < 4; k++) {
            h = (h ^ l[k]) * 0x100000001b3ull;
            h ^= h >> 29;
        }
    }
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        std::memcpy(&v, b + i, 8);
        h = (h ^ v) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    for (; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

uint64_t Stream::alloc(uint64_t n, uint64_t align) {
    const uint64_t off = (used + align - 1) & ~(align - 1);
    if (off + n > capacity) return UINT64_MAX;
    used = off + n;
    return off;
}

uint64_t stream_alloc(Stream& s, uint64_t n, uint64_t align) {
    uint64_t off = s.alloc(n, align);
    if (off == UINT64_MAX && n <= s.capacity) {
        flush_batch();
        off = s.alloc(n, align);
    }
    return off;
}

WGPUCommandEncoder encoder() {
    if (!pending_clears.empty()) flush_pending_clears();
    if (!enc) enc = encoding_deferred() ? fake_encoder() : wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    return enc;
}

namespace {
struct PassResult {
    uint64_t rw_offset, size, gpu_va;
};
std::vector<PassResult> pass_results;   // storage the open pass's pixel programs write (copies in stream_rw)
} // namespace

/// Whether a pixel program of the open pass writes storage overlapping this range (into its stream_rw copy: the
/// buffer cache gets it when the pass ends)
bool storage_written_this_pass(uint64_t gpu_va, uint64_t size) {
    for (const PassResult& r : pass_results)
        if (r.gpu_va < gpu_va + size && r.gpu_va + r.size > gpu_va) return true;
    return false;
}

bool pass_wrote_storage() { return !pass_results.empty(); }

void storage_written_in_pass(uint64_t rw_offset, uint64_t size, uint64_t gpu_va) {
    for (const PassResult& r : pass_results)
        if (r.gpu_va == gpu_va && r.rw_offset == rw_offset) return;
    pass_results.push_back({rw_offset, size, gpu_va});
}

void split_long_pass() {
    static const uint32_t limit = [] {
        const char* e = getenv("RX_PASS_SPLIT");
        return e ? (uint32_t)std::atoi(e) : 256u;
    }();
    static uint64_t seen = ~0ull;
    static uint32_t draws = 0;
    if (!limit || !pass) return;
    if (passes_begun != seen) {
        seen = passes_begun;
        draws = 0;
    }
    // (the pass's storage results pending, as in TOTK's G-buffer pass, its biggest: the copy at the split is an early
    // one; the draws after it write the same copy, draw_storage's rw_copies are per batch, and the final end copies
    // it all again. A read-only binding of that memory after the split still breaks the pass at its first write.)
    if (++draws < limit) return;
    end_render_pass("pe_split");
}

void end_render_pass(const char* why) {
    if (!pass) return;
    if (why) count(why);
    rp_end(pass);
    pass = nullptr;
    pass_at = {};
    // what the pass's pixel programs wrote goes to the buffer cache, for the draws and dispatches after it (TOTK's
    // G-buffer shaders count each tile's materials in a buffer the lighting reads)
    if (!pass_results.empty()) {
        std::vector<PassResult> results;
        results.swap(pass_results);
        for (const PassResult& r : results) {
            const BufferRef ref = guest_buffer(r.gpu_va, r.size);
            if (!ref.buffer) continue;
            if (frame_dump_active()) {
                char b[128];
                std::snprintf(b, sizeof b, "pass result %llx+%llx -> buffer cache", (unsigned long long)r.gpu_va,
                              (unsigned long long)r.size);
                dump_draw(b);
            }
            ce_copy_buffer_to_buffer(encoder(), stream_rw.buffer, r.rw_offset, ref.buffer, ref.offset,
                                                 (r.size + 3) & ~3ull);
            buffer_cache_gpu_written(ref);
            count("pass_results_cached", (int64_t)r.size);
        }
    }
}

uint64_t pass_count() {
    return passes_begun;
}

const Attachments& current_attachments() {
    return pass_at;
}

WGPURenderPassEncoder render_pass(const Attachments& at) {
    if (pass && pass_at == at) return pass;
    if (pass) count("pe_switch");
    dump_pass(at, "draw");
    std::array<WGPURenderPassColorAttachment, 8> colors{};
    for (uint32_t i = 0; i < 8; i++) colors[i] = color_attachment(at.color[i], WGPULoadOp_Load, {});
    WGPURenderPassDepthStencilAttachment depth{};
    if (at.depth) depth = depth_attachment(at.depth, WGPULoadOp_Load, WGPULoadOp_Load, 0.0f, 0);
    begin_pass(at, colors.data(), &depth);
    return pass;
}

// ---- the submit thread: wgpu turns a command encoder into Vulkan commands at its finish (with the queue writes
// and the submit, a third of the GPU thread's time), so batches are finished and submitted there, in order, while
// the GPU thread goes on recording the next ----
namespace {
/// Bytes for a buffer (guest memory for the buffer cache, ...), at `src` in its batch's upload_bytes.
struct Upload {
    WGPUBuffer buffer;
    uint64_t offset, src, size;
};
struct Writeback {
    uint64_t rw_offset, size, gpu_va;
};
/// GPU results on their way back to guest memory: a staging copy of the written ranges, and what they held before.
struct Readback {
    WGPUBuffer staging = nullptr;
    uint64_t total = 0, serial = 0;   // serial: its batch's
    std::vector<Writeback> ranges;
    std::vector<uint8_t> before;   // the ranges' contents as uploaded, concatenated (4-byte aligned each)
};
struct Batch {
    WGPUCommandEncoder enc = nullptr;
    uint64_t serial = 0;            // flush order (the native NVN layer's fences name one: wait_batch)
    bool deferred = false;          // its commands are a recording, replayed into a new encoder here
    Recording* rec = nullptr;
    std::vector<Writeback> writebacks;
    std::vector<Upload> uploads;   // before the stream writes and the submit
    std::vector<uint8_t> upload_bytes;
    uint8_t *ro = nullptr, *rw = nullptr;
    uint64_t ro_used = 0, rw_used = 0;
    std::function<void()> after;   // after the submit (a present)
    bool present = false;
    std::vector<WGPUBindGroup> group_releases;   // after the replay (release_bind_group_later)
    std::vector<WGPUBuffer> buffer_releases;
    std::vector<WGPURenderPipeline> pipeline_releases;
    std::vector<std::function<void(bool)>> afters;   // after the submit (true: it was submitted): writebacks' readbacks
};
std::vector<Upload> open_uploads;
std::vector<uint8_t> open_upload_bytes;
std::mutex upload_pool_mu;
std::vector<std::vector<uint8_t>> upload_pool;   // (upload_bytes given back by run_batch: no regrowing each batch)
std::vector<std::function<void(bool)>> open_afters;
std::vector<WGPUBindGroup> open_group_releases;
std::vector<WGPUBuffer> open_buffer_releases;
std::vector<WGPURenderPipeline> open_pipeline_releases;
std::vector<Writeback> open_writebacks;
/// Ranges shaders wrote in submitted batches whose results are not in guest memory yet (rx_gpu_writes_pending).
std::mutex inflight_mu;
std::vector<Writeback> inflight;
std::atomic<int> readbacks_pending;
std::multiset<uint64_t> readback_serials;   // the batches of the readbacks not applied yet (under inflight_mu)

// Batch serials: the native NVN layer's fences wait for one batch (and the batches before it) to have run on the
// GPU and its readbacks to be in guest memory, not for everything submitted since (wait_batch)
std::atomic<uint64_t> batch_serial{0};       // the last batch flushed (flush_batch, under mu)
std::atomic<uint64_t> batch_serial_run{0};   // the last one the submit thread submitted
std::atomic<uint64_t> gpu_done_serial{0};    // the last one the GPU finished (its queue's work-done callback)
std::mutex index_mu;
struct SerialIndex {
    uint64_t serial = 0, index = 0;
};
SerialIndex serial_index[1024];   // a batch's last submission (the queue's index: wgpuDevicePoll waits for one)
uint64_t last_index = 0;          // (the submit thread's)

void retire_inflight(const std::vector<Writeback>& done) {
    std::lock_guard lk(inflight_mu);
    for (const Writeback& w : done)
        for (size_t i = 0; i < inflight.size(); i++)
            if (inflight[i].gpu_va == w.gpu_va && inflight[i].size == w.size) {
                inflight[i] = inflight.back();
                inflight.pop_back();
                break;
            }
}
std::mutex submit_mu;
std::condition_variable submit_cv, submitted_cv;
std::deque<std::unique_ptr<Batch>> submit_queue;
int submit_busy;          // batches popped and not done yet
int presents_queued;      // presents in the queue (frames the GPU thread is ahead by)

/// The words a batch's shaders changed in the ranges, written into guest memory (the map callback).
void apply_readback(Readback* rb, WGPUMapAsyncStatus status) {
    if (status == WGPUMapAsyncStatus_Success) {
        const auto* gpu = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(rb->staging, 0, rb->total));
        uint64_t pos = 0;
        int64_t words = 0;
        for (const Writeback& w : rb->ranges) {
            const uint64_t n = (w.size + 3) & ~3ull;
            uint64_t avail = 0;
            uint8_t* guest = hos_gpu_ptr(w.gpu_va, &avail);
            if (gpu && guest) words += readback_to_guest(guest, gpu + pos, rb->before.data() + pos, std::min(n, avail) & ~3ull);
            pos += n;
        }
        st_writeback_words += words;
        wgpuBufferUnmap(rb->staging);
    }
    retire_inflight(rb->ranges);
    {
        std::lock_guard lk(inflight_mu);
        if (auto it = readback_serials.find(rb->serial); it != readback_serials.end()) readback_serials.erase(it);
    }
    readbacks_pending--;
    wgpuBufferRelease(rb->staging);
    delete rb;
}

/// After a submitted batch: its writable storage ranges are copied out of stream_rw (before the next batch's writes
/// replace it) and mapped; apply_readback writes the changes back.
void start_readback(Batch& b) {
    auto* rb = new Readback();
    rb->serial = b.serial;
    {
        std::lock_guard lk(inflight_mu);
        readback_serials.insert(b.serial);
    }
    rb->ranges = std::move(b.writebacks);
    for (const Writeback& w : rb->ranges) {
        const uint64_t n = (w.size + 3) & ~3ull;
        rb->before.insert(rb->before.end(), b.rw + w.rw_offset, b.rw + w.rw_offset + n);
        rb->total += n;
    }
    WGPUBufferDescriptor d{};
    d.label = sv("rx readback");
    d.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    d.size = rb->total;
    rb->staging = wgpuDeviceCreateBuffer(dev.device, &d);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    uint64_t pos = 0;
    for (const Writeback& w : rb->ranges) {
        const uint64_t n = (w.size + 3) & ~3ull;
        wgpuCommandEncoderCopyBufferToBuffer(e, stream_rw.buffer, w.rw_offset, rb->staging, pos, n);
        pos += n;
    }
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
#if RX_WEB
    wgpuQueueSubmit(dev.queue, 1, &cb);
#else
    last_index = wgpuQueueSubmitForIndex(dev.queue, 1, &cb);
#endif
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(e);
    WGPUBufferMapCallbackInfo mc{};
    mc.mode = WGPUCallbackMode_AllowSpontaneous;
    mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        apply_readback(static_cast<Readback*>(u1), status);
    };
    mc.userdata1 = rb;
    readbacks_pending++;
    wgpuBufferMapAsync(rb->staging, WGPUMapMode_Read, 0, rb->total, mc);
}

/// wgpu turns a command encoder into Vulkan commands at its finish (most of the submit thread: 9 ms a frame in the
/// rainy open world), so a large batch is replayed into several encoders, split where passes end, and finished on
/// helper threads at once; the command buffers go to the queue in one submit, in order. RX_FINISH_THREADS=<n>
/// (default 3; 1: one encoder, as before).
struct FinishPart {
    Recording* rec = nullptr;
    size_t from = 0, to = 0;
    WGPUCommandBuffer cb = nullptr;
    std::string error;
    GpuTimer* timer = nullptr;   // RX_GPU_TIMING
    int64_t ns = 0;              // (its replay and finish)
};
std::mutex finish_mu;
std::condition_variable finish_cv, finish_done_cv;
std::deque<FinishPart*> finish_jobs;
int finish_left;

void finish_part(FinishPart& p) {
    const int64_t t0 = hos_now_ns();
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    if (p.rec && gpu_timing()) p.timer = gpu_timer_take();
    if (p.rec) recording_replay(p.rec, e, p.from, p.to, p.timer);
    gpu_timer_resolve(p.timer, e);
    // an invalid command buffer must not reach the queue (wgpu-native aborts the process on a failed submit)
    scope_push();
    p.cb = wgpuCommandEncoderFinish(e, nullptr);
    p.error = scope_pop();
    wgpuCommandEncoderRelease(e);
    p.ns = hos_now_ns() - t0;
}

void finisher_main() {
    hos_host_thread_register_self("GPU finish");
    hos_thread_boost(1);
    for (;;) {
        FinishPart* p;
        {
            std::unique_lock lk(finish_mu);
            finish_cv.wait(lk, [] { return !finish_jobs.empty(); });
            p = finish_jobs.front();
            finish_jobs.pop_front();
        }
        finish_part(*p);
        std::lock_guard lk(finish_mu);
        if (--finish_left == 0) finish_done_cv.notify_one();
    }
}

uint32_t finish_threads() {
    static const uint32_t n = [] {
        const char* e = getenv("RX_FINISH_THREADS");
        const int v = RX_WEB ? 1 : e ? std::atoi(e) : 3;
        return (uint32_t)std::clamp(v, 1, 8);
    }();
    return n;
}

void run_batch(Batch& b) {
    // (one queue write each: copying them from one written buffer at the start of the batch's first command buffer
    // was slower, renderer CPU +3% and -7% fps in an in-run A/B at the castle, 2026-10-25)
    for (const Upload& u : b.uploads) {
        wgpuQueueWriteBuffer(dev.queue, u.buffer, u.offset, b.upload_bytes.data() + u.src, u.size);
#if RX_WEB_MT
        serve_gpu_calls_now_if_any();
#endif
    }
    const int64_t t0 = hos_now_ns();
    bool submitted = false;
    std::vector<FinishPart> parts;
    if (b.deferred) {
        const auto ranges = b.rec ? recording_parts(b.rec, finish_threads(), 4096)
                                  : std::vector<std::pair<size_t, size_t>>{{0, 0}};
        parts.resize(ranges.size());
        for (size_t i = 0; i < ranges.size(); i++) parts[i] = {b.rec, ranges[i].first, ranges[i].second};
        if (parts.size() > 1) {
            static bool started;
            std::unique_lock lk(finish_mu);
            if (!started) {
                started = true;
                for (uint32_t i = 1; i < finish_threads(); i++) std::thread(finisher_main).detach();
            }
            finish_left = (int)parts.size() - 1;
            for (size_t i = 1; i < parts.size(); i++) finish_jobs.push_back(&parts[i]);
            lk.unlock();
            finish_cv.notify_all();
        }
        finish_part(parts[0]);
        if (parts.size() > 1) {
            std::unique_lock lk(finish_mu);
            finish_done_cv.wait(lk, [] { return finish_left == 0; });
        }
    } else if (b.enc) {
        parts.resize(1);
        scope_push();
        parts[0].cb = wgpuCommandEncoderFinish(b.enc, nullptr);
        parts[0].error = scope_pop();
        wgpuCommandEncoderRelease(b.enc);
    }
    if (!parts.empty()) {
        const int64_t t1 = hos_now_ns();
        st_finish_ns += t1 - t0;
        if (parts.size() > 1) count_parts += (int64_t)parts.size();
        static const char* const part_keys[8] = {"t_part0", "t_part1", "t_part2", "t_part3", "t_part4", "t_part5",
                                                 "t_part6", "t_part7"};
        for (size_t i = 0; i < parts.size() && i < 8; i++) count(part_keys[i], parts[i].ns);   // (balance)
        if (static const bool plog = env_flag("RX_PART_LOG"); plog && frame_number % 600 == 0) {   // (balance, a frame now and then)
            std::string line;
            for (const FinishPart& p : parts) {
                char b[64];
                std::snprintf(b, sizeof b, " %zu words %.2f ms;", p.to - p.from, p.ns / 1e6);
                line += b;
            }
            hos_log("rx: batch at frame %lld: %zu parts:%s", (long long)frame_number, parts.size(), line.c_str());
        }
        std::string error;
        for (const FinishPart& p : parts)
            if (!p.error.empty() && error.empty()) error = p.error;
        if (error.empty()) {
            if (b.ro_used) wgpuQueueWriteBuffer(dev.queue, stream_ro.buffer, 0, b.ro, (b.ro_used + 3) & ~3ull);
            if (b.rw_used) wgpuQueueWriteBuffer(dev.queue, stream_rw.buffer, 0, b.rw, (b.rw_used + 3) & ~3ull);
            const int64_t t2 = hos_now_ns();
            st_write_ns += t2 - t1;
            WGPUCommandBuffer cbs[8];
            for (size_t i = 0; i < parts.size(); i++) cbs[i] = parts[i].cb;
#if RX_WEB
            wgpuQueueSubmit(dev.queue, parts.size(), cbs);
#else
            last_index = wgpuQueueSubmitForIndex(dev.queue, parts.size(), cbs);
#endif
            st_submit_ns += hos_now_ns() - t2;
            submitted = true;
            if (!b.writebacks.empty() && b.rw) start_readback(b);
        } else {
            static int dropped;
            if (dropped++ < 20) hos_log("rx: batch dropped: %s", error.c_str());
        }
        for (const FinishPart& p : parts) {
            if (p.cb) wgpuCommandBufferRelease(p.cb);
            gpu_timer_read(p.timer);
        }
    }
    for (auto& f : b.afters) f(submitted);
    recording_free(b.rec);   // (what it held: the encoders have their own references now)
    b.rec = nullptr;
    for (WGPUBindGroup g : b.group_releases) wgpuBindGroupRelease(g);
    for (WGPUBuffer x : b.buffer_releases) wgpuBufferRelease(x);
    for (WGPURenderPipeline x : b.pipeline_releases) wgpuRenderPipelineRelease(x);
    if (!b.writebacks.empty()) retire_inflight(b.writebacks);   // (no readback started)
    if (RX_WEB_MT) {   // (copies: flush_batch)
        std::free(b.ro);
        std::free(b.rw);
    } else {
        if (b.ro) give_stream_memory(stream_ro.capacity, b.ro, b.ro_used);
        if (b.rw) give_stream_memory(stream_rw.capacity, b.rw, b.rw_used);
    }
    if (b.upload_bytes.capacity() && b.upload_bytes.capacity() <= (4u << 20)) {   // (a load's big ones not kept)
        b.upload_bytes.clear();
        std::lock_guard lk(upload_pool_mu);
        if (upload_pool.size() < 8) upload_pool.push_back(std::move(b.upload_bytes));
    }
    if (b.after) {
        const int64_t t0 = hos_now_ns();
        b.after();
        st_after_ns += hos_now_ns() - t0;
    }
    if (b.serial) {
        {
            std::lock_guard lk(index_mu);
            serial_index[b.serial % 1024] = {b.serial, last_index};
        }
        WGPUQueueWorkDoneCallbackInfo wd{};
        wd.mode = WGPUCallbackMode_AllowSpontaneous;
        wd.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* u1, void*) {
            const uint64_t serial = (uint64_t)(uintptr_t)u1;
            uint64_t cur = gpu_done_serial.load();
            while (cur < serial && !gpu_done_serial.compare_exchange_weak(cur, serial)) {}
        };
        wd.userdata1 = (void*)(uintptr_t)b.serial;
        wgpuQueueOnSubmittedWorkDone(dev.queue, wd);
    }
    wgpuDevicePoll(dev.device, false, nullptr);
}

/// After a frame's present (the submit thread): wait until the GPU has finished the frame RX_GPU_FRAMES_AHEAD back
/// (default 3; 0: off). Only the batches not yet submitted were capped (presents_queued); what was submitted could
/// pile up without bound when the GPU is slower than the game: with another game on the GPU, committed memory grew
/// ~2-3 GB a minute (each submission holds its staging) and so did the latency. A window's swapchain throttled
/// most of it; headless runs not at all. A fast GPU is never 3 frames behind: no wait.
void throttle_gpu_frames() {
#if !RX_WEB
    static const int ahead = getenv("RX_GPU_FRAMES_AHEAD") ? atoi(getenv("RX_GPU_FRAMES_AHEAD")) : 3;
    static std::deque<uint64_t> frames;   // (the presents' last submissions, oldest first)
    if (ahead <= 0 || !last_index) return;
    if (frames.empty() || frames.back() != last_index) frames.push_back(last_index);
    if ((int)frames.size() <= ahead) return;
    const WGPUSubmissionIndex i = frames.front();
    frames.pop_front();
    const int64_t t0 = hos_now_ns();
    wgpuDevicePoll(dev.device, true, &i);
    const int64_t waited = hos_now_ns() - t0;
    // (RX_LOG_GPU_AHEAD=1: every 300 frames, how long the submit thread waited and how many submissions ran ahead)
    static const bool log_ahead = env_flag("RX_LOG_GPU_AHEAD");
    static int64_t sum, most, nframes;
    sum += waited, most = std::max(most, waited);
    if (log_ahead && ++nframes % 300 == 0) {
        hos_log("rx: GPU frames ahead: waited %.1f ms in 300 frames (longest %.1f ms); submissions %llu, frame %d back "
                "is %llu", sum / 1e6, most / 1e6, (unsigned long long)last_index, ahead, (unsigned long long)i);
        sum = most = 0;
    }
#endif
}

void submit_main() {
    hos_host_thread_register_self("GPU submit");
    hos_thread_boost(1);
    for (;;) {
        std::unique_ptr<Batch> b;
        {
            std::unique_lock lk(submit_mu);
            submit_cv.wait(lk, [] { return !submit_queue.empty(); });
            b = std::move(submit_queue.front());
            submit_queue.pop_front();
            submit_busy++;
        }
        run_batch(*b);
        {
            std::lock_guard lk(submit_mu);
            submit_busy--;
            if (b->present) presents_queued--;
            if (b->serial > batch_serial_run) batch_serial_run = b->serial;
        }
        submitted_cv.notify_all();
        if (b->present) throttle_gpu_frames();
    }
}

#if RX_WEB_MT
}  // namespace
void post_to_gpu_thread(void (*fn)(void*), void* arg);   // rx_web_mt.cpp
/// The threaded web build: one queued batch run on the WebGPU thread (rx_web_mt.cpp), as the native submit thread
/// runs them. False when none is queued.
bool gpu_thread_batch() {
    std::unique_ptr<Batch> b;
    {
        std::lock_guard lk(submit_mu);
        if (submit_queue.empty()) return false;
        b = std::move(submit_queue.front());
        submit_queue.pop_front();
        submit_busy++;
    }
    run_batch(*b);
    {
        std::lock_guard lk(submit_mu);
        submit_busy--;
        if (b->present) presents_queued--;
        if (b->serial > batch_serial_run) batch_serial_run = b->serial;
    }
    submitted_cv.notify_all();
    return true;
}
namespace {
#endif

void push_batch(std::unique_ptr<Batch> b) {
    if (RX_WEB && !RX_WEB_MT) {   // (one thread makes every WebGPU call: the batch is finished and submitted here)
        run_batch(*b);
        if (b->serial > batch_serial_run) batch_serial_run = b->serial;
        return;
    }
    static bool started_thread;
    if (!started_thread && !RX_WEB_MT) {   // (the threaded web build: the WebGPU thread takes the batches)
        started_thread = true;
        std::thread(submit_main).detach();
    }
    const int64_t t0 = hos_now_ns();
    std::unique_lock lk(submit_mu);
    // the GPU thread stays at most 2 frames (and 64 batches) ahead of the queue
    // (the threaded web build: at most 8 batches, each holds a copy of the stream bytes it used (flush_batch); the heap is
    // ~570 MB)
    submitted_cv.wait(lk, [] { return submit_queue.size() < (RX_WEB_MT ? 8u : 64u) && presents_queued < 2; });
    st_wait_ns += hos_now_ns() - t0;
    if (b->present) presents_queued++;
    submit_queue.push_back(std::move(b));
    lk.unlock();
    submit_cv.notify_one();
#if RX_WEB_MT
    // (a job in the WebGPU thread's queue, so the batch runs in order with the calls this thread made before it:
    // reference counts, writes, creations. Run apart, a batch released objects before AddRefs queued ahead of it.)
    post_to_gpu_thread([](void*) { gpu_thread_batch(); }, nullptr);
#endif
}
} // namespace

void queue_upload(WGPUBuffer buffer, uint64_t offset, const void* data, uint64_t size) {
    if (!open_upload_bytes.capacity()) {
        std::lock_guard lk(upload_pool_mu);
        if (!upload_pool.empty()) {
            open_upload_bytes.swap(upload_pool.back());
            upload_pool.pop_back();
        }
    }
    const uint64_t src = open_upload_bytes.size();
    const auto* p = static_cast<const uint8_t*>(data);
    open_upload_bytes.insert(open_upload_bytes.end(), p, p + size);
    open_upload_bytes.resize((open_upload_bytes.size() + 3) & ~size_t{3});
    open_uploads.push_back({buffer, offset, src, size});
}

void writeback_after_batch(uint64_t rw_offset, uint64_t size, uint64_t gpu_va) {
    open_writebacks.push_back({rw_offset, size, gpu_va});
}

void wait_submitted() {
    const int64_t t0 = hos_now_ns();
    std::unique_lock lk(submit_mu);
    submitted_cv.wait(lk, [] { return submit_queue.empty() && !submit_busy; });
    spike.wait_ns += hos_now_ns() - t0;
}

FrameSpike spike;

void flush_batch() {
    flush_pending_clears();
    if (!enc && open_uploads.empty() && open_afters.empty()) return;
    const int64_t t0 = hos_now_ns();
    count("upload_bytes", (int64_t)(stream_ro.used + stream_rw.used));
    end_render_pass("pe_flush");
    auto b = std::make_unique<Batch>();
    b->serial = ++batch_serial;
    const bool has_commands = enc != nullptr;
    if (encoding_deferred()) {
        b->deferred = has_commands;
        b->rec = recording_take();
    } else {
        b->enc = enc;
    }
    enc = nullptr;
    b->uploads.swap(open_uploads);
    b->upload_bytes.swap(open_upload_bytes);
    b->writebacks.swap(open_writebacks);
    b->group_releases.swap(open_group_releases);
    b->buffer_releases.swap(open_buffer_releases);
    b->pipeline_releases.swap(open_pipeline_releases);
    b->afters.swap(open_afters);
    if (!b->writebacks.empty()) {
        std::lock_guard lk(inflight_mu);
        inflight.insert(inflight.end(), b->writebacks.begin(), b->writebacks.end());
    }
    if (has_commands) {
        b->ro = stream_ro.cpu;
        b->ro_used = stream_ro.used;
        b->rw = stream_rw.cpu;
        b->rw_used = stream_rw.used;
        if (RX_WEB_MT) {
            // (the threaded web build: the batch takes a copy of what it used, a few MB, and the blocks (48 MB) stay
            // here: a block for each batch in flight ran the web's host heap out during a load)
            auto copy_used = [](const uint8_t* m, uint64_t n) -> uint8_t* {
                if (!n) return nullptr;
                auto* c = static_cast<uint8_t*>(std::malloc((n + 3) & ~3ull));
                std::memcpy(c, m, n);
                return c;
            };
            b->ro = copy_used(stream_ro.cpu, stream_ro.used);
            b->rw = copy_used(stream_rw.cpu, stream_rw.used);
        } else if (RX_WEB) {
            stream_ro.cpu = stream_rw.cpu = nullptr;
        } else {
            stream_ro.cpu = take_stream_memory(stream_ro.capacity);
            stream_rw.cpu = take_stream_memory(stream_rw.capacity);
        }
    }
    stream_ro.used = stream_rw.used = 0;
    push_batch(std::move(b));
    if (!stream_ro.cpu) stream_ro.cpu = take_stream_memory(stream_ro.capacity);
    if (!stream_rw.cpu) stream_rw.cpu = take_stream_memory(stream_rw.capacity);
    buffer_cache_new_batch();
    count("t_flush", hos_now_ns() - t0);
}

/// The window's present (the submit thread): the frame's image blitted into the surface texture.
void present_pacing() {
    static int64_t first, last, window_start;
    static std::vector<int64_t> gaps;
    const int64_t now = hos_now_ns();
    if (!first) first = window_start = now;
    if (last) gaps.push_back(now - last);
    last = now;
    if (now - window_start < 5000000000LL || gaps.empty()) return;
    const double avg = (double)(now - window_start) / (double)gaps.size();
    const double limit = (avg < 22e6 ? 1e9 / 60 : 1e9 / 30) * 1.5;
    int stutters = 0;
    int64_t longest = 0;
    for (int64_t g : gaps) {
        stutters += g > limit;
        longest = std::max(longest, g);
    }
    if (stutters)
        hos_log("rx: presents at %.0f s: %.1f fps, %d stutters (gaps over %.0f ms), longest %.1f ms", (now - first) / 1e9,
                (double)gaps.size() * 1e9 / (double)(now - window_start), stutters, limit / 1e6, longest / 1e6);
    gaps.clear();
    window_start = now;
}

/// The frame into the window and presented (the submit thread, or with frame interpolation the display thread:
/// rx_framegen.cpp); `view` is released. One present at a time.
void surface_present(WGPUTextureView view, uint32_t sw, uint32_t sh, int32_t ww, int32_t wh, bool debug_present) {
    static std::mutex present_mu;
    std::lock_guard present_lk(present_mu);
    if (ww > 0 && wh > 0 && ((uint32_t)ww != surface_w || (uint32_t)wh != surface_h))
        configure_surface((uint32_t)ww, (uint32_t)wh);
    WGPUSurfaceTexture st{};
    wgpuSurfaceGetCurrentTexture(surface, &st);
    if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        static int logged;
        if (logged++ < 8) hos_log("rx: no surface texture (status %d)", (int)st.status);
        if (st.texture) wgpuTextureRelease(st.texture);
        configure_surface(surface_w, surface_h, true);
        if (view) wgpuTextureViewRelease(view);
        return;
    }
    WGPUTextureView target = wgpuTextureCreateView(st.texture, nullptr);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    WGPUBindGroup bg = nullptr;
    if (view && !debug_present) {   // (the frame, its aspect kept, upscaled with FSR 1: rx_present_fx.cpp)
        present_fx_draw(e, view, sw, sh, target, surface_format, surface_w, surface_h, WGPUColor{0, 0, 0, 1});
    } else {
        WGPURenderPassColorAttachment ca{};
        ca.view = target;
        ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        ca.loadOp = WGPULoadOp_Clear;
        ca.storeOp = WGPUStoreOp_Store;
        ca.clearValue = debug_present ? WGPUColor{1.0, 0.0, 1.0, 1.0} : WGPUColor{0.15, 0.0, 0.15, 1.0};
        WGPURenderPassDescriptor pd{};
        pd.colorAttachmentCount = 1;
        pd.colorAttachments = &ca;
        WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(e, &pd);
        if (view) {
            WGPUBindGroupEntry be[2]{};
            be[0].binding = 0;
            be[0].textureView = view;
            be[1].binding = 1;
            be[1].sampler = blit_sampler;
            WGPUBindGroupDescriptor bd{};
            bd.layout = blit_layout;
            bd.entryCount = 2;
            bd.entries = be;
            bg = wgpuDeviceCreateBindGroup(dev.device, &bd);
            wgpuRenderPassEncoderSetPipeline(p, blit_pipeline);
            wgpuRenderPassEncoderSetBindGroup(p, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(p, 3, 1, 0, 0);
        }
        wgpuRenderPassEncoderEnd(p);
        wgpuRenderPassEncoderRelease(p);
    }
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(e);
    if (bg) wgpuBindGroupRelease(bg);
#ifdef __EMSCRIPTEN__
    rx_web_present();   // (the page counts frames)
    rx_web_yield();     // (back to the event loop: the browser shows the canvas and runs WebGPU's callbacks)
#else
    wgpuSurfacePresent(surface);
#endif
    present_pacing();
    wgpuTextureViewRelease(target);
    wgpuTextureRelease(st.texture);
    if (view) wgpuTextureViewRelease(view);
}

void present_on_submit_thread(WGPUTextureView view, uint32_t sw, uint32_t sh, LONG ww, LONG wh, bool debug_present) {
    // frame interpolation (F7): the frame and the ones made between it and the last go to the display thread
    if (view && !debug_present && present_framegen() && framegen_frame(view, sw, sh, ww, wh)) return;
    surface_present(view, sw, sh, ww, wh, debug_present);
}

/// Queues a present of the frame (after its batches): runs on the submit thread.
void queue_present(std::function<void()> fn) {
    flush_batch();
    auto b = std::make_unique<Batch>();
    b->after = std::move(fn);
    b->present = true;
    push_batch(std::move(b));
}

/// RX_FLASH_LOG=<dir> (diagnostic): every presented frame is read back (asynchronously, a ring of staging buffers)
/// and reduced to a 16x9 grid of mean luma on a worker. A frame whose cells jump away from both neighbours while the
/// neighbours agree is a one-frame flash: logged ("rx: flash ...") and the three frames saved as PNGs (the first 40).
/// RX_FLASH_DELTA=<0-255>: the cell jump that counts (default 48); RX_FLASH_CELLS: how many cells must jump (default 8:
/// a small bright thing moving fast, such as a fairy, jumps in 2-5); frames are dropped when the ring is full.
namespace {
struct FlashFrame {
    int64_t frame = 0, t_ns = 0;
    float cell[16 * 9]{};
    float mean = 0;
    std::vector<uint8_t> px;   // (kept for the saved PNGs)
    uint32_t w = 0, h = 0;
    size_t stride = 0;
};
struct FlashSlot {
    WGPUBuffer buf = nullptr;
    size_t size = 0;
    uint32_t w = 0, h = 0;
    size_t stride = 0;
    int64_t frame = 0, t_ns = 0;
    std::atomic<int> state{0};   // 0 free, 1 copying/mapping, 2 mapped (the worker's)
};
FlashSlot flash_slots[6];
std::mutex flash_mu;
std::condition_variable flash_cv;
std::deque<int> flash_ready;

void flash_worker() {
    hos_host_thread_register_self("flash probe");
    static const char* dir = getenv("RX_FLASH_LOG");
    static const float delta = getenv("RX_FLASH_DELTA") ? (float)atof(getenv("RX_FLASH_DELTA")) : 48.0f;
    static const int min_cells = getenv("RX_FLASH_CELLS") ? atoi(getenv("RX_FLASH_CELLS")) : 8;
    static const int64_t t0 = hos_now_ns();
    std::deque<FlashFrame> hist;   // the last three frames
    int saved = 0, flashes = 0;
    int64_t frames = 0, last_report = 0;
    for (;;) {
        int i;
        {
            std::unique_lock lk(flash_mu);
            flash_cv.wait(lk, [] { return !flash_ready.empty(); });
            i = flash_ready.front();
            flash_ready.pop_front();
        }
        FlashSlot& s = flash_slots[i];
        FlashFrame f;
        f.frame = s.frame, f.t_ns = s.t_ns, f.w = s.w, f.h = s.h, f.stride = s.stride;
        const auto* p = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(s.buf, 0, s.size));
        if (p) {
            double sum[16 * 9]{};
            int n[16 * 9]{};
            for (uint32_t y = 0; y < s.h; y += 4)
                for (uint32_t x = 0; x < s.w; x += 4) {
                    const uint8_t* q = p + (size_t)y * s.stride + (size_t)x * 4;
                    const int c = (int)(y * 9 / s.h) * 16 + (int)(x * 16 / s.w);
                    sum[c] += 0.2126 * q[0] + 0.7152 * q[1] + 0.0722 * q[2];
                    n[c]++;
                }
            double m = 0;
            for (int c = 0; c < 16 * 9; c++) {
                f.cell[c] = n[c] ? (float)(sum[c] / n[c]) : 0.0f;
                m += f.cell[c];
            }
            f.mean = (float)(m / (16 * 9));
            if (saved < 40) f.px.assign(p, p + s.size);
        }
        wgpuBufferUnmap(s.buf);
        s.state.store(0);
        frames++;
        hist.push_back(std::move(f));
        if (hist.size() > 3) hist.pop_front();
        if (hist.size() == 3 && hist[0].frame + 2 == hist[2].frame) {   // (consecutive frames only)
            const FlashFrame &a = hist[0], &b = hist[1], &c = hist[2];
            int cells = 0;
            float worst = 0;
            for (int k = 0; k < 16 * 9; k++) {
                const float up = std::min(std::fabs(b.cell[k] - a.cell[k]), std::fabs(b.cell[k] - c.cell[k]));
                const float spike = up - std::fabs(c.cell[k] - a.cell[k]);
                if (spike > delta) cells++;
                worst = std::max(worst, spike);
            }
            if (cells >= min_cells) {
                flashes++;
                hos_log("rx: flash at frame %lld (%.1f s into the probe): %d of 144 cells, jump %.0f; mean luma %.0f -> %.0f -> %.0f",
                        (long long)b.frame, (b.t_ns - t0) / 1e9, cells, worst, a.mean, b.mean, c.mean);
                if (dir && saved < 40 && !a.px.empty() && !b.px.empty() && !c.px.empty()) {
                    for (const FlashFrame* g : {&a, &b, &c}) {
                        std::vector<uint8_t> copy = g->px;
                        for (size_t j = 3; j < copy.size(); j += 4) copy[j] = 255;
                        char path[512];
                        std::snprintf(path, sizeof path, "%s/flash_%06lld_f%lld.png", dir, (long long)b.frame,
                                      (long long)g->frame);
                        write_png(path, copy.data(), g->w, g->h, g->stride);
                    }
                    saved++;
                }
            }
        }
        if (frames - last_report >= 600) {
            last_report = frames;
            hos_log("rx: flash probe: %lld frames read, %d flashes", (long long)frames, flashes);
        }
    }
}
}  // namespace

void flash_probe(Image* im) {
    static const char* dir = getenv("RX_FLASH_LOG");
    if (!dir || !im || (im->format != WGPUTextureFormat_RGBA8Unorm && im->format != WGPUTextureFormat_RGBA8UnormSrgb))
        return;
    static std::once_flag started;
    std::call_once(started, [] { std::thread(flash_worker).detach(); });
    int i = 0;
    while (i < 6 && flash_slots[i].state.load() != 0) i++;
    if (i == 6) return count("flash_probe_dropped");
    FlashSlot& s = flash_slots[i];
    const size_t stride = ((size_t)im->width * 4 + 255) & ~size_t(255), size = stride * im->height;
    if (s.size != size) {
        if (s.buf) wgpuBufferRelease(s.buf);
        WGPUBufferDescriptor bd{};
        bd.size = size;
        bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        s.buf = wgpuDeviceCreateBuffer(dev.device, &bd);
        s.size = size;
    }
    s.w = im->width, s.h = im->height, s.stride = stride, s.frame = frame_number, s.t_ns = hos_now_ns();
    s.state.store(1);
    WGPUTexture tex = im->tex;
    wgpuTextureAddRef(tex);
    // after the frame's batches, on the submit thread (queue order)
    flush_batch();
    auto b = std::make_unique<Batch>();
    b->after = [i, tex] {
        FlashSlot& s = flash_slots[i];
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
        WGPUTexelCopyTextureInfo src{};
        src.texture = tex;
        WGPUTexelCopyBufferInfo dst{};
        dst.buffer = s.buf;
        dst.layout.bytesPerRow = (uint32_t)s.stride;
        dst.layout.rowsPerImage = s.h;
        const WGPUExtent3D extent{s.w, s.h, 1};
        wgpuCommandEncoderCopyTextureToBuffer(e, &src, &dst, &extent);
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
        wgpuQueueSubmit(dev.queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(e);
        wgpuTextureRelease(tex);
        WGPUBufferMapCallbackInfo mc{};
        mc.mode = WGPUCallbackMode_AllowSpontaneous;
        mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
            const int i = (int)(intptr_t)u1;
            if (status != WGPUMapAsyncStatus_Success) {
                flash_slots[i].state.store(0);
                return;
            }
            flash_slots[i].state.store(2);
            {
                std::lock_guard lk(flash_mu);
                flash_ready.push_back(i);
            }
            flash_cv.notify_one();
        };
        mc.userdata1 = (void*)(intptr_t)i;
        wgpuBufferMapAsync(s.buf, WGPUMapMode_Read, 0, s.size, mc);
    };
    push_batch(std::move(b));
}

// ---- images ----
namespace {
std::vector<WGPUTextureView> retired;
}

void retire_view(WGPUTextureView view) {
    retired.push_back(view);
}

bool retired_views_pending() {
    return retired.size() > 256;
}

void release_retired_views() {
    for (WGPUTextureView v : retired) wgpuTextureViewRelease(v);
    retired.clear();
}

const std::vector<WGPUTextureView>& retired_view_list() {
    return retired;
}

void release_buffer_later(WGPUBuffer b) {
    if (encoding_deferred()) open_buffer_releases.push_back(b);
    else wgpuBufferRelease(b);
}

void release_bind_group_later(WGPUBindGroup g) {
    if (encoding_deferred()) open_group_releases.push_back(g);
    else wgpuBindGroupRelease(g);
}

void release_pipeline_later(WGPURenderPipeline p) {
    if (encoding_deferred()) open_pipeline_releases.push_back(p);
    else wgpuRenderPipelineRelease(p);
}

Image::~Image() {
    if (rendered)
        for (auto it = pending_clears.begin(); it != pending_clears.end();)
            it = it->im == this ? pending_clears.erase(it) : std::next(it);
    for (auto& v : views) retire_view(v.second);
    if (attachment) retire_view(attachment);
    if (tex) wgpuTextureRelease(tex);
}

const std::unordered_map<uint64_t, std::unique_ptr<Image>>& all_images() {
    return images;
}

uint64_t images_generation() {
    return images_changed;
}

bool images_changed_in(uint64_t since, uint64_t lo, uint64_t hi) {
    if (images_changed - since >= 256) return true;   // (older than the log)
    for (uint64_t g = since + 1; g <= images_changed; g++) {
        const ImageChange& c = image_changes[g & 255];
        if (c.generation != g || (c.guest >= lo && c.guest < hi)) return true;
    }
    return false;
}

Image* image_at(uint64_t guest) {
    // a direct-mapped cache in front of the map, valid while no image was added or dropped (the map's lookup was
    // ~4% of the GPU thread: per texture binding, subresource and attachment)
    struct Slot {
        uint64_t guest = ~0ull, generation = 0;
        Image* image = nullptr;
    };
    thread_local Slot slots[1024];
    Slot& s = slots[(guest >> 9 ^ guest >> 19) & 1023];
    if (s.guest == guest && s.generation == images_changed) return s.image;
    auto it = images.find(guest);
    s = {guest, images_changed, it == images.end() ? nullptr : it->second.get()};
    return s.image;
}

namespace {
uint64_t barrier_epoch = 1;   // texture barriers so far (feedback snapshots taken since the last one stay valid)
} // namespace

void texture_barrier(uint32_t method, uint32_t arg) {
    barrier_epoch++;
    count("texture_barriers");
    if (frame_dump_active()) {
        char b[64];
        std::snprintf(b, sizeof b, "barrier %#x = %#x", method, arg);
        dump_draw(b);
    }
}

Image* snapshot_for_feedback(Image& image) {
    if (!image.sampled_copy) {
        auto copy = std::make_unique<Image>();
        copy->width = image.width;
        copy->height = image.height;
        copy->layers = image.layers;
        copy->levels = image.levels;
        copy->pixel_format = image.pixel_format;
        copy->format = image.format;
        copy->dimension = image.dimension;
        WGPUTextureDescriptor d{};
        d.label = sv("rx attachment feedback snapshot");
        d.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        d.dimension = image.dimension;
        d.size = {image.width, image.height, image.layers};
        d.format = image.format;
        d.mipLevelCount = image.levels;
        d.sampleCount = 1;
        scope_push();
        copy->tex = wgpuDeviceCreateTexture(dev.device, &d);
        const std::string error = scope_pop();
        if (!error.empty()) {
            log_limited("feedback_create", 8, "rx: attachment feedback snapshot not created: %s", error.c_str());
            return nullptr;
        }
        image.sampled_copy = std::move(copy);
    }
    // Refreshed once per texture barrier: the guest GPU sees its own render target writes through the texture
    // cache only after a barrier, and TOTK puts one before a group of feedback draws (its characters' G-buffer
    // draws read the G-buffer: 34 per frame), not between them. A copy before each draw split the G-buffer pass
    // that many times (each copy ~20 MB). RX_SNAPSHOT_EVERY_DRAW=1: the old behavior.
    static const bool every_draw = env_flag("RX_SNAPSHOT_EVERY_DRAW");
    if (!every_draw && image.snapshot_epoch == barrier_epoch) {
        count("feedback_snapshot_reused");
        return image.sampled_copy.get();
    }
    image.snapshot_epoch = barrier_epoch;
    count("feedback_snapshots");
    end_render_pass("pe_feedback");
    for (uint32_t level = 0; level < image.levels; ++level) {
        WGPUTexelCopyTextureInfo source{}, destination{};
        source.texture = image.tex;
        destination.texture = image.sampled_copy->tex;
        source.mipLevel = destination.mipLevel = level;
        const WGPUExtent3D extent{std::max(1u, image.width >> level), std::max(1u, image.height >> level),
                                 image.dimension == WGPUTextureDimension_3D
                                     ? std::max(1u, image.layers >> level) : image.layers};
        ce_copy_texture_to_texture(encoder(), &source, &destination, &extent);
    }
    return image.sampled_copy.get();
}

Image* find_render_target(uint64_t gpu_va, uint32_t width, uint32_t height, PixelFormat pf, uint64_t key) {
    const uint64_t guest = key ? key : hos_gpu_translate(gpu_va);
    if (!guest || !width || !height || width > dev.limits.maxTextureDimension2D ||
        height > dev.limits.maxTextureDimension2D)
        return nullptr;
    if (Image* im = image_at(guest); im && im->width == width && im->height == height && im->pixel_format == pf) {
        im->last_use = frame_number;   // (the usual case: the same surface again)
        return im;
    }
    auto it = images.find(guest);
    if (it != images.end()) {
        Image* im = it->second.get();
        if (im->width == width && im->height == height && im->pixel_format == pf) {
            im->last_use = frame_number;
            return im;
        }
        // the memory now holds another surface: the old image steps aside (ending a pass that draws into it), and
        // a spare of this shape takes its place
        bool attached = pass_at.depth == im;
        for (Image* c : pass_at.color) attached |= c == im;
        if (attached) end_render_pass("pe_alias");
        std::unique_ptr<Image> old = std::move(it->second);
        images.erase(it);
        note_image_change(guest);
        if (env_flag("RX_NO_SPARES")) old.reset();   // (debugging: every new surface is a new, cleared texture)
        auto [a, b] = spare_images.equal_range(guest);
        for (auto s = a; s != b; ++s) {
            Image* sp = s->second.get();
            if (sp->width == width && sp->height == height && sp->pixel_format == pf) {
                std::unique_ptr<Image> back = std::move(s->second);
                spare_images.erase(s);
                if (old) spare_images.emplace(guest, std::move(old));
                back->last_use = frame_number;
                Image* raw = back.get();
                images[guest] = std::move(back);
                count("rt_spare_reused");
                return raw;
            }
        }
        if (old) spare_images.emplace(guest, std::move(old));
    }
    WGPUTextureFormat format = wgpu_format(pf);
    static const bool norm16_rt = env_flag("RX_NORM16_RT");
    if (!norm16_rt || !dev.norm16) switch (pf) {
    case PixelFormat::R16_UNORM:
    case PixelFormat::R16_SNORM: format = WGPUTextureFormat_R16Float; break;
    case PixelFormat::R16G16_UNORM:
    case PixelFormat::R16G16_SNORM: format = WGPUTextureFormat_RG16Float; break;
    case PixelFormat::R16G16B16A16_UNORM:
    case PixelFormat::R16G16B16A16_SNORM: format = WGPUTextureFormat_RGBA16Float; break;
    default: break;
    }
    if (format == WGPUTextureFormat_Undefined) {
        log_limited("rt_format", 16, "rx: render target format %d has no WebGPU format", (int)pf);
        return nullptr;
    }
    auto im = std::make_unique<Image>();
    im->gpu_va = gpu_va;
    im->guest = guest;
    im->width = width;
    im->height = height;
    im->pixel_format = pf;
    im->format = format;
    im->rendered = true;
    im->last_use = frame_number;
    WGPUTextureDescriptor d{};
    d.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc |
              WGPUTextureUsage_CopyDst;
    d.dimension = WGPUTextureDimension_2D;
    d.size = {width, height, 1};
    d.format = format;
    d.mipLevelCount = 1;
    d.sampleCount = 1;
    const WGPUTextureFormat pair = srgb_pair(format);
    if (pair != WGPUTextureFormat_Undefined) {
        d.viewFormatCount = 1;
        d.viewFormats = &pair;
    }
    scope_push();
    im->tex = wgpuDeviceCreateTexture(dev.device, &d);
    im->attachment = wgpuTextureCreateView(im->tex, nullptr);
    if (const std::string error = scope_pop(); !error.empty()) {
        log_limited("rt_create", 16, "rx: render target %ux%u format %d not created: %s", width, height, (int)pf,
                    error.c_str());
        return nullptr;
    }
    Image* raw = im.get();
    images[guest] = std::move(im);
    note_image_change(guest);
    static int logged;
    if (logged++ < 64)
        hos_log("rx: render target %ux%u format %d (WebGPU %#x) at GPU %#llx (guest %#llx)", width, height, (int)pf,
                (unsigned)format, (unsigned long long)gpu_va, (unsigned long long)guest);
    return raw;
}

WGPUTextureFormat srgb_pair(WGPUTextureFormat f) {
    switch (f) {
    case WGPUTextureFormat_RGBA8Unorm: return WGPUTextureFormat_RGBA8UnormSrgb;
    case WGPUTextureFormat_RGBA8UnormSrgb: return WGPUTextureFormat_RGBA8Unorm;
    case WGPUTextureFormat_BGRA8Unorm: return WGPUTextureFormat_BGRA8UnormSrgb;
    case WGPUTextureFormat_BGRA8UnormSrgb: return WGPUTextureFormat_BGRA8Unorm;
    case WGPUTextureFormat_BC1RGBAUnorm: return WGPUTextureFormat_BC1RGBAUnormSrgb;
    case WGPUTextureFormat_BC1RGBAUnormSrgb: return WGPUTextureFormat_BC1RGBAUnorm;
    case WGPUTextureFormat_BC2RGBAUnorm: return WGPUTextureFormat_BC2RGBAUnormSrgb;
    case WGPUTextureFormat_BC2RGBAUnormSrgb: return WGPUTextureFormat_BC2RGBAUnorm;
    case WGPUTextureFormat_BC3RGBAUnorm: return WGPUTextureFormat_BC3RGBAUnormSrgb;
    case WGPUTextureFormat_BC3RGBAUnormSrgb: return WGPUTextureFormat_BC3RGBAUnorm;
    case WGPUTextureFormat_BC7RGBAUnorm: return WGPUTextureFormat_BC7RGBAUnormSrgb;
    case WGPUTextureFormat_BC7RGBAUnormSrgb: return WGPUTextureFormat_BC7RGBAUnorm;
    default: return WGPUTextureFormat_Undefined;
    }
}

bool is_depth_format(WGPUTextureFormat f) {
    switch (f) {
    case WGPUTextureFormat_Depth16Unorm:
    case WGPUTextureFormat_Depth24Plus:
    case WGPUTextureFormat_Depth24PlusStencil8:
    case WGPUTextureFormat_Depth32Float:
    case WGPUTextureFormat_Depth32FloatStencil8:
    case WGPUTextureFormat_Stencil8:
        return true;
    default:
        return false;
    }
}

bool has_stencil(WGPUTextureFormat f) {
    return f == WGPUTextureFormat_Depth24PlusStencil8 || f == WGPUTextureFormat_Depth32FloatStencil8 ||
           f == WGPUTextureFormat_Stencil8;
}

WGPUTextureFormat wgpu_format(PixelFormat pf) {
    using P = PixelFormat;
    switch (pf) {
    case P::A8B8G8R8_UNORM: return WGPUTextureFormat_RGBA8Unorm;
    case P::A8B8G8R8_SNORM: return WGPUTextureFormat_RGBA8Snorm;
    case P::A8B8G8R8_SINT: return WGPUTextureFormat_RGBA8Sint;
    case P::A8B8G8R8_UINT: return WGPUTextureFormat_RGBA8Uint;
    case P::A8B8G8R8_SRGB: return WGPUTextureFormat_RGBA8UnormSrgb;
    case P::B8G8R8A8_UNORM: return WGPUTextureFormat_BGRA8Unorm;
    case P::B8G8R8A8_SRGB: return WGPUTextureFormat_BGRA8UnormSrgb;
    // 16-bit packed formats: no WebGPU equivalent; drawn as RGBA8 (textures are converted on upload)
    case P::R5G6B5_UNORM:
    case P::B5G6R5_UNORM:
    case P::A1R5G5B5_UNORM:
    case P::A1B5G5R5_UNORM:
    case P::A5B5G5R1_UNORM:
    case P::A4B4G4R4_UNORM:
    case P::G4R4_UNORM:
        return WGPUTextureFormat_RGBA8Unorm;
    case P::A2B10G10R10_UNORM: return WGPUTextureFormat_RGB10A2Unorm;
    case P::A2B10G10R10_UINT: return WGPUTextureFormat_RGB10A2Uint;
    case P::A2R10G10B10_UNORM: return WGPUTextureFormat_RGB10A2Unorm;
    case P::R8_UNORM: return WGPUTextureFormat_R8Unorm;
    case P::R8_SNORM: return WGPUTextureFormat_R8Snorm;
    case P::R8_SINT: return WGPUTextureFormat_R8Sint;
    case P::R8_UINT: return WGPUTextureFormat_R8Uint;
    case P::R8G8_UNORM: return WGPUTextureFormat_RG8Unorm;
    case P::R8G8_SNORM: return WGPUTextureFormat_RG8Snorm;
    case P::R8G8_SINT: return WGPUTextureFormat_RG8Sint;
    case P::R8G8_UINT: return WGPUTextureFormat_RG8Uint;
    case P::R16G16B16A16_FLOAT: return WGPUTextureFormat_RGBA16Float;
    case P::R16G16B16X16_FLOAT: return WGPUTextureFormat_RGBA16Float;
    case P::R16G16B16A16_UNORM: return dev.norm16 ? WGPUTextureFormat_RGBA16Unorm : WGPUTextureFormat_RGBA16Float;
    case P::R16G16B16A16_SNORM: return dev.norm16 ? WGPUTextureFormat_RGBA16Snorm : WGPUTextureFormat_RGBA16Float;
    case P::R16G16B16A16_SINT: return WGPUTextureFormat_RGBA16Sint;
    case P::R16G16B16A16_UINT: return WGPUTextureFormat_RGBA16Uint;
    case P::B10G11R11_FLOAT: return WGPUTextureFormat_RG11B10Ufloat;
    case P::E5B9G9R9_FLOAT: return WGPUTextureFormat_RGB9E5Ufloat;
    case P::R32G32B32A32_UINT: return WGPUTextureFormat_RGBA32Uint;
    case P::R32G32B32A32_FLOAT: return WGPUTextureFormat_RGBA32Float;
    case P::R32G32B32A32_SINT: return WGPUTextureFormat_RGBA32Sint;
    case P::R32G32B32_FLOAT: return WGPUTextureFormat_RGBA32Float;   // converted on upload
    case P::R32G32_FLOAT: return WGPUTextureFormat_RG32Float;
    case P::R32G32_SINT: return WGPUTextureFormat_RG32Sint;
    case P::R32G32_UINT: return WGPUTextureFormat_RG32Uint;
    case P::R32_FLOAT: return WGPUTextureFormat_R32Float;
    case P::R32_UINT: return WGPUTextureFormat_R32Uint;
    case P::R32_SINT: return WGPUTextureFormat_R32Sint;
    case P::R16_FLOAT: return WGPUTextureFormat_R16Float;
    case P::R16_UNORM: return dev.norm16 ? WGPUTextureFormat_R16Unorm : WGPUTextureFormat_R16Float;
    case P::R16_SNORM: return dev.norm16 ? WGPUTextureFormat_R16Snorm : WGPUTextureFormat_R16Float;
    case P::R16_UINT: return WGPUTextureFormat_R16Uint;
    case P::R16_SINT: return WGPUTextureFormat_R16Sint;
    case P::R16G16_UNORM: return dev.norm16 ? WGPUTextureFormat_RG16Unorm : WGPUTextureFormat_RG16Float;
    case P::R16G16_FLOAT: return WGPUTextureFormat_RG16Float;
    case P::R16G16_UINT: return WGPUTextureFormat_RG16Uint;
    case P::R16G16_SINT: return WGPUTextureFormat_RG16Sint;
    case P::R16G16_SNORM: return dev.norm16 ? WGPUTextureFormat_RG16Snorm : WGPUTextureFormat_RG16Float;
    case P::BC1_RGBA_UNORM: return dev.bc ? WGPUTextureFormat_BC1RGBAUnorm : WGPUTextureFormat_RGBA8Unorm;
    case P::BC1_RGBA_SRGB: return dev.bc ? WGPUTextureFormat_BC1RGBAUnormSrgb : WGPUTextureFormat_RGBA8UnormSrgb;
    case P::BC2_UNORM: return dev.bc ? WGPUTextureFormat_BC2RGBAUnorm : WGPUTextureFormat_RGBA8Unorm;
    case P::BC2_SRGB: return dev.bc ? WGPUTextureFormat_BC2RGBAUnormSrgb : WGPUTextureFormat_RGBA8UnormSrgb;
    case P::BC3_UNORM: return dev.bc ? WGPUTextureFormat_BC3RGBAUnorm : WGPUTextureFormat_RGBA8Unorm;
    case P::BC3_SRGB: return dev.bc ? WGPUTextureFormat_BC3RGBAUnormSrgb : WGPUTextureFormat_RGBA8UnormSrgb;
    case P::BC4_UNORM: return dev.bc ? WGPUTextureFormat_BC4RUnorm : WGPUTextureFormat_R8Unorm;
    case P::BC4_SNORM: return dev.bc ? WGPUTextureFormat_BC4RSnorm : WGPUTextureFormat_R8Snorm;
    case P::BC5_UNORM: return dev.bc ? WGPUTextureFormat_BC5RGUnorm : WGPUTextureFormat_RG8Unorm;
    case P::BC5_SNORM: return dev.bc ? WGPUTextureFormat_BC5RGSnorm : WGPUTextureFormat_RG8Snorm;
    case P::BC7_UNORM: return dev.bc ? WGPUTextureFormat_BC7RGBAUnorm : WGPUTextureFormat_RGBA8Unorm;
    case P::BC7_SRGB: return dev.bc ? WGPUTextureFormat_BC7RGBAUnormSrgb : WGPUTextureFormat_RGBA8UnormSrgb;
    case P::BC6H_UFLOAT: return dev.bc ? WGPUTextureFormat_BC6HRGBUfloat : WGPUTextureFormat_RGBA16Float;
    case P::BC6H_SFLOAT: return dev.bc ? WGPUTextureFormat_BC6HRGBFloat : WGPUTextureFormat_RGBA16Float;
    case P::D32_FLOAT: return WGPUTextureFormat_Depth32Float;
    case P::D16_UNORM: return WGPUTextureFormat_Depth16Unorm;
    case P::X8_D24_UNORM: return WGPUTextureFormat_Depth24Plus;
    case P::S8_UINT: return WGPUTextureFormat_Stencil8;
    case P::D24_UNORM_S8_UINT:
    case P::S8_UINT_D24_UNORM:
        return WGPUTextureFormat_Depth24PlusStencil8;
    case P::D32_FLOAT_S8_UINT:
        return dev.depth32_stencil8 ? WGPUTextureFormat_Depth32FloatStencil8 : WGPUTextureFormat_Depth24PlusStencil8;
    default:
        if (VideoCore::Surface::IsPixelFormatASTC(pf))   // decoded on upload
            return VideoCore::Surface::IsPixelFormatSRGB(pf) ? WGPUTextureFormat_RGBA8UnormSrgb
                                                             : WGPUTextureFormat_RGBA8Unorm;
        return WGPUTextureFormat_Undefined;
    }
}

// ---- clears ----
namespace {

/// A clear of part of a target (the scissor), of some components only, or of some stencil bits: a triangle over the
/// target with the clear value, drawn with that scissor and write mask (suyu: BlitImageHelper::ClearColor /
/// ClearDepthStencil). One pipeline per format, mask and value (TOTK uses a handful).
WGPURenderPipeline partial_clear_pipeline(WGPUTextureFormat color_format, uint32_t write_mask, const float* color,
                                          WGPUTextureFormat depth_format, bool depth, bool stencil, float depth_value,
                                          uint32_t stencil_write_mask) {
    uint32_t cb[4]{}, db = 0;
    if (color) std::memcpy(cb, color, sizeof cb);
    std::memcpy(&db, &depth_value, 4);
    char key[160];
    std::snprintf(key, sizeof key, "%u/%u/%08x%08x%08x%08x/%u/%d%d/%08x/%u", (unsigned)color_format, write_mask, cb[0],
                  cb[1], cb[2], cb[3], (unsigned)depth_format, (int)depth, (int)stencil, db, stencil_write_mask);
    static std::unordered_map<std::string, WGPURenderPipeline> cache;
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    std::string wgsl = "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {\n"
                       "  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));\n"
                       "  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);\n}\n";
    char b[320];
    if (color_format != WGPUTextureFormat_Undefined)
        std::snprintf(b, sizeof b,
                      "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(bitcast<f32>(%uu), "
                      "bitcast<f32>(%uu), bitcast<f32>(%uu), bitcast<f32>(%uu)); }\n", cb[0], cb[1], cb[2], cb[3]);
    else
        std::snprintf(b, sizeof b, "@fragment fn fs() -> @builtin(frag_depth) f32 { return bitcast<f32>(%uu); }\n",
                      db);
    wgsl += b;
    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = WGPUStringView{wgsl.data(), wgsl.size()};
    WGPUShaderModuleDescriptor md{};
    md.nextInChain = &src.chain;
    scope_push();
    WGPUShaderModule m = wgpuDeviceCreateShaderModule(dev.device, &md);
    WGPUColorTargetState t{};
    t.format = color_format;
    t.writeMask = write_mask;
    WGPUFragmentState fs{};
    fs.module = m;
    fs.entryPoint = sv("fs");
    fs.targetCount = color_format != WGPUTextureFormat_Undefined ? 1 : 0;
    fs.targets = &t;
    WGPUDepthStencilState ds{};
    WGPURenderPipelineDescriptor d{};
    d.vertex.module = m;
    d.vertex.entryPoint = sv("vs");
    d.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    d.multisample.count = 1;
    d.multisample.mask = 0xFFFFFFFF;
    d.fragment = &fs;
    if (depth_format != WGPUTextureFormat_Undefined) {
        const bool st = stencil && has_stencil(depth_format);
        ds.format = depth_format;
        ds.depthWriteEnabled = depth ? WGPUOptionalBool_True : WGPUOptionalBool_False;
        ds.depthCompare = WGPUCompareFunction_Always;
        ds.stencilFront = ds.stencilBack = {WGPUCompareFunction_Always, WGPUStencilOperation_Keep,
                                            WGPUStencilOperation_Keep,
                                            st ? WGPUStencilOperation_Replace : WGPUStencilOperation_Keep};
        ds.stencilReadMask = st ? 0xFF : 0;
        ds.stencilWriteMask = st ? stencil_write_mask : 0;
        d.depthStencil = &ds;
    }
    WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(dev.device, &d);
    const std::string error = scope_pop();
    wgpuShaderModuleRelease(m);
    if (!error.empty()) {
        log_limited("partial_clear", 8, "rx: partial clear pipeline not created: %s", error.c_str());
        if (p) wgpuRenderPipelineRelease(p);
        p = nullptr;
    }
    cache[key] = p;
    return p;
}

/// The rectangle a clear covers in a w x h target: the scissor when the clear uses it (suyu's GetScissorState).
/// False when it is empty.
bool clear_rect(const Regs& r, uint32_t w, uint32_t h, uint32_t rect[4]) {
    uint32_t x0 = 0, y0 = 0, x1 = w, y1 = h;
    if (r.clear_control.use_scissor && r.scissor_test[0].enable) {
        const auto& s = r.scissor_test[0];
        x0 = std::min<uint32_t>(s.min_x, w);
        y0 = std::min<uint32_t>(s.min_y, h);
        x1 = std::min<uint32_t>(s.max_x, w);
        y1 = std::min<uint32_t>(s.max_y, h);
    }
    rect[0] = x0;
    rect[1] = y0;
    rect[2] = x1 > x0 ? x1 - x0 : 0;
    rect[3] = y1 > y0 ? y1 - y0 : 0;
    return rect[2] && rect[3];
}

void draw_partial_clear(WGPURenderPipeline p, const uint32_t rect[4], uint32_t stencil_ref) {
    rp_set_pipeline(pass, p);
    rp_set_scissor(pass, rect[0], rect[1], rect[2], rect[3]);
    rp_set_stencil_reference(pass, stencil_ref);
    rp_draw(pass, 3, 1, 0, 0);
}

} // namespace

void clear(Channel& ch, const Regs& r) {
    const auto& cs = r.clear_surface;
    // A target past rt_control.count is not bound, and clearing it does nothing (suyu: FindColorBuffer gives no view
    // for index >= rt_control.count). TOTK clears targets 2 and 3 of its half-resolution lighting pass with only
    // target 0 enabled, right before the full-resolution composite reads them: clearing them took the cave's ambient
    // light off Link's skin (left orange by the subsurface term). RX_CLEAR_ALL_RTS=1: the old behavior.
    static const bool all_rts = env_flag("RX_CLEAR_ALL_RTS");
    const bool any_color = cs.R || cs.G || cs.B || cs.A;
    const bool color = any_color && (all_rts || cs.RT < r.rt_control.count);
    if (any_color && !color) count("clear_unbound_rt");
    const bool depth = cs.Z, stencil = cs.S;
    bool forced = false;
    if (!ch.render_on) {   // RX_COND_FORCE=<guest hex | WxH>,...: clears of those targets run anyway (debugging)
        static std::vector<std::pair<uint64_t, uint64_t>> force;   // (guest, 0) or (0, w << 32 | h)
        static bool parsed;
        if (!parsed) {
            parsed = true;
            if (const char* s = getenv("RX_COND_FORCE"))
                for (const char* p = s; *p;) {
                    char* end;
                    const uint64_t v = strtoull(p, &end, 16);
                    if (end == p) break;
                    if (*end == 'x') {   // a size: decimal
                        char* e2;
                        const uint64_t w = strtoull(p, nullptr, 10), h = strtoull(end + 1, &e2, 10);
                        force.push_back({0, w << 32 | h});
                        end = e2;
                    } else {
                        force.push_back({v, 0});
                    }
                    p = *end == ',' ? end + 1 : end;
                }
        }
        const Image* rt = cs.RT < 8 && (cs.R || cs.G || cs.B || cs.A) ? rt_image(r, cs.RT) : nullptr;
        const Image* z = cs.Z || cs.S ? zeta_image(r) : nullptr;
        for (const auto& [g, wh] : force)
            for (const Image* im : {rt, z})
                if (im && (g ? im->guest == g : ((uint64_t)im->width << 32 | im->height) == wh)) forced = true;
    }
    if (!ch.render_on && !forced) {
        count("clear_conditional_off");
        if (static const uint64_t watch = getenv("RX_WATCH_RT") ? strtoull(getenv("RX_WATCH_RT"), nullptr, 16) : 0;
            watch && cs.RT < 8 && (uint32_t)r.rt[cs.RT].Address() == (uint32_t)watch) {
            static int logged;
            uint64_t v[4]{};
            gpu_read(r.render_enable.Address(), v, sizeof v);
            if (logged++ < 200)
                hos_log("rx: watch %08llx: clear skipped (render enable %u at %llx: %llx / %llx) frame %lld",
                        (unsigned long long)watch, (unsigned)r.render_enable.mode,
                        (unsigned long long)r.render_enable.Address(), (unsigned long long)v[0],
                        (unsigned long long)v[2], (long long)frame_number);
        }
        if (frame_dump_active()) {
            uint64_t v[4]{};
            gpu_read(r.render_enable.Address(), v, sizeof v);
            char b[320];
            const Image* rt = cs.RT < 8 ? rt_image(r, cs.RT) : nullptr;
            const Image* z = zeta_image(r);
            std::snprintf(b, sizeof b, "clear skipped (render enable %u at %llx: %llx %llx / %llx %llx): raw %#x rt %u %08llx "
                          "%ux%u zeta %08llx", (unsigned)r.render_enable.mode, (unsigned long long)r.render_enable.Address(),
                          (unsigned long long)v[0], (unsigned long long)v[1], (unsigned long long)v[2],
                          (unsigned long long)v[3], cs.raw, (unsigned)cs.RT,
                          (unsigned long long)(rt ? rt->guest : 0), rt ? rt->width : 0, rt ? rt->height : 0,
                          (unsigned long long)(z ? z->guest : 0));
            dump_draw(b);
        }
        return;
    }
    st_clears++;
    static const bool whole = env_flag("RX_WHOLE_CLEARS");   // (debugging: the old behavior, every clear whole)
    if (color && cs.RT < 8) {
        if (Image* im = rt_image(r, cs.RT); im && !whole) {
            uint32_t rect[4];
            if (!clear_rect(r, im->width, im->height, rect)) {
                count("clear_empty");
                goto depth_part;
            }
            const bool part = rect[0] || rect[1] || rect[2] < im->width || rect[3] < im->height;
            const bool all = cs.R && cs.G && cs.B && cs.A;
            if ((part || !all) && !VideoCore::Surface::IsPixelFormatInteger(im->pixel_format)) {
                const uint32_t mask = (cs.R ? WGPUColorWriteMask_Red : 0) | (cs.G ? WGPUColorWriteMask_Green : 0) |
                                      (cs.B ? WGPUColorWriteMask_Blue : 0) | (cs.A ? WGPUColorWriteMask_Alpha : 0);
                if (WGPURenderPipeline p = partial_clear_pipeline(im->format, mask, r.clear_color.data(),
                                                                  WGPUTextureFormat_Undefined, false, false, 0, 0)) {
                    Attachments at;
                    at.color[0] = im;
                    at.width = im->width;
                    at.height = im->height;
                    std::array<WGPURenderPassColorAttachment, 8> c{};
                    c[0] = color_attachment(im, WGPULoadOp_Load, {});
                    dump_pass(at, "clear");
                    if (frame_dump_active()) {
                        char b[200];
                        std::snprintf(b, sizeof b, "clear color %g %g %g %g (mask %#x) rect %u,%u %ux%u",
                                      r.clear_color[0], r.clear_color[1], r.clear_color[2], r.clear_color[3], cs.raw,
                                      rect[0], rect[1], rect[2], rect[3]);
                        dump_draw(b);
                    }
                    begin_pass(at, c.data(), nullptr);
                    draw_partial_clear(p, rect, 0);
                    end_render_pass("pe_clear");
                    count("clear_partial");
                    goto depth_part;
                }
            } else if (part || !all) {
                log_limited("clear_int_part", 4, "rx: partial clear of an integer target (%#x): cleared whole",
                            cs.raw);
            }
        }
    }
    if (color && cs.RT < 8) {
        if (Image* im = rt_image(r, cs.RT)) {
            Attachments at;
            at.color[0] = im;
            at.width = im->width;
            at.height = im->height;
            if (clears_deferred()) {
                PendingClear p{};
                p.im = im;
                p.color = true;
                p.value = {r.clear_color[0], r.clear_color[1], r.clear_color[2], r.clear_color[3]};
                add_pending_clear(p);
                goto depth_part;
            }
            std::array<WGPURenderPassColorAttachment, 8> c{};
            c[0] = color_attachment(im, WGPULoadOp_Clear,
                                    {r.clear_color[0], r.clear_color[1], r.clear_color[2], r.clear_color[3]});
            dump_pass(at, "clear");
            if (frame_dump_active()) {
                char b[256];
                uint64_t v[3]{};
                if (r.render_enable.mode != Regs::RenderEnable::Mode::True) gpu_read(r.render_enable.Address(), v, sizeof v);
                std::snprintf(b, sizeof b, "clear color %g %g %g %g (mask %#x) render %u at %llx: %llx / %llx",
                              r.clear_color[0], r.clear_color[1], r.clear_color[2], r.clear_color[3], cs.raw,
                              (unsigned)r.render_enable.mode, (unsigned long long)r.render_enable.Address(),
                              (unsigned long long)v[0], (unsigned long long)v[2]);
                dump_draw(b);
            }
            begin_pass(at, c.data(), nullptr);
            end_render_pass("pe_clear");
        }
    }
depth_part:
    if ((depth || stencil) && !whole) {
        if (Image* im = zeta_image(r)) {
            uint32_t rect[4];
            if (!clear_rect(r, im->width, im->height, rect)) {
                count("clear_empty");
                return;
            }
            const bool part = rect[0] || rect[1] || rect[2] < im->width || rect[3] < im->height;
            // (suyu: a stencil write mask other than 0xFF, or 0, clears only those bits)
            const uint32_t smask = r.stencil_front_mask & 0xFF;
            const bool stencil_part = stencil && has_stencil(im->format) && smask != 0xFF && smask != 0;
            if (part || stencil_part) {
                const bool has_d = im->format != WGPUTextureFormat_Stencil8;
                if (WGPURenderPipeline p = partial_clear_pipeline(WGPUTextureFormat_Undefined, 0, nullptr, im->format,
                                                                  depth && has_d, stencil,
                                                                  depth_clear_value(im, r.clear_depth),
                                                                  stencil_part ? smask : 0xFF)) {
                    Attachments at;
                    at.depth = im;
                    at.width = im->width;
                    at.height = im->height;
                    WGPURenderPassDepthStencilAttachment d =
                        depth_attachment(im, WGPULoadOp_Load, WGPULoadOp_Load, 0.0f, 0);
                    dump_pass(at, "clear");
                    if (frame_dump_active()) {
                        char b[200];
                        std::snprintf(b, sizeof b, "clear depth %g (%d) stencil %u (%d) mask %#x rect %u,%u %ux%u",
                                      r.clear_depth, (int)depth, r.clear_stencil & 0xFF, (int)stencil, smask, rect[0],
                                      rect[1], rect[2], rect[3]);
                        dump_draw(b);
                    }
                    begin_pass(at, nullptr, &d);
                    draw_partial_clear(p, rect, r.clear_stencil & 0xFF);
                    end_render_pass("pe_clear");
                    count("clear_partial");
                    return;
                }
            }
        }
    }
    if (depth || stencil) {
        if (Image* im = zeta_image(r)) {
            Attachments at;
            at.depth = im;
            at.width = im->width;
            at.height = im->height;
            if (clears_deferred()) {
                PendingClear p{};
                p.im = im;
                p.depth = depth && im->format != WGPUTextureFormat_Stencil8;
                p.stencil = stencil && has_stencil(im->format);
                p.depth_value = depth_clear_value(im, r.clear_depth);
                p.stencil_value = r.clear_stencil & 0xFF;
                if (p.depth || p.stencil) add_pending_clear(p);
                return;
            }
            WGPURenderPassDepthStencilAttachment d =
                depth_attachment(im, depth ? WGPULoadOp_Clear : WGPULoadOp_Load,
                                 stencil ? WGPULoadOp_Clear : WGPULoadOp_Load, depth_clear_value(im, r.clear_depth),
                                 r.clear_stencil & 0xFF);
            dump_pass(at, "clear");
            if (frame_dump_active()) {
                char b[256];
                std::snprintf(b, sizeof b, "clear depth %g (%d) stencil %u (%d) raw %#x zeta %llx %ux%u fmt %#x "
                              "control %#x zwrite %u ro %#x scissor %u %u..%u %u..%u render %u/%u at %llx", r.clear_depth, (int)depth,
                              r.clear_stencil & 0xFF, (int)stencil, cs.raw, (unsigned long long)r.zeta.Address(),
                              r.zeta_size.width, r.zeta_size.height, (unsigned)r.zeta.format,
                              *(const uint32_t*)&r.clear_control, (unsigned)r.depth_write_enabled,
                              *(const uint32_t*)&r.zeta_read_only, (unsigned)r.scissor_test[0].enable,
                              (unsigned)r.scissor_test[0].min_x, (unsigned)r.scissor_test[0].max_x,
                              (unsigned)r.scissor_test[0].min_y, (unsigned)r.scissor_test[0].max_y,
                              (unsigned)r.render_enable.mode, (unsigned)r.render_enable_override,
                              (unsigned long long)r.render_enable.Address());
                dump_draw(b);
            }
            begin_pass(at, nullptr, &d);
            end_render_pass("pe_clear");
        }
    }
}

// ---- the native NVN layer: render targets and clears ----
Image* nvn_target(const nvn_tex* t, const nvn_view* v) {
    if (!t || !t->gpu) return nullptr;
    const nvn_texdesc& d = t->d;
    const unsigned level = v && v->set_levels ? v->base_level : 0;
    const unsigned layer = v && v->set_layers ? v->min_layer : 0;
    const int format = v && v->set_format ? v->format : d.format;
    const PixelFormat pf = nvn_pixel_format(format);
    if (pf == PixelFormat::Invalid) {
        log_limited("nvn_rt_format", 16, "rx: NVN render target format %#x has no pixel format", format);
        return nullptr;
    }
    const bool one_row = d.target == NVN_TEXTURE_TARGET_1D || d.target == NVN_TEXTURE_TARGET_1D_ARRAY;
    uint32_t w = std::max(1u, d.width >> level), h = one_row ? 1u : std::max(1u, d.height >> level);
    // a compressed texture drawn through an uncompressed view of its block's size (compression on the GPU: TOTK
    // renders its gloom noise volumes, 64x64x64 BC4, as RGBA16UI, a texel a 4x4 block): the target is the texture's
    // size in blocks, as the GPU draws it, and is written back where the texture is decoded from (writeback_targets)
    const PixelFormat tex_pf = format != d.format ? nvn_pixel_format(d.format) : pf;
    const uint32_t bw = VideoCore::Surface::DefaultBlockWidth(tex_pf), bh = VideoCore::Surface::DefaultBlockHeight(tex_pf);
    const bool block_view = tex_pf != PixelFormat::Invalid && (bw > 1 || bh > 1) &&
                            VideoCore::Surface::DefaultBlockWidth(pf) == 1 &&
                            VideoCore::Surface::BytesPerBlock(tex_pf) == VideoCore::Surface::BytesPerBlock(pf);
    if (block_view) w = (w + bw - 1) / bw, h = one_row ? 1u : (h + bh - 1) / bh;
    // (each level and layer is an image at its own address, as the targets of a layered or mipmapped texture are
    // drawn one by one: the texture sampled whole is assembled from them, rx_texture.cpp)
    uint64_t gpu = t->gpu + nvn_tex_level_offset(&d, level);
    // a 3D texture's slices interleave in its memory (block linear): each slice drawn is an image of its own, filed
    // under nvn_slice_key, and the 3D texture is assembled from them where it is sampled (assemble_3d_slices). They
    // were all drawn as slice 0: TOTK renders its colour-grading LUT slice by slice (the loading map lost its blues).
    uint64_t key = 0;
    if (d.target == NVN_TEXTURE_TARGET_3D) {
        const uint64_t guest = nvn_gpu_guest(gpu);
        if (!guest) return nullptr;
        key = nvn_slice_key(guest, layer);
    } else if (layer) {
        gpu += (uint64_t)layer * t->L.layer_size;
    }
    Image* im = find_render_target(gpu, w, h, pf, key);
    if (im) {
        im->block_view = block_view;
        if (key) im->slice = layer, im->volume_depth = std::max(1u, d.depth >> level);
        im->pitch_linear = nvn_tex_is_linear(&d);
        unsigned th = 0, td = 0;
        nvn_tex_level_tiles(&d, level, &th, &td);
        im->bl_height = th;
        im->bl_depth = td;
        im->reg_width = im->pitch_linear ? nvn_tex_linear_stride(&d) : w;
        // (what a write-back may cover: this level of this layer)
        const uint64_t lo = nvn_tex_level_offset(&d, level);
        const uint64_t hi = level + 1 < std::max(1u, d.levels) ? nvn_tex_level_offset(&d, level + 1)
                                                               : (t->L.layer_size ? t->L.layer_size : t->L.size);
        im->storage_limit = hi > lo ? hi - lo : 0;
    }
    return im;
}

Attachments nvn_attachments(NvnState& s) {
    // (images can be replaced or made by any lookup: the answer holds while none was, in this frame)
    if (!s.at_dirty && s.at_images == images_changed && s.at_frame == frame_number) return s.at;
    Attachments at;
    for (int i = 0; i < std::min(s.rt.n, 8); i++)
        at.color[i] = nvn_target(s.rt.color[i], s.rt.has_cview[i] ? &s.rt.cview[i] : nullptr);
    at.depth = nvn_target(s.rt.depth, s.rt.has_dview ? &s.rt.dview : nullptr);
    for (int i = 0; i < 8; i++)   // (one image attached once: two targets can name one storage)
        for (int j = 0; j < i && at.color[i]; j++)
            if (at.color[j] == at.color[i]) at.color[i] = nullptr;
    if (at.depth)
        for (Image*& c : at.color)
            if (c == at.depth) c = nullptr;
    for (Image* im : at.color)
        if (im && !at.width) at.width = im->width, at.height = im->height;
    if (!at.width && at.depth) at.width = at.depth->width, at.height = at.depth->height;
    for (auto& im : at.color)
        if (im && (im->width != at.width || im->height != at.height)) {
            log_limited("nvn_rt_size", 8, "rx: NVN color target %ux%u dropped from a %ux%u pass", im->width, im->height,
                        at.width, at.height);
            im = nullptr;
        }
    if (at.depth && (at.depth->width != at.width || at.depth->height != at.height)) {
        log_limited("nvn_zt_size", 8, "rx: NVN depth target %ux%u dropped from a %ux%u pass", at.depth->width,
                    at.depth->height, at.width, at.height);
        at.depth = nullptr;
    }
    s.at = at;
    s.at_dirty = false;
    s.at_images = images_changed;   // (after the lookups: they can make images)
    s.at_frame = frame_number;
    return at;
}

namespace {
/// The rectangle of a w x h target an NVN clear covers: the scissor, when one was set (upper-left origin)
bool nvn_clear_rect(uint32_t w, uint32_t h, const int32_t* sc, uint32_t rect[4]) {
    int64_t x0 = 0, y0 = 0, x1 = w, y1 = h;
    if (sc && sc[2] > 0 && sc[3] > 0) {
        x0 = std::max<int64_t>(x0, sc[0]);
        y0 = std::max<int64_t>(y0, sc[1]);
        x1 = std::min<int64_t>(x1, (int64_t)sc[0] + sc[2]);
        y1 = std::min<int64_t>(y1, (int64_t)sc[1] + sc[3]);
    }
    if (x1 <= x0 || y1 <= y0) return false;
    rect[0] = (uint32_t)x0, rect[1] = (uint32_t)y0, rect[2] = (uint32_t)(x1 - x0), rect[3] = (uint32_t)(y1 - y0);
    return true;
}
}  // namespace

void nvn_clear_color(Image* im, const float rgba[4], uint32_t mask, const int32_t* scissor) {
    if (!im || !(mask & 15)) return;
    st_clears++;
    uint32_t rect[4];
    if (!nvn_clear_rect(im->width, im->height, scissor, rect)) {
        count("clear_empty");
        return;
    }
    const bool part = rect[0] || rect[1] || rect[2] < im->width || rect[3] < im->height;
    const bool all = (mask & 15) == 15;
    const bool integer = VideoCore::Surface::IsPixelFormatInteger(im->pixel_format);
    if ((part || !all) && !integer) {
        const uint32_t wmask = (mask & 1 ? WGPUColorWriteMask_Red : 0) | (mask & 2 ? WGPUColorWriteMask_Green : 0) |
                               (mask & 4 ? WGPUColorWriteMask_Blue : 0) | (mask & 8 ? WGPUColorWriteMask_Alpha : 0);
        if (WGPURenderPipeline p = partial_clear_pipeline(im->format, wmask, rgba, WGPUTextureFormat_Undefined, false,
                                                          false, 0, 0)) {
            Attachments at;
            at.color[0] = im;
            at.width = im->width;
            at.height = im->height;
            std::array<WGPURenderPassColorAttachment, 8> c{};
            c[0] = color_attachment(im, WGPULoadOp_Load, {});
            begin_pass(at, c.data(), nullptr);
            draw_partial_clear(p, rect, 0);
            end_render_pass("pe_clear");
            count("clear_partial");
            return;
        }
    } else if (part || !all) {
        log_limited("nvn_clear_int_part", 4, "rx: partial clear of an NVN integer target: cleared whole");
    }
    // (integer targets: the values are the integers' bits)
    WGPUColor value{rgba[0], rgba[1], rgba[2], rgba[3]};
    if (integer) {
        int32_t v[4];
        std::memcpy(v, rgba, 16);
        const bool sign = VideoCore::Surface::IsPixelFormatSignedInteger(im->pixel_format);
        value = sign ? WGPUColor{(double)v[0], (double)v[1], (double)v[2], (double)v[3]}
                     : WGPUColor{(double)(uint32_t)v[0], (double)(uint32_t)v[1], (double)(uint32_t)v[2],
                                 (double)(uint32_t)v[3]};
    }
    if (clears_deferred()) {
        PendingClear p{};
        p.im = im;
        p.color = true;
        p.value = value;
        add_pending_clear(p);
        return;
    }
    Attachments at;
    at.color[0] = im;
    at.width = im->width;
    at.height = im->height;
    std::array<WGPURenderPassColorAttachment, 8> c{};
    c[0] = color_attachment(im, WGPULoadOp_Clear, value);
    begin_pass(at, c.data(), nullptr);
    end_render_pass("pe_clear");
}

void nvn_clear_depth_stencil(Image* im, bool depth, float dv, bool stencil, uint32_t sv, uint32_t smask,
                             const int32_t* scissor) {
    if (!im) return;
    depth = depth && im->format != WGPUTextureFormat_Stencil8;
    stencil = stencil && (smask & 0xFF) && has_stencil(im->format);
    if (!depth && !stencil) return;
    st_clears++;
    uint32_t rect[4];
    if (!nvn_clear_rect(im->width, im->height, scissor, rect)) {
        count("clear_empty");
        return;
    }
    const bool part = rect[0] || rect[1] || rect[2] < im->width || rect[3] < im->height;
    const bool stencil_part = stencil && (smask & 0xFF) != 0xFF;
    if (part || stencil_part) {
        if (WGPURenderPipeline p = partial_clear_pipeline(WGPUTextureFormat_Undefined, 0, nullptr, im->format, depth,
                                                          stencil, depth_clear_value(im, dv),
                                                          stencil_part ? smask & 0xFF : 0xFF)) {
            Attachments at;
            at.depth = im;
            at.width = im->width;
            at.height = im->height;
            WGPURenderPassDepthStencilAttachment d = depth_attachment(im, WGPULoadOp_Load, WGPULoadOp_Load, 0.0f, 0);
            begin_pass(at, nullptr, &d);
            draw_partial_clear(p, rect, sv & 0xFF);
            end_render_pass("pe_clear");
            count("clear_partial");
            return;
        }
    }
    if (clears_deferred()) {
        PendingClear p{};
        p.im = im;
        p.depth = depth;
        p.stencil = stencil;
        p.depth_value = depth_clear_value(im, dv);
        p.stencil_value = sv & 0xFF;
        add_pending_clear(p);
        return;
    }
    Attachments at;
    at.depth = im;
    at.width = im->width;
    at.height = im->height;
    WGPURenderPassDepthStencilAttachment d = depth_attachment(im, depth ? WGPULoadOp_Clear : WGPULoadOp_Load,
                                                              stencil ? WGPULoadOp_Clear : WGPULoadOp_Load,
                                                              depth_clear_value(im, dv), sv & 0xFF);
    begin_pass(at, nullptr, &d);
    end_render_pass("pe_clear");
}

} // namespace rx

using namespace rx;

extern "C" {

int rx_start(void* hwnd) {
    stats_on = getenv("RX_STATS") && std::atoi(getenv("RX_STATS")) != 0;
    frame_dump_enabled = getenv("RX_DUMP_FRAME") || getenv("RX_DUMP_AT");
#ifdef __EMSCRIPTEN__
    on_render_thread = true;   // (the web: rx_start runs on the render thread)
#endif
    std::lock_guard lk(mu);
    if (started) return 1;
    if (getenv("RX_OFF")) return 0;
#if RX_WEB_MT
    bool made = false;   // (the device belongs to the WebGPU thread: made there)
    on_gpu("init_device", [&] { made = init_device(static_cast<HWND>(hwnd)); });
    if (!made) {
#else
    if (!init_device(static_cast<HWND>(hwnd))) {
#endif
        hos_log("rx: no WebGPU device: the window shows the window buffers' memory");
        return 0;
    }
    started = true;
    return 1;
}

int rx_active(void) {
    return started;
}

void rx_resize(int width, int height) {
    InterlockedExchange(&want_w, width);
    InterlockedExchange(&want_h, height);
    present_window_size(width, height);
}

uint64_t rx_channel_zpass(void* channel) {
    return channel ? static_cast<Channel*>(channel)->zpass : 0;
}

} // extern "C"
namespace rx {
namespace {
struct PendingWriteback { uint64_t guest, id; };
std::vector<PendingWriteback> pending_writebacks;
} // namespace

void queue_target_writeback(Image* im) {
    static const bool off = env_flag("RX_NO_TARGET_WRITEBACK");
    // (a 3D slice, filed under nvn_slice_key: only a block view's, written into its volume)
    if (off || !im || im->writeback_queued || ((im->guest >> 32) && !im->block_view)) return;
    im->writeback_queued = true;
    pending_writebacks.push_back({im->guest, im->id});
}

/// What a render target writeback needs once its pixels are read back (a copy: the image may be gone by then)
struct TargetWriteback {
    uint64_t guest = 0, gpu_va = 0, storage_limit = 0;
    uint32_t w = 0, h = 0, bpp = 0, row = 0, reg_width = 0, bl_height = 0, bl_depth = 0;
    uint32_t slice = 0, depth = 1;   // a 3D texture's slice: which, of how many
    bool pitch_linear = false, block_view = false;
    int pf = 0;
    uint8_t norm16 = 0;   // the texture is the float16 stand-in of a 16-bit unorm (1) / snorm (2) format
};

static float half_value(uint16_t h) {
    const uint32_t e = h >> 10 & 31, m = h & 1023;
    const float f = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1.0f + m / 1024.0f, (int)e - 15);
    return h >> 15 ? -f : f;
}

/// 1 / 2 when <texture> is the float16 stand-in rx draws a 16-bit unorm / snorm target into (no 16-bit norm
/// render targets without RX_NORM16_RT): same size per texel, values to convert. 0 otherwise.
static uint8_t norm16_stand_in(PixelFormat pf, WGPUTextureFormat texture) {
    using P = PixelFormat;
    switch (pf) {
    case P::R16_UNORM: return texture == WGPUTextureFormat_R16Float ? 1 : 0;
    case P::R16G16_UNORM: return texture == WGPUTextureFormat_RG16Float ? 1 : 0;
    case P::R16G16B16A16_UNORM: return texture == WGPUTextureFormat_RGBA16Float ? 1 : 0;
    case P::R16_SNORM: return texture == WGPUTextureFormat_R16Float ? 2 : 0;
    case P::R16G16_SNORM: return texture == WGPUTextureFormat_RG16Float ? 2 : 0;
    case P::R16G16B16A16_SNORM: return texture == WGPUTextureFormat_RGBA16Float ? 2 : 0;
    default: return 0;
    }
}

/// The read-back rows q (row bytes apart) into guest memory, in the target's layout (pitch or block linear).
/// counted: false off the renderer's thread (count() is that thread's)
void apply_target_writeback(const TargetWriteback& t, const uint8_t* q, bool counted) {
    const uint32_t w = t.w, h = t.h, bpp = t.bpp, row = t.row;
    std::vector<uint8_t> conv;
    if (t.norm16) {   // (float16 back to the guest's 16-bit norm values)
        conv.assign(q, q + (size_t)row * h);
        for (uint32_t y = 0; y < h; y++) {
            uint8_t* p = conv.data() + (size_t)y * row;
            for (uint32_t i = 0; i < w * bpp / 2; i++) {
                uint16_t v;
                std::memcpy(&v, p + 2 * i, 2);
                float f = half_value(v);
                if (f != f) f = 0;
                v = t.norm16 == 1 ? (uint16_t)std::lround(std::clamp(f, 0.0f, 1.0f) * 65535.0f)
                                  : (uint16_t)(int16_t)std::lround(std::clamp(f, -1.0f, 1.0f) * 32767.0f);
                std::memcpy(p + 2 * i, &v, 2);
            }
        }
        q = conv.data();
        if (counted) count("rt_writeback_norm16");
    }
    uint64_t avail = 0;
    uint8_t* out = const_cast<uint8_t*>(hos_gpu_ptr(t.gpu_va, &avail));
    if (t.pitch_linear) {
        const uint32_t pitch = std::max(t.reg_width, w * bpp);
        const uint64_t need = (uint64_t)pitch * (h - 1) + (uint64_t)w * bpp;
        if (out && avail >= need && (!t.storage_limit || need <= t.storage_limit))
            for (uint32_t y = 0; y < h; y++) std::memcpy(out + (uint64_t)y * pitch, q + (uint64_t)y * row, (size_t)w * bpp);
    } else {
        const uint32_t sw = std::max(t.reg_width, w);   // (the surface's width: its GOB row stride)
        const uint32_t depth = std::max(1u, t.depth);
        const size_t size = Tegra::Texture::CalculateSize(true, bpp, sw, h, depth, t.bl_height, t.bl_depth);
        if (t.storage_limit && size > t.storage_limit) {   // (a native NVN target: never past its storage)
            log_limited("wb_limit", 16, "rx: writeback of %ux%u (%zu bytes, block height %u) is larger than its "
                        "storage (%llu bytes): skipped", w, h, size, t.bl_height, (unsigned long long)t.storage_limit);
            if (counted) count("rt_writeback_too_big");
        } else if (out && avail >= size && t.slice < depth) {
            // (a 3D texture's slice: its volume as memory holds it, the slice replaced; slices interleave in memory)
            std::vector<uint8_t> lin((size_t)sw * h * depth * bpp);
            if (depth > 1)
                Tegra::Texture::UnswizzleTexture(lin, std::span<const uint8_t>(out, size), bpp, sw, h, depth,
                                                 t.bl_height, t.bl_depth);
            uint8_t* at = lin.data() + (size_t)t.slice * sw * h * bpp;
            for (uint32_t y = 0; y < h; y++) std::memcpy(at + (size_t)y * sw * bpp, q + (uint64_t)y * row, (size_t)w * bpp);
            // (the surface's own bytes first: the GOBs' unused corners keep what memory held)
            std::vector<uint8_t> tiled(out, out + size);
            Tegra::Texture::SwizzleTexture(tiled, lin, bpp, sw, h, depth, t.bl_height, t.bl_depth);
            std::memcpy(out, tiled.data(), size);
            if (counted) count("rt_writeback_bytes", (int64_t)size);
            // (a block view: the compressed texture is decoded again from what it drew)
            if (t.block_view && counted) textures_written(t.guest & ((1ull << 40) - 1), size);
        }
    }
    if (counted) count("rt_writebacks");
    // (RX_LOG_WRITEBACKS=<first frame>)
    static const int64_t log_wb = getenv("RX_LOG_WRITEBACKS") ? atoll(getenv("RX_LOG_WRITEBACKS")) : -1;
    static int logged;
    if (log_wb >= 0 && frame_number >= log_wb && logged++ < 2000)
        hos_log("rx: writeback %08llx %ux%u format %d frame %lld", (unsigned long long)t.guest, w, h, t.pf,
                (long long)frame_number);
}

/// (the threaded web build: writebacks recorded into a batch and not in guest memory yet; the native NVN layer's
/// drains wait for them, rx_target_writebacks_pending, and the game's memcpy for the ones it reads,
/// rx_wait_writebacks_reading)
std::atomic<int> target_writebacks_pending{0};
std::mutex wb_spans_mu;
std::vector<std::pair<uintptr_t, uintptr_t>> wb_spans;   // (the host memory each pending one writes)
void drop_wb_span(std::pair<uintptr_t, uintptr_t> s) {
    std::lock_guard lk(wb_spans_mu);
    if (auto it = std::find(wb_spans.begin(), wb_spans.end(), s); it != wb_spans.end()) wb_spans.erase(it);
}

/// The queued render targets' contents to guest memory, laid out as the target registers said (block linear or
/// pitch), when the texture holds the guest format's bytes as they are (no 16-bit norm substitute, color only).
/// The threaded web build: the copy is recorded into the batch and read back after its submit, on the WebGPU thread,
/// instead of waiting here for every batch before it and for the GPU (~35 a second at the castle, each one stopping
/// the replay); the native NVN layer's fences and drains wait for it instead.
void writeback_targets() {
    if (pending_writebacks.empty()) return;
    std::vector<PendingWriteback> list;
    list.swap(pending_writebacks);
    if (!RX_WEB_MT) wait_submitted();
    for (const PendingWriteback& p : list) {
        Image* im = image_at(p.guest);
        if (!im || im->id != p.id) continue;
        im->writeback_queued = false;
        const PixelFormat pf = im->pixel_format;
        // (an sRGB target's texture may be its linear twin, viewed as sRGB: the same bytes)
        const bool same_bytes = wgpu_format(pf) == im->format || srgb_pair(wgpu_format(pf)) == im->format;
        const uint8_t norm16 = same_bytes ? 0 : norm16_stand_in(pf, im->format);
        if (im->layers != 1 || im->dimension != WGPUTextureDimension_2D || (!same_bytes && !norm16) ||
            VideoCore::Surface::GetFormatType(pf) != VideoCore::Surface::SurfaceType::ColorTexture ||
            VideoCore::Surface::IsPixelFormatASTC(pf) || VideoCore::Surface::DefaultBlockWidth(pf) != 1) {
            count("rt_writeback_skipped");
            log_limited("rt_writeback_skipped", 16, "rx: writeback of %08llx %ux%u skipped: layers %u dim %d pixel format "
                        "%d texture format %#x (guest's %#x)", (unsigned long long)im->guest, im->width, im->height,
                        im->layers, (int)im->dimension, (int)pf, (unsigned)im->format, (unsigned)wgpu_format(pf));
            continue;
        }
        TargetWriteback t;
        t.guest = im->guest, t.gpu_va = im->gpu_va, t.storage_limit = im->storage_limit;
        t.bpp = VideoCore::Surface::BytesPerBlock(pf);
        t.w = im->width, t.h = im->height;
        t.row = (t.w * t.bpp + 255) & ~255u;
        t.reg_width = im->reg_width, t.bl_height = im->bl_height, t.bl_depth = im->bl_depth;
        t.pitch_linear = im->pitch_linear;
        t.block_view = im->block_view;
        if (im->guest >> 32) t.slice = im->slice, t.depth = std::max(1u, im->volume_depth);
        t.pf = (int)pf;
        t.norm16 = norm16;
        WGPUBufferDescriptor bd{};
        bd.label = sv("rx target writeback");
        bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        bd.size = (uint64_t)t.row * t.h;
        WGPUBuffer staging = wgpuDeviceCreateBuffer(dev.device, &bd);
        WGPUTexelCopyTextureInfo src{};
        src.texture = im->tex;
        src.aspect = WGPUTextureAspect_All;
        WGPUTexelCopyBufferInfo dst{};
        dst.buffer = staging;
        dst.layout.bytesPerRow = t.row;
        dst.layout.rowsPerImage = t.h;
        const WGPUExtent3D extent{t.w, t.h, 1};
        if (RX_WEB_MT) {
            end_render_pass("pe_writeback");
            ce_copy_texture_to_buffer(encoder(), &src, &dst, &extent);
            // (the memory it will write: as apply_target_writeback lays it out, at most)
            uint64_t avail = 0;
            const uintptr_t at = (uintptr_t)hos_gpu_ptr(t.gpu_va, &avail);
            const uint64_t span = t.pitch_linear ? (uint64_t)std::max(t.reg_width, t.w * t.bpp) * t.h
                                                 : Tegra::Texture::CalculateSize(true, t.bpp, std::max(t.reg_width, t.w), t.h,
                                                                                 1, t.bl_height, t.bl_depth);
            const std::pair<uintptr_t, uintptr_t> s{at, at + span};
            {
                std::lock_guard lk(wb_spans_mu);
                wb_spans.push_back(s);
            }
            target_writebacks_pending++;
            struct Pending {
                TargetWriteback t;
                WGPUBuffer staging;
                uint64_t size;
                std::pair<uintptr_t, uintptr_t> span;
            };
            auto* pw = new Pending{t, staging, bd.size, s};
            open_afters.push_back([pw](bool submitted) {   // (the WebGPU thread, after the batch's submit)
                if (!submitted) {
                    wgpuBufferRelease(pw->staging);
                    drop_wb_span(pw->span);
                    delete pw;
                    target_writebacks_pending--;
                    return;
                }
                WGPUBufferMapCallbackInfo mc{};
                mc.mode = WGPUCallbackMode_AllowSpontaneous;
                mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
                    auto* pw = static_cast<Pending*>(u1);
                    if (status == WGPUMapAsyncStatus_Success) {
                        apply_target_writeback(
                            pw->t, static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(pw->staging, 0, pw->size)),
                            false);
                        wgpuBufferUnmap(pw->staging);
                    }
                    wgpuBufferRelease(pw->staging);
                    drop_wb_span(pw->span);
                    delete pw;
                    target_writebacks_pending--;
                };
                mc.userdata1 = pw;
                wgpuBufferMapAsync(pw->staging, WGPUMapMode_Read, 0, pw->size, mc);
            });
            continue;
        }
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
        wgpuCommandEncoderCopyTextureToBuffer(e, &src, &dst, &extent);
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
        wgpuQueueSubmit(dev.queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(e);
        struct Done { bool complete = false; WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error; } done;
        WGPUBufferMapCallbackInfo mc{};
        mc.mode = WGPUCallbackMode_AllowSpontaneous;
        mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
            auto* dn = static_cast<Done*>(u1);
            dn->status = status;
            dn->complete = true;
        };
        mc.userdata1 = &done;
        wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, bd.size, mc);
        while (!done.complete) wgpuDevicePoll(dev.device, true, nullptr);
        if (done.status == WGPUMapAsyncStatus_Success) {
            apply_target_writeback(t, static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(staging, 0, bd.size)), true);
            wgpuBufferUnmap(staging);
        }
        wgpuBufferRelease(staging);
    }
    if (RX_WEB_MT) flush_batch();   // (the copies go now)
}
int rx_target_writebacks_pending() { return target_writebacks_pending.load(); }

extern "C" void rx_wait_writebacks_reading(const void* p, uint64_t n) {
    if (target_writebacks_pending.load(std::memory_order_relaxed) <= 0) return;
    const uintptr_t a = (uintptr_t)p, b = a + n;
    const int64_t t0 = hos_now_ns();
    bool waited = false;
    for (;;) {
        bool hit = false;
        {
            std::lock_guard lk(wb_spans_mu);
            for (const auto& [lo, hi] : wb_spans) hit |= a < hi && lo < b;
        }
        if (!hit) break;
        waited = true;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
#if RX_WEB_MT
    if (waited) mt_note_wait("(memcpy: writebacks)", hos_now_ns() - t0);
#endif
}
uint64_t last_batch_serial() {
    return batch_serial.load();
}

bool batch_done(uint64_t serial) {
    if (!serial) return true;
    if (batch_serial_run.load() < serial) return false;
    wgpuDevicePoll(dev.device, false, nullptr);
    if (gpu_done_serial.load() < serial) return false;
    std::lock_guard lk(inflight_mu);
    return readback_serials.empty() || *readback_serials.begin() > serial;
}

void wait_batch(uint64_t serial) {
    if (!started || !serial) return;
    {
        std::unique_lock lk(submit_mu);
        submitted_cv.wait(lk, [&] { return batch_serial_run.load() >= serial; });
    }
#if RX_WEB_MT
    while (!batch_done(serial)) rx_web_yield();   // (the WebGPU thread's event loop runs the done callbacks)
#elif RX_WEB
    while (!batch_done(serial)) wgpuInstanceProcessEvents(dev.instance);
#else
    uint64_t index = 0;
    {
        std::lock_guard lk(index_mu);
        // (this batch's last submission, or a later batch's: submissions finish in order)
        for (uint64_t s = serial; s < serial + 1024 && s <= batch_serial_run.load(); s++)
            if (serial_index[s % 1024].serial == s && serial_index[s % 1024].index) {
                index = serial_index[s % 1024].index;
                break;
            }
    }
    const int64_t t0 = hos_now_ns();
    if (index) {
        const WGPUSubmissionIndex i = index;
        wgpuDevicePoll(dev.device, true, &i);
    }
    while (!batch_done(serial)) wgpuDevicePoll(dev.device, true, nullptr);
    count("t_fence_wait", hos_now_ns() - t0);
#endif
}
} // namespace rx
extern "C" {

void* rx_channel_new(void) {
    return new Channel();
}

int64_t rx_stat_draws(void) {   // (hos_prof.c's HOS_AB: is the game in the world yet)
    return st_draws;
}

const uint8_t* rx_3d_methods(void) {
    static const std::array<uint8_t, 0x1000> table = method_table();
    return table.data();
}

void rx_3d_method(void* channel, const uint32_t* words, uint32_t method, uint32_t arg) {
    std::lock_guard lk(mu);
    Channel& ch = *static_cast<Channel*>(channel);
    const Regs& r = *reinterpret_cast<const Regs*>(words);
    const uint32_t bg = RX_REG(bind_groups);
    if (method >= bg && method < bg + 5 * 8 && (method - bg) % 8 == 4) {
        const uint32_t stage = (method - bg) / 8;
        const auto& b = r.bind_groups[stage];
        ConstBuffer& cb = ch.cbufs[stage][b.shader_slot % 18];
        cb.enabled = b.valid != 0;
        cb.address = r.const_buffer.Address();
        cb.size = r.const_buffer.size;
        static int binding_trace_count;
        static const bool trace_bindings = env_flag("RX_TRACE_BINDINGS");
        if (trace_bindings && binding_trace_count++ < 128)
            hos_log("rx: cbuf bind stage %u slot %u valid %u address %#llx size %#x before device %u",
                    stage, (uint32_t)b.shader_slot, cb.enabled, (unsigned long long)cb.address, cb.size, !started);
        return;
    }
    if (!started) return;
    if (method == 0xFFFF) {   // a macro finished: its instanced draws run
        draw_deferred(ch, r);
        return;
    }
    switch (method) {
    case RX_REG(invalidate_texture_data_cache):
    case RX_REG(tiled_cache_barrier):
    case RX_REG(invalidate_texture_cache_no_wfi):
    case RX_REG(invalidate_texture_data_cache_lines):
        texture_barrier(method, arg);
        return;
    case RX_REG(clear_report_value):
        if (arg == (uint32_t)Regs::ClearReport::ZPassPixelCount) ch.zpass = 0;
        return;
    case RX_REG(render_enable.mode):
    case RX_REG(render_enable_override):
        ch.render_on = render_enabled(r);
        if (frame_dump_active()) {
            uint64_t v[4]{};
            if (r.render_enable.mode >= Regs::RenderEnable::Mode::Conditional) gpu_read(r.render_enable.Address(), v, sizeof v);
            char b[200];
            std::snprintf(b, sizeof b, "render enable %u override %u at %llx (%llx %llx / %llx %llx): %s",
                          (unsigned)r.render_enable.mode, (unsigned)r.render_enable_override,
                          (unsigned long long)r.render_enable.Address(), (unsigned long long)v[0],
                          (unsigned long long)v[1], (unsigned long long)v[2], (unsigned long long)v[3],
                          ch.render_on ? "on" : "off");
            dump_draw(b);
        }
        return;
    case RX_REG(clear_surface):
        if (rt_history_on()) rt_history_note(r, ch.render_on);
        clear(ch, r);
        return;
    case RX_REG(draw.begin):
        if (rt_history_on()) rt_history_note(r, ch.render_on);
        if (static const uint64_t on_rt = getenv("RX_DUMP_ON_RT") ? strtoull(getenv("RX_DUMP_ON_RT"), nullptr, 16) : 0;
            on_rt)
            for (uint32_t i = 0; i < 8; i++)
                if (r.rt[i].format != Tegra::RenderTargetFormat::NONE &&
                    (r.rt[i].Address() == on_rt ||   // (or a size: RX_DUMP_ON_RT=<width>0000<height> in hex, e.g. 120000104)
                     (on_rt < 0x100000000ull && on_rt >> 16 == r.rt[i].width && (on_rt & 0xFFFF) == r.rt[i].height)))
                    dump_start_now();
        draw_begin(ch, r);
        return;
    case RX_REG(draw.end):
        draw_end(ch, r, 1, false);
        return;
    case RX_REG(index_buffer.count):
        ch.draw_indexed = true;
        return;
    case RX_REG(index_buffer32_subsequent):
    case RX_REG(index_buffer16_subsequent):
    case RX_REG(index_buffer8_subsequent):
        ch.instance_count++;
        [[fallthrough]];
    case RX_REG(index_buffer32_first):
    case RX_REG(index_buffer16_first):
    case RX_REG(index_buffer8_first):
        draw_index_small(ch, r, arg);
        return;
    case RX_REG(draw_inline_index):
        push_inline_index(ch, arg);
        return;
    case RX_REG(inline_index_2x16.even):
        push_inline_index(ch, arg & 0xFFFF);
        push_inline_index(ch, arg >> 16);
        return;
    case RX_REG(inline_index_4x8.index0):
        for (int i = 0; i < 4; i++) push_inline_index(ch, (arg >> (8 * i)) & 0xFF);
        return;
    case RX_REG(vertex_array_instance_first):
        draw_array_instanced(ch, r, r.vertex_array_instance_first, false);
        return;
    case RX_REG(vertex_array_instance_subsequent):
        draw_array_instanced(ch, r, r.vertex_array_instance_subsequent, true);
        return;
    case RX_REG(draw_texture.src_y0):
        log_limited("draw_texture", 4, "rx: draw_texture not drawn yet");
        return;
    default:
        break;
    }
}

void rx_flush(void) {
    if (!started) return;
    std::lock_guard lk(mu);
    flush_batch();
    writeback_targets();
}

/// Every batch so far submitted and run, and the readbacks after them done (the native NVN layer's fences).
void rx_wait_idle(void) {
    if (!started) return;
    {
        std::lock_guard lk(mu);
        flush_batch();
    }
    wait_submitted();
    for (;;) {
        {
            std::lock_guard lk(inflight_mu);
            if (inflight.empty()) break;
        }
        if (readbacks_pending <= 0) break;
        wgpuDevicePoll(dev.device, true, nullptr);
    }
}

void rx_begin_submission(void) {
    if (!started) return;
    std::lock_guard lk(mu);
    flush_batch();   // (none should be open)
    buffer_cache_scan();
    count("submissions");
}

int rx_gpu_writes_pending(uint64_t gpu_va, uint64_t size) {
    if (!started) return 0;
    auto overlaps = [&](const Writeback& w) { return w.gpu_va < gpu_va + size && gpu_va < w.gpu_va + w.size; };
    {
        std::lock_guard lk(mu);
        for (const Writeback& w : open_writebacks)
            if (overlaps(w)) return 1;
        if (gpu_written(gpu_va, size)) return 1;   // a dispatch wrote it in the buffer cache
    }
    std::lock_guard lk(inflight_mu);
    for (const Writeback& w : inflight)
        if (overlaps(w)) return 1;
    return 0;
}

#ifdef __EMSCRIPTEN__
// The web: WebGPU objects belong to the render thread, so a sync asked for by another thread (the parsing thread,
// before the CPU reads what a shader wrote) is done there, between its records (rx_web_idle), and waited for.
namespace {
struct SyncRequest {
    uint64_t gpu_va, size;
    bool done = false;
};
std::mutex sync_mu;
std::condition_variable sync_cv;
std::deque<SyncRequest*> sync_requests;
}   // namespace
#endif

#ifdef __EMSCRIPTEN__
std::atomic<volatile uint32_t*> idle_word{nullptr};   // (rx_web_idle's wake word: sync requests bump it)
#endif

void rx_sync_gpu_writes(uint64_t gpu_va, uint64_t size) {
    if (!started) return;
#ifdef __EMSCRIPTEN__
    if (!on_render_thread) {
        SyncRequest r{gpu_va, size};
        std::unique_lock lk(sync_mu);
        sync_requests.push_back(&r);
        if (volatile uint32_t* w = idle_word.load()) {   // (the render thread may be waiting for work)
            __atomic_add_fetch(w, 1, __ATOMIC_SEQ_CST);
            emscripten_futex_wake((void*)w, 1);
        }
        sync_cv.wait(lk, [&] { return r.done; });
        return;
    }
#endif
    const int64_t t0 = hos_now_ns();
    {
        std::lock_guard lk(mu);
        flush_batch();
    }
    wait_submitted();
    for (;;) {
        {
            std::lock_guard lk(inflight_mu);
            if (inflight.empty()) break;
        }
        if (readbacks_pending <= 0) break;
        wgpuDevicePoll(dev.device, true, nullptr);
    }
    {
        std::lock_guard lk(mu);
        if (gpu_written(gpu_va, size)) buffer_cache_sync_to_guest(gpu_va, size);
    }
    count("gpu_write_syncs");
    count("t_gpu_write_sync", hos_now_ns() - t0);
}

#ifdef __EMSCRIPTEN__
/// The render thread with no records to run (web): serves sync requests, then waits in its event loop (WebGPU's
/// callbacks: readbacks run there) until the parse thread bumps *wake (its condition variable's sequence: new
/// records), a sync request comes, or 2 ms pass.
extern "C" void rx_web_idle(volatile uint32_t* wake, uint32_t seen) {
    idle_word = wake;
    on_render_thread = true;
    for (;;) {
        SyncRequest* r = nullptr;
        {
            std::lock_guard lk(sync_mu);
            if (!sync_requests.empty()) {
                r = sync_requests.front();
                sync_requests.pop_front();
            }
        }
        if (!r) break;
        rx_sync_gpu_writes(r->gpu_va, r->size);
        {
            std::lock_guard lk(sync_mu);
            r->done = true;
        }
        sync_cv.notify_all();
    }
    {
        std::lock_guard lk(sync_mu);
        if (!sync_requests.empty()) return;
    }
    rx_web_wait_word((uint32_t)(uintptr_t)wake, seen, 2.0);
}
#endif

uint32_t* rx_collect_written(int64_t* count) {
    *count = -1;
    if (!started) return nullptr;
    std::vector<uint32_t> pages;
    if (!collect_written_pages(pages)) return nullptr;
    auto* out = static_cast<uint32_t*>(malloc((pages.size() + 1) * sizeof(uint32_t)));
    if (!pages.empty()) std::memcpy(out, pages.data(), pages.size() * sizeof(uint32_t));
    *count = (int64_t)pages.size();
    return out;
}

void rx_begin_submission_written(uint32_t* pages, int64_t n) {
    if (!started) {
        free(pages);
        return;
    }
    std::lock_guard lk(mu);
    flush_batch();   // (none should be open)
    if (n >= 0) buffer_cache_apply_written(pages, (size_t)n);
    else buffer_cache_scan();
    count("submissions");
    free(pages);
}

void rx_memory_written(uint64_t gpu_va, uint64_t size) {
    if (!started) return;
    std::lock_guard lk(mu);
    buffer_cache_written(gpu_va, size);
}

/// HOS_FRAME_HIST: called after each present with the render thread's busy time for the frame.
void rx_frame_spike(int64_t busy_ns, int64_t threshold_ns) {
    static int64_t logged_frame = -1000;
    if (busy_ns > threshold_ns && (threshold_ns <= 18000000 || frame_number - logged_frame >= 120) &&
        (logged_frame = frame_number, true))
        hos_log("rx: slow frame %lld: %.1f ms busy: %lld draws, decode %.1f ms, waits %.1f ms (trims %.1f), %lld KB "
                "texture uploads, %lld bind groups made, %lld translations queued; in place: translations %.1f ms, "
                "pipelines %.1f ms",
                (long long)frame_number, busy_ns / 1e6, (long long)spike.draws, spike.decode_ns / 1e6,
                spike.wait_ns / 1e6, spike.trim_ns / 1e6, (long long)(spike.upload_bytes >> 10),
                (long long)spike.bind_groups, (long long)spike.translations, spike.translate_ns / 1e6,
                spike.pipeline_ns / 1e6);
    spike = {};
}

int rx_present(uint64_t guest, uint32_t width, uint32_t height) {
    if (!started) return 0;
    std::lock_guard lk(mu);
    flush_batch();
#if !RX_WEB
    rdoc_present();
#endif
    if (static const bool watch_buf = getenv("RX_WATCH_BUF") != nullptr; watch_buf) watch_buffer_frame();
    if (frame_dump_active()) {
        wait_submitted();
        const char* dir = getenv("RX_DUMP_DIR");
        dump_frame_images(dir ? dir : "rx_frame");
    }
    const int64_t present_start = hos_now_ns();
    struct PresentTimer {
        int64_t t0;
        ~PresentTimer() { count("t_present", hos_now_ns() - t0); }
    } present_timer{present_start};
    sync_dispatch_writes();
    install_decoded_textures();
#if RX_WEB
    wgpuInstanceProcessEvents(dev.instance);   // (pipelines made asynchronously: their callbacks)
    pipecache_step();
#endif
    native_cache_step();
    report_period();
#if !RX_WEB
    if (static const bool ab = getenv("HOS_AB") != nullptr; ab) {   // (the patched wgpu flips its switch with HOS_AB)
        using SetPhase = void (*)(uint32_t);
        static const auto set_phase =
            reinterpret_cast<SetPhase>(GetProcAddress(GetModuleHandleA("wgpu_native.dll"), "wgpuTotkSetAbPhase"));
        if (set_phase) set_phase(ab_flipped() ? 1u : 0u);
    }
#endif
    // RX_DUMP_ON_DRAWS=<n> (with RX_DUMP_FRAME=99999999): the dump starts RX_DUMP_ON_DELAY frames after the first
    // frame with more than n draws (gameplay: thousands; title and loading screens: ~100)
    if (static const int64_t on_draws = getenv("RX_DUMP_ON_DRAWS") ? atoll(getenv("RX_DUMP_ON_DRAWS")) : 0; on_draws) {
        static int64_t last_draws = 0, fired = -1;
        static const int64_t delay = getenv("RX_DUMP_ON_DELAY") ? atoll(getenv("RX_DUMP_ON_DELAY")) : 0;
        static const int64_t arm_ns = getenv("RX_DUMP_ON_ARM") ? (int64_t)(atof(getenv("RX_DUMP_ON_ARM")) * 1e9) : 0;
        static const int64_t armed_from = hos_now_ns() + arm_ns;   // (and RX_DUMP_ON_ARM seconds in)
        if (fired < 0 && st_draws - last_draws > on_draws && hos_now_ns() >= armed_from) fired = frame_number;
        last_draws = st_draws;
        frame_number++;
        if (fired >= 0 && frame_number == fired + 1 + delay) dump_start_now();   // (the frame starting now)
    } else {
        frame_number++;
    }
    trim_decoded_textures();
    if (frame_number % 600 == 300) retire_unused_segments();
    if (frame_number % 600 == 450) reset_idle_stream_memory();
    if (frame_number % 600 == 150) trim_pipelines();
    if (frame_number % 3600 == 900) memory_report();
    if (frame_number % 600 == 0)   // spare render targets not drawn for 10 s go
        for (auto it = spare_images.begin(); it != spare_images.end();)
            it = it->second->last_use < frame_number - 600 ? spare_images.erase(it) : std::next(it);
    Image* im = image_at(guest);
    const bool debug_present = env_flag("RX_DEBUG_PRESENT");
    if (debug_present) im = nullptr;
    static int present_trace;
    if (env_flag("RX_TRACE") && present_trace++ < 12)
        hos_log("rx: present #%d guest %#llx target %p %ux%u format %d rendered %d", present_trace,
                (unsigned long long)guest, (void*)im, width, height, im ? (int)im->format : -1,
                im ? im->rendered : 0);
    if (!im) {
        log_limited("present_none", 8, "rx: present of %ux%u at guest %#llx: nothing drawn there", width, height,
                    (unsigned long long)guest);
    } else if (is_depth_format(im->format)) {
        im = nullptr;
    }
    if (env_flag("RX_READBACK_TARGET") || env_flag("RX_DEBUG_TARGET")) wait_submitted();
    static int64_t next_readback_frame = 120;
    if (im && frame_number >= next_readback_frame && env_flag("RX_READBACK_TARGET")) {
        next_readback_frame += 240;
        const uint32_t channels = im->format == WGPUTextureFormat_R8Unorm ? 1 :
                                  im->format == WGPUTextureFormat_RG8Unorm ? 2 :
                                  im->format == WGPUTextureFormat_RGBA8Unorm ||
                                          im->format == WGPUTextureFormat_RGBA8UnormSrgb
                                      ? 4 : 0;
        if (!channels) {
            log_limited("readback_format", 4, "rx: target readback unsupported format %#x", (unsigned)im->format);
        }
        const size_t bytes_per_pixel = channels ? channels : 4;
        const size_t bytes_per_row = ((size_t)im->width * bytes_per_pixel + 255) & ~size_t(255);
        const size_t read_size = bytes_per_row * im->height;
        WGPUBufferDescriptor buffer_desc{};
        buffer_desc.size = read_size;
        buffer_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        WGPUBuffer buffer = wgpuDeviceCreateBuffer(dev.device, &buffer_desc);
        WGPUCommandEncoder read_encoder = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
        WGPUTexelCopyTextureInfo source{};
        source.texture = im->tex;
        WGPUTexelCopyBufferInfo destination{};
        destination.buffer = buffer;
        destination.layout.bytesPerRow = (uint32_t)bytes_per_row;
        destination.layout.rowsPerImage = im->height;
        const WGPUExtent3D extent{im->width, im->height, 1};
        wgpuCommandEncoderCopyTextureToBuffer(read_encoder, &source, &destination, &extent);
        WGPUCommandBuffer read_cb = wgpuCommandEncoderFinish(read_encoder, nullptr);
        wgpuQueueSubmit(dev.queue, 1, &read_cb);
        wgpuCommandBufferRelease(read_cb);
        wgpuCommandEncoderRelease(read_encoder);
        struct ReadResult {
            std::mutex mutex;
            std::condition_variable ready;
            bool complete = false;
            WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
        } result;
        WGPUBufferMapCallbackInfo callback{};
        callback.mode = WGPUCallbackMode_AllowSpontaneous;
        callback.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata, void*) {
            auto& result = *static_cast<ReadResult*>(userdata);
            {
                std::lock_guard lock(result.mutex);
                result.status = status;
                result.complete = true;
            }
            result.ready.notify_one();
        };
        callback.userdata1 = &result;
        wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, read_size, callback);
        wgpuDevicePoll(dev.device, true, nullptr);
        {
            std::unique_lock lock(result.mutex);
            result.ready.wait(lock, [&] { return result.complete; });
        }
        if (result.status == WGPUMapAsyncStatus_Success) {
            const uint8_t* pixels = (const uint8_t*)wgpuBufferGetConstMappedRange(buffer, 0, read_size);
            if (pixels) {
                uint64_t visible = 0;
                uint8_t max_channel = 0;
                for (uint32_t y = 0; y < im->height; y++) {
                    for (uint32_t x = 0; x < im->width; x++) {
                        const uint8_t* pixel = pixels + (size_t)y * bytes_per_row + (size_t)x * bytes_per_pixel;
                        const uint8_t g = channels >= 2 ? pixel[1] : pixel[0];
                        const uint8_t b = channels >= 3 ? pixel[2] : channels == 1 ? pixel[0] : 0;
                        visible += pixel[0] > 2 || g > 2 || b > 2;
                        max_channel = std::max({max_channel, pixel[0], g, b});
                    }
                }
                hos_log("rx: target readback guest %#llx %ux%u format %#x: %llu nonblack pixels, max RGB %u",
                        (unsigned long long)guest, im->width, im->height, (unsigned)im->format,
                        (unsigned long long)visible, max_channel);
            }
            wgpuBufferUnmap(buffer);
        } else {
            hos_log("rx: target readback map failed (%d)", (int)result.status);
        }
        wgpuBufferRelease(buffer);
    }
    if (im && env_flag("RX_DEBUG_TARGET")) {
        WGPUCommandEncoder debug_encoder = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
        WGPURenderPassColorAttachment debug_color{};
        debug_color.view = im->attachment;
        debug_color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        debug_color.loadOp = WGPULoadOp_Clear;
        debug_color.storeOp = WGPUStoreOp_Store;
        debug_color.clearValue = {0.0, 1.0, 0.0, 1.0};
        WGPURenderPassDescriptor debug_pass_desc{};
        debug_pass_desc.colorAttachmentCount = 1;
        debug_pass_desc.colorAttachments = &debug_color;
        WGPURenderPassEncoder debug_pass = wgpuCommandEncoderBeginRenderPass(debug_encoder, &debug_pass_desc);
        wgpuRenderPassEncoderEnd(debug_pass);
        wgpuRenderPassEncoderRelease(debug_pass);
        WGPUCommandBuffer debug_cb = wgpuCommandEncoderFinish(debug_encoder, nullptr);
        wgpuQueueSubmit(dev.queue, 1, &debug_cb);
        wgpuCommandBufferRelease(debug_cb);
        wgpuCommandEncoderRelease(debug_encoder);
    }
    screenshot(im);
    flash_probe(im);
    if (!surface) {   // headless: the frame's end still queued (nothing presented), so the frames-ahead caps hold
        queue_present([] {});
        return im != nullptr;
    }
    // the blit into the window and the present run on the submit thread, after the frame's batches
    WGPUTextureView view = nullptr;
    if (im) {
        // the bytes as they are: an sRGB window buffer holds encoded values, and the surface is not sRGB
        const WGPUTextureFormat raw = srgb_pair(im->format);
        const bool srgb = im->format == WGPUTextureFormat_RGBA8UnormSrgb || im->format == WGPUTextureFormat_BGRA8UnormSrgb;
        view = srgb ? im->view(WGPUTextureViewDimension_2D, false, 0, 0, 1, raw) : im->attachment;
        wgpuTextureViewAddRef(view);
    }
    const LONG ww = want_w, wh = want_h;
    const uint32_t sw = im ? im->width : 0, sh = im ? im->height : 0;
    queue_present([view, sw, sh, ww, wh, debug_present] { present_on_submit_thread(view, sw, sh, ww, wh, debug_present); });
    return im != nullptr;
}

const char* rx_describe(void) {
    return started ? dev.description.c_str() : "off";
}

void rx_stats(int64_t* draws, int64_t* skipped, int64_t* pipelines) {
    *draws = st_draws;
    *skipped = st_skipped;
    *pipelines = st_pipelines;
}

} // extern "C"

namespace rx {
std::string rt_history_find(uint64_t lo, uint64_t hi) {
    std::string out;
    std::vector<std::pair<uint64_t, RtSeen>> sorted;
    for (const auto& [va, h] : rt_history)
        if (va >= lo && va < hi) sorted.push_back({va, h});
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [va, h] : sorted) {
            char b[120];
            std::snprintf(b, sizeof b, " named %#llx %ux%u f%#x frame %lld%s", (unsigned long long)va, h.width, h.height,
                          h.format, (long long)h.frame, h.on ? "" : " (render off)");
            out += b;
        }
    return out;
}
} // namespace rx

/// (debugging, any thread, no lock: a guess) the render target images over guest [lo, lo + len)
extern "C" void rx_debug_images_at(uint64_t lo, uint64_t len, char* out, size_t cap) {
    size_t n = 0;
    out[0] = 0;
    for (const auto& [g, im] : rx::all_images())
        if (im && im->guest < lo + len && lo < im->guest + (uint64_t)im->width * im->height * 4 && n + 80 < cap)
            n += std::snprintf(out + n, cap - n, " rt %08llx %ux%u fmt %d", (unsigned long long)im->guest, im->width,
                               im->height, (int)im->format);
}

namespace rx {
bool guest_watch_hit(uint64_t va, uint64_t size) {
    static uint64_t lo = 1, hi = 0;
    if (lo == 1) {
        lo = 0;
        if (const char* v = getenv("RX_GUEST_WATCH")) {
            char* e;
            lo = strtoull(v, &e, 16);
            hi = *e == '-' ? strtoull(e + 1, nullptr, 16) : 0;
        }
    }
    if (!hi) return false;
    const uint64_t g = hos_gpu_translate(va);
    return g && g < hi && g + size > lo;
}
} // namespace rx

/// the live input file's "!shots <ms>" (hos/nn_hid.c): another screenshot interval from now on
extern "C" void rx_set_screenshot_ms(int64_t ms) { rx::screenshot_interval = ms * 1000000LL; }

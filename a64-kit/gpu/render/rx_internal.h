// The renderer's shared state and helpers (gpu/render/rx_*.cpp). See rx.h.
#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <memory>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <webgpu/webgpu.h>
#ifdef __EMSCRIPTEN__
// The browser build (the browser's WebGPU through emdawnwebgpu): wgpu-native's extras stand in here. Every WebGPU
// call is made on the render thread (objects belong to the worker that made them); waits on the GPU yield to that
// worker's event loop (JSPI), where the browser runs WebGPU's callbacks.
#define RX_WEB 1
#define WGPUNativeFeature_TextureFormat16bitNorm 0x7FFF0001
bool wgpuDevicePoll(WGPUDevice device, bool wait, const void* submission);   // rx_core.cpp
#else
#define RX_WEB 0
#include <webgpu/wgpu.h>
#endif
#ifndef RX_WEB_MT
#define RX_WEB_MT 0   // (the threaded web build: rx_web_mt.h)
#endif

#include "shader_recompiler/shader_info.h"
#include "video_core/maxwell_3d_regs.h"
#include "video_core/surface.h"
#include "video_core/textures/texture.h"

#include "rx.h"
#include "rx_nvn.h"

extern "C" {
void hos_log(const char* fmt, ...);
void hos_process_memory(uint64_t* private_bytes, uint64_t* working_set);
void hos_memory_map_log(void);
uint64_t hos_gpu_translate(uint64_t va);
uint8_t* hos_gpu_ptr(uint64_t va, uint64_t* avail);
int64_t hos_now_ns(void);
void hos_host_thread_register_self(const char* name);
void hos_thread_boost(int level);
}

namespace rx {

using Regs = Tegra::Engines::Maxwell3D::Regs;
using PixelFormat = VideoCore::Surface::PixelFormat;
#define RX_REG(field) (offsetof(::rx::Regs, field) / sizeof(uint32_t))

inline WGPUStringView sv(const char* s) {
    return WGPUStringView{s, WGPU_STRLEN};
}
inline std::string str(WGPUStringView v) {
    return v.data ? std::string(v.data, v.length == WGPU_STRLEN ? std::strlen(v.data) : v.length) : std::string{};
}

// ---- guest memory, through the GPU's address space (hos/nvdrv.c's mappings) ----
/// Reads n bytes at a GPU VA; unmapped bytes read as 0. False when any byte was unmapped.
bool gpu_read(uint64_t va, void* dst, uint64_t n);
/// A cheap nanosecond clock for the renderer's own time statistics (the CPU's invariant TSC, scaled once against
/// hos_now_ns): ten reads a draw through QueryPerformanceCounter were ~4% of the GPU thread.
int64_t clock_ns();
extern "C" volatile long maps_generation;   // hos/nvdrv.c: bumped by every GPU mapping change
bool render_enabled(const Regs& r);
/// A host pointer to n contiguous bytes at a GPU VA, or null when they are not all mapped in one piece.
const uint8_t* gpu_span(uint64_t va, uint64_t n);
template <class T>
T gpu_get(uint64_t va) {
    T v{};
    gpu_read(va, &v, sizeof v);
    return v;
}

// ---- images: render targets and textures, by the guest address of their memory ----
inline uint64_t next_image_id = 1;
struct Image {
    uint64_t id = next_image_id++;          // unique (a cache that remembers an image compares this, not the pointer)
    uint64_t gpu_va = 0, guest = 0;
    uint32_t width = 0, height = 0, layers = 1, levels = 1;
    PixelFormat pixel_format{};
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    WGPUTextureDimension dimension = WGPUTextureDimension_2D;
    WGPUTexture tex = nullptr;
    WGPUTextureView attachment = nullptr;   // level 0, layer 0: what render passes draw into
    bool rendered = false;                  // the GPU wrote it: its contents live only in the texture
    bool storage = false;                   // decoded by a compute shader (ASTC): views ask for sampling only
    // a render target: its memory layout (from the target registers), for writing it back to guest memory
    uint32_t bl_height = 0, bl_depth = 0, reg_width = 0;
    bool pitch_linear = false;
    uint64_t storage_limit = 0;             // a native NVN target: the bytes its texture's storage has there (0: unknown)
    bool writeback_queued = false;          // a one-off draw rendered it: copied to guest memory at the submission's end
    uint32_t slice = 0, volume_depth = 0;   // a 3D texture's slice drawn (nvn_slice_key): which, of how many
    bool block_view = false;                // drawn through an uncompressed view of a compressed texture: a texel a block
    uint64_t content_hash = 0;              // a texture decoded from memory: samples of the memory it came from
    int64_t checked_frame = -1;
    uint64_t decoded_seq = 0;               // memory_seq() when it was decoded (0: no tracking: content_hash)
    uint64_t guest_bytes = 0;               // a texture decoded from memory: how much it covers
    bool stale = false;                     // ...and its memory was written since
    int64_t last_use = 0;
    std::vector<std::pair<uint64_t, WGPUTextureView>> views;
    uint64_t version = 0;                    // a render target: bumped by each pass that may write it
    std::vector<uint64_t> sub_versions;      // a texture assembled from render targets: the version copied into
                                             // each (layer, level), layer-major
    std::vector<uint64_t> sub_addresses;     // ...where each starts in guest memory
    std::vector<Image*> sub_images;          // ...image_at(each address), as of sub_images_generation
    uint64_t sub_images_generation = 0;
    uint64_t sub_end = 0;                    // ...the end of the last subresource's start (the range image changes
                                             // must touch for sub_images to change)
    std::vector<uint32_t> sub_present;       // ...the subresources with an image (the ones a pass may need to copy)
    uint64_t assembled_passes = ~0ull;       // ...pass_count() when last brought up to date
    std::unique_ptr<Image> sampled_copy;     // reusable pre-draw snapshot for attachment feedback
    uint64_t snapshot_epoch = 0;             // the texture barrier epoch it was taken in (snapshot_for_feedback)
    /// A view of the image as a shader binds it (cached).
    WGPUTextureView view(WGPUTextureViewDimension dim, bool depth_aspect, uint32_t base_layer = 0,
                         uint32_t base_level = 0, uint32_t level_count = 0,
                         WGPUTextureFormat view_format = WGPUTextureFormat_Undefined);
    ~Image();
};
/// Views of destroyed images are kept until no cached bind group can name them (a released handle's address could
/// come back as a new view's).
void retire_view(WGPUTextureView view);
bool retired_views_pending();
void release_retired_views();
const std::vector<WGPUTextureView>& retired_view_list();
/// A bind group the recorded batches may still name: released once they are replayed (at once when not deferred).
void release_bind_group_later(WGPUBindGroup g);
void release_pipeline_later(WGPURenderPipeline p);
void trim_pipelines();
/// The image a render target register set describes (created, or replaced when its format/size changed).
/// key: where the image is filed instead of the translated address (a 3D texture's slice: nvn_slice_key)
Image* find_render_target(uint64_t gpu_va, uint32_t width, uint32_t height, PixelFormat format, uint64_t key = 0);
/// The key a slice of a 3D render target's images are filed under: above any guest address (guest memory is below
/// 4 GB), so no texture lookup meets it, and never written back. The 3D texture is assembled from them when sampled.
inline uint64_t nvn_slice_key(uint64_t guest, uint32_t slice) { return guest | (uint64_t)(slice + 1) << 40; }
/// Every render target, by guest address (debugging).
const std::unordered_map<uint64_t, std::unique_ptr<Image>>& all_images();
/// A PNG of RGBA8 rows.
void write_png(const char* path, const uint8_t* rgba, uint32_t w, uint32_t h, size_t stride);
/// The image whose memory starts at this guest address, if any.
Image* image_at(uint64_t guest);
/// Changes whenever image_at could answer differently for some address (a cache of its answers is valid while this
/// stays the same).
uint64_t images_generation();
/// Whether image_at could answer differently for an address in [lo, hi) than at generation since.
bool images_changed_in(uint64_t since, uint64_t lo, uint64_t hi);
/// Ends the current pass and snapshots an attachment before a draw samples and writes it.
Image* snapshot_for_feedback(Image& image);
/// A texture barrier (texture data cache invalidation, tiled cache barrier) in the command stream.
void texture_barrier(uint32_t method, uint32_t arg);

WGPUTextureFormat wgpu_format(PixelFormat format);
/// The same format with the other sRGB encoding (RGBA8 <-> RGBA8 sRGB, ...), or Undefined. Textures are created
/// with it as a view format: what the game writes or samples as sRGB and what it reads raw are one memory.
WGPUTextureFormat srgb_pair(WGPUTextureFormat f);
bool is_depth_format(WGPUTextureFormat f);
bool has_stencil(WGPUTextureFormat f);

// ---- per-channel state ----
struct ConstBuffer {
    uint64_t address = 0;
    uint32_t size = 0;
    bool enabled = false;
};
struct Channel {
    std::array<std::array<ConstBuffer, 18>, 5> cbufs{};
    // the draw being assembled (suyu's DrawManager state)
    enum class DrawMode { General, Instance, InlineIndex } draw_mode = DrawMode::General;
    bool draw_indexed = false;
    uint32_t instance_count = 0;
    std::vector<uint8_t> inline_index;
    uint32_t topology = 0;
    // the render enable condition (draws and clears run while it holds), evaluated when its mode or override is
    // written (suyu's ProcessQueryCondition): the game can change the values it compares before the draws it covers
    bool render_on = true;
    // ZPASS_PIXEL_COUNT: an estimate of the samples that passed (draws while counting is enabled add their size),
    // reset by CLEAR_REPORT_VALUE. TOTK culls objects by occlusion queries on boxes: a report of 0 hides them.
    uint64_t zpass = 0;
};

/// One draw, as the 3D engine describes it at the moment it runs.
struct DrawParams {
    bool indexed = false;
    uint32_t topology = 0;
    uint32_t first = 0, count = 0;         // vertices, or indices of the index buffer
    uint32_t instance_count = 1, base_instance = 0;
    int32_t base_vertex = 0;
    const std::vector<uint8_t>* inline_index = nullptr;   // u32 indices given in the command stream
    uint64_t index_va = 0;                                // otherwise the index buffer
    uint32_t index_size = 4;                              // bytes per index
};
void draw(Channel& ch, const Regs& regs, const DrawParams& p);
void clear(Channel& ch, const Regs& regs);

// ---- recording: one command encoder per batch, render passes inside it ----
struct Attachments {
    std::array<Image*, 8> color{};
    Image* depth = nullptr;
    uint32_t width = 0, height = 0;
    bool operator==(const Attachments& o) const {
        return color == o.color && depth == o.depth;
    }
};
/// The render targets a draw writes (one size: WebGPU requires it).
Attachments draw_attachments(const Regs& r);
/// RX_DUMP_FRAME=<n> (debugging): frame n's passes are logged and its render targets saved as PNGs at its present.
extern bool frame_dump_enabled;
bool frame_dump_active_slow();
inline bool frame_dump_active() { return frame_dump_enabled && frame_dump_active_slow(); }
void dump_pass(const Attachments& at, const char* kind);
void dump_on_targets(const Attachments& at);
void watch_buffer_frame();
void watch_buffer_after_draw(const char* what);
void dump_draw(const char* what);
/// RX_DUMP_SAMPLED=1 (with a frame dump): render targets sampled by draws are saved with the frame's targets.
void dump_sampled(const Image* im);
/// A dispatch wrote this range in the buffer cache (RX_SYNC_DISPATCH_WRITES=1: copied to guest memory at the present).
void note_dispatch_write(uint64_t gpu_va, uint64_t size);
void sync_dispatch_writes();
/// The native NVN layer's fences: the last batch flushed; whether one ran on the GPU with its readbacks in guest
/// memory; waiting for that (rx_core.cpp)
uint64_t last_batch_serial();
bool batch_done(uint64_t serial);
void wait_batch(uint64_t serial);
void dump_frame_images(const char* dir);
void note_texels_written(uint64_t guest);   // (rx_texture.cpp: RX_DUMP_TEX=texels)
/// RX_DUMP_TEX=<GPU VA hex>[,...] (with a frame dump): the decoded textures at those addresses, saved like targets.
void dump_tex_list(const char* dir);
void dump_texture(WGPUTexture tex, uint64_t va, uint32_t width, uint32_t height, WGPUTextureFormat format,
                  const char* dir, int order);
void dump_after_draw(uint64_t fs_address, Image* target);
void dump_before_draw(uint64_t fs_address, const Attachments& at);
void dump_pixel_after(const Attachments& at, const char* what);
/// The pass draws go into: (re)started when the attachments differ from the current pass's.
WGPURenderPassEncoder render_pass(const Attachments& at);
/// Ends the current render pass (a clear or a copy comes next).
void end_render_pass(const char* why = nullptr);   // (why: a stats counter when a pass was open)
/// Before a draw: a pass that has had RX_PASS_SPLIT draws (default 256; 0: never) is ended, so the draw begins it again
/// with its attachments loaded. wgpu encodes a pass in one encoder, and the submit thread's encoders (one each of up to
/// RX_FINISH_THREADS, cut where passes end) were 5.1, 4.3 and 0.4 ms a frame at the castle: one pass was most of it.
void split_long_pass();
/// Render passes begun so far (a render target can have changed only if this did).
uint64_t pass_count();
WGPUCommandEncoder encoder();
int rx_target_writebacks_pending();
/// The window's present (rx_present_fx.cpp): where the frame goes in the window (its aspect kept), and the passes
/// that put it there (FSR 1 when that is larger than the frame)
struct PresentRect {
    uint32_t x, y, w, h;
};
PresentRect present_rect(uint32_t sw, uint32_t sh, uint32_t tw, uint32_t th);
bool present_framegen();   // (the player's frame interpolation setting, F7)
void present_window_size(int32_t w, int32_t h);   // (rx_resize: ultrawide's aspect)
void surface_present(WGPUTextureView view, uint32_t sw, uint32_t sh, int32_t ww, int32_t wh, bool debug_present);
/// Frame interpolation (rx_framegen.cpp): the frame (its view released) and the frames made between it and the last
/// handed to the display thread; false: not done (the caller presents it as usual)
bool framegen_frame(WGPUTextureView view, uint32_t sw, uint32_t sh, int32_t ww, int32_t wh);
/// (the threaded web build's WebGPU thread: its loop, between jobs and between a batch's commands) the next made frame
/// shown if due; ns until the one after (-1: none waiting). framegen_web_due: when it is due (0: none).
int64_t framegen_web_tick();
extern int64_t framegen_web_due;
void present_fx_draw(WGPUCommandEncoder e, WGPUTextureView src, uint32_t sw, uint32_t sh, WGPUTextureView target,
                     WGPUTextureFormat target_format, uint32_t tw, uint32_t th, const WGPUColor& clear);   // (the threaded web build: render target writebacks not in guest memory yet)
/// The attachments of the current pass (valid while render_pass's result is).
const Attachments& current_attachments();

/// Per-batch upload memory: GPU buffers written once before the batch's commands are submitted.
struct Stream {
    WGPUBuffer buffer = nullptr;
    uint8_t* cpu = nullptr;   // this batch's contents (each batch gets its own: the submit thread writes them)
    uint64_t used = 0, capacity = 0;
    WGPUBufferUsage usage{};
    const char* name = "";
    /// Space for n bytes; the offset, or UINT64_MAX when the batch must be flushed first.
    uint64_t alloc(uint64_t n, uint64_t align);
    uint8_t* at(uint64_t offset) {
        return cpu + offset;
    }
};
/// A buffer write that must reach the GPU after the batches already flushed and before the open one (the buffer
/// cache's page uploads): it goes to the queue with the open batch, on the submit thread.
void queue_upload(WGPUBuffer buffer, uint64_t offset, const void* data, uint64_t size);
/// Waits until every flushed batch was submitted (before work that reads results back or bypasses the batches).
void wait_submitted();
/// RenderDoc frame captures (rx_debug.cpp: "!rdoc", RX_RDOC_AT): at each present, after its batch was flushed
void rdoc_present();
/// A writable storage binding the open batch copied into stream_rw at rw_offset from guest memory at gpu_va: after the
/// batch runs, the words the GPU changed there are written back into guest memory (the game reads what its shaders
/// write: visibility flags from material shaders, ...).
void writeback_after_batch(uint64_t rw_offset, uint64_t size, uint64_t gpu_va);
/// A pixel program of the open pass writes this stream_rw copy of guest memory: at the pass's end it is copied into
/// the buffer cache (later passes read the results there).
void storage_written_in_pass(uint64_t rw_offset, uint64_t size, uint64_t gpu_va);
// ---- the buffer cache (rx_buffer.cpp): guest memory mirrored in GPU buffers ----
struct BufferRef {
    WGPUBuffer buffer = nullptr;
    uint64_t offset = 0, size = 0;
};
/// A guest range as a binding of a cached buffer, its contents current; empty when it cannot be (too big,
/// unmapped, spans mappings).
BufferRef guest_buffer(uint64_t gpu_va, uint64_t size);
bool buffer_cache_active();
/// Change tracking of guest memory (the buffer cache's): a tick (0 when there is no tracking), and whether any page
/// of a guest range was written after a tick.
uint64_t memory_seq();
bool memory_written_since(uint64_t guest, uint64_t size, uint64_t seq);
/// Guest memory [guest, guest + size) was written: textures decoded from it are stale (rx_texture.cpp).
void textures_written(uint64_t guest, uint64_t size);
void buffer_cache_new_batch();
/// The open batch (changes at each flush).
int64_t batch_number();
/// Whether a dispatch wrote some of this range (its contents are then only on the GPU).
bool gpu_written(uint64_t gpu_va, uint64_t size);
/// A readback's changed words (now against was) into guest memory, as the GPU's results rather than CPU writes (any
/// thread).
/// Whether a pixel program of the open render pass writes storage overlapping this range.
bool storage_written_this_pass(uint64_t gpu_va, uint64_t size);
bool pass_wrote_storage();   // (a pixel program of the open pass wrote storage: its results are pending)
int64_t readback_to_guest(uint8_t* guest, const uint8_t* now, const uint8_t* was, uint64_t n);
/// Guest memory gets what dispatches wrote into the range (and the GPU-written pages next to it) in the buffer cache:
/// a GPU round trip, after the batches that wrote it were submitted.
void buffer_cache_sync_to_guest(uint64_t gpu_va, uint64_t size);
void buffer_cache_scan();
bool collect_written_pages(std::vector<uint32_t>& out);
void buffer_cache_apply_written(const uint32_t* pages, size_t n);
void buffer_cache_written(uint64_t gpu_va, uint64_t size);
/// The GPU wrote a cached range (a dispatch's storage buffer): the other segment holding it gets the new contents.
void buffer_cache_gpu_written(const BufferRef& ref);
/// Bumped by every dispatch that writes storage: the draws' shared copies of writable storage (one per range and
/// batch) are made again after it, from what the dispatch wrote.
extern uint64_t dispatch_write_epoch;
/// The draw being made goes to targets no draw of the last frames used (a map made once, at loading, say): its
/// programs are translated and its pipeline compiled before it is drawn, not skipped meanwhile (RX_SKIP_ONE_OFF=1:
/// skipped as any other)
extern bool draw_must_run;
/// RX_RT_HISTORY=1 (debugging): the color target addresses in [lo, hi) any draw or clear named, with their last frame
std::string rt_history_find(uint64_t lo, uint64_t hi);
void dump_start_now();
/// A one-off draw rendered into it: its contents go to guest memory at the end of the submission (before its fence),
/// as the console's memory would hold them (TOTK's terrain copies the maps it renders with the CPU)
void queue_target_writeback(Image* im);
void writeback_targets();
/// RX_GUEST_WATCH=<lo>-<hi> guest hex (debugging): whether GPU memory [va, va + size) touches that guest range
bool guest_watch_hit(uint64_t va, uint64_t size);

extern Stream stream_ro;   // uniforms, vertices, indices, read-only storage
extern Stream stream_rw;   // writable storage
/// Allocates, flushing the batch first when the stream is full. UINT64_MAX when n can never fit.
uint64_t stream_alloc(Stream& s, uint64_t n, uint64_t align);
void flush_batch();

// ---- the device ----
struct Device {
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPULimits limits{};
    bool bc = false, float32_filterable = false, depth32_stencil8 = false, norm16 = false, depth_clip_control = false,
         early_depth_test = false, timestamps = false;
    bool is_intel = false, is_intel_arc = false;
    std::string description;
};
extern Device dev;
/// The renderer's lock: recursive, over a Windows SRW lock (std::recursive_mutex was ~3.5% of the GPU thread: it is
/// taken for every renderer method, several a draw, nearly always by the GPU thread itself).
class RecursiveLock {
    alignas(8) unsigned char srw_[64] = {};   // an SRWLOCK (SRWLOCK_INIT is all zero; the web's is a pthread rwlock)
    std::atomic<unsigned long> owner_{0};
    unsigned depth_ = 0;
public:
    void lock();
    void unlock();
};
extern RecursiveLock mu;
extern int64_t frame_number;

// statistics
extern int64_t st_draws, st_skipped, st_pipelines, st_clears;
/// Validation error scopes: the error the calls in between made (empty: none).
void scope_push();
std::string scope_pop();
/// An environment switch, read once (the hot paths test them per draw).
bool env_flag(const char* name);
/// HOS_AB_SWITCH=<name> with HOS_AB=<ms> (hos_prof.c: an A/B inside one run, for a busy machine): true while the
/// switch of that name is in its flipped phase. Callers keep `ab_named(name)` in a static and test `ab && ab_flipped()`.
bool ab_named(const char* name);
bool ab_flipped();
/// A translated program's module. RX_SHADER_CHECKS=<bits> (native, experiment): the runtime checks naga adds to it
/// (WGPUShaderRuntimeChecks: 1 bounds checks, 2 loop bounding); unset: wgpu's default, all of them.
WGPUShaderModule program_module(const WGPUShaderModuleDescriptor* d);
void background_thread();
/// Statistics of the current period, by key (skipped draws by reason, ...): logged with RX_STATS.
void count(const char* key, int64_t n = 1);
/// Logs a message at most `limit` times for its key.
void log_limited(const char* key, int limit, const char* fmt, ...);

// ---- shaders and pipelines (rx_draw.cpp, rx_shader.cpp) ----
struct ShaderModule {
    WGPUShaderModule module = nullptr;
    Shader::Info info{};
    std::string wgsl;                    // (until interned: then text, and this is emptied)
    std::string* text = nullptr;   // the interned WGSL (graphics stages: one copy per distinct text; a pair
                                         // each held its own, ~7000 pairs x 2 x ~15 KB of host memory)
    const std::string& source() const { return text ? *text : wgsl; }
    bool failed = false;
    const char* what = "";   // (graphics stages: created by the pipeline that first needs them, ensure_module)
    uint64_t key = 0;
    uint32_t text_id = 0;   // the same for the same WGSL (module sharing; ids are per session)
    /// info.loads.Generic(i) for the 32 attributes as one mask, made once (each draw testing 128 bits was ~1.6% of the
    /// replay thread; info is set when the module is made and not changed after)
    uint32_t generic_loads() const {
        if (!generic_loads_ok) {
            uint32_t m = 0;
            for (uint32_t i = 0; i < 32; i++) m |= (info.loads.Generic(i) ? 1u : 0u) << i;
            generic_loads_mask = m, generic_loads_ok = true;
        }
        return generic_loads_mask;
    }
    mutable uint32_t generic_loads_mask = 0;
    mutable bool generic_loads_ok = false;
    uint64_t text_hash = 0;   // the interned text's hash: pipelines are keyed by these (the same in every session)
    uint64_t cache_hash = 0;
    std::shared_ptr<std::mutex> creating = std::make_shared<std::mutex>();   // (one creation; others wait)
};
/// The module from the WGSL (once; any thread: the pipeline compilers create the graphics stages' modules, as
/// creating them waits for the device while pipelines compile). False when it does not compile.
bool ensure_module(ShaderModule& m);
struct StageKey {
    uint64_t address = 0;   // GPU VA of the program header
    uint64_t code_hash = 0;
    bool operator==(const StageKey& o) const {
        return address == o.address && code_hash == o.code_hash;
    }
};
/// The vertex + fragment modules of a draw's programs (translated together: their interfaces must agree).
struct ProgramPair {
    ShaderModule vs, fs;
    bool has_fs = false;
    bool failed = false;
};
ProgramPair* programs_for_draw(Channel& ch, const Regs& regs);
/// Vertex attributes the vertex stage reads from memory itself (layouts WebGPU's vertex fetch cannot describe):
/// up to 3 pulled streams, as the WGSL backend's Options::vertex_pull wants them.
struct PullPlan {
    struct Attr {
        bool enabled = false;
        uint8_t stream = 0;   // the pulled stream (0-2)
        bool instanced = false;
        uint32_t attribute = 0, stride = 0;
    } attr[32];
    uint32_t streams[3]{};    // the vertex stream (0-15) of each pulled stream
    uint32_t count = 0;
};
PullPlan pull_plan(const Regs& r, const std::bitset<32>& loads);
/// How the vertex stage unpacks an attribute fetched as one u32 (Shader::Backend::WGSL::VertexUnpack; 0: it is
/// fetched in its own format).
uint8_t vertex_unpack(const Regs::VertexAttribute& a);

/// A compute dispatch as the launch descriptor (QMD) and the compute engine's registers describe it.
struct ComputeLaunch {
    uint64_t code_base = 0;         // code_loc
    uint32_t program_start = 0;
    uint32_t grid[3]{}, block[3]{};
    uint32_t shared_size = 0, local_size = 0, texture_bound = 0;
    bool linked_tsc = false;
    struct Cbuf {
        uint64_t address = 0;
        uint32_t size = 0;
        bool enabled = false;
    } cbufs[8];
    const Regs* tex_regs = nullptr;   // the texture/sampler pools (the compute engine's, in the 3D layout)
};
/// The compute module of a launch (cached by program); null when it does not translate.
/// ssbo_alias (web): storage descriptors bound to one same range use one binding (see WGSL::ssbo_alias_next)
ShaderModule* compute_module(const ComputeLaunch& launch, const std::array<uint8_t, 64>* ssbo_alias = nullptr,
                             const std::array<uint32_t, 64>* ssbo_alias_words = nullptr);

// ---- textures and samplers (rx_texture.cpp) ----
/// How a texture binding is declared in a pipeline layout (the format decides; part of the pipeline key).
enum class TexKind : uint8_t { Float, UnfilterableFloat, Depth, Uint, Sint };
struct BoundTexture {
    WGPUTextureView view = nullptr;
    WGPUSampler sampler = nullptr;
    TexKind kind = TexKind::Float;
    Image* image = nullptr;   // a render target sampled (must not be an attachment of the same pass)
    char source = 'd';        // (debugging) d placeholder, t texture from memory, r render target
    uint64_t guest = 0;
    uint32_t width = 0, height = 0;
    int format = 0;
    uint8_t slots[32]{};      // the header's swizzle and the sampler's border/compare/bias, as the uniform block takes them
};
BoundTexture texture_for_handle(const Regs& regs, uint32_t raw_handle, const Shader::TextureDescriptor& desc);
void bc7_encode(const uint8_t* rgba, uint32_t width, uint32_t height, uint32_t pitch, uint8_t* out);
/// The placeholder a binding gets when its texture cannot be used.
BoundTexture dummy_texture(const Shader::TextureDescriptor& desc);
Tegra::Texture::TICEntry read_tic(const Regs& regs, uint32_t tic_index);
Tegra::Texture::TSCEntry read_tsc_for_dump(const Regs& regs, uint32_t raw_handle);
/// A texture's two uniform block slots (the WGSL backend's): the TIC's component swizzle with the sampler's border
/// axes and depth compare function, then the border color.
void texture_slots(uint8_t* dst, const Tegra::Texture::TICEntry& tic, const Tegra::Texture::TSCEntry& tsc);
WGPUTextureViewDimension view_dimension(Shader::TextureType type);

uint64_t hash_bytes(const void* p, size_t n, uint64_t seed = 0xcbf29ce484222325ull);
/// The browser's pipeline cache (rx_pipecache.cpp): pipelines recorded as made, made again in the background when the
/// page opens next (start: at device creation; step: each present).
std::string web_shader_source(const std::string& wgsl);   // (the WGSL as the device gets it)
void pipecache_start();
bool sampler_nonfiltering(WGPUSampler s);   // (debugging)
void pipecache_step();
void pipecache_record_render(const WGPURenderPipelineDescriptor& d, const ShaderModule& vs, const ShaderModule* fs,
                             const std::vector<WGPUBindGroupLayoutEntry>* layouts, int nlayouts);
uint64_t pipecache_text_hash(const std::string& wgsl);
bool pipecache_has_module(uint64_t h);
void release_module_text(ShaderModule& m);
void pipecache_record_compute(const std::string& wgsl, const std::vector<WGPUBindGroupLayoutEntry>& layout);

/// A render pipeline as the native pipeline cache keeps it (rx_pipecache_native.cpp): its key in the pipeline map,
/// the WGSL texts of its stages (by interned-text hash) and everything its creation reads.
struct PipelineDesc {
    uint64_t key = 0, vs_text = 0, fs_text = 0;
    std::vector<uint32_t> layout[2];   // packed bind group layouts (pack_layout)
    int nlayouts = 1;
    std::vector<WGPUVertexBufferLayout> buffers;   // (attributes: below, one list per buffer)
    std::vector<std::vector<WGPUVertexAttribute>> attributes;
    std::vector<WGPUColorTargetState> targets;   // (blend: has_blend)
    std::vector<uint8_t> has_blend;
    std::array<WGPUBlendState, 8> blends{};
    WGPUPrimitiveState primitive{};
    bool has_depth = false;
    WGPUDepthStencilState depth{};
};
void native_cache_start();   // (render thread, at device creation: reads the cache in the background)
void native_cache_step();    // (render thread, each frame: saves new records, makes cached pipelines a few at a time)
void native_cache_text(uint64_t h, const std::string& text);   // a new interned WGSL text (any thread)
void native_cache_pipeline(const PipelineDesc& d);              // a new pipeline (render thread)
void native_cache_used(uint64_t key);   // a draw used this pipeline (render thread; once a session per pipeline)
bool native_cache_on();
void native_cache_compute_text(uint64_t h, const std::string& text);   // a compute program's WGSL that compiled
WGPUShaderModule native_cache_compute_module(uint64_t h);   // its module, made by the cache's thread (or null)
/// The module of a cached WGSL text (interned as a translation would: draws of the same text share the module)
ShaderModule* prewarm_stage_module(uint64_t h, const std::string& text, const char* what);
/// Makes a cached pipeline, unless the map has its key (render thread); false when there was nothing to make
bool prewarm_pipeline(const PipelineDesc& d, ShaderModule* vs, ShaderModule* fs);
int prewarm_in_flight();   // cached pipelines still compiling

// ---- textures made by the GPU from guest memory as it is (rx_texture_gpu.cpp) ----
struct BlockLinearLevel {
    uint32_t offset;                              // the level's first byte in the guest texture
    uint32_t tiles_w, tiles_h;                    // its size in blocks
    uint32_t block_size, x_shift, block_height;   // its block-linear layout (as the CPU unswizzle's)
};
/// Whether ASTC textures are decoded by the GPU (native; RX_ASTC_CPU=1: by the CPU)
bool astc_gpu_on();
/// Decodes every level of a 2D (array) ASTC texture in guest layout into tex (RGBA8Unorm, storage binding), submitted
/// at once (any thread). False: nothing was done (the CPU decodes it).
bool astc_gpu_decode(WGPUTexture tex, const uint8_t* src, uint64_t total, const BlockLinearLevel* levels, int nlevels,
                     uint32_t layers, uint32_t layer_stride, uint32_t block_w, uint32_t block_h);
/// Whether other block-linear textures are unswizzled by the GPU (native; RX_UNSWIZZLE_CPU=1: by the CPU)
bool unswizzle_gpu_on();
/// Unswizzles every level of a 2D (array) texture of 4, 8 or 16 byte blocks (bpp_log2 2-4) into tex, as it is
/// (submitted at once, any thread). False: nothing was done.
bool unswizzle_gpu(WGPUTexture tex, const uint8_t* src, uint64_t total, const BlockLinearLevel* levels, int nlevels,
                   uint32_t layers, uint32_t layer_stride, uint32_t bpp_log2, uint32_t block_w, uint32_t block_h);
/// The bytes of one level and layer, row_bytes x rows (debugging: waits for the GPU; width/height: the copy's extent)
std::vector<uint8_t> read_level(WGPUTexture tex, uint32_t level, uint32_t layer, uint32_t width, uint32_t height,
                                uint32_t row_bytes, uint32_t rows);

// ---- the native NVN layer (HOS_NVN_NATIVE=1): rx_nvn.cpp replays the game's NVN command records with these, each
// built from NVN state directly (no Switch GPU registers or command buffers in between). See rx_nvn.h. ----
PixelFormat nvn_pixel_format(int nvn_format);
/// The queue's state as the recordings set it (one queue; it carries across submits, as the console's does).
struct NvnState {
    nvn_prog prog{};   // the bound program, copied at the bind (the game can set new shaders on the object later)
    bool has_prog = false;
    // per NVN stage (0 vertex, 1 fragment, 2 geometry, 3 tess control, 4 tess evaluation, 5 compute)
    nvn_p_bind_buf ubo[6][16]{}, ssbo[6][16]{};
    uint64_t tex[6][128]{}, samp[6][128]{}, img[6][32]{};
    nvn_blend_st blend[8]{};   // by target
    nvn_color_st color{};
    nvn_chmask_st chmask{0xffffffffu};
    nvn_ms_st ms{};
    nvn_poly_st poly{};
    nvn_ds_st ds{};
    uint32_t vattr[16]{};      // nvn_va_st
    int nvattr = 0;
    nvn_vs_st vstream[16]{};
    NVNbufferRange vbuf[16]{};
    nvn_p_set_rt rt{};
    Attachments at;            // the render targets' images (made from rt when at_dirty)
    bool at_dirty = true;
    uint64_t at_images = ~0ull;
    int64_t at_frame = -1;
    float viewport[16][4]{};   // x, y, width, height
    float depth_near = 0.0f, depth_far = 1.0f;
    int32_t scissor[4]{0, 0, 0, 0};   // (width 0: none set)
    uint32_t stencil_value_mask[2]{0xff, 0xff}, stencil_mask[2]{0xff, 0xff}, stencil_ref[2]{0, 0};   // front, back
    float poly_offset[3]{};    // factor, units, clamp
    float blend_color[4]{};
    bool restart = false;
    uint32_t restart_index = 0;
    bool depth_clamp = false, raster_discard = false;
    bool render_enabled = true;
    int64_t copy_stride = 0;   // SetCopyRowStride: the buffer side's bytes per row (0: packed)
    uint32_t grid[3]{};        // the dispatch being made (compute programs read it from c0)
    // bumped when what picks a program variant changes (program, vertex attributes / streams, texture, sampler and
    // image handles; a rebind of the same values does not count): nvn_programs reuses an answer while it stays.
    uint64_t prog_epoch = 1;
    // the vertex attributes and streams by content (set when they change): the program variants and pull plans
    // memoized for each, as draws of different materials take turns (an epoch bumped at every program bind was not)
    uint64_t vtx_hash = 0;
    // the occlusion counter (SAMPLES_PASSED) the queue reports: it grows with each draw that rasterizes (no real
    // count is read back), so the difference between two reports says whether anything was drawn between them
    uint64_t samples_passed = 0;
};
inline uint64_t nvn_vertex_hash(const NvnState& s) {   // (NvnState::vtx_hash)
    return hash_bytes(s.vattr, sizeof s.vattr, hash_bytes(s.vstream, sizeof s.vstream, (uint64_t)s.nvattr));
}
/// The images a render target set draws into (one size: WebGPU requires it)
Attachments nvn_attachments(NvnState& s);
/// A render target: the texture's level / layer the view selects, as an image at its storage address
Image* nvn_target(const nvn_tex* t, const nvn_view* v);
struct NvnDraw {
    bool indexed = false;
    int prim = 0, index_type = 0;   // NVNdrawPrimitive, NVNindexType
    uint32_t first = 0, count = 0, base_instance = 0, instances = 1;
    int32_t base_vertex = 0;
    uint64_t index_gpu = 0;
};
void nvn_draw(NvnState& s, const NvnDraw& d);                              // rx_draw.cpp
void nvn_dispatch(NvnState& s, uint32_t x, uint32_t y, uint32_t z);        // rx_compute.cpp
void nvn_clear_color(Image* im, const float rgba[4], uint32_t mask, const int32_t* scissor);           // rx_core.cpp
void nvn_clear_depth_stencil(Image* im, bool depth, float d, bool stencil, uint32_t sv, uint32_t smask,
                             const int32_t* scissor);
/// An NVN vertex attribute (nvn_va_st) in the encoding the vertex fetch and the WGSL backend take
Regs::VertexAttribute nvn_attribute(uint32_t va);                          // rx_draw.cpp
/// The attributes the vertex program reads from memory itself (see PullPlan), for the bound NVN vertex state
PullPlan nvn_pull_plan(const NvnState& s, const std::bitset<32>& loads);
/// The program pair of the bound program (vertex + fragment, translated together) for these target output types
/// (Shader::AttributeType of each color target); null: not drawable
ProgramPair* nvn_programs(const NvnState& s, const uint8_t out_types[8]);  // rx_shader.cpp
/// The c1 constants (the shader package's own constant bank) of the bound program's vertex (0) / fragment (1) stage
const std::vector<uint8_t>* nvn_c1(const NvnState& s, int stage);
/// The bound program's compute code key (its RX_DUMP_WGSL name nvn_cs_<key>), 0 without one.
uint64_t nvn_cs_key(const NvnState& s);
ShaderModule* nvn_compute_module(const NvnState& s, const std::array<uint8_t, 64>* ssbo_alias = nullptr,
                                 const std::array<uint32_t, 64>* ssbo_alias_words = nullptr);
/// A texture binding from NVN handles (texture handle; separate sampler handle, 0: the texture handle's sampler)
BoundTexture nvn_texture(const NvnState& s, uint64_t tex_handle, uint64_t samp_handle,
                         const Shader::TextureDescriptor& desc);           // rx_texture.cpp
void nvn_register_texture(int id, const nvn_tex& t, const nvn_view* v);
void nvn_register_sampler(int id, const nvn_samp_par& p);
void nvn_texture_released(uint64_t serial);
/// The registered texture of an id, its storage resolved to the texture living there now (false: none)
bool nvn_texture_of(int id, nvn_tex* t, nvn_view* v, bool* has_view);
// per handle, bumped when what the handle resolves to may change (rx_nvn.cpp; read without a lock)
uint32_t nvn_texture_version(int id);
uint32_t nvn_sampler_version(int id);
// the handle's texture was finalized: it names whatever texture lives at that storage now (may change any frame)
bool nvn_texture_follows_storage(int id);
bool nvn_sampler_of(int id, nvn_samp_par* p);   // rx_nvn.cpp: the registered sampler (false: none)
/// The driver's constant bank c0 (storage buffer table) / c2 (texture and sampler handle tables) of a stage as the
/// NVN driver lays them out (0xA00 bytes): what translated programs read for their bindings
void nvn_driver_bank(const NvnState& s, int bank, int stage, uint8_t out[0xA00]);

/// What a shader can read of a uniform buffer bound with `size` bytes: the console's driver binds whole 256-byte
/// blocks (RX_NVN_UBO_EXACT=1: only the bytes bound)
inline uint64_t nvn_ubo_size(uint64_t size) {
    static const bool exact = getenv("RX_NVN_UBO_EXACT") != nullptr;
    return exact ? size : (size + 255) & ~255ull;
}
/// One stage's constant banks as an NVN draw or dispatch sees them: c0 / c2 the driver's, c1 the package's, c3+ uniform buffers
uint32_t nvn_driver_bank_word(const NvnState& s, int bank, int stage, uint32_t offset);
/// Native programs: uniform buffers they read past this many bytes are bound in place as read-only storage, not
/// copied into each draw's uniform block (RX_NVN_UB_LIMIT=<bytes>, default 512; 0: as before). The translation
/// (RuntimeInfo::cbuf_ub_limit) and nvn_stage_resources lay the block out with it.
uint32_t nvn_ub_limit();
struct NvnBanks {
    uint8_t bank0[0xA00], bank2[0xA00];
    bool have0 = false, have2 = false;
    const std::vector<uint8_t>* c1 = nullptr;
    /// up to n bytes of constant bank `index` from `offset` (the rest zeros); the bank's size
    uint64_t read(const NvnState& s, int stage, uint32_t index, uint64_t offset, void* dst, uint64_t n) {
        if ((index == 0 || index == 2) && (n == 4 || n == 8) && !(offset & 3) && offset + n <= 0xA00) {
            // (a handle or a buffer address: the words themselves, not the whole bank built)
            const uint32_t w[2] = {nvn_driver_bank_word(s, (int)index, stage, (uint32_t)offset),
                                   n == 8 ? nvn_driver_bank_word(s, (int)index, stage, (uint32_t)offset + 4) : 0u};
            std::memcpy(dst, w, n);
            return 0xA00;
        }
        std::memset(dst, 0, n);
        if (index == 0 || index == 2) {
            uint8_t* b = index == 0 ? bank0 : bank2;
            bool& have = index == 0 ? have0 : have2;
            if (!have) nvn_driver_bank(s, (int)index, stage, b), have = true;
            if (offset < 0xA00) std::memcpy(dst, b + offset, std::min<uint64_t>(n, 0xA00 - offset));
            return 0xA00;
        }
        if (index == 1) {
            if (c1 && offset < c1->size()) std::memcpy(dst, c1->data() + offset, std::min<uint64_t>(n, c1->size() - offset));
            return c1 ? c1->size() : 0;
        }
        if (index >= 3 && index < 3 + 16) {
            const nvn_p_bind_buf& u = s.ubo[stage][index - 3];
            const uint64_t size = nvn_ubo_size(u.size);
            if (u.addr && offset < size) gpu_read(u.addr + offset, dst, std::min<uint64_t>(n, size - offset));
            return u.addr ? size : 0;
        }
        return 0;
    }
};
uint32_t nvn_driver_bank_word(const NvnState& s, int bank, int stage, uint32_t offset);

/// What a frame spent on the usual causes of slow frames (logged for frames over 18 ms with HOS_FRAME_HIST).
struct FrameSpike {
    int64_t decode_ns, wait_ns, trim_ns, upload_bytes, bind_groups, draws, translations;
    int64_t translate_ns, pipeline_ns;   // one-off draws: programs translated and pipelines made in place
};
extern FrameSpike spike;
/// Set on the renderer's helper threads (translation, texture decoding): count() and log_limited() are the render
/// thread's alone and do nothing there.
extern thread_local bool rx_side_thread;
/// Textures decoded on the worker since the last call go into the decoded map (render thread, each frame).
void install_decoded_textures();
void trim_decoded_textures();
uint64_t image_bytes(const Image& im);
void decoded_texture_stats(size_t* n, uint64_t* bytes);
size_t buffer_segments();
void retire_unused_segments();
bool segments_retired_pending();
std::vector<WGPUBuffer> take_retired_segment_buffers();
void release_buffer_later(WGPUBuffer b);
size_t pipeline_count();
void wgsl_text_stats(size_t* n, uint64_t* bytes);


// ---- deferred command encoding (rx_record.cpp) ----
struct Recording;
bool encoding_deferred();
WGPUCommandEncoder fake_encoder();
Recording* recording_take();
void recording_replay(Recording* r, WGPUCommandEncoder e);
/// A replay in parts (words [from, to): each part whole passes), for encoders finished on several threads. With a
/// GPU timer (RX_GPU_TIMING), each pass writes timestamps into it.
struct GpuTimer;
void recording_replay(Recording* r, WGPUCommandEncoder e, size_t from, size_t to, GpuTimer* t = nullptr);
/// RX_GPU_TIMING=1 (profiling): each pass's GPU time, summed by what it draws into and logged every 5 s. begin_pass
/// tags the next pass (its first target's address; its size and formats); a timer is taken per encoder, resolved into
/// it before its finish and read after its submit.
bool gpu_timing();
void ce_pass_tag(uint64_t va, uint64_t desc);
GpuTimer* gpu_timer_take();
void gpu_timer_resolve(GpuTimer* t, WGPUCommandEncoder e);
void gpu_timer_read(GpuTimer* t);
/// Up to `parts` ranges of about equal size, split where passes end (one range when it is small).
std::vector<std::pair<size_t, size_t>> recording_parts(const Recording* r, uint32_t parts, size_t min_words);
void recording_free(Recording* r);
WGPURenderPassEncoder ce_begin_render_pass(WGPUCommandEncoder e, const WGPURenderPassDescriptor* pd);
void ce_copy_buffer_to_buffer(WGPUCommandEncoder e, WGPUBuffer src, uint64_t src_offset, WGPUBuffer dst,
                              uint64_t dst_offset, uint64_t size);
void ce_copy_texture_to_texture(WGPUCommandEncoder e, const WGPUTexelCopyTextureInfo* src,
                                const WGPUTexelCopyTextureInfo* dst, const WGPUExtent3D* extent);
void ce_copy_texture_to_buffer(WGPUCommandEncoder e, const WGPUTexelCopyTextureInfo* src,
                               const WGPUTexelCopyBufferInfo* dst, const WGPUExtent3D* extent);
WGPUComputePassEncoder ce_begin_compute_pass(WGPUCommandEncoder e);
void rp_end(WGPURenderPassEncoder p);
void rp_set_viewport(WGPURenderPassEncoder p, float x, float y, float w, float h, float zmin, float zmax);
void rp_set_pipeline(WGPURenderPassEncoder p, WGPURenderPipeline pipeline);
void rp_set_bind_group(WGPURenderPassEncoder p, uint32_t index, WGPUBindGroup g, size_t n, const uint32_t* offsets);
void rp_set_vertex_buffer(WGPURenderPassEncoder p, uint32_t slot, WGPUBuffer b, uint64_t offset, uint64_t size);
void rp_set_index_buffer(WGPURenderPassEncoder p, WGPUBuffer b, WGPUIndexFormat format, uint64_t offset, uint64_t size);
void rp_set_scissor(WGPURenderPassEncoder p, uint32_t x, uint32_t y, uint32_t w, uint32_t h);
void rp_set_blend_constant(WGPURenderPassEncoder p, const WGPUColor* c);
void rp_set_stencil_reference(WGPURenderPassEncoder p, uint32_t ref);
void rp_draw(WGPURenderPassEncoder p, uint32_t vertices, uint32_t instances, uint32_t first, uint32_t first_instance);
void rp_draw_indexed(WGPURenderPassEncoder p, uint32_t indices, uint32_t instances, uint32_t first_index,
                     int32_t base_vertex, uint32_t first_instance);
void cp_set_pipeline(WGPUComputePassEncoder p, WGPUComputePipeline pipeline);
void cp_set_bind_group(WGPUComputePassEncoder p, uint32_t index, WGPUBindGroup g, size_t n, const uint32_t* offsets);
void cp_set_bind_group_desc(WGPUComputePassEncoder p, uint32_t index, WGPUBindGroupLayout layout,
                            const WGPUBindGroupEntry* entries, size_t n);
void cp_dispatch(WGPUComputePassEncoder p, uint32_t x, uint32_t y, uint32_t z);
void cp_end(WGPUComputePassEncoder p);

} // namespace rx

#include "rx_web_mt.h"   // (last: its macros wrap the WebGPU calls of the files that include this)

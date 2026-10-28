#include <bit>
// Shaders: a draw's Maxwell programs (read from guest memory) through suyu's shader recompiler and the WGSL
// backend, into WebGPU shader modules. The vertex and pixel programs are translated together: the vertex stage's
// output interpolation and the pixel stage's inputs must agree.
#include <atomic>
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_set>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <mutex>
#include <iterator>
#include <optional>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/wgsl/emit_wgsl.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/texture_cache/format_lookup_table.h"

#include "rx_internal.h"

namespace rx {
namespace {

Shader::Profile current_shader_profile() {
    Shader::Profile profile{};
    if (dev.is_intel || dev.is_intel_arc) {
        profile.need_fastmath_off = true;      // Prevent NaN/Inf precision issues on Intel Arc ALUs
        profile.has_broken_spirv_clamp = true; // Intel driver clamp workaround
        profile.has_gl_bool_ref_bug = true;    // Workaround Intel boolean reference bugs
    }
    return profile;
}

Shader::TextureType convert_texture_type(const Tegra::Texture::TICEntry& entry) {
    using T = Tegra::Texture::TextureType;
    switch (entry.texture_type) {
    case T::Texture1D: return Shader::TextureType::Color1D;
    case T::Texture2D:
    case T::Texture2DNoMipmap:
        return entry.normalized_coords ? Shader::TextureType::Color2D : Shader::TextureType::Color2DRect;
    case T::Texture3D: return Shader::TextureType::Color3D;
    case T::TextureCubemap: return Shader::TextureType::ColorCube;
    case T::Texture1DArray: return Shader::TextureType::ColorArray1D;
    case T::Texture2DArray: return Shader::TextureType::ColorArray2D;
    case T::Texture1DBuffer: return Shader::TextureType::Buffer;
    case T::TextureCubeArray: return Shader::TextureType::ColorArrayCube;
    default: return Shader::TextureType::Color2D;
    }
}

/// suyu's GraphicsEnvironment over guest memory: the program at program_region + offset, the channel's constant
/// buffers, the texture headers of the bound pool.
class GuestEnvironment final : public Shader::Environment {
public:
    GuestEnvironment(const Regs& r, const Channel& ch, Shader::Stage stage_, uint32_t stage_index_, uint32_t offset)
        : regs{r}, channel{ch}, stage_index{stage_index_}, base{r.program_region.Address()} {
        stage = stage_;
        start_address = offset;
        code.resize(0x10000 / 8);
        gpu_read(base + offset, code.data(), code.size() * 8);
        std::memcpy(&sph, code.data(), sizeof(sph));
        local_memory = static_cast<uint32_t>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
        texture_bound = r.bindless_texture_const_buffer_slot;
        is_proprietary_driver = texture_bound == 2;
    }
    u64 ReadInstruction(u32 address) override {
        const uint64_t rel = address - start_address;
        if (rel / 8 < code.size()) return code[rel / 8];
        return gpu_get<uint64_t>(base + address);
    }
    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override {
        const ConstBuffer& cb = channel.cbufs[stage_index][cbuf_index % 18];
        return cbuf_offset < cb.size ? gpu_get<uint32_t>(cb.address + cbuf_offset) : 0;
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return convert_texture_type(tic(handle));
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        const auto e = tic(handle);
        return static_cast<Shader::TexturePixelFormat>(VideoCommon::PixelFormatFromTextureInfo(
            e.format, e.r_type, e.g_type, e.b_type, e.a_type, e.srgb_conversion));
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        return VideoCore::Surface::IsPixelFormatInteger(static_cast<PixelFormat>(ReadTexturePixelFormat(handle)));
    }
    u32 ReadViewportTransformState() override {
        return regs.viewport_scale_offset_enabled;
    }
    u32 TextureBoundBuffer() const override {
        return texture_bound;
    }
    u32 LocalMemorySize() const override {
        return local_memory;
    }
    u32 SharedMemorySize() const override {
        return 0;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {1, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    Tegra::Texture::TICEntry tic(u32 handle) {
        const bool via_header = regs.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding;
        return read_tic(regs, Tegra::Texture::TexturePair(handle, via_header).first);
    }
    const Regs& regs;
    const Channel& channel;
    uint32_t stage_index;
    uint64_t base;
    std::vector<uint64_t> code;
    uint32_t local_memory = 0, texture_bound = 0;
};

/// suyu's ComputeEnvironment over guest memory: the program at code_loc + program_start (no program header),
/// constant buffers from the launch descriptor.
class ComputeGuestEnvironment final : public Shader::Environment {
public:
    explicit ComputeGuestEnvironment(const ComputeLaunch& l) : launch{l} {
        stage = Shader::Stage::Compute;
        start_address = l.program_start;
        code.resize(0x10000 / 8);
        gpu_read(l.code_base + l.program_start, code.data(), code.size() * 8);
        is_proprietary_driver = l.texture_bound == 2;
    }
    u64 ReadInstruction(u32 address) override {
        const uint64_t rel = address - start_address;
        if (rel / 8 < code.size()) return code[rel / 8];
        return gpu_get<uint64_t>(launch.code_base + address);
    }
    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override {
        const auto& cb = launch.cbufs[cbuf_index % 8];
        return cb.enabled && cbuf_offset < cb.size ? gpu_get<uint32_t>(cb.address + cbuf_offset) : 0;
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return convert_texture_type(tic(handle));
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        const auto e = tic(handle);
        return static_cast<Shader::TexturePixelFormat>(VideoCommon::PixelFormatFromTextureInfo(
            e.format, e.r_type, e.g_type, e.b_type, e.a_type, e.srgb_conversion));
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        return VideoCore::Surface::IsPixelFormatInteger(static_cast<PixelFormat>(ReadTexturePixelFormat(handle)));
    }
    u32 ReadViewportTransformState() override {
        return 0;
    }
    u32 TextureBoundBuffer() const override {
        return launch.texture_bound;
    }
    u32 LocalMemorySize() const override {
        return launch.local_size;
    }
    u32 SharedMemorySize() const override {
        return launch.shared_size;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {launch.block[0], launch.block[1], launch.block[2]};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    Tegra::Texture::TICEntry tic(u32 handle) {
        return read_tic(*launch.tex_regs, Tegra::Texture::TexturePair(handle, launch.linked_tsc).first);
    }
    const ComputeLaunch& launch;
    std::vector<uint64_t> code;
};

struct Translated {
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
    Shader::ObjectPool<Shader::IR::Inst> insts{8192};
    Shader::ObjectPool<Shader::IR::Block> blocks{32};
    Shader::IR::Program program;
};

thread_local bool on_translation_worker;   // (the worker: no count(), no log_limited(): not thread-safe)
/// Marks the thread as a translation worker while it lives (build_pair's helper threads: pool threads)
struct AsWorker {
    bool was_worker = on_translation_worker, was_side = rx_side_thread;
    AsWorker() { on_translation_worker = rx_side_thread = true; }
    ~AsWorker() {
        on_translation_worker = was_worker;
        rx_side_thread = was_side;
    }
};

std::unique_ptr<Translated> translate(const Regs& r, const Channel& ch, Shader::Stage stage, uint32_t stage_index,
                                      uint32_t offset) {
    struct TranslateTimer {
        int64_t t0 = hos_now_ns();
        ~TranslateTimer() {
            if (!on_translation_worker) count("t_translate", hos_now_ns() - t0);
        }
    } translate_timer;
    auto t = std::make_unique<Translated>();
    GuestEnvironment env{r, ch, stage, stage_index, offset};
    Shader::Maxwell::Flow::CFG cfg{env, t->flow_blocks,
                                   env.StartAddress() + static_cast<u32>(sizeof(Shader::ProgramHeader))};
    Shader::HostTranslateInfo host{};
    host.support_float64 = false;
    host.support_int64 = false;
    host.min_ssbo_alignment = dev.limits.minStorageBufferOffsetAlignment;
    // RX_DUMP_IR=<dir> (debugging): each program's IR before constant folding, as <dir>/<vs|fs>_<address>.pre.txt
    static const char* ir_dir = getenv("RX_DUMP_IR");
    if (ir_dir) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/%s_%010llx", ir_dir, stage == Shader::Stage::Fragment ? "fs" : "vs",
                      (unsigned long long)(r.program_region.Address() + offset));
        Shader::Maxwell::ir_dump_path = path;
    }
    t->program = Shader::Maxwell::TranslateProgram(t->insts, t->blocks, env, cfg, host);
    if (ir_dir) Shader::Maxwell::ir_dump_path.clear();
    return t;
}

/// What the stage programs at these addresses use, from their first translation (for the pair keys).
struct StageInfo {
    std::bitset<32> loads, stores;
    bool failed = false;
};
std::unordered_map<uint64_t, StageInfo> stage_infos;   // by StageKey hash
std::unordered_map<uint64_t, std::unique_ptr<ProgramPair>> pairs;
std::mutex prog_mu;   // stage_infos, pairs and the WGSL text ids: the translation workers add to them
std::atomic<size_t> wgsl_texts;
std::atomic<uint64_t> wgsl_bytes;
struct InternedText {
    uint32_t id = 0;
    uint64_t check = 0, cache_hash = 0;
    std::string text;
};
std::unordered_map<uint64_t, InternedText> interned_texts;

uint64_t stage_hash(uint64_t base, uint32_t offset) {
    // (not cached by address and page writes: tried 2026-10-13, programs went stale somehow and gameplay broke.
    // Cached by address and validated by comparing the bytes themselves: a memcmp is several times cheaper than
    // hashing them, and exact)
    constexpr size_t kBytes = 0x50 + 0x400;
    struct Entry {
        uint8_t bytes[kBytes];
        uint64_t hash;
    };
    static std::unordered_map<uint64_t, std::unique_ptr<Entry>> cache;
    const uint64_t va = base + offset;
    static const bool no_cache = env_flag("RX_NO_STAGE_CACHE");   // (debugging)
    if (no_cache) {
        uint8_t b[kBytes];
        gpu_read(va, b, kBytes);
        return hash_bytes(b, kBytes, va);
    }
    uint64_t avail = 0;
    const uint8_t* host = hos_gpu_ptr(va, &avail);
    uint8_t buf[kBytes];
    const uint8_t* now = host;
    if (!host || avail < kBytes) {
        gpu_read(va, buf, kBytes);
        now = buf;
    }
    auto& e = cache[va];
    if (e && std::memcmp(e->bytes, now, kBytes) == 0) {
        static const bool verify = env_flag("RX_VERIFY_STAGE_CACHE");   // (debugging)
        if (verify) {
            uint8_t b[kBytes];
            gpu_read(va, b, kBytes);
            const uint64_t fresh = hash_bytes(b, kBytes, va);
            static int logged;
            if (fresh != e->hash && logged++ < 20)
                hos_log("rx: stage cache %#llx: cached %016llx fresh %016llx (memcmp of the copy %d)",
                        (unsigned long long)va, (unsigned long long)e->hash, (unsigned long long)fresh,
                        std::memcmp(b, e->bytes, kBytes));
        }
        return e->hash;
    }
    if (!e) e = std::make_unique<Entry>();
    std::memcpy(e->bytes, now, kBytes);
    e->hash = hash_bytes(e->bytes, kBytes, va);   // (the copy: the guest may be writing the live bytes)
    return e->hash;
}

Shader::AttributeType attribute_type(const Regs::VertexAttribute& a) {
    if (a.constant) return Shader::AttributeType::Float;
    switch (a.type) {
    case Regs::VertexAttribute::Type::SInt: return Shader::AttributeType::SignedInt;
    case Regs::VertexAttribute::Type::UInt: return Shader::AttributeType::UnsignedInt;
    case Regs::VertexAttribute::Type::SScaled: return Shader::AttributeType::SignedScaled;
    case Regs::VertexAttribute::Type::UScaled: return Shader::AttributeType::UnsignedScaled;
    default: return Shader::AttributeType::Float;
    }
}

Shader::AttributeType output_type(const Regs& r, uint32_t i) {
    if (i >= r.rt_control.count) return Shader::AttributeType::Float;
    const uint32_t target = r.rt_control.Map(i);
    if (target >= 8) return Shader::AttributeType::Float;
    const auto f = r.rt[target].format;
    if (f == Tegra::RenderTargetFormat::NONE) return Shader::AttributeType::Float;
    const PixelFormat pf = VideoCore::Surface::PixelFormatFromRenderTargetFormat(f);
    if (!VideoCore::Surface::IsPixelFormatInteger(pf)) return Shader::AttributeType::Float;
    return VideoCore::Surface::IsPixelFormatSignedInteger(pf) ? Shader::AttributeType::SignedInt
                                                              : Shader::AttributeType::UnsignedInt;
}

bool create_module(ShaderModule& m, const char* what, uint64_t key) {
    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    const std::string& wgsl = m.source();
    src.code = WGPUStringView{wgsl.data(), wgsl.size()};
    // the browser (Tint) enforces derivative uniformity, which wgpu (naga) does not: the translated programs sample
    // in control flow it cannot prove uniform
    std::string web_wgsl;
    if (RX_WEB) {
        web_wgsl = web_shader_source(wgsl);
        src.code = WGPUStringView{web_wgsl.data(), web_wgsl.size()};
    }
    WGPUShaderModuleDescriptor d{};
    d.nextInChain = &src.chain;
    static const bool times = env_flag("RX_MODULE_TIMES");
    const auto t0 = times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    scope_push();
    m.module = program_module(&d);
    const std::string error = scope_pop();
    if (times) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "rx: module %s %016llx %zu bytes %.1f ms\n", what, (unsigned long long)key,
                     wgsl.size(), ms);
        if (const char* dir = getenv("RX_DUMP_SLOW_WGSL"); dir && ms > 40) {
            char path[512];
            std::snprintf(path, sizeof path, "%s/slow_%s_%016llx.wgsl", dir, what, (unsigned long long)key);
            std::ofstream(path, std::ios::binary) << wgsl;
        }
    }
    if (error.empty()) return true;
    m.failed = true;
    log_limited("wgsl_error", 20, "rx: %s shader %016llx does not compile: %s", what, (unsigned long long)key,
                error.c_str());
    if (const char* dir = getenv("RX_DUMP_WGSL")) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/bad_%s_%016llx.wgsl", dir, what, (unsigned long long)key);
        std::ofstream(path, std::ios::binary) << wgsl;
    }
    return false;
}

} // namespace

ShaderModule* prewarm_stage_module(uint64_t h, const std::string& text, const char* what) {
    static std::deque<ShaderModule> modules;   // (live as long as the program pairs: modules are never released)
    std::lock_guard lk(prog_mu);
    const uint64_t check = hash_bytes(text.data(), text.size(), 0xc0ffee5ull);
    if (hash_bytes(text.data(), text.size(), 0x1d7e57ull) != h) return nullptr;   // (a damaged record)
    auto [it, inserted] = interned_texts.try_emplace(h);
    InternedText& t = it->second;
    if (inserted) {
        t.id = (uint32_t)interned_texts.size();
        t.check = check;
        t.text = text;
        wgsl_texts++;
        wgsl_bytes += t.text.size();
    } else if (t.check != check) {
        return nullptr;
    }
    ShaderModule& m = modules.emplace_back();
    m.text_id = t.id;
    m.text_hash = h;
    m.text = &t.text;
    m.what = what;
    m.key = h;
    return &m;
}

void release_module_text(ShaderModule& m) {
    if (!m.text_id || !m.text) return;
    std::lock_guard g(prog_mu);
    wgsl_bytes -= m.text->size();
    std::string().swap(*m.text);
}

bool ensure_module(ShaderModule& m) {
    // per module: WGSL parsing and validation (most of the compilers' time: 36 s of 42 at the Great Sky Island
    // spawn) runs on several compilers at once
    std::lock_guard lk(*m.creating);
    if (m.module || m.failed) return !m.failed;
    // and once per distinct WGSL: a program pair is per vertex + pixel program combination and vertex format, so
    // the same stage translates to the same text in many pairs (Great Sky Island spawn: 6954 modules, 773 distinct)
    struct Shared {
        std::mutex creating;
        bool done = false, ok = false;
        WGPUShaderModule module = nullptr;
    };
    static std::mutex cache_mutex;
    static std::unordered_map<uint32_t, std::shared_ptr<Shared>> by_text_id;
    static std::unordered_map<std::string, std::shared_ptr<Shared>> by_text;
    static const bool no_share = env_flag("RX_NO_MODULE_SHARING");
    if (no_share) return create_module(m, m.what, m.key);
    std::shared_ptr<Shared> s;
    {
        std::lock_guard g(cache_mutex);
        auto& e = m.text_id ? by_text_id[m.text_id] : by_text[m.source()];
        if (!e) e = std::make_shared<Shared>();
        s = e;
    }
    std::lock_guard sl(s->creating);
    if (!s->done) {
        s->ok = create_module(m, m.what, m.key);
        s->module = m.module;   // (modules live as long as the program pairs: never released)
        s->done = true;
        if (!RX_WEB) release_module_text(m);
        return s->ok;
    }
    m.module = s->module;
    m.failed = !s->ok;
    return s->ok;
}

uint8_t vertex_unpack(const Regs::VertexAttribute& a) {
    using U = Shader::Backend::WGSL::VertexUnpack;
    using S = Regs::VertexAttribute::Size;
    using T = Regs::VertexAttribute::Type;
    if (a.constant) return (uint8_t)U::None;
    if (a.size == S::Size_B10_G11_R11) return (uint8_t)U::B10G11R11_FLOAT;
    if (a.size != S::Size_A2_B10_G10_R10) return (uint8_t)U::None;
    switch (a.type) {
    case T::SNorm: return (uint8_t)U::A2B10G10R10_SNORM;
    case T::UInt: return (uint8_t)U::A2B10G10R10_UINT;
    case T::SInt: return (uint8_t)U::A2B10G10R10_SINT;
    case T::UScaled: return (uint8_t)U::A2B10G10R10_USCALED;
    case T::SScaled: return (uint8_t)U::A2B10G10R10_SSCALED;
    default: return (uint8_t)U::None;   // unorm: WebGPU's own format
    }
}

bool fetchable(const Regs::VertexAttribute& a, const Regs::VertexStream& vs);   // rx_draw.cpp

PullPlan pull_plan(const Regs& r, const std::bitset<32>& loads) {
    PullPlan plan;
    for (uint32_t i = 0; i < 32; i++) {
        if (!loads[i]) continue;
        const auto& a = r.vertex_attrib_format[i];
        const auto& vs = r.vertex_streams[a.buffer % 16];
        if (a.constant || !vs.IsEnabled() || fetchable(a, vs)) continue;
        uint32_t k = 0;
        while (k < plan.count && plan.streams[k] != a.buffer % 16) k++;
        if (k == plan.count) {
            if (plan.count == 3) continue;   // (more: the CPU converts them)
            plan.streams[plan.count++] = a.buffer % 16;
        }
        auto& p = plan.attr[i];
        p.enabled = true;
        p.stream = (uint8_t)k;
        p.instanced = r.vertex_stream_instances.IsInstancingEnabled(a.buffer % 16);
        p.attribute = a.hex;
        p.stride = vs.stride;
    }
    // more streams than vertex buffers (browsers allow 8; TOTK draws with 9): the last ones are pulled too
    // (RX_MAX_VERTEX_BUFFERS=<n>: as if the device allowed n, to test the browser's limit natively)
    static const uint32_t forced = getenv("RX_MAX_VERTEX_BUFFERS") ? (uint32_t)atoi(getenv("RX_MAX_VERTEX_BUFFERS")) : 0;
    const uint32_t max_buffers = forced ? forced : dev.limits.maxVertexBuffers ? dev.limits.maxVertexBuffers : 16;
    uint32_t fetched = 0;
    bool constants = false;
    for (uint32_t i = 0; i < 32; i++) {
        if (!loads[i]) continue;
        const auto& a = r.vertex_attrib_format[i];
        if (a.constant || !r.vertex_streams[a.buffer % 16].IsEnabled()) constants = true;
        else if (!plan.attr[i].enabled) fetched |= 1u << (a.buffer % 16);
    }
    uint32_t buffers = (uint32_t)std::popcount(fetched) + (constants ? 1u : 0u);
    for (int st = 15; st >= 0 && buffers > max_buffers && plan.count < 3; st--) {
        if (!(fetched >> st & 1)) continue;
        const uint32_t k = plan.count++;
        plan.streams[k] = (uint32_t)st;
        for (uint32_t i = 0; i < 32; i++) {
            const auto& a = r.vertex_attrib_format[i];
            if (!loads[i] || a.constant || a.buffer % 16 != (uint32_t)st || plan.attr[i].enabled) continue;
            auto& p = plan.attr[i];
            p.enabled = true;
            p.stream = (uint8_t)k;
            p.instanced = r.vertex_stream_instances.IsInstancingEnabled(st);
            p.attribute = a.hex;
            p.stride = r.vertex_streams[st].stride;
        }
        buffers--;
    }
    return plan;
}

ShaderModule* compute_module(const ComputeLaunch& l, const std::array<uint8_t, 64>* ssbo_alias,
                             const std::array<uint32_t, 64>* ssbo_alias_words) {
    static std::unordered_map<uint64_t, std::unique_ptr<ShaderModule>> modules;
    uint8_t head[0x400];
    gpu_read(l.code_base + l.program_start, head, sizeof head);
    uint64_t key = hash_bytes(head, sizeof head, l.code_base + l.program_start);
    const uint32_t shape[5] = {l.block[0], l.block[1], l.block[2], l.shared_size, l.local_size};
    key = hash_bytes(shape, sizeof shape, key);
    if (ssbo_alias) key = hash_bytes(ssbo_alias->data(), ssbo_alias->size(), key ^ 0xA11A5);
    if (ssbo_alias_words) key = hash_bytes(ssbo_alias_words->data(), ssbo_alias_words->size() * 4, key);
    auto it = modules.find(key);
    if (it != modules.end()) return it->second->failed ? nullptr : it->second.get();
    auto m = std::make_unique<ShaderModule>();
    try {
        struct TranslateTimer {
            int64_t t0 = hos_now_ns();
            ~TranslateTimer() { count("t_translate", hos_now_ns() - t0); }
        } translate_timer;
        Translated t;
        ComputeGuestEnvironment env{l};
        Shader::Maxwell::Flow::CFG cfg{env, t.flow_blocks, env.StartAddress()};
        Shader::HostTranslateInfo host{};
        host.support_float64 = false;
        host.support_int64 = false;
        host.min_ssbo_alignment = dev.limits.minStorageBufferOffsetAlignment;
        t.program = Shader::Maxwell::TranslateProgram(t.insts, t.blocks, env, cfg, host);
        Shader::Profile profile = current_shader_profile();
        Shader::RuntimeInfo rt{};
        Shader::Backend::Bindings b{};
        if (ssbo_alias) Shader::Backend::WGSL::ssbo_alias_next = *ssbo_alias;
        if (ssbo_alias_words) Shader::Backend::WGSL::ssbo_alias_words_next = *ssbo_alias_words;
        m->wgsl = Shader::Backend::WGSL::EmitWGSL(profile, rt, t.program, b);
        m->info = t.program.info;
    } catch (const std::exception& e) {
        log_limited("translate_cs", 30, "rx: compute program %#llx does not translate: %s",
                    (unsigned long long)(l.code_base + l.program_start), e.what());
        m->failed = true;
    }
    if (!m->failed) {
        if (const char* dir = getenv("RX_DUMP_WGSL")) {
            char path[512];
            std::snprintf(path, sizeof path, "%s/cs_%010llx.wgsl", dir,
                          (unsigned long long)(l.code_base + l.program_start));
            std::ofstream(path, std::ios::binary) << m->wgsl;
        }
        // (made by the pipeline cache's thread when a session before compiled the same text)
        const uint64_t th = hash_bytes(m->wgsl.data(), m->wgsl.size(), 0x1d7e57ull);
        if (WGPUShaderModule cached = native_cache_compute_module(th)) {
            m->module = cached;
            count("cs_module_cached");
        } else if (create_module(*m, "compute", key)) {
            native_cache_compute_text(th, m->wgsl);
        }
    }
    ShaderModule* raw = m.get();
    modules[key] = std::move(m);
    return raw->failed ? nullptr : raw;
}

/// RX_DEBUG_SHADOWS_LIT=1 (debugging): every depth-compare sample of the pixel programs reads 1 (lit).
void debug_compare_lit(std::string& wgsl) {
    for (const char* call : {"textureSampleCompareLevel(", "textureSampleCompare("}) {
        size_t at = 0;
        while ((at = wgsl.find(call, at)) != std::string::npos) {
            if (at >= 10 && wgsl.compare(at - 10, 10, "(0.0*(1.0)") == 0) {   // (already wrapped)
                at += std::strlen(call);
                continue;
            }
            size_t i = at + std::strlen(call), depth = 1;
            while (i < wgsl.size() && depth) {
                if (wgsl[i] == '(') depth++;
                else if (wgsl[i] == ')') depth--;
                i++;
            }
            const std::string inner = wgsl.substr(at, i - at);
            const std::string repl = "(0.0*(1.0)*" + inner + "+1.0)";
            wgsl.replace(at, i - at, repl);
            at += repl.size();
        }
    }
}

/// A pair's key: its two programs, the types of the attributes the vertex program reads and of the targets the pixel
/// program writes (known once each program was translated). Caller holds prog_mu.
uint64_t pair_key(const Regs& r, uint64_t vs_key, uint64_t fs_key, bool has_fs, bool* complete, bool* failed) {
    auto vs_info = stage_infos.find(vs_key);
    auto fs_info = has_fs ? stage_infos.find(fs_key) : stage_infos.end();
    *failed = (vs_info != stage_infos.end() && vs_info->second.failed) ||
              (fs_info != stage_infos.end() && fs_info->second.failed);
    uint64_t key = hash_bytes(&vs_key, 8, fs_key);
    if (vs_info != stage_infos.end()) {
        for (uint32_t i = 0; i < 32; i++)
            if (vs_info->second.loads[i]) {
                const uint8_t t[2] = {(uint8_t)attribute_type(r.vertex_attrib_format[i]),
                                      vertex_unpack(r.vertex_attrib_format[i])};
                key = hash_bytes(t, 2, key ^ i);
            }
        const PullPlan plan = pull_plan(r, vs_info->second.loads);
        for (uint32_t i = 0; i < 32; i++)
            if (plan.attr[i].enabled) {
                // (field by field: the struct's padding byte is whatever was on the stack, which differs between the
                // translation workers and the render thread: a pair translated on a worker was then never found)
                const auto& a = plan.attr[i];
                const uint32_t f[3] = {a.stream | (a.instanced ? 0x100u : 0u), a.attribute, a.stride};
                key = hash_bytes(f, sizeof f, key ^ (i + 200));
            }
    }
    if (fs_info != stage_infos.end())
        for (uint32_t i = 0; i < 8; i++) {
            const uint8_t t = (uint8_t)output_type(r, i);
            key = hash_bytes(&t, 1, key ^ (i + 100));
        }
    *complete = vs_info != stage_infos.end() && (!has_fs || fs_info != stage_infos.end());
    return key;
}

/// Translates a pair and makes its WGSL (prog_mu is held only to read and publish), once: null if it does not
/// translate. Runs on a translation worker with copies of the registers and channel, or on the render thread
/// (parallel: the pixel stage is translated and emitted on another thread meanwhile: a one-off draw waits for it;
/// RX_SERIAL_STAGES=1 off).
ProgramPair* build_pair(const Channel& ch, const Regs& r, uint64_t vs_key, uint64_t fs_key, bool parallel = false) {
    const uint64_t base = r.program_region.Address();
    const uint32_t vs_offset = r.pipelines[1].offset;
    const bool has_fs = r.pipelines[5].enable != 0;
    const uint32_t fs_offset = r.pipelines[5].offset;
    auto pair = std::make_unique<ProgramPair>();
    pair->has_fs = has_fs;
    uint64_t key = 0;
    try {
        static const bool serial_stages = env_flag("RX_SERIAL_STAGES");
        // (not in the browser: a pthread started there needs the page's main thread, which may be the one waiting)
        const bool split = !RX_WEB && parallel && has_fs && !serial_stages;
        std::unique_ptr<Translated> vs, fs;
        if (split) {
            // (the future's destructor waits for it: a vertex stage that throws leaves no thread behind)
            auto fs_job = std::async(std::launch::async, [&] {
                AsWorker as_worker;
                return translate(r, ch, Shader::Stage::Fragment, 4, fs_offset);
            });
            vs = translate(r, ch, Shader::Stage::VertexB, 0, vs_offset);
            fs = fs_job.get();
        } else {
            vs = translate(r, ch, Shader::Stage::VertexB, 0, vs_offset);
            if (has_fs) fs = translate(r, ch, Shader::Stage::Fragment, 4, fs_offset);
        }
        std::bitset<32> loads;
        {
            std::lock_guard lk(prog_mu);
            StageInfo& vi = stage_infos[vs_key];
            for (uint32_t i = 0; i < 32; i++) {
                vi.loads[i] = vs->program.info.loads.Generic(i);
                vi.stores[i] = vs->program.info.stores.Generic(i);
            }
            if (fs) {
                StageInfo& fi = stage_infos[fs_key];
                for (uint32_t i = 0; i < 32; i++) fi.loads[i] = fs->program.info.loads.Generic(i);
            }
            bool complete, failed;
            key = pair_key(r, vs_key, fs_key, has_fs, &complete, &failed);
            if (auto it = pairs.find(key); it != pairs.end()) return it->second->failed ? nullptr : it->second.get();
            loads = vi.loads;
        }
        Shader::Profile profile = current_shader_profile();
        Shader::Backend::WGSL::Options options{};
        Shader::RuntimeInfo vs_rt{};
        // VertexB consumes vertex fetch, not an earlier shader. An empty mask makes the backend replace every
        // attribute read with (0,0,0,1), collapsing all vertices. Match suyu's first-stage runtime configuration.
        vs_rt.previous_stage_stores.mask.set();
        const PullPlan plan = pull_plan(r, loads);
        for (uint32_t i = 0; i < 32; i++) {
            vs_rt.generic_input_types[i] = attribute_type(r.vertex_attrib_format[i]);
            options.vertex_unpack[i] = (Shader::Backend::WGSL::VertexUnpack)vertex_unpack(r.vertex_attrib_format[i]);
            auto& vp = options.vertex_pull[i];
            vp.enabled = plan.attr[i].enabled;
            vp.stream = plan.attr[i].stream;
            vp.instanced = plan.attr[i].instanced;
            vp.attribute = plan.attr[i].attribute;
            vp.stride = plan.attr[i].stride;
        }
        if (fs)
            for (uint32_t i = 0; i < 32; i++) options.output_interpolation[i] = fs->program.info.interpolation[i];
        auto emit_fs = [&] {
            Shader::RuntimeInfo fs_rt{};
            fs_rt.previous_stage_stores = vs->program.info.stores;
            for (uint32_t i = 0; i < 8; i++) fs_rt.color_output_types[i] = output_type(r, i);
            Shader::Backend::Bindings fb{};
            pair->fs.wgsl = Shader::Backend::WGSL::EmitWGSL(profile, fs_rt, fs->program, fb);
            static const bool shadows_lit = env_flag("RX_DEBUG_SHADOWS_LIT");
            if (shadows_lit) debug_compare_lit(pair->fs.wgsl);
            pair->fs.info = fs->program.info;
        };
        std::future<void> fs_emit;
        if (fs && split)   // (reads vs->program.info only)
            fs_emit = std::async(std::launch::async, [&] {
                AsWorker as_worker;
                emit_fs();
            });
        Shader::Backend::Bindings vb{};
        pair->vs.wgsl = Shader::Backend::WGSL::EmitWGSL(profile, vs_rt, vs->program, vb, options);
        pair->vs.info = vs->program.info;
        if (fs_emit.valid()) fs_emit.get();
        else if (fs) emit_fs();
    } catch (const std::exception& e) {
        static std::atomic<int> logged;
        if (logged++ < 40)
            hos_log("rx: programs vs %#llx / fs %#llx do not translate: %s", (unsigned long long)(base + vs_offset),
                    (unsigned long long)(base + fs_offset), e.what());
        std::lock_guard lk(prog_mu);
        stage_infos[vs_key].failed = true;   // (whichever failed: both stop being tried)
        if (has_fs) stage_infos[fs_key].failed = true;
        return nullptr;
    }
    // RX_WGSL_OVERRIDE=<dir> (debugging): a program's WGSL is replaced by <dir>/vs_<address>.wgsl or fs_<address>.wgsl
    // when that file exists (e.g. an RX_DUMP_WGSL file edited to output an intermediate value)
    if (const char* dir = getenv("RX_WGSL_OVERRIDE")) {
        auto override_from = [&](const char* kind, uint64_t address, std::string& wgsl) {
            char path[512];
            std::snprintf(path, sizeof path, "%s/%s_%010llx.wgsl", dir, kind, (unsigned long long)address);
            std::ifstream f(path, std::ios::binary);
            if (!f) return;
            wgsl.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            hos_log("rx: %s program %#llx: WGSL from %s", kind, (unsigned long long)address, path);
        };
        override_from("vs", base + vs_offset, pair->vs.wgsl);
        if (has_fs) override_from("fs", base + fs_offset, pair->fs.wgsl);
    }
    if (const char* dir = getenv("RX_DUMP_WGSL")) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/vs_%010llx.wgsl", dir, (unsigned long long)(base + vs_offset));
        std::ofstream(path, std::ios::binary) << pair->vs.wgsl;
        if (has_fs) {
            std::snprintf(path, sizeof path, "%s/fs_%010llx.wgsl", dir, (unsigned long long)(base + fs_offset));
            std::ofstream(path, std::ios::binary) << pair->fs.wgsl;
        }
    }
    // (the modules are created by the pipeline compilers: ensure_module)
    pair->vs.what = "vertex";
    pair->vs.key = vs_key;
    pair->fs.what = "pixel";
    pair->fs.key = fs_key;
    pair->failed = false;
    std::lock_guard lk(prog_mu);
    auto intern = [](ShaderModule& m) {
        const uint64_t h = hash_bytes(m.wgsl.data(), m.wgsl.size(), 0x1d7e57ull);
        const uint64_t check = hash_bytes(m.wgsl.data(), m.wgsl.size(), 0xc0ffee5ull);
        auto [it, inserted] = interned_texts.try_emplace(h);
        InternedText& t = it->second;
        if (inserted) {
            t.id = (uint32_t)interned_texts.size();
            t.check = check;
            t.text = std::move(m.wgsl);
            if (RX_WEB) t.cache_hash = pipecache_text_hash(t.text);
            else native_cache_text(h, t.text);
            wgsl_texts++;
            wgsl_bytes += t.text.size();
        } else if (t.check != check) {
            return;
        }
        m.text_id = t.id;
        m.text_hash = h;
        m.cache_hash = t.cache_hash;
        m.text = &t.text;
        std::string().swap(m.wgsl);
    };
    intern(pair->vs);
    if (has_fs) intern(pair->fs);
    if (auto it = pairs.find(key); it != pairs.end()) return it->second.get();   // (another worker was first)
    ProgramPair* raw = pair.get();
    pairs[key] = std::move(pair);
    return raw;
}

/// New programs are translated by two workers while the draws that need them are skipped (as for pipelines still
/// compiling): translating on the render thread was most of the stutter after a load. A job carries copies of the
/// registers and the channel (constant buffer bindings). RX_SYNC_TRANSLATE=1: on the render thread, as before.
struct TranslationJob {
    std::unique_ptr<Regs> regs;
    Channel ch;
    uint64_t vs_key = 0, fs_key = 0, pending = 0;
};
std::mutex job_mu;
std::condition_variable job_cv;
std::deque<TranslationJob> translation_jobs;
std::unordered_set<uint64_t> translations_pending;

void translation_worker() {
    on_translation_worker = true;
    rx_side_thread = true;
    background_thread();
    for (;;) {
        TranslationJob j;
        {
            std::unique_lock lk(job_mu);
            job_cv.wait(lk, [] { return !translation_jobs.empty(); });
            j = std::move(translation_jobs.front());
            translation_jobs.pop_front();
        }
        build_pair(j.ch, *j.regs, j.vs_key, j.fs_key);
        std::lock_guard lk(job_mu);
        translations_pending.erase(j.pending);
    }
}

void queue_translation(const Channel& ch, const Regs& r, uint64_t vs_key, uint64_t fs_key) {
    // (one job for the same programs and formats at a time)
    uint32_t ingredients[32 + 16 + 9];
    for (uint32_t i = 0; i < 32; i++) ingredients[i] = r.vertex_attrib_format[i].hex;
    for (uint32_t i = 0; i < 16; i++)
        ingredients[32 + i] = (r.vertex_streams[i].stride & 0xFFF) | (r.vertex_streams[i].IsEnabled() ? 1u << 12 : 0u) |
                              (r.vertex_stream_instances.IsInstancingEnabled(i) ? 1u << 13 : 0u);
    for (uint32_t i = 0; i < 8; i++) ingredients[48 + i] = (uint32_t)r.rt[i].format;
    std::memcpy(&ingredients[56], &r.rt_control, 4);
    const uint64_t pending = hash_bytes(ingredients, sizeof ingredients, hash_bytes(&vs_key, 8, fs_key));
    std::lock_guard lk(job_mu);
    if (!translations_pending.insert(pending).second) return;
    count("translation_jobs");

    static bool started;
    if (!started) {
        started = true;
        for (int i = 0; i < 2; i++) std::thread(translation_worker).detach();
    }
    TranslationJob j;
    j.regs = std::make_unique<Regs>(r);
    j.ch = ch;
    j.vs_key = vs_key;
    j.fs_key = fs_key;
    j.pending = pending;
    translation_jobs.push_back(std::move(j));
    job_cv.notify_one();
}

ProgramPair* programs_for_draw_uncached(Channel& ch, const Regs& r) {
    if (r.pipelines[0].enable) {
        log_limited("vertex_a", 4, "rx: dual vertex programs (VertexA) not drawn yet");
        return nullptr;
    }
    for (int i = 2; i <= 4; i++)
        if (r.pipelines[i].enable) {
            log_limited("tess_geom", 4, "rx: tessellation/geometry program (stage %d): WebGPU has none", i);
            return nullptr;
        }
    const uint64_t base = r.program_region.Address();
    const bool has_fs = r.pipelines[5].enable != 0;
    const uint64_t vs_key = stage_hash(base, r.pipelines[1].offset);
    const uint64_t fs_key = has_fs ? stage_hash(base, r.pipelines[5].offset) : 0;
    {
        std::lock_guard lk(prog_mu);
        bool complete, failed;
        const uint64_t key = pair_key(r, vs_key, fs_key, has_fs, &complete, &failed);
        if (failed) return nullptr;
        if (complete)
            if (auto it = pairs.find(key); it != pairs.end()) return it->second->failed ? nullptr : it->second.get();
    }
    static const bool sync = env_flag("RX_SYNC_TRANSLATE");
    if (sync || frame_dump_active() || draw_must_run) {
        const int64_t t0 = hos_now_ns();
        ProgramPair* p = build_pair(ch, r, vs_key, fs_key, /*parallel=*/true);
        spike.translate_ns += hos_now_ns() - t0;
        return p;
    }
    queue_translation(ch, r, vs_key, fs_key);
    spike.translations++;
    count("program_translating");
    return nullptr;
}

/// Consecutive draws mostly use the same programs with the same vertex and target formats: a memo keyed by exactly
/// the registers the pair's key is made from (program offsets, attribute formats, stream strides/enables/instancing,
/// target formats and mapping), still checked against the programs' code bytes (RX_NO_PROGRAM_MEMO=1: off).
ProgramPair* programs_for_draw(Channel& ch, const Regs& r) {
    static const bool no_memo = env_flag("RX_NO_PROGRAM_MEMO");
    if (no_memo || frame_dump_active()) return programs_for_draw_uncached(ch, r);
    struct State {
        uint64_t base;
        uint32_t vs, fs, enables;
        uint32_t attribs[32];
        uint32_t streams[16];
        uint32_t targets[8];
        uint32_t rt_control;
    } st;
    std::memset(&st, 0, sizeof st);
    st.base = r.program_region.Address();
    st.vs = r.pipelines[1].offset;
    st.fs = r.pipelines[5].offset;
    for (uint32_t i = 0; i < 6; i++) st.enables |= (r.pipelines[i].enable ? 1u : 0u) << i;
    for (uint32_t i = 0; i < 32; i++) st.attribs[i] = r.vertex_attrib_format[i].hex;
    for (uint32_t i = 0; i < 16; i++) {
        const auto& vs = r.vertex_streams[i];
        st.streams[i] = (vs.stride & 0xFFF) | (vs.IsEnabled() ? 1u << 12 : 0u) |
                        (r.vertex_stream_instances.IsInstancingEnabled(i) ? 1u << 13 : 0u);
    }
    for (uint32_t i = 0; i < 8; i++) st.targets[i] = (uint32_t)r.rt[i].format;
    std::memcpy(&st.rt_control, &r.rt_control, sizeof st.rt_control);
    constexpr size_t kCode = 0x50 + 0x400;   // (as stage_hash)
    struct Memo {
        State st;
        ProgramPair* pair = nullptr;
        uint64_t seq = 0;   // memory_seq() when the code was last compared
        uint8_t code[2][kCode];
    };
    static std::unique_ptr<Memo[]> memo(new Memo[2048]);
    const uint64_t h = hash_bytes(&st, sizeof st);
    Memo& m = memo[h & 2047];
    const bool has_fs = (st.enables >> 5) & 1;
    auto code_now = [&](uint32_t offset, uint8_t* scratch) -> const uint8_t* {
        uint64_t avail = 0;
        const uint8_t* p = hos_gpu_ptr(st.base + offset, &avail);
        if (p && avail >= kCode) return p;
        gpu_read(st.base + offset, scratch, kCode);
        return scratch;
    };
    // the code bytes need no compare when no page of them was written since they last matched (CPU writes, applied
    // at each submission, and the GPU's own writes both bump the pages' sequence numbers): the compares were ~3% of
    // the render thread
    static const bool always_compare = env_flag("RX_PROGRAM_MEMO_COMPARE");
    const uint64_t seq_now = memory_seq();
    if (m.pair && m.seq && seq_now && !always_compare && !std::memcmp(&m.st, &st, sizeof st)) {
        const uint64_t gvs = hos_gpu_translate(st.base + st.vs), gfs = has_fs ? hos_gpu_translate(st.base + st.fs) : 0;
        if (gvs && !memory_written_since(gvs, kCode, m.seq) && (!has_fs || (gfs && !memory_written_since(gfs, kCode, m.seq)))) {
            count("program_memo_hits");
            return m.pair;
        }
    }
    uint8_t scratch[2][kCode];
    const uint8_t* vs_code = code_now(st.vs, scratch[0]);
    const uint8_t* fs_code = has_fs ? code_now(st.fs, scratch[1]) : nullptr;
    if (m.pair && !std::memcmp(&m.st, &st, sizeof st) && !std::memcmp(m.code[0], vs_code, kCode) &&
        (!has_fs || !std::memcmp(m.code[1], fs_code, kCode))) {
        count("program_memo_hits");
        m.seq = seq_now;
        return m.pair;
    }
    ProgramPair* pair = programs_for_draw_uncached(ch, r);
    if (pair) {   // (failures are not kept: they are rare and retried)
        m.st = st;
        m.pair = pair;
        m.seq = seq_now;
        std::memcpy(m.code[0], vs_code, kCode);
        if (has_fs) std::memcpy(m.code[1], fs_code, kCode);
    }
    return pair;
}

void wgsl_text_stats(size_t* n, uint64_t* bytes) {
    *n = wgsl_texts.load();
    *bytes = wgsl_bytes.load();
}

// ---- the native NVN layer: programs from NVN shader packages ----
// A stage of an NVN program is a control section (0x880 bytes, CPU memory: code area length at +0x704, the c1
// constants' offset in the area at +0x700, the NVN stage at +0x714, a compute stage's workgroup size at +0x748) and a
// code area (GPU memory): an NVN header of 0x30 bytes, then the Maxwell program (a graphics stage: its header and its
// instructions), then the c1 constants (indirect branch tables). Translated with the same recompiler as any Maxwell
// program, answering its questions from NVN state: c0 / c2 from the driver banks (nvn_driver_bank), c1 from the code
// area, user uniform buffers (c3-c16: slots 0-13) from their memory, texture types from the registered textures.
namespace {

Shader::TextureType nvn_texture_type(int target) {
    switch (target) {
    case NVN_TEXTURE_TARGET_1D: return Shader::TextureType::Color1D;
    case NVN_TEXTURE_TARGET_3D: return Shader::TextureType::Color3D;
    case NVN_TEXTURE_TARGET_1D_ARRAY: return Shader::TextureType::ColorArray1D;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
        return Shader::TextureType::ColorArray2D;
    case NVN_TEXTURE_TARGET_RECTANGLE: return Shader::TextureType::Color2DRect;
    case NVN_TEXTURE_TARGET_CUBEMAP: return Shader::TextureType::ColorCube;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: return Shader::TextureType::ColorArrayCube;
    case NVN_TEXTURE_TARGET_BUFFER: return Shader::TextureType::Buffer;
    default: return Shader::TextureType::Color2D;
    }
}

/// The texture type an NVN texture handle names (its registration's view target, else its texture's)
Shader::TextureType nvn_handle_type(uint32_t handle) {
    nvn_tex t;
    nvn_view v;
    bool has_view = false;
    if (!handle || !nvn_texture_of((int)(handle & 0xFFFFF), &t, &v, &has_view)) return Shader::TextureType::Color2D;
    return nvn_texture_type(has_view && v.set_target ? v.target : t.d.target);
}

PixelFormat nvn_handle_format(uint32_t handle) {
    nvn_tex t;
    nvn_view v;
    bool has_view = false;
    if (!handle || !nvn_texture_of((int)(handle & 0xFFFFF), &t, &v, &has_view)) return PixelFormat::A8B8G8R8_UNORM;
    const PixelFormat pf = nvn_pixel_format(has_view && v.set_format ? v.format : t.d.format);
    return pf == PixelFormat::Invalid ? PixelFormat::A8B8G8R8_UNORM : pf;
}

/// A stage's package, read from guest memory
struct NvnCode {
    int stage = -1;          // NVN stage
    uint64_t data = 0, key = 0;
    std::vector<uint64_t> code;   // the code area from its start (the program at 0x30: instruction bundles are aligned
                                  // in memory, and the recompiler finds their control words by address)
    std::vector<uint8_t> c1;
    uint32_t local[3]{};
    uint32_t shared = 0, local_memory = 0;   // compute: workgroup shared memory, local memory bytes
    bool ok = false;
};

NvnCode read_nvn_code(const nvn_stage_data& sd) {
    NvnCode c;
    uint8_t ctl[0x880];
    if (!sd.control || !sd.data || !gpu_read(sd.control, ctl, sizeof ctl)) return c;
    uint32_t c1o, len, st;
    std::memcpy(&c1o, ctl + 0x700, 4);
    std::memcpy(&len, ctl + 0x704, 4);
    std::memcpy(&st, ctl + 0x714, 4);
    if (len <= 0x30 || len > (16u << 20) || c1o > len || st > 5) {
        log_limited("nvn_code", 8, "rx: NVN shader package at %#llx: code area %#x, c1 at %#x, stage %u",
                    (unsigned long long)sd.data, len, c1o, st);
        return c;
    }
    std::vector<uint8_t> area(len);
    if (!gpu_read(sd.data, area.data(), len)) return c;
    c.key = hash_bytes(area.data(), len, hash_bytes(ctl, sizeof ctl));
    // RX_NVN_CODE_LOG=1 (debugging): the first packages' control fields and first words
    if (static const bool log_code = env_flag("RX_NVN_CODE_LOG"); log_code) {
        static int n;
        if (n++ < 24) {
            uint32_t w[24];
            std::memcpy(w, area.data(), std::min<size_t>(sizeof w, len));
            hos_log("rx: NVN package %016llx stage %u len %#x c1 %#x data %#llx: %08x %08x %08x %08x %08x %08x %08x %08x "
                    "%08x %08x %08x %08x | %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x",
                    (unsigned long long)c.key, st, len, c1o, (unsigned long long)sd.data, w[0], w[1], w[2], w[3], w[4],
                    w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15], w[16], w[17], w[18], w[19],
                    w[20], w[21], w[22], w[23]);
        }
    }
    c.stage = (int)st;
    c.data = sd.data;
    const uint32_t end = c1o > 0x30 ? c1o : len;
    c.code.assign((end + 7) / 8 + 64, 0);   // (and a few words past it: the control flow reads ahead)
    std::memcpy(c.code.data(), area.data(), end);
    c.c1.assign(area.begin() + c1o, area.end());
    if (st == 5) std::memcpy(c.local, ctl + 0x748, 12);
    if (st == 5) {
        // shared memory at +0x754 (0x800 and 0x180 in the two of TOTK's first 16 kernels whose programs use it: the
        // others hold 0) and local memory at +0x760 (also at +0x710)
        std::memcpy(&c.shared, ctl + 0x754, 4);
        std::memcpy(&c.local_memory, ctl + 0x760, 4);
        if (c.shared > (64u << 10) || c.local_memory > (1u << 20)) {
            log_limited("nvn_cs_sizes", 8, "rx: NVN compute package %016llx: shared memory %#x, local memory %#x?",
                        (unsigned long long)c.key, c.shared, c.local_memory);
            c.shared = std::min(c.shared, 64u << 10), c.local_memory = std::min(c.local_memory, 1u << 20);
        }
        // RX_NVN_CS_LOG=1 (debugging): compute control sections (where the shared memory size is kept)
        if (static const bool log_cs = env_flag("RX_NVN_CS_LOG"); log_cs) {
            static int n;
            if (n++ < 16) {
                std::string line;
                char w[16];
                for (uint32_t o = 0x6c0; o < 0x7c0; o += 4) {
                    uint32_t v;
                    std::memcpy(&v, ctl + o, 4);
                    std::snprintf(w, sizeof w, "%s%x", o % 0x40 ? " " : " | ", v);
                    line += w;
                }
                line += " || area";
                for (uint32_t o = 0; o < 0xc0 && o + 4 <= len; o += 4) {
                    uint32_t v;
                    std::memcpy(&v, area.data() + o, 4);
                    std::snprintf(w, sizeof w, " %08x", v);
                    line += w;
                }
                hos_log("rx: NVN compute package %016llx len %#x c1 %#x local %ux%ux%u:%s", (unsigned long long)c.key, len,
                        c1o, c.local[0], c.local[1], c.local[2], line.c_str());
            }
        }
    }
    c.ok = true;
    return c;
}

/// suyu's Environment over an NVN stage: what the recompiler asks, answered from the NVN state of the draw
class NvnEnvironment final : public Shader::Environment {
public:
    NvnEnvironment(const NvnState& state, const NvnCode& c, Shader::Stage st) : s{state}, nvn{c} {
        stage = st;
        start_address = 0x30;
        if (st != Shader::Stage::Compute) {
            std::memcpy(&sph, reinterpret_cast<const uint8_t*>(nvn.code.data()) + 0x30, sizeof sph);
            local_memory = static_cast<uint32_t>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
        } else {
            local_memory = nvn.local_memory;
        }
        is_proprietary_driver = true;
    }
    u64 ReadInstruction(u32 address) override {
        if (address / 8 < nvn.code.size()) return nvn.code[address / 8];
        return gpu_get<uint64_t>(nvn.data + address);
    }
    u32 ReadCbufValue(u32 index, u32 offset) override {
        if (index == 1) {
            uint32_t v = 0;
            if ((size_t)offset + 4 <= nvn.c1.size()) std::memcpy(&v, nvn.c1.data() + offset, 4);
            return v;
        }
        if (index == 0 || index == 2) {
            if (index == 2 && offset < 0x568) c2_reads.push_back(offset);
            return nvn_driver_bank_word(s, (int)index, nvn.stage, offset);
        }
        if (index >= 3 && index < 3 + 16) {
            const nvn_p_bind_buf& u = s.ubo[nvn.stage][index - 3];
            return u.addr && (uint64_t)offset + 4 <= nvn_ubo_size(u.size) ? gpu_get<uint32_t>(u.addr + offset) : 0;
        }
        return 0;
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return nvn_handle_type(handle);
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        return static_cast<Shader::TexturePixelFormat>(nvn_handle_format(handle));
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        return VideoCore::Surface::IsPixelFormatInteger(nvn_handle_format(handle));
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    u32 TextureBoundBuffer() const override {
        return 2;
    }
    u32 LocalMemorySize() const override {
        return local_memory;
    }
    u32 SharedMemorySize() const override {
        return nvn.shared;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {std::max(1u, nvn.local[0]), std::max(1u, nvn.local[1]), std::max(1u, nvn.local[2])};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

    std::vector<uint32_t> c2_reads;   // texture handle words the translation read (what its answers depend on)

private:
    const NvnState& s;
    const NvnCode& nvn;
    uint32_t local_memory = 0;
};

std::unique_ptr<Translated> nvn_translate(NvnEnvironment& env, bool graphics) {
    struct TranslateTimer {
        int64_t t0 = hos_now_ns();
        ~TranslateTimer() {
            if (!on_translation_worker) count("t_translate", hos_now_ns() - t0);
        }
    } translate_timer;
    auto t = std::make_unique<Translated>();
    Shader::Maxwell::Flow::CFG cfg{env, t->flow_blocks,
                                   env.StartAddress() + (graphics ? static_cast<u32>(sizeof(Shader::ProgramHeader)) : 0u)};
    Shader::HostTranslateInfo host{};
    host.support_float64 = false;
    host.support_int64 = false;
    host.min_ssbo_alignment = dev.limits.minStorageBufferOffsetAlignment;
    t->program = Shader::Maxwell::TranslateProgram(t->insts, t->blocks, env, cfg, host);
    return t;
}

uint8_t nvn_type_at(const NvnState& s, int stage, uint32_t offset) {
    return (uint8_t)nvn_handle_type(nvn_driver_bank_word(s, 2, stage, offset));
}

/// A translated pair and what it was translated for
struct NvnVariant {
    std::unique_ptr<ProgramPair> pair;
    bool failed = false;   // (does not translate whatever the state: not tried again)
    std::bitset<32> loads;
    uint64_t attr_sig = 0, plan_sig = 0;
    uint8_t out_types[8]{};
    std::vector<std::pair<uint32_t, uint8_t>> answers[2];
};
struct NvnProgCodes {
    NvnCode vs, fs;
    bool has_fs = false, ok = false;
};
std::unordered_map<uint64_t, std::unique_ptr<NvnProgCodes>> nvn_codes;                 // by program serial
std::unordered_map<uint64_t, std::vector<std::unique_ptr<NvnVariant>>> nvn_variants;   // by the stages' code keys

uint64_t nvn_attr_sig(const NvnState& s, const std::bitset<32>& loads) {
    uint64_t h = 0x5eed;
    for (uint32_t i = 0; i < 16; i++)
        if (loads[i]) {
            const Regs::VertexAttribute a = nvn_attribute(s.vattr[i]);
            const uint8_t t[2] = {(uint8_t)attribute_type(a), vertex_unpack(a)};
            h = hash_bytes(t, 2, h ^ i);
        }
    return h;
}
uint64_t nvn_plan_sig(const PullPlan& plan) {
    uint64_t h = 0x91a7;
    for (uint32_t i = 0; i < 32; i++)
        if (plan.attr[i].enabled) {
            const auto& a = plan.attr[i];
            const uint32_t f[3] = {a.stream | (a.instanced ? 0x100u : 0u), a.attribute, a.stride};
            h = hash_bytes(f, sizeof f, h ^ (i + 200));
        }
    return h;
}

void intern_module(ShaderModule& m) {   // (prog_mu held) the same WGSL text: one copy, one module
    const uint64_t h = hash_bytes(m.wgsl.data(), m.wgsl.size(), 0x1d7e57ull);
    const uint64_t check = hash_bytes(m.wgsl.data(), m.wgsl.size(), 0xc0ffee5ull);
    auto [it, inserted] = interned_texts.try_emplace(h);
    InternedText& t = it->second;
    if (inserted) {
        t.id = (uint32_t)interned_texts.size();
        t.check = check;
        t.text = std::move(m.wgsl);
        if (RX_WEB) t.cache_hash = pipecache_text_hash(t.text);
        else native_cache_text(h, t.text);
        wgsl_texts++;
        wgsl_bytes += t.text.size();
    } else if (t.check != check) {
        return;
    }
    m.text_id = t.id;
    m.text_hash = h;
    m.cache_hash = t.cache_hash;
    m.text = &t.text;
    std::string().swap(m.wgsl);
}


/// An NVN program's compute stage: its package (once per program object) and a module per answer to the texture
/// types it reads
struct NvnComputeVariant {
    std::unique_ptr<ShaderModule> module;
    std::vector<std::pair<uint32_t, uint8_t>> answers;
    std::array<uint8_t, 64> alias{};   // (web: storage buffers read through another's binding, see compute_module)
    std::array<uint32_t, 64> alias_words{};
};
std::unordered_map<uint64_t, std::unique_ptr<NvnCode>> nvn_cs_codes;                              // by program serial
std::unordered_map<uint64_t, std::vector<std::unique_ptr<NvnComputeVariant>>> nvn_cs_variants;   // by code key

const NvnCode* nvn_cs_code(const NvnState& s) {
    auto& c = nvn_cs_codes[s.prog.serial];
    if (!c) {
        c = std::make_unique<NvnCode>();
        for (uint32_t i = 0; i < s.prog.nstage && i < 6; i++) {
            NvnCode k = read_nvn_code(s.prog.sd[i]);
            if (k.ok && k.stage == 5) {
                *c = std::move(k);
                break;
            }
        }
    }
    return c->ok ? c.get() : nullptr;
}

}  // namespace

uint64_t nvn_cs_key(const NvnState& s) {
    const NvnCode* code = s.has_prog ? nvn_cs_code(s) : nullptr;
    return code ? code->key : 0;
}

ShaderModule* nvn_compute_module(const NvnState& s, const std::array<uint8_t, 64>* ssbo_alias,
                                 const std::array<uint32_t, 64>* ssbo_alias_words) {
    if (!s.has_prog) return nullptr;
    const NvnCode* code = nvn_cs_code(s);
    if (!code) return nullptr;
    static const std::array<uint8_t, 64> no_alias{};
    static const std::array<uint32_t, 64> no_words{};
    const auto& alias = ssbo_alias ? *ssbo_alias : no_alias;
    const auto& alias_words = ssbo_alias_words ? *ssbo_alias_words : no_words;
    auto& list = nvn_cs_variants[code->key];
    for (const auto& v : list) {
        bool same = v->alias == alias && v->alias_words == alias_words;
        for (const auto& [off, type] : v->answers)
            if (nvn_type_at(s, 5, off) != type) {
                same = false;
                break;
            }
        if (same) return v->module->failed ? nullptr : v->module.get();
    }
    auto var = std::make_unique<NvnComputeVariant>();
    var->module = std::make_unique<ShaderModule>();
    var->alias = alias;
    var->alias_words = alias_words;
    ShaderModule& m = *var->module;
    try {
        NvnEnvironment env{s, *code, Shader::Stage::Compute};
        auto t = nvn_translate(env, false);
        for (uint32_t off : env.c2_reads) var->answers.push_back({off, nvn_type_at(s, 5, off)});
        Shader::Profile profile = current_shader_profile();
        Shader::RuntimeInfo rt{};
        Shader::Backend::Bindings b{};
        Shader::Backend::WGSL::ssbo_alias_next = alias;
        Shader::Backend::WGSL::ssbo_alias_words_next = alias_words;
        m.wgsl = Shader::Backend::WGSL::EmitWGSL(profile, rt, t->program, b);
        m.info = t->program.info;
    } catch (const std::exception& e) {
        log_limited("nvn_translate_cs", 30, "rx: NVN compute program %016llx does not translate: %s",
                    (unsigned long long)code->key, e.what());
        m.failed = true;
    }
    if (!m.failed) {
        if (const char* dir = getenv("RX_DUMP_WGSL")) {
            char path[512];
            std::snprintf(path, sizeof path, "%s/nvn_cs_%016llx.wgsl", dir, (unsigned long long)code->key);
            std::ofstream(path, std::ios::binary) << m.wgsl;
        }
        const uint64_t th = hash_bytes(m.wgsl.data(), m.wgsl.size(), 0x1d7e57ull);
        if (WGPUShaderModule cached = native_cache_compute_module(th)) {
            m.module = cached;
            count("cs_module_cached");
        } else if (create_module(m, "compute", code->key)) {
            native_cache_compute_text(th, m.wgsl);
        } else {
            m.failed = true;
        }
        if (!m.failed) count("nvn_compute_programs");
    }
    ShaderModule* out = m.failed ? nullptr : &m;
    list.push_back(std::move(var));
    return out;
}

namespace {
}  // namespace

ProgramPair* nvn_programs_uncached(const NvnState& s, const uint8_t out_types[8], const NvnVariant** chosen);

uint32_t nvn_ub_limit() {
    static const uint32_t limit = getenv("RX_NVN_UB_LIMIT") ? (uint32_t)atoi(getenv("RX_NVN_UB_LIMIT")) : 512;
    return limit;
}

const std::vector<uint8_t>* nvn_c1(const NvnState& s, int stage) {
    if (stage == 5) {
        auto cs = nvn_cs_codes.find(s.prog.serial);
        return cs != nvn_cs_codes.end() && cs->second && cs->second->ok ? &cs->second->c1 : nullptr;
    }
    auto it = nvn_codes.find(s.prog.serial);
    if (it == nvn_codes.end() || !it->second) return nullptr;
    const NvnCode& c = stage == 0 ? it->second->vs : it->second->fs;
    return c.ok ? &c.c1 : nullptr;
}

ProgramPair* nvn_programs(const NvnState& s, const uint8_t out_types[8]) {
    if (!s.has_prog) return nullptr;
    // an answer for this program, vertex state and targets while nothing that picks the variant changed (the variant
    // checks below cost ~8% of the replay thread; draws of different materials take turns, so one entry each).
    // Per frame: texture pool entries can change a handle's type.
    struct Memo {
        const NvnState* s = nullptr;
        uint64_t epoch = 0, vtx = 0, serial = 0;
        uint32_t vbufs = 0;
        int64_t frame = -1;
        uint8_t out[8]{};
        ProgramPair* pair = nullptr;
        const NvnVariant* var = nullptr;
    };
    static Memo memos[512];
    static const bool no_memo = env_flag("RX_NO_NVN_PROGRAM_MEMO");
    uint32_t vbufs = 0;   // (which streams have a buffer: the pull plan depends on it)
    for (int i = 0; i < 16; i++) vbufs |= (s.vbuf[i].address != 0 ? 1u : 0u) << i;
    uint64_t out_bits;
    std::memcpy(&out_bits, out_types, 8);
    Memo& memo = memos[hash_bytes(&out_bits, 8, s.prog.serial * 0x9e3779b97f4a7c15ull ^ s.vtx_hash ^ vbufs) & 511];
    if (!no_memo && memo.pair && memo.s == &s && memo.serial == s.prog.serial && memo.vtx == s.vtx_hash &&
        memo.vbufs == vbufs && memo.frame == frame_number && !std::memcmp(memo.out, out_types, 8)) {
        if (memo.epoch == s.prog_epoch) return memo.pair;
        // only handles changed (or another program was bound between): the variant holds while the texture types it
        // was translated for do
        if (memo.var) {
            bool same = true;
            for (int st = 0; st < 2 && same; st++)
                for (const auto& [off, type] : memo.var->answers[st])
                    if (nvn_type_at(s, st, off) != type) {
                        same = false;
                        break;
                    }
            if (same) {
                memo.epoch = s.prog_epoch;
                count("nvn_program_memo_rechecked");
                return memo.pair;
            }
        }
    }
    const NvnVariant* var = nullptr;
    ProgramPair* found = nvn_programs_uncached(s, out_types, &var);
    memo.s = &s, memo.epoch = s.prog_epoch, memo.vtx = s.vtx_hash, memo.serial = s.prog.serial;
    memo.vbufs = vbufs, memo.frame = frame_number;
    std::memcpy(memo.out, out_types, 8);
    memo.pair = found;
    memo.var = var;
    return found;
}

ProgramPair* nvn_programs_uncached(const NvnState& s, const uint8_t out_types[8], const NvnVariant** chosen) {
    *chosen = nullptr;
    if (!s.has_prog) return nullptr;
    // the stages' packages (once per program object's shaders)
    auto& codes = nvn_codes[s.prog.serial];
    if (!codes) {
        codes = std::make_unique<NvnProgCodes>();
        for (uint32_t i = 0; i < s.prog.nstage && i < 6; i++) {
            NvnCode c = read_nvn_code(s.prog.sd[i]);
            if (!c.ok) continue;
            if (c.stage == 0) codes->vs = std::move(c);
            else if (c.stage == 1) codes->fs = std::move(c), codes->has_fs = true;
            else log_limited("nvn_stage", 8, "rx: NVN program with a stage %d: not drawn", c.stage);
        }
        codes->ok = codes->vs.ok;
    }
    if (!codes->ok) return nullptr;
    const uint64_t vkey = codes->vs.key, fkey = codes->has_fs ? codes->fs.key : 0;
    auto& list = nvn_variants[hash_bytes(&vkey, 8, fkey)];
    // a variant translated for what the state answers now
    for (const auto& v : list) {
        if (v->failed) return nullptr;
        if (std::memcmp(v->out_types, out_types, 8)) continue;
        if (v->attr_sig != nvn_attr_sig(s, v->loads)) continue;
        if (v->plan_sig != nvn_plan_sig(nvn_pull_plan(s, v->loads))) continue;
        bool same = true;
        for (int st = 0; st < 2 && same; st++)
            for (const auto& [off, type] : v->answers[st])
                if (nvn_type_at(s, st, off) != type) {
                    same = false;
                    break;
                }
        if (same) {
            *chosen = v.get();
            return v->pair ? v->pair.get() : nullptr;
        }
    }
    // a new one
    auto var = std::make_unique<NvnVariant>();
    std::memcpy(var->out_types, out_types, 8);
    try {
        NvnEnvironment venv{s, codes->vs, Shader::Stage::VertexB};
        auto vs = nvn_translate(venv, true);
        std::unique_ptr<Translated> fs;
        std::unique_ptr<NvnEnvironment> fenv;
        if (codes->has_fs) {
            fenv = std::make_unique<NvnEnvironment>(s, codes->fs, Shader::Stage::Fragment);
            fs = nvn_translate(*fenv, true);
        }
        for (uint32_t i = 0; i < 32; i++) var->loads[i] = vs->program.info.loads.Generic(i);
        for (uint32_t off : venv.c2_reads) var->answers[0].push_back({off, nvn_type_at(s, 0, off)});
        if (fenv)
            for (uint32_t off : fenv->c2_reads) var->answers[1].push_back({off, nvn_type_at(s, 1, off)});
        var->attr_sig = nvn_attr_sig(s, var->loads);
        const PullPlan plan = nvn_pull_plan(s, var->loads);
        var->plan_sig = nvn_plan_sig(plan);
        auto pair = std::make_unique<ProgramPair>();
        pair->has_fs = fs != nullptr;
        Shader::Profile profile = current_shader_profile();
        Shader::Backend::WGSL::Options options{};
        Shader::RuntimeInfo vs_rt{};
        vs_rt.previous_stage_stores.mask.set();
        for (uint32_t i = 0; i < 32; i++) {
            const Regs::VertexAttribute a = i < 16 ? nvn_attribute(s.vattr[i]) : Regs::VertexAttribute{};
            vs_rt.generic_input_types[i] = attribute_type(a);
            options.vertex_unpack[i] = (Shader::Backend::WGSL::VertexUnpack)vertex_unpack(a);
            auto& vp = options.vertex_pull[i];
            vp.enabled = plan.attr[i].enabled;
            vp.stream = plan.attr[i].stream;
            vp.instanced = plan.attr[i].instanced;
            vp.attribute = plan.attr[i].attribute;
            vp.stride = plan.attr[i].stride;
        }
        if (fs)
            for (uint32_t i = 0; i < 32; i++) options.output_interpolation[i] = fs->program.info.interpolation[i];
        vs_rt.cbuf_ub_limit = nvn_ub_limit();
        Shader::Backend::Bindings vb{};
        pair->vs.wgsl = Shader::Backend::WGSL::EmitWGSL(profile, vs_rt, vs->program, vb, options);
        pair->vs.info = vs->program.info;
        if (fs) {
            Shader::RuntimeInfo fs_rt{};
            fs_rt.previous_stage_stores = vs->program.info.stores;
            for (uint32_t i = 0; i < 8; i++) fs_rt.color_output_types[i] = (Shader::AttributeType)out_types[i];
            fs_rt.cbuf_ub_limit = nvn_ub_limit();
            // RX_EARLY_Z_STORERS=1 (experiment): pixel programs that store to memory test depth before they run
            if (static const bool early = env_flag("RX_EARLY_Z_STORERS"); early && dev.early_depth_test)
                for (const auto& d : fs->program.info.storage_buffers_descriptors) fs_rt.force_early_z |= d.is_written;
            Shader::Backend::Bindings fb{};
            pair->fs.wgsl = Shader::Backend::WGSL::EmitWGSL(profile, fs_rt, fs->program, fb);
            pair->fs.info = fs->program.info;
        }
        // RX_WGSL_OVERRIDE=<dir> (debugging): a program's WGSL is replaced by <dir>/nvn_vs_<key>.wgsl or
        // nvn_fs_<key>.wgsl when that file exists (e.g. an RX_DUMP_WGSL file edited to output an intermediate value)
        if (const char* dir = getenv("RX_WGSL_OVERRIDE")) {
            auto override_from = [&](const char* kind, uint64_t key, std::string& wgsl) {
                char path[512];
                std::snprintf(path, sizeof path, "%s/nvn_%s_%016llx.wgsl", dir, kind, (unsigned long long)key);
                std::ifstream f(path, std::ios::binary);
                if (!f) return;
                wgsl.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                hos_log("rx: %s program %016llx: WGSL from %s", kind, (unsigned long long)key, path);
            };
            override_from("vs", vkey, pair->vs.wgsl);
            if (fs) override_from("fs", fkey, pair->fs.wgsl);
        }
        if (const char* dir = getenv("RX_DUMP_WGSL")) {
            char path[512];
            auto banks = [](const Shader::Info& info) {   // (the uniform block's banks, in order)
                std::string c;
                for (const auto& d : info.constant_buffer_descriptors) {
                    char b[96];
                    std::snprintf(b, sizeof b, "// c%u used %#x\n", d.index, info.constant_buffer_used_sizes[d.index]);
                    c += b;
                }
                return c;
            };
            std::snprintf(path, sizeof path, "%s/nvn_vs_%016llx.wgsl", dir, (unsigned long long)vkey);
            std::ofstream(path, std::ios::binary) << banks(pair->vs.info) << pair->vs.wgsl;
            if (fs) {
                std::snprintf(path, sizeof path, "%s/nvn_fs_%016llx.wgsl", dir, (unsigned long long)fkey);
                std::ofstream(path, std::ios::binary) << banks(pair->fs.info) << pair->fs.wgsl;
            }
        }
        pair->vs.what = "vertex";
        pair->vs.key = vkey;
        pair->fs.what = "pixel";
        pair->fs.key = fkey;
        std::lock_guard lk(prog_mu);
        intern_module(pair->vs);
        if (pair->has_fs) intern_module(pair->fs);
        var->pair = std::move(pair);
        count("nvn_programs");
    } catch (const std::exception& e) {
        log_limited("nvn_translate", 40, "rx: NVN program (vertex %016llx, pixel %016llx) does not translate: %s",
                    (unsigned long long)vkey, (unsigned long long)fkey, e.what());
        var->failed = true;
    }
    ProgramPair* out = var->pair ? var->pair.get() : nullptr;
    *chosen = var.get();
    list.push_back(std::move(var));
    return out;
}

} // namespace rx

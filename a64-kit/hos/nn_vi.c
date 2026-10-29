/* nn::vi and nv::: the display, its layer, the native window, the vsync event, and the GPU driver's setup calls.

   Display and layer are small guest objects the game only passes back to us; the native window is the ANativeWindow
   NVN presents through (below). Vsync is a 60 Hz host thread signaling the display's vsync system event; games pace
   their frames on it. */
#include "hos.h"
#include "perf_menu.h"
#include <string.h>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../gpu/render/rx.h"
uint64_t hos_nvmap_address(uint32_t handle);   /* nvdrv.c */
void hos_gpu_present(uint64_t address, uint32_t width, uint32_t height, int64_t at_vsync);   /* gpu.c */

void hos_system_event_init(uint64_t key, int autoclear);
void hos_system_event_signal(uint64_t key);

static uint64_t display, layer, native_window;
static uint64_t vsync_events[8];
static volatile LONG nvsync;
static HANDLE vsync_thread;
volatile int64_t hos_vsync_count;

static SRWLOCK vsync_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE vsync_cv = CONDITION_VARIABLE_INIT;

#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include <dwmapi.h>
static int display_timing(int64_t *vblank_ns, int64_t *period_ns) {
    typedef HRESULT(WINAPI * Fn)(HWND, DWM_TIMING_INFO *);
    static Fn fn;
    static int looked;
    static LARGE_INTEGER freq;
    if (!looked) {
        looked = 1;
        HMODULE m = LoadLibraryA("dwmapi.dll");
        if (m) fn = (Fn)(void *)GetProcAddress(m, "DwmGetCompositionTimingInfo");
        QueryPerformanceFrequency(&freq);
    }
    if (!fn || !freq.QuadPart) return 0;
    DWM_TIMING_INFO ti;
    memset(&ti, 0, sizeof ti);
    ti.cbSize = sizeof ti;
    if (FAILED(fn(NULL, &ti)) || !ti.qpcRefreshPeriod || !ti.qpcVBlank) return 0;
    const int64_t f = freq.QuadPart, rp = (int64_t)ti.qpcRefreshPeriod, vb = (int64_t)ti.qpcVBlank;
    *period_ns = rp / f * 1000000000LL + rp % f * 1000000000LL / f;
    *vblank_ns = vb / f * 1000000000LL + vb % f * 1000000000LL / f;
    return 1;
}
#endif

static int realtime_on(void);

/* The emulated vsync (the display's vsync event, and with the game's own frame step the frame pacing): 60 Hz,
   locked to a whole number of display refreshes when it can be. */
static double vsync_target_hz(void) {
    static double hz = -1;   /* HOS_VSYNC_HZ=<hz> (experiment): another rate (240 changed nothing in 10-20 tests) */
    if (hz < 0) hz = getenv("HOS_VSYNC_HZ") && atof(getenv("HOS_VSYNC_HZ")) >= 30 ? atof(getenv("HOS_VSYNC_HZ")) : 60;
    return hz;
}

/* With real-time steps frames are not held to vsync ticks (a grid of ticks turns a 15 ms frame into two ticks: at a
   90 Hz grid, 45 fps): each frame may start 1/cap after the one before, and goes out as soon as it is done (mailbox
   presents: no tearing). HOS_FPS_CAP=<hz>; default the display's refresh rate, at most 120 (60 when unknown). */
static volatile LONG64 display_period_ns;   /* (the vsync thread reads the display's timing) */
static double frame_cap_hz(void) {
    static double hz = -1;
    if (hz < 0) {
        const char *e = getenv("HOS_FPS_CAP");
        if (e && atof(e) >= 20) hz = atof(e) > 480 ? 480 : atof(e);
        else {
            const int64_t p = display_period_ns;
            if (!p) return 60;   /* (not known yet: ask again next frame) */
            hz = 1e9 / (double)p;
            if (hz > 120) hz = 120;
            if (hz < 50) hz = 60;
        }
        hos_log("vi: frames at most %.0f a second (%s), each shown as soon as it is done", hz,
                e && atof(e) >= 20 ? "HOS_FPS_CAP" : "the display's refresh rate, at most 120");
    }
    return hz;
}
static void frame_limit(void) {
    static int64_t next;
    const int64_t override_hz = perf_menu_fps_cap;
    const int64_t period = override_hz > 0 ? (int64_t)(1e9 / (double)override_hz) : (int64_t)(1e9 / frame_cap_hz());
    int64_t now = hos_now_ns();
    if (next && now < next) {
        hos_sleep_ns(next - now);
        now = hos_now_ns();
    }
    next = !next || now - next > period ? now + period : next + period;
}

static DWORD WINAPI vsync_main(void *p) {
    (void)p;
    hos_thread_boost(2);
    const int64_t target = (int64_t)(1e9 / vsync_target_hz());
    int64_t period = target, next = hos_now_ns() + period;
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
    const int free_run = getenv("HOS_VSYNC_FREE") != NULL;
    int64_t check_at = 0;
    int locked = 0;
#endif
    for (;;) {
        hos_sleep_ns(next - hos_now_ns());
        next += period;
        AcquireSRWLockExclusive(&vsync_lock);
        hos_vsync_count++;
        ReleaseSRWLockExclusive(&vsync_lock);
        WakeAllConditionVariable(&vsync_cv);
        for (LONG i = 0; i < nvsync; i++) hos_system_event_signal(vsync_events[i]);
        if (hos_vsync_count % 60 == 0) {
            extern volatile int64_t hos_hid_last_read;
            extern volatile LONG64 hos_present_count;
            static int reported;
            const int64_t last = hos_hid_last_read, idle = last ? hos_now_ns() - last : 0;
            if (idle > 3000000000LL && !reported) {
                reported = 1;
                hos_log("hid: the game has not read the controller for %.1f s (%lld frames presented): input is not "
                        "reaching it", idle / 1e9, (long long)hos_present_count);
            } else if (idle < 1000000000LL && reported) {
                reported = 0;
                hos_log("hid: the game reads the controller again");
            }
        }
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
        if (free_run || hos_vsync_count < check_at) continue;
        check_at = hos_vsync_count + 30;
        int64_t vb, rp;
        if (!display_timing(&vb, &rp) || rp <= 0) continue;
        InterlockedExchange64(&display_period_ns, rp);
        /* the fewest refreshes a tick that is not shorter than the target (5% slack: 59.94 Hz for 60) */
        int64_t k = (target * 95 / 100 + rp - 1) / rp;
        if (k < 1) k = 1;
        const int64_t tick = rp * k;
        if (k <= 8 && tick >= target * 95 / 100 && tick <= target * 2) {
            if (!locked) hos_log("vi: vsync follows the display (%.3f Hz, every %lld refresh%s): at most %.1f fps",
                                 1e9 / rp, (long long)k, k > 1 ? "es" : "", 1e9 / tick);
            locked = 1;
            period = tick;
            int64_t grid = next - vb >= 0 ? vb + (next - vb) / tick * tick : vb - (vb - next + tick - 1) / tick * tick;
            if (next - grid > tick / 2) grid += tick;
            next += (grid - next) / 4;
        } else if (locked) {
            locked = 0;
            period = target;
            hos_log("vi: the display runs at %.3f Hz: vsync on its own %.0f Hz clock", 1e9 / rp, 1e9 / target);
        }
#endif
    }
}

/* Blocks until the vsync count reaches `until` (waits on the vsync thread rather than polling: the game's
   presentation thread spun here, holding a lock its main thread wanted). */
static void wait_vsync(int64_t until) {
    AcquireSRWLockExclusive(&vsync_lock);
    while (hos_vsync_count < until) SleepConditionVariableSRW(&vsync_cv, &vsync_lock, 50, 0);
    ReleaseSRWLockExclusive(&vsync_lock);
}
void hos_vsync_wait_until(int64_t until) { wait_vsync(until); }

/* Where frames are held to the swap interval (HOS_VI_MODE): 0 dequeue waits for its buffer's release; 1 dequeue
   returns at once and queue waits for the frame's vsync; 2 dequeue returns at once and the GPU thread holds the
   present until the frame's vsync (the console's GPU waits on the buffer's fence, not the CPU). */
static int vi_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = getenv("HOS_VI_MODE");
        mode = e ? atoi(e) : 0;
    }
    return mode;
}

HLE(_ZN2nn2vi10InitializeEv) {}
/* OpenDefaultDisplay(Display**) -> Result */
HLE(_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE) {
    if (!display) display = hos_sys_alloc(0x100, 0x10);
    G64(ARG(0)) = display;
    RET(0);
}
/* CreateLayer(Layer**, Display*) -> Result */
HLE(_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE) {
    if (!layer) layer = hos_sys_alloc(0x100, 0x10);
    G64(ARG(0)) = layer;
    RET(0);
}
/* ---- the native window: an Android ANativeWindow ---------------------------------------------------------------------
   nn::vi hands NVN an android::Surface. NVN's window goes through NVIDIA's WSI layer (sdk NvWsi*), whose "Android"
   backend accepts a window by its magic and then drives only the ANativeWindow interface (system/window.h, LP64):
     +0x00 magic '_wnd'  +0x04 version (sizeof, 0xc0)  +0x28 incRef(base)  +0x30 decRef(base)
     +0x38 flags  +0x3c minSwapInterval  +0x40 maxSwapInterval  +0x44 xdpi  +0x48 ydpi
     +0x70 setSwapInterval(win, n)         +0x90 query(win, what, int *out)     +0x98 perform(win, op, ...)
     +0xa8 dequeueBuffer(win, ANativeWindowBuffer **, int *fence_fd)
     +0xb0 queueBuffer(win, buffer, fence_fd)                                   +0xb8 cancelBuffer(win, buffer, fence_fd)
     (+0x78 +0x80 +0x88 +0xa0: the deprecated fence-less entries, never called)
   The buffers are NVN's own window textures. nvnWindowInitialize registers each one with Nintendo's perform 0x13
   (slot, nvgr handle, width, height, format, usage, [stride]); a NULL handle unregisters the slot. Dequeue and queue
   then trade ANativeWindowBuffers for those slots: magic '_bfr', +0x38 width, height, stride, format, usage,
   +0x60 the nvgr handle, which the backend matches against the handles it registered. We implement the window
   ourselves rather than run the SDK's Surface over an emulated vi binder service: these few entries are all the
   driver touches.

   A queued buffer is a finished frame (milestone 5 shows it); for now presents are counted. Queue paces like the
   console's FIFO: it returns at the swap interval's next vsync (the 60 Hz thread above). The fence the driver passes
   with each queue is ignored: our GPU (hos/nvdrv.c) completes submissions before the queue arrives. */
uint64_t hos_hostcall(const char *name, void (*fn)(Ctx *, void *), void *user);
enum { NW_SLOTS = 8, ANW_SIZE = 0xc0, ANB_SIZE = 0xa8 };
enum { NW_WIDTH = 1920, NW_HEIGHT = 1080 };
static struct {
    uint64_t buffer[NW_SLOTS];          /* guest ANativeWindowBuffer per slot, 0 = unregistered */
    int64_t free_at[NW_SLOTS];          /* vsync from which the slot may be dequeued again */
    int queued[NW_SLOTS];               /* in the queue or on screen: not free before free_at says */
    int format, usage, interval, transform, front;
    int64_t last_display;               /* vsync the last queued frame goes on screen at */
} nw = { .format = 1, .interval = 1, .front = -1 };
static SRWLOCK nw_lock = SRWLOCK_INIT;
volatile LONG64 hos_present_count;
volatile int hos_present_slot = -1;     /* slot of the frame on screen */
/* The window buffer of a slot, for the display (hos/display.c): its nvgr handle describes the surface. */
uint64_t hos_window_buffer_handle(int slot) {
    return slot >= 0 && slot < NW_SLOTS && nw.buffer[slot] ? G64(nw.buffer[slot] + 0x60) : 0;
}
static int64_t present_window_start, presents_in_window;

static void nw_nop(Ctx *c, void *u) { (void)c; (void)u; }
/* TOTK (v1.0.0) picks its own swap interval: a controller in its game framework (main+0x8f6bec) measures each
   frame's work in vsyncs, sets interval = clamp(measured + 1, min, max) (min/max at +0/+4 of the controller, which
   is at framework + 0x248, current at +0x14) and scales its simulation step by interval / 2 every frame, so the game
   runs at the right speed at any interval it really gets (checked: the same walk at 30 and at 60 fps ends at the
   same spot). Showing frames faster than the interval makes the game itself faster (2-3x, HOS_UNCAPPED=1).
   The game's measurement includes its waits on GPU fences, which our pacing delays: once at 3 (20 fps) it never came
   back. So the interval is ours: 2 (30 fps) or 1 (60 fps), switched by how long frames really take (frame_done);
   HOS_TARGET_INTERVAL=n holds n, HOS_DYNAMIC_INTERVAL=1 leaves the game's controller alone. We find the controller on
   its own call: the caller's saved registers on the guest stack (the SDK's SetPresentInterval, main+0x4b61260,
   saves x20/x19 = controller/framework at its sp + 0x10; NvWsi's setter below it has a 0x10-byte frame), checked
   against each other before anything is written. */
static uint64_t totk_ctl, totk_fw;
static int interval_mode = -1;   /* 0 the game's, n > 0 held at n, -1 not read yet, -2 automatic (1 or 2) */

/* Real-time frame step (default; HOS_REALTIME=0 the old 30/60 interval): the game steps by the time its frames really take, so any frame rate runs
   at the right speed, instead of by its swap interval (1 or 2 vsyncs: 30/60 and nothing between). The step is set
   in two places (the patch MacNativeGames makes to v1.0.0, github.com/ly05010419/MacNativeGames, MIT): the
   variable-frame-rate step main+0x7a83e4 (stores ratio = interval * 0.5 at +8 and frame seconds = interval / 60 at
   +12 of its object, then derives the rest from +8) and the global ratio stored after the frame-rate controller
   (main+0x8f68cc). Both take ratio = seconds * 30 and seconds from the time between presents (the mean of the last
   5, 1/240..1/15 s). The swap interval is then always 1: frames go out at every vsync they are ready for. */
static volatile LONG64 frame_step_bits;   /* low: ratio (float), high: seconds (float); 0 until the first use */
static int realtime_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("HOS_REALTIME");
        on = !(e && *e == '0');
    }
    return on;
}
static uint64_t step_bits(float seconds) {
    const float ratio = seconds * 30.0f;
    uint32_t lo, hi;
    memcpy(&lo, &ratio, 4);
    memcpy(&hi, &seconds, 4);
    return (uint64_t)lo | (uint64_t)hi << 32;
}
/* The ability wheel (hold L) captures the screen for its blurred background in the frame the world stops drawing,
   unless the game steps by whole 1/30 s frames: at any faster step (its own 60 fps mode too) the world was already
   gone and the background came out black. While L is held (at most 3 s a press) the step is 1/30 s: the wheel pauses
   the world anyway, and a tap of L runs a few frames at a 30 fps step. HOS_WHEEL_STEP30=0: off. */
static volatile int64_t step30_until;
void hos_frame_step_buttons(uint64_t buttons) {
    static int on = -1;
    static int64_t held_since;
    if (on < 0) on = !(getenv("HOS_WHEEL_STEP30") && *getenv("HOS_WHEEL_STEP30") == '0');
    if (!on || !realtime_on()) return;
    const int64_t now = hos_now_ns();
    if (!(buttons >> 6 & 1)) {   /* (L: NpadButton bit 6) */
        held_since = 0;
        return;
    }
    if (!held_since) held_since = now;
    if (now - held_since < 3000000000LL) step30_until = now + 100000000;
}
static uint64_t frame_step(void) {
    if (hos_now_ns() < step30_until) return step_bits(1.0f / 30.0f);
    uint64_t v = (uint64_t)frame_step_bits;
    if (!v) {
        v = step_bits(1.0f / 60.0f);
        InterlockedCompareExchange64(&frame_step_bits, (LONG64)v, 0);
    }
    return v;
}
static void frame_step_present(int64_t now) {
    static int64_t last;
    static double samples[5];
    static int n, next;
    if (!realtime_on()) return;
    const double dt = last ? (now - last) * 1e-9 : 0;
    last = now;
    if (dt <= 0.001 || dt > 0.25) {   /* (a loading hitch: start over, the step stays) */
        n = next = 0;
        return;
    }
    samples[next++ % 5] = dt < 1.0 / 240 ? 1.0 / 240 : dt > 1.0 / 15 ? 1.0 / 15 : dt;
    if (n < 5) n++;
    double sum = 0;
    for (int i = 0; i < n; i++) sum += samples[i];
    InterlockedExchange64(&frame_step_bits, (LONG64)step_bits((float)(sum / n)));
}
SPLICE_AFTER(0x087a83e4, "int64_t _o=8; { uint64_t _p0=0,_p1=0;") {
    if (!realtime_on()) return;
    const uint64_t v = frame_step();
    G32(c->x[0] + 8) = (uint32_t)v;
    G32(c->x[0] + 12) = (uint32_t)(v >> 32);
    c->vreg[0][0] = (uint32_t)v;
    c->vreg[1][0] = (uint32_t)(v >> 32);
}
SPLICE_AFTER(0x088f689c, "recomp_store32(c,(c->x[8]+0ULL),_v); } }") {
    if (!realtime_on()) return;
    G32(c->x[8]) = (uint32_t)frame_step();
}

static void totk_set_interval(int n) {
    if (!totk_ctl) return;
    G32(totk_ctl + 0) = (uint32_t)n;      /* min */
    G32(totk_ctl + 4) = (uint32_t)n;      /* max */
    G32(totk_ctl + 0x14) = (uint32_t)n;   /* current: the simulation step is current / 2 */
    G32(totk_fw + 0x88) = (uint32_t)n;
    nw.interval = n;
}

static int pin_interval(uint64_t ctl, uint64_t fw, int requested);
static void read_interval_mode(void) {
    if (interval_mode != -1) return;
    const char *e = getenv("HOS_TARGET_INTERVAL");
    interval_mode = getenv("HOS_DYNAMIC_INTERVAL") ? 0 : e ? atoi(e) : -2;
    if (interval_mode > 4) interval_mode = 4;
}
static int pin_totk_interval(Ctx *c, int requested) {
    read_interval_mode();
    if (interval_mode == 0) return requested;
    const uint64_t sp = c->x[31];
    const uint64_t ctl = G64(sp + 0x20), fw = G64(sp + 0x28);
    if (!ctl || ctl != fw + 0x248 || (int)G32(ctl + 0x14) != requested) {
        static int warned;
        if (!warned++) hos_log("vi: swap interval %d: not TOTK's frame-rate controller, left as asked", requested);
        return requested;
    }
    return pin_interval(ctl, fw, requested);
}
static int pin_interval(uint64_t ctl, uint64_t fw, int requested) {
    const int first = !totk_ctl;
    totk_ctl = ctl;
    totk_fw = fw;
    /* automatic starts at 30 fps; later calls (the game's controller following a change of ours) keep our choice;
       real-time steps: always 1 */
    const int n = realtime_on() ? 1 : interval_mode > 0 ? interval_mode : first ? 2 : nw.interval;
    totk_set_interval(n);
    if (first)
        hos_log("vi: TOTK frame rate is ours: interval %d (%d fps)%s; it asked for %d", n, 60 / n,
                interval_mode < 0 ? ", 30 or 60 by load" : "", requested);
    return n;
}

/* Automatic interval (each queued frame): at 30 fps, frames that leave the presentation thread more than 19 ms of a
   33 ms period waiting (they take under ~14 ms) go to 60; at 60 fps, more than 25% of frames late for their vsync
   over 2 s go back to 30. After going back, 60 is tried again in 3 s (a loading hitch passes), then 6, 12, ... up to
   30 s while it keeps failing; 10 s held at 60 resets that. */
static void frame_done(int64_t waited_ns, int64_t now_vsync) {
    static int64_t window_start_vsync, frames, slow_frames, waited_total, last_vsync;
    static int64_t hold_until_vsync, backoff = 180, at60_since;
    if (interval_mode != -2 || !totk_ctl || realtime_on()) return;
    const int64_t period = last_vsync ? now_vsync - last_vsync : nw.interval;
    last_vsync = now_vsync;
    frames++;
    waited_total += waited_ns;
    if (period > nw.interval) slow_frames++;
    if (!window_start_vsync) window_start_vsync = now_vsync;
    if (now_vsync - window_start_vsync < 120) return;   /* 2 s */
    const double avg_wait_ms = waited_total / 1e6 / frames, slow = (double)slow_frames / frames;
    static int hist = -1;
    if (hist < 0) hist = getenv("HOS_FRAME_HIST") != NULL;
    if (hist) hos_log("vi: %lld frames, %lld late, dequeue waited %.1f ms on average", (long long)frames,
                      (long long)slow_frames, avg_wait_ms);
    if (now_vsync >= hold_until_vsync) {
        if (nw.interval == 2 && avg_wait_ms > 19.0) {
            totk_set_interval(1);
            hos_log("vi: frames take ~%.0f ms: 60 fps", 33.3 - avg_wait_ms);
            hold_until_vsync = now_vsync + 300;
            at60_since = now_vsync;
        } else if (nw.interval == 1 && slow > 0.25) {
            /* (a few late frames at 60 beat a steady 30: e.g. 12% late in the Room of Awakening's hall) */
            totk_set_interval(2);
            if (now_vsync - at60_since > 600) backoff = 180;   /* (it held 60 a while: this is a new hitch) */
            hos_log("vi: %.0f%% of frames late at 60 fps: 30 fps for %lld s", slow * 100, (long long)(backoff / 60));
            hold_until_vsync = now_vsync + backoff;
            backoff = backoff * 2 > 1800 ? 1800 : backoff * 2;
        }
    }
    window_start_vsync = now_vsync;
    frames = slow_frames = waited_total = 0;
}

static void nw_set_swap_interval(Ctx *c, void *u) {
    (void)u;
    static int uncapped = -1;
    if (uncapped < 0) uncapped = getenv("HOS_UNCAPPED") != NULL;
    static int logged;
    if (logged++ < 4) {
        hos_log("vi: the game sets swap interval %d%s", (int)ARG(1), uncapped ? " (uncapped: 1)" : "");
        if (getenv("HOS_VI_TRACE")) { void hos_backtrace(void); hos_backtrace(); }
    }
    nw.interval = uncapped ? 1 : pin_totk_interval(c, (int)ARG(1));
    RET(0);
}
static void nw_query(Ctx *c, void *u) {
    (void)u;
    int what = (int)ARG(1), v;
    switch (what) {
    case 0: case 6: v = NW_WIDTH; break;            /* WIDTH, DEFAULT_WIDTH */
    case 1: case 7: v = NW_HEIGHT; break;           /* HEIGHT, DEFAULT_HEIGHT */
    case 2: v = nw.format; break;                   /* FORMAT */
    case 3: v = 1; break;                           /* MIN_UNDEQUEUED_BUFFERS */
    case 4: v = 1; break;                           /* QUEUES_TO_WINDOW_COMPOSER */
    case 5: v = 1; break;                           /* CONCRETE_TYPE: NATIVE_WINDOW_SURFACE */
    default: v = 0; break;                          /* transform hint, consumer behind, dataspace, ... */
    }
    *(int32_t *)GP(ARG(2)) = v;
    RET(0);
}
static void nw_perform(Ctx *c, void *u) {
    (void)u;
    int op = (int)ARG(1);
    static int perform_trace = -1;
    if (perform_trace < 0) perform_trace = getenv("HOS_VI_PERFORM") != NULL;
    if (perform_trace) {
        static volatile LONG n;
        if (InterlockedIncrement(&n) <= 400)
            hos_log("vi: perform op %d (%#llx, %#llx, %#llx) at vsync %lld", op, (unsigned long long)ARG(2),
                    (unsigned long long)ARG(3), (unsigned long long)ARG(4), (long long)hos_vsync_count);
    }
    switch (op) {
    case 0: nw.usage = (int)ARG(2); break;          /* SET_USAGE */
    case 6: nw.transform = (int)ARG(2); break;      /* SET_BUFFERS_TRANSFORM */
    case 9: nw.format = (int)ARG(2); break;         /* SET_BUFFERS_FORMAT */
    case 0x13: {                                    /* Nintendo: (un)register a preallocated buffer */
        uint64_t slot = ARG(2), handle = ARG(3);
        if (slot >= NW_SLOTS) { RET((uint64_t)-22); return; }
        if (!handle) { nw.buffer[slot] = 0; break; }
        uint64_t b = nw.buffer[slot];
        if (!b) {
            b = hos_sys_alloc(ANB_SIZE, 0x10);
            G32(b) = 0x5f626672;                    /* '_bfr' */
            G32(b + 4) = ANB_SIZE;
            G64(b + 0x28) = G64(b + 0x30) = hos_hostcall("vi.NativeWindow.bufferRef", nw_nop, NULL);
        }
        G32(b + 0x38) = (uint32_t)ARG(4);           /* width */
        G32(b + 0x3c) = (uint32_t)ARG(5);           /* height */
        G32(b + 0x40) = G32(c->x[31]);              /* stride, the 9th argument */
        G32(b + 0x44) = (uint32_t)ARG(6);           /* format */
        G32(b + 0x48) = (uint32_t)ARG(7);           /* usage */
        G64(b + 0x60) = handle;
        nw.buffer[slot] = b;
        hos_log("vi: window buffer %d = %ux%u format %u (handle %#llx)", (int)slot, G32(b + 0x38), G32(b + 0x3c),
                G32(b + 0x44), (unsigned long long)handle);
        if (getenv("HOS_VI_TRACE"))
            for (int i = 0; i < 0x80; i += 16)
                hos_log("vi:   handle +%02x: %08x %08x %08x %08x", i, G32(handle + i), G32(handle + i + 4),
                        G32(handle + i + 8), G32(handle + i + 12));
        break;
    }
    default: break;  /* connect/disconnect, dimensions, crop, scaling, timestamps: no effect on us */
    }
    RET(0);
}
/* Buffer queue timing, as the console's: a queued frame goes on screen at the first vsync at least `interval` after
   the previous one (and after the queue); the frame it replaces is released then. Queue never blocks; dequeue
   blocks until its buffer is released. With two buffers the game renders one frame ahead of the display. */
static int64_t last_dequeue_wait_ns;   /* how long the last dequeue waited for its buffer (frame_done) */
static void nw_dequeue(Ctx *c, void *u) {
    (void)u;
    static int no_pacing = -1;
    if (no_pacing < 0) no_pacing = getenv("HOS_NO_PACING") != NULL;
    AcquireSRWLockExclusive(&nw_lock);
    int best = -1;
    for (int s = 0; s < NW_SLOTS; s++)   /* the registered slot released first (front buffer last) */
        if (nw.buffer[s] && s != nw.front && (best < 0 || nw.free_at[s] < nw.free_at[best])) best = s;
    if (best < 0)
        for (int s = 0; s < NW_SLOTS; s++)
            if (nw.buffer[s]) { best = s; break; }
    int64_t until = best >= 0 ? nw.free_at[best] : 0;
    ReleaseSRWLockExclusive(&nw_lock);
    if (best < 0) {
        hos_log("vi: dequeueBuffer with no registered buffers");
        RET((uint64_t)-19);                         /* -ENODEV */
        return;
    }
    int64_t v0 = hos_vsync_count;
    const int64_t wait_t0 = hos_now_ns();
    if (!no_pacing && vi_mode() == 0) {
        if (realtime_on()) frame_limit();
        else wait_vsync(until);
    }
    last_dequeue_wait_ns = hos_now_ns() - wait_t0;
    static int logged;
    if (getenv("HOS_VI_TRACE") && logged++ < 40)
        hos_log("vi: dequeue slot %d at vsync %lld, free at %lld, waited to %lld", best, (long long)v0, (long long)until,
                (long long)hos_vsync_count);
    G64(ARG(1)) = nw.buffer[best];
    *(int32_t *)GP(ARG(2)) = -1;                    /* no fence: the slot is free now */
    RET(0);
}
static void nw_queue(Ctx *c, void *u) {
    (void)u;
    int slot = -1;
    for (int i = 0; i < NW_SLOTS; i++)
        if (nw.buffer[i] == ARG(1)) slot = i;
    AcquireSRWLockExclusive(&nw_lock);
    int64_t now = hos_vsync_count, at = now + 1;
    if (nw.interval > 0 && nw.last_display + nw.interval > at) at = nw.last_display + nw.interval;
    nw.last_display = at;
    static int qlogged;
    if (getenv("HOS_VI_TRACE") && qlogged++ < 40)
        hos_log("vi: queue slot %d at vsync %lld, on screen at %lld (interval %d)", slot, (long long)now, (long long)at,
                nw.interval);
    /* The frame this one replaces is free from the vsync before this one shows: the renderer has its own copy of
       every frame it presents, so unlike the console's scan-out, nothing reads it after this queue. Freeing it at
       `at` itself (as the console does) made each dequeue wait for the next vsync: frames that took 21 ms waited
       until 33 ms, with the game's main thread blocked behind the presentation thread. Queues still come at most
       one per `interval` vsyncs (at grows past now + 1 when frames come faster). */
    if (nw.front >= 0) nw.free_at[nw.front] = at - 1;
    if (slot >= 0) { nw.front = slot; nw.free_at[slot] = INT64_MAX; }
    ReleaseSRWLockExclusive(&nw_lock);
    hos_present_slot = slot;
    if (slot >= 0 && rx_active()) {   /* the renderer presents the GPU's copy of the buffer */
        uint64_t h = hos_window_buffer_handle(slot);
        if (h && G32(h + 0x18) == 0xdaffcaff)
            hos_gpu_present(hos_nvmap_address(G32(h + 0x10)), G32(h + 0x40), G32(h + 0x44), vi_mode() == 2 ? at : 0);
    }
    if (vi_mode() == 1) wait_vsync(at);
    frame_done(last_dequeue_wait_ns, now);
    int64_t n = InterlockedIncrement64(&hos_present_count), t = hos_now_ns();
    frame_step_present(t);
    if (!present_window_start) present_window_start = t;
    presents_in_window++;
    if (t - present_window_start >= 5000000000LL) {
        hos_log("vi: %lld frames presented (%.1f fps over the last %.1f s)", (long long)n,
                presents_in_window * 1e9 / (double)(t - present_window_start), (t - present_window_start) / 1e9);
        present_window_start = t;
        presents_in_window = 0;
    }
    RET(0);
}
static void nw_cancel(Ctx *c, void *u) { (void)u; RET(0); }

/* The native NVN layer's window (HOS_NVN_NATIVE=1, hos/nvn_native.c): the game acquires and presents NVN window
   textures itself instead of going through NvWsi and this ANativeWindow. The same pacing and frame bookkeeping:
   acquire waits as dequeue does, present counts and times the frame as queue does. */
static int64_t native_next_vsync;
/* the native NVN layer's nvnWindowSetPresentInterval: the game calls it with its frame-rate controller in x20 (the
   emulated path finds the controller in the frame of the window's perform call) */
int hos_vi_native_interval(Ctx *c, int requested) {
    uint64_t hos_guest_size(void);
    read_interval_mode();
    if (interval_mode == 0) return requested;
    const uint64_t ctl = c->x[20];
    if (ctl != totk_ctl) {
        MEMORY_BASIC_INFORMATION mi;
        if (ctl < 0x8000000 || ctl + 0x20 >= hos_guest_size() || !VirtualQuery(GP(ctl), &mi, sizeof mi) ||
            mi.State != MEM_COMMIT || (int)G32(ctl + 0x14) != requested || G32(ctl) > G32(ctl + 4) || G32(ctl + 4) > 4) {
            static int warned;
            if (!warned++) hos_log("vi: present interval %d: not TOTK's frame-rate controller, left as asked", requested);
            return requested;
        }
    }
    return pin_interval(ctl, ctl - 0x248, requested);
}
void hos_vi_native_acquire(void) {
    static int no_pacing = -1;
    if (no_pacing < 0) no_pacing = getenv("HOS_NO_PACING") != NULL;
    const int64_t wait_t0 = hos_now_ns();
    if (!no_pacing) {
        if (realtime_on()) frame_limit();
        else wait_vsync(native_next_vsync);
    }
    last_dequeue_wait_ns = hos_now_ns() - wait_t0;
}
void hos_vi_native_presented(void) {
    const int64_t now = hos_vsync_count;
    AcquireSRWLockExclusive(&nw_lock);
    int64_t at = now + 1;
    if (nw.interval > 0 && nw.last_display + nw.interval > at) at = nw.last_display + nw.interval;
    nw.last_display = at;
    ReleaseSRWLockExclusive(&nw_lock);
    native_next_vsync = at - 1;
    frame_done(last_dequeue_wait_ns, now);
    int64_t n = InterlockedIncrement64(&hos_present_count), t = hos_now_ns();
    frame_step_present(t);
    if (!present_window_start) present_window_start = t;
    presents_in_window++;
    if (t - present_window_start >= 5000000000LL) {
        hos_log("vi: %lld frames presented (%.1f fps over the last %.1f s)", (long long)n,
                presents_in_window * 1e9 / (double)(t - present_window_start), (t - present_window_start) / 1e9);
        present_window_start = t;
        presents_in_window = 0;
    }
}
static void nw_deprecated(Ctx *c, void *u) { (void)u; RET((uint64_t)-38); }   /* -ENOSYS */

/* GetNativeWindow(void**, Layer*) -> Result: the ANativeWindow NVN's window is created on */
HLE(_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE) {
    if (!native_window) {
        uint64_t w = hos_sys_alloc(ANW_SIZE, 0x10);
        G32(w) = 0x5f776e64;                        /* '_wnd' */
        G32(w + 4) = ANW_SIZE;
        G64(w + 0x28) = G64(w + 0x30) = hos_hostcall("vi.NativeWindow.ref", nw_nop, NULL);
        G32(w + 0x3c) = 0;                          /* minSwapInterval */
        G32(w + 0x40) = 4;                          /* maxSwapInterval */
        float dpi = 160.0f;
        memcpy(GP(w + 0x44), &dpi, 4);
        memcpy(GP(w + 0x48), &dpi, 4);
        G64(w + 0x70) = hos_hostcall("vi.NativeWindow.setSwapInterval", nw_set_swap_interval, NULL);
        uint64_t dep = hos_hostcall("vi.NativeWindow.deprecated", nw_deprecated, NULL);
        G64(w + 0x78) = G64(w + 0x80) = G64(w + 0x88) = G64(w + 0xa0) = dep;
        G64(w + 0x90) = hos_hostcall("vi.NativeWindow.query", nw_query, NULL);
        G64(w + 0x98) = hos_hostcall("vi.NativeWindow.perform", nw_perform, NULL);
        G64(w + 0xa8) = hos_hostcall("vi.NativeWindow.dequeueBuffer", nw_dequeue, NULL);
        G64(w + 0xb0) = hos_hostcall("vi.NativeWindow.queueBuffer", nw_queue, NULL);
        G64(w + 0xb8) = hos_hostcall("vi.NativeWindow.cancelBuffer", nw_cancel, NULL);
        native_window = w;
    }
    G64(ARG(0)) = native_window;
    RET(0);
}
/* GetDisplayVsyncEvent(os::SystemEventType*, Display*) -> Result */
HLE(_ZN2nn2vi20GetDisplayVsyncEventEPNS_2os15SystemEventTypeEPNS0_7DisplayE) {
    hos_system_event_init(ARG(0), 1);
    if (nvsync < 8) vsync_events[InterlockedIncrement(&nvsync) - 1] = ARG(0);
    if (!vsync_thread) vsync_thread = CreateThread(NULL, 0, vsync_main, NULL, 0, NULL);
    RET(0);
}

/* nv::InitializeGraphics hands the driver a transfer memory over IPC: not needed by our nvdrv (hos/nvdrv.c), so it
   only logs. The allocator setters (nv::SetGraphicsAllocator ...) run as the SDK's own code: NvOs allocates through
   the game's callbacks. */
HLE(_ZN2nv18InitializeGraphicsEPvm) {
    hos_log("nv::InitializeGraphics(%#llx, %llu MB)", (unsigned long long)ARG(0), (unsigned long long)(ARG(1) >> 20));
}

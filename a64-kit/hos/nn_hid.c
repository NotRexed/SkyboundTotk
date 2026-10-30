/* nn::hid: one Pro Controller (FullKey style) on Npad No.1, from every XInput gamepad and the keyboard and mouse at
   once. Buttons map by position (the Xbox bottom face button is the Switch's bottom one, B).

   Keyboard: WASD left stick, IJKL right stick (camera), Space X (jump), E A (action), Shift/Q B (run/cancel),
   F Y (attack), R R, Z L, C ZL, V ZR, Ctrl left-stick click (crouch), Enter Plus, Tab Minus, arrows d-pad. Only
   while the game's window is in front (the key state is global).
   Mouse (after a click in the window, display.c; Esc releases): its motion is the right stick, deflection by speed.
   - and = change its sensitivity (x0.8 / x1.25, shown in the title bar, kept in <user>/mouse_sens.txt);
   HOS_MOUSE_SENS=<x> sets it, HOS_MOUSE_INVERT_Y=1 flips up/down.

   HOS_INPUT_SCRIPT (test runs): timed presses added to the input, "<seconds>:<button>[+<button>...][/<ms held>],..."
   with buttons A B X Y L R ZL ZR PLUS MINUS LS RS LEFT UP RIGHT DOWN and the stick directions LUP LDOWN LLEFT
   LRIGHT RUP RDOWN RLEFT RRIGHT; held 200 ms unless given. Example: "50:A,52:A,80:LUP/4000".

   Npad states share one layout (NpadCommonState): { s64 sampling; u64 buttons; s32 lx, ly; s32 rx, ry; u32 attributes;
   pad } = 0x28 bytes. */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <xinput.h>

enum { B_A = 0, B_B, B_X, B_Y, B_LS, B_RS, B_L, B_R, B_ZL, B_ZR, B_PLUS, B_MINUS, B_LEFT, B_UP, B_RIGHT, B_DOWN };
#define STATE_SIZE 0x28
#define NPAD_NO1 0
#define STYLE_FULLKEY 1u
#define ATTR_CONNECTED 1u

typedef DWORD(WINAPI *XInputGetStateFn)(DWORD, XINPUT_STATE *);
static XInputGetStateFn xinput_get;
static int64_t sampling;

static int key(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

int hos_mouse_take(int32_t *dx, int32_t *dy);   /* display.c */

/* The captured mouse as right-stick deflection: motion over the last >= 8 ms as a speed (the game reads the pad
   several times a frame: those reads see one value), smoothed a little; at sensitivity 1 full deflection at 1000
   counts/s. The smallest motions start past the game's dead zone. */
double hos_mouse_sens = 0;   /* (display.c shows it in the title) */
extern const char *hos_user_dir;

static void mouse_sens_save(void) {
    char path[512];
    snprintf(path, sizeof path, "%s/mouse_sens.txt", hos_user_dir ? hos_user_dir : "user");
    FILE *f = fopen(path, "w");
    if (f) { fprintf(f, "%.3f\n", hos_mouse_sens); fclose(f); }
}

static void mouse_sens_keys(void) {
    static int prev_minus, prev_plus;
    const int minus = key(VK_OEM_MINUS) || key(VK_SUBTRACT), plus = key(VK_OEM_PLUS) || key(VK_ADD);
    if (minus && !prev_minus) { hos_mouse_sens *= 0.8; mouse_sens_save(); }
    if (plus && !prev_plus) { hos_mouse_sens *= 1.25; mouse_sens_save(); }
    if (hos_mouse_sens < 0.05) hos_mouse_sens = 0.05;
    if (hos_mouse_sens > 20) hos_mouse_sens = 20;
    prev_minus = minus;
    prev_plus = plus;
}

static int mouse_stick(int32_t *rx, int32_t *ry) {
    static int64_t last;
    static double vx, vy;
    static int captured;
    static int invert = -1;
    if (invert < 0) {
        invert = getenv("HOS_MOUSE_INVERT_Y") != NULL;
        hos_mouse_sens = 1.0;
        char path[512];
        snprintf(path, sizeof path, "%s/mouse_sens.txt", hos_user_dir ? hos_user_dir : "user");
        FILE *f = fopen(path, "r");
        if (f) {
            double v;
            if (fscanf(f, "%lf", &v) == 1 && v > 0) hos_mouse_sens = v;
            fclose(f);
        }
        const char *e = getenv("HOS_MOUSE_SENS");
        if (e && atof(e) > 0) hos_mouse_sens = atof(e);
    }
    mouse_sens_keys();
    const double scale = hos_mouse_sens / 1000.0;
    const int64_t now = hos_now_ns();
    if (!last) last = now;
    if (now - last >= 8000000) {
        int32_t dx, dy;
        captured = hos_mouse_take(&dx, &dy);
        double dt = (now - last) / 1e9;
        if (dt > 0.1) dt = 0.1;
        last = now;
        const double tx = dx / dt * scale, ty = (invert ? dy : -dy) / dt * scale;
        vx = captured ? vx * 0.2 + tx * 0.8 : 0;
        vy = captured ? vy * 0.2 + ty * 0.8 : 0;
    }
    if (!captured) return 0;
    const double v[2] = {vx, vy};
    int32_t *out[2] = {rx, ry};
    for (int i = 0; i < 2; i++) {
        double a = v[i] < 0 ? -v[i] : v[i];
        if (a < 0.002) { *out[i] = 0; continue; }
        a = 0.2 + 0.8 * (a > 1 ? 1 : a);
        *out[i] = (int32_t)((v[i] < 0 ? -a : a) * 32767);
    }
    return 1;
}

static int game_in_front(void) {
    DWORD pid = 0;
    HWND w = GetForegroundWindow();
    if (w) GetWindowThreadProcessId(w, &pid);
    int front = pid == GetCurrentProcessId();
#ifndef __EMSCRIPTEN__
    static int last_normal_was_us = 1, was_front = 1;
    if (w && !front) {
        const LONG_PTR ex = GetWindowLongPtrA(w, GWL_EXSTYLE);
        const int overlay = (ex & (WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)) != 0;
        if (overlay && last_normal_was_us) front = 1;
        else last_normal_was_us = 0;
    } else if (front) {
        last_normal_was_us = 1;
    }
    if (front != was_front) {
        char cls[64] = "";
        if (w) GetClassNameA(w, cls, sizeof cls);
        hos_log("hid: input %s (foreground window: process %lu, class \"%s\")", front ? "back" : "paused: the game "
                "window is not in front", (unsigned long)pid, cls);
        was_front = front;
    }
#endif
    return front;
}

/* HOS_INPUT_SCRIPT, and HOS_INPUT_LIVE=<file>: lines appended to that file while the game runs are presses in the
   script's syntax, their times counted from when the line is read ("0:LUP/2000,2:A"): a test can steer the game
   by looking at its screenshots. */
typedef struct { int64_t start_ms, end_ms; uint64_t buttons; int32_t stick[4]; } Press;
#define MAX_PRESSES 1024
static Press presses[MAX_PRESSES];
static int npresses = -1;
static int64_t script_t0, script_end_ms;

static int parse_presses(const char *s, int64_t base_ms, int live);

static void parse_script(void) {
    {   /* start SDL's controller thread now: pads are open by the time the game window comes to the front */
        int hos_sdl_pads(uint64_t *buttons, int32_t stick[4]);
        uint64_t b = 0;
        int32_t st[4] = {0};
        hos_sdl_pads(&b, st);
    }
    npresses = 0;
    script_t0 = hos_now_ns();
    parse_presses(getenv("HOS_INPUT_SCRIPT"), 0, 0);
    if (npresses)
        hos_log("hid: input script of %d presses: the player's devices are ignored until %.0f s", npresses,
                (script_end_ms + 5000) / 1000.0);
}

static void live_input(void) {
    static const char *path;
    static int64_t next_check, offset;
    static int looked;
    if (!looked) { looked = 1; path = getenv("HOS_INPUT_LIVE"); }
    if (!path) return;
    const int64_t now = hos_now_ns();
    if (now < next_check) return;
    next_check = now + 30000000;
    FILE *f = fopen(path, "rb");
    if (!f) return;
    if (fseek(f, (long)offset, SEEK_SET) == 0) {
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            const size_t n = strlen(line);
            if (!n || line[n - 1] != '\n') break;   /* a line still being written: read it next time */
            offset += (int64_t)n;
            const int64_t at = (now - script_t0) / 1000000;
            if (line[0] == '!') {   /* commands: !dump = a frame dump of the next frame (as RX_DUMP_AT) */
                void rx_request_frame_dump(void);   /* gpu/render/rx_debug.cpp */
                if (!strncmp(line, "!dump", 5)) rx_request_frame_dump();
                extern volatile int hos_nvn_sync_log_on;   /* nvn_native.c: "!synclog" */
                if (!strncmp(line, "!synclog", 8)) hos_nvn_sync_log_on = 1;
                void rx_set_screenshot_ms(int64_t ms);   /* gpu/render/rx_core.cpp: "!shots <ms>" */
                if (!strncmp(line, "!shots ", 7)) rx_set_screenshot_ms(atoll(line + 7));
                void rx_request_rdoc_capture(int frames);   /* gpu/render/rx_debug.cpp: "!rdoc [frames]" */
                if (!strncmp(line, "!rdoc", 5)) rx_request_rdoc_capture(atoi(line + 5));
                hos_log("hid: live command at %.1f s: %.*s", at / 1000.0, (int)(n > 1 ? n - 1 : 0), line);
                continue;
            }
            const int got = parse_presses(line, at, 1);
            hos_log("hid: live input at %.1f s: %d presses (%.*s)", at / 1000.0, got, (int)(n > 1 ? n - 1 : 0), line);
        }
    }
    fclose(f);
}

static int parse_presses(const char *s, int64_t base_ms, int live) {
    static const struct { const char *name; int b; int axis; int dir; } names[] = {
        {"A", B_A}, {"B", B_B}, {"X", B_X}, {"Y", B_Y}, {"LS", B_LS}, {"RS", B_RS}, {"L", B_L}, {"R", B_R},
        {"ZL", B_ZL}, {"ZR", B_ZR}, {"PLUS", B_PLUS}, {"MINUS", B_MINUS}, {"LEFT", B_LEFT}, {"UP", B_UP},
        {"RIGHT", B_RIGHT}, {"DOWN", B_DOWN}, {"LUP", -1, 1, 1}, {"LDOWN", -1, 1, -1}, {"LLEFT", -1, 0, -1},
        {"LRIGHT", -1, 0, 1}, {"RUP", -1, 3, 1}, {"RDOWN", -1, 3, -1}, {"RLEFT", -1, 2, -1}, {"RRIGHT", -1, 2, 1}};
    int got = 0;
    const int64_t now_ms = (hos_now_ns() - script_t0) / 1000000;
    while (s && *s) {
        while (*s == ' ' || *s == '\t') s++;
        Press p;
        memset(&p, 0, sizeof p);
        char *end;
        double t = strtod(s, &end);
        if (end == s || *end != ':') break;
        s = end + 1;
        int64_t hold = 200;
        for (;;) {
            size_t n = strcspn(s, "+/,\r\n");
            for (unsigned i = 0; i < sizeof names / sizeof *names; i++)
                if (strlen(names[i].name) == n && !strncmp(s, names[i].name, n)) {
                    if (names[i].b >= 0) p.buttons |= 1ULL << names[i].b;
                    else p.stick[names[i].axis] = names[i].dir * 32767;
                }
            s += n;
            if (*s != '+') break;
            s++;
        }
        if (*s == '/') hold = strtol(s + 1, (char **)&s, 10);
        p.start_ms = base_ms + (int64_t)(t * 1000);
        p.end_ms = p.start_ms + hold;
        int slot = npresses < MAX_PRESSES ? npresses : -1;
        if (slot < 0)   /* full: reuse a press that is over */
            for (int i = 0; i < MAX_PRESSES && slot < 0; i++)
                if (presses[i].end_ms < now_ms) slot = i;
        if (slot < 0) break;
        if (!live && p.end_ms > script_end_ms) script_end_ms = p.end_ms;
        presses[slot] = p;
        if (slot == npresses) npresses++;
        got++;
        if (*s == ',') s++;
    }
    return got;
}

static void scripted_input(uint64_t *buttons, int32_t stick[4]) {
    if (npresses < 0) parse_script();
    live_input();
    const int64_t ms = (hos_now_ns() - script_t0) / 1000000;
    for (int i = 0; i < npresses; i++)
        if (ms >= presses[i].start_ms && ms < presses[i].end_ms) {
            *buttons |= presses[i].buttons;
            for (int k = 0; k < 4; k++)
                if (presses[i].stick[k]) stick[k] = presses[i].stick[k];
        }
}

static void stick_max(int32_t stick[4], const int32_t v[4]) {
    for (int i = 0; i < 4; i++)
        if ((v[i] < 0 ? -v[i] : v[i]) > (stick[i] < 0 ? -stick[i] : stick[i])) stick[i] = v[i];
}

static void pads_read(uint64_t *buttons, int32_t stick[4]) {
    static int64_t next_read, retry[XUSER_MAX_COUNT];
    static int was_connected[XUSER_MAX_COUNT];
    static uint64_t cached_buttons;
    static int32_t cached_stick[4];
    const int64_t now = hos_now_ns();
    if (now < next_read) {
        *buttons |= cached_buttons;
        stick_max(stick, cached_stick);
        return;
    }
    next_read = now + 2000000;
    uint64_t b = 0;
    int32_t s[4] = {0};
    {   /* every controller SDL knows (DualSense, DualShock 4, Switch Pro, ...: hos/pad_sdl.c) */
        int hos_sdl_pads(uint64_t *buttons, int32_t stick[4]);
        static int was;
        const int n = hos_sdl_pads(&b, s);
        if (n != was) hos_log("hid: %d controller%s through SDL", n, n == 1 ? "" : "s");
        was = n;
    }
#ifdef __EMSCRIPTEN__
    xinput_get = XInputGetState;
#endif
    if (!xinput_get) {
        static int64_t load_retry;
        if (now >= load_retry) {
            load_retry = now + 5000000000LL;
            HMODULE x = LoadLibraryA("xinput1_4.dll");
            if (!x) x = LoadLibraryA("xinput9_1_0.dll");
            if (x) xinput_get = (XInputGetStateFn)GetProcAddress(x, "XInputGetState");
        }
    }
    static const struct { WORD x; int b; } map[] = {
        {XINPUT_GAMEPAD_A, B_B}, {XINPUT_GAMEPAD_B, B_A}, {XINPUT_GAMEPAD_X, B_Y}, {XINPUT_GAMEPAD_Y, B_X},
        {XINPUT_GAMEPAD_LEFT_THUMB, B_LS}, {XINPUT_GAMEPAD_RIGHT_THUMB, B_RS},
        {XINPUT_GAMEPAD_LEFT_SHOULDER, B_L}, {XINPUT_GAMEPAD_RIGHT_SHOULDER, B_R},
        {XINPUT_GAMEPAD_START, B_PLUS}, {XINPUT_GAMEPAD_BACK, B_MINUS},
        {XINPUT_GAMEPAD_DPAD_LEFT, B_LEFT}, {XINPUT_GAMEPAD_DPAD_UP, B_UP},
        {XINPUT_GAMEPAD_DPAD_RIGHT, B_RIGHT}, {XINPUT_GAMEPAD_DPAD_DOWN, B_DOWN}};
    for (int p = 0; p < XUSER_MAX_COUNT && xinput_get; p++) {
        if (now < retry[p]) continue;
        XINPUT_STATE xs;
        if (xinput_get((DWORD)p, &xs) != ERROR_SUCCESS) {
            if (was_connected[p]) hos_log("hid: controller %d disconnected", p);
            was_connected[p] = 0;
            retry[p] = now + 1000000000LL;
            continue;
        }
        if (!was_connected[p]) hos_log("hid: controller %d connected", p);
        was_connected[p] = 1;
        const WORD w = xs.Gamepad.wButtons;
        for (unsigned i = 0; i < sizeof map / sizeof *map; i++)
            if (w & map[i].x) b |= 1ULL << map[i].b;
        if (xs.Gamepad.bLeftTrigger > 64) b |= 1ULL << B_ZL;
        if (xs.Gamepad.bRightTrigger > 64) b |= 1ULL << B_ZR;
        const int16_t raw[4] = {xs.Gamepad.sThumbLX, xs.Gamepad.sThumbLY, xs.Gamepad.sThumbRX, xs.Gamepad.sThumbRY};
        int32_t v[4];
        for (int i = 0; i < 4; i++) {
            const int a = raw[i] < -32767 ? -32767 : raw[i];
            v[i] = (a > 7849 || a < -7849) ? a : 0;
        }
        stick_max(s, v);
    }
    cached_buttons = b;
    memcpy(cached_stick, s, sizeof s);
    *buttons |= b;
    stick_max(stick, s);
}

static void read_input(uint64_t *buttons, int32_t stick[4]) {
    *buttons = 0;
    memset(stick, 0, 4 * sizeof *stick);
    scripted_input(buttons, stick);
    /* A scripted run (a test) takes no input from the player's devices until 5 s after its last press (a variable
       left set in a shell no longer leaves the game without controls), and otherwise the pads count only
       while the game's window is in front, like the keyboard: XInput is global, so a test run in the background
       used to follow the player's pad (and pressed things in its own menus). */
    if ((npresses > 0 && (hos_now_ns() - script_t0) / 1000000 < script_end_ms + 5000) || !game_in_front()) return;
    pads_read(buttons, stick);
    struct { int vk; int b; } kmap[] = {
        {VK_SPACE, B_X}, {'E', B_A}, {VK_SHIFT, B_B}, {'Q', B_B}, {'F', B_Y}, {'R', B_R}, {'Z', B_L}, {'C', B_ZL},
        {'V', B_ZR}, {VK_CONTROL, B_LS}, {VK_RETURN, B_PLUS}, {VK_TAB, B_MINUS}, {VK_LEFT, B_LEFT}, {VK_UP, B_UP},
        {VK_RIGHT, B_RIGHT}, {VK_DOWN, B_DOWN}};
    for (unsigned i = 0; i < sizeof kmap / sizeof *kmap; i++)
        if (key(kmap[i].vk)) *buttons |= 1ULL << kmap[i].b;
    int32_t mouse[4] = {0};
    if (mouse_stick(&mouse[2], &mouse[3])) stick_max(stick, mouse);
    const int32_t keys[4] = {(key('D') - key('A')) * 32767, (key('W') - key('S')) * 32767,
                             (key('L') - key('J')) * 32767, (key('I') - key('K')) * 32767};
    stick_max(stick, keys);
}

static void vibration_send(uint64_t handle, uint64_t value);
volatile int64_t hos_hid_last_read;

/* GetNpadStates(State* out, int count, unsigned int const& npadId): the latest state first */
static void npad_states(Ctx *c) {
    uint64_t out = ARG(0);
    int count = (int)ARG(1);
    uint32_t id = G32(ARG(2));
    uint64_t buttons = 0;
    int32_t stick[4] = {0};
    int connected = id == NPAD_NO1;
    {
        static int64_t last_read;
        static uint32_t seen_ids[16];
        static int nseen;
        const int64_t now = hos_now_ns();
        if (connected && last_read && now - last_read > 2000000000LL)
            hos_log("hid: the game read no controller for %.1f s (lr %#llx)", (now - last_read) / 1e9,
                    (unsigned long long)c->x[30]);
        if (connected) last_read = hos_hid_last_read = now;
        int known = 0;
        for (int i = 0; i < nseen; i++) known |= seen_ids[i] == id;
        if (!known && nseen < 16) {
            seen_ids[nseen++] = id;
            hos_log("hid: the game reads npad %#x (%s, lr %#llx)", id, connected ? "No.1" : "not connected",
                    (unsigned long long)c->x[30]);
        }
    }
    if (connected) read_input(&buttons, stick);
    uint64_t hos_title_input(uint64_t buttons);   /* totk_title.c: the title menu's credits entries */
    if (connected) buttons = hos_title_input(buttons);
    uint64_t hos_frame_step_buttons(uint64_t buttons);   /* nn_vi.c: the frame step of the wheel and quick menus */
    if (connected) buttons = hos_frame_step_buttons(buttons);
    int64_t s = ++sampling;
    for (int i = 0; i < count; i++) {
        uint64_t e = out + (uint64_t)i * STATE_SIZE;
        memset(GP(e), 0, STATE_SIZE);
        G64(e) = (uint64_t)(s - i);
        G64(e + 8) = buttons;
        memcpy(GP(e + 16), stick, 16);
        G32(e + 32) = connected ? ATTR_CONNECTED : 0;
    }
    RET(count);
}
HLE(_ZN2nn3hid13GetNpadStatesEPNS0_16NpadFullKeyStateEiRKj) { npad_states(c); }
HLE(_ZN2nn3hid13GetNpadStatesEPNS0_17NpadHandheldStateEiRKj) { npad_states(c); }
HLE(_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyDualStateEiRKj) { npad_states(c); }
HLE(_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyLeftStateEiRKj) { npad_states(c); }
HLE(_ZN2nn3hid13GetNpadStatesEPNS0_17NpadJoyRightStateEiRKj) { npad_states(c); }
/* GetNpadStyleSet(unsigned int const&) -> BitFlagSet<32, NpadStyleTag> */
HLE(_ZN2nn3hid15GetNpadStyleSetERKj) { RET(G32(ARG(0)) == NPAD_NO1 ? STYLE_FULLKEY : 0); }

HLE(_ZN2nn3hid14InitializeNpadEv) {}
HLE(_ZN2nn3hid18InitializeKeyboardEv) {}
HLE(_ZN2nn3hid15InitializeMouseEv) {}
HLE(_ZN2nn3hid14DisconnectNpadERKj) {}
HLE(_ZN2nn3hid18SetNpadJoyHoldTypeENS0_15NpadJoyHoldTypeE) {}
HLE(_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm) {}
HLE(_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE) {}
HLE(_ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE) {}

/* ShowControllerSupport(ControllerSupportResultInfo*, ControllerSupportArg const&) -> Result: one player, No.1 */
HLE(_ZN2nn3hid21ShowControllerSupportEPNS0_27ControllerSupportResultInfoERKNS0_20ControllerSupportArgE) {
    memset(GP(ARG(0)), 0, 8);
    *(uint8_t *)GP(ARG(0)) = 1;
    G32(ARG(0) + 4) = NPAD_NO1;
    RET(0);
}

/* Handles: six-axis ones are opaque u32s; a vibration handle is {u8 style type (3 = full key), u8 npad id, u8 device
   index (0 left, 1 right), u8 reserved}, which the SDK's vibration classes read. Vibration goes nowhere yet. */
HLE(_ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE) {
    for (int i = 0; i < (int)ARG(1); i++) G32(ARG(0) + 4 * (uint64_t)i) = 0x100u + (uint32_t)i;
    RET((int)ARG(1));
}
HLE(_ZN2nn3hid22GetSixAxisSensorStatesEPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE) {
    static int64_t six_sampling;
    const uint64_t out = ARG(0);
    const int count = (int)ARG(1) < 1 ? 0 : (int)ARG(1) > 32 ? 32 : (int)ARG(1);
    const int64_t s = ++six_sampling;
    static const float at_rest[18] = {0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < count; i++) {
        const uint64_t e = out + (uint64_t)i * 0x60;
        memset(GP(e), 0, 0x60);
        G64(e) = 5000000;
        G64(e + 8) = (uint64_t)(s - i);
        memcpy(GP(e + 0x10), at_rest, sizeof at_rest);
        G32(e + 0x58) = 1;
    }
    RET(count);
}
HLE(_ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE) {
    int n = (int)ARG(1) < 2 ? (int)ARG(1) : 2;
    for (int i = 0; i < n; i++) G32(ARG(0) + 4 * (uint64_t)i) = 3u | (G32(ARG(2)) & 0xFF) << 8 | (uint32_t)i << 16;
    RET(n);
}
HLE(_ZN2nn3hid25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE) {}
HLE(_ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE) { vibration_send(ARG(0), ARG(1)); }

/* nn::hid::detail vibration: what the SDK's own classes (VibrationNode, VibrationTarget, VibrationPlayer) call; on
   hardware each is an IPC to hid. A device is an LRA on its side of the controller, always mounted and permitted. */
/* A vibration value for one of the two motors of the pad (handle +2 = device index): the pads that rumble (SDL) get
   the stronger of the two sides' low and high band amplitudes. */
static void vibration_send(uint64_t handle, uint64_t value) {
    static float amp[2][2];
    float v[4];
    memcpy(v, GP(value), sizeof v);
    const int dev = (G32(handle) >> 16) & 1;
    amp[dev][0] = v[0];
    amp[dev][1] = v[2];
    void hos_sdl_rumble(float low, float high);   /* pad_sdl.c */
    hos_sdl_rumble(amp[0][0] > amp[1][0] ? amp[0][0] : amp[1][0], amp[0][1] > amp[1][1] ? amp[0][1] : amp[1][1]);
}
static void vibration_value_idle(uint64_t v) {   /* VibrationValue {f32 amp_low, freq_low, amp_high, freq_high} */
    float idle[4] = {0.0f, 160.0f, 0.0f, 320.0f};
    memcpy(GP(v), idle, sizeof idle);
}
HLE(_ZN2nn3hid6detail15PermitVibrationEb) { RET(0); }
HLE(_ZN2nn3hid6detail20IsVibrationPermittedEPb) { *(uint8_t *)GP(ARG(0)) = 1; RET(0); }
HLE(_ZN2nn3hid6detail27BeginPermitVibrationSessionEv) { RET(0); }
HLE(_ZN2nn3hid6detail25EndPermitVibrationSessionEv) { RET(0); }
HLE(_ZN2nn3hid6detail25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE) { RET(0); }
/* GetVibrationDeviceInfo(VibrationDeviceInfo* {u32 type; u32 position}, handle const&): LRA (1), left 1 / right 2 */
HLE(_ZN2nn3hid6detail22GetVibrationDeviceInfoEPNS0_19VibrationDeviceInfoERKNS0_21VibrationDeviceHandleE) {
    G32(ARG(0)) = 1;
    G32(ARG(0) + 4) = ((G32(ARG(1)) >> 16) & 1) + 1;
    RET(0);
}
HLE(_ZN2nn3hid6detail18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE) {
    vibration_send(ARG(0), ARG(1));
    RET(0);
}
HLE(_ZN2nn3hid6detail19SendVibrationValuesEPKNS0_21VibrationDeviceHandleEPKNS0_14VibrationValueEi) {
    for (int i = 0; i < (int)ARG(2) && i < 8; i++) vibration_send(ARG(0) + 4 * (uint64_t)i, ARG(1) + 16 * (uint64_t)i);
    RET(0);
}
HLE(_ZN2nn3hid6detail23GetActualVibrationValueEPNS0_14VibrationValueERKNS0_21VibrationDeviceHandleE) {
    vibration_value_idle(ARG(0));
    RET(0);
}
HLE(_ZN2nn3hid6detail25SendVibrationGcErmCommandERKNS0_21VibrationDeviceHandleERKNS0_21VibrationGcErmCommandE) { RET(0); }
HLE(_ZN2nn3hid6detail30GetActualVibrationGcErmCommandEPNS0_21VibrationGcErmCommandERKNS0_21VibrationDeviceHandleE) {
    G32(ARG(0)) = 0;   /* Stop */
    RET(0);
}
HLE(_ZN2nn3hid6detail24IsVibrationDeviceMountedEPbRKNS0_21VibrationDeviceHandleE) { *(uint8_t *)GP(ARG(0)) = 1; RET(0); }
HLE(_ZN2nn3hid6detail21EnqueueVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE) {
    vibration_send(ARG(0), ARG(1));
    RET(0);
}
HLE(_ZN2nn3hid6detail28FlushEnqueuedVibrationValuesEv) { RET(0); }

/* Game controllers through SDL2 (third_party/sdl2/SDL2.dll, zlib license, loaded at run time): DualSense,
   DualShock 4, Switch Pro, Xbox and most other pads, over USB or Bluetooth. XInput alone sees only Xbox-style pads.

   A thread of its own opens every controller SDL knows (and ones plugged in later), polls them every 4 ms and keeps
   the merged state; nn_hid.c ORs it into the input with XInput, keyboard and mouse. Without SDL2.dll nothing
   changes. HOS_NO_SDL=1 leaves SDL out. Buttons map by position, as for XInput: the bottom face button is the
   Switch's B (cross on a DualSense, A on an Xbox pad), the right one A, left Y, top X. */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

void hos_host_thread_register_self(const char *name);   /* hos_prof.c */

/* the few SDL2 entry points used (SDL.h's types: SDL_GameController* is opaque, SDL_bool/Uint8/Sint16) */
typedef int (*InitFn)(uint32_t);
typedef int (*SetHintFn)(const char *, const char *);
typedef int (*NumJoysticksFn)(void);
typedef int (*IsGameControllerFn)(int);
typedef void *(*OpenFn)(int);
typedef void (*CloseFn)(void *);
typedef int (*GetAttachedFn)(void *);
typedef void (*UpdateFn)(void);
typedef uint8_t (*GetButtonFn)(void *, int);
typedef int16_t (*GetAxisFn)(void *, int);
typedef const char *(*NameFn)(void *);
typedef void *(*GetJoystickFn)(void *);
typedef int32_t (*InstanceIdFn)(void *);
typedef int32_t (*DeviceInstanceIdFn)(int);
typedef int (*RumbleFn)(void *, uint16_t, uint16_t, uint32_t);
typedef const char *(*GetErrorFn)(void);

static struct {
    InitFn Init;
    SetHintFn SetHint;
    NumJoysticksFn NumJoysticks;
    IsGameControllerFn IsGameController;
    OpenFn Open;
    CloseFn Close;
    GetAttachedFn GetAttached;
    UpdateFn Update;
    GetButtonFn GetButton;
    GetAxisFn GetAxis;
    NameFn Name;
    GetJoystickFn GetJoystick;
    InstanceIdFn JoystickInstanceID;
    DeviceInstanceIdFn JoystickGetDeviceInstanceID;
    RumbleFn Rumble;
    GetErrorFn GetError;
} sdl;

enum { SDL_INIT_GAMECONTROLLER = 0x2000 };
enum { BTN_A, BTN_B, BTN_X, BTN_Y, BTN_BACK, BTN_GUIDE, BTN_START, BTN_LSTICK, BTN_RSTICK, BTN_LSHOULDER, BTN_RSHOULDER,
       BTN_UP, BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_MISC1, BTN_N = 21 };
enum { AX_LX, AX_LY, AX_RX, AX_RY, AX_LT, AX_RT };
/* nn_hid.c's button bits (B_A = 0 ...) */
enum { H_A = 0, H_B, H_X, H_Y, H_LS, H_RS, H_L, H_R, H_ZL, H_ZR, H_PLUS, H_MINUS, H_LEFT, H_UP, H_RIGHT, H_DOWN };

#define MAX_PADS 8
static struct { void *gc; int32_t id; } pads[MAX_PADS];
static SRWLOCK state_lock = SRWLOCK_INIT;
static uint64_t state_buttons;
static int32_t state_stick[4];
static int state_count;
static volatile LONG rumble_low, rumble_high;   /* 0..65535, set by the vibration HLE */

static int load(void) {
    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, path, sizeof path);
    while (n && path[n - 1] != '\\' && path[n - 1] != '/') n--;
    snprintf(path + n, sizeof path - n, "SDL2.dll");
    HMODULE m = LoadLibraryA(path);
    if (!m) m = LoadLibraryA("SDL2.dll");
    if (!m) return 0;
#define GET(field, name) if (!(*(FARPROC *)&sdl.field = GetProcAddress(m, name))) return 0
    GET(Init, "SDL_Init");
    GET(SetHint, "SDL_SetHint");
    GET(NumJoysticks, "SDL_NumJoysticks");
    GET(IsGameController, "SDL_IsGameController");
    GET(Open, "SDL_GameControllerOpen");
    GET(Close, "SDL_GameControllerClose");
    GET(GetAttached, "SDL_GameControllerGetAttached");
    GET(Update, "SDL_GameControllerUpdate");
    GET(GetButton, "SDL_GameControllerGetButton");
    GET(GetAxis, "SDL_GameControllerGetAxis");
    GET(Name, "SDL_GameControllerName");
    GET(GetJoystick, "SDL_GameControllerGetJoystick");
    GET(JoystickInstanceID, "SDL_JoystickInstanceID");
    GET(JoystickGetDeviceInstanceID, "SDL_JoystickGetDeviceInstanceID");
    GET(Rumble, "SDL_GameControllerRumble");
    GET(GetError, "SDL_GetError");
#undef GET
    return 1;
}

static int32_t stick_value(int16_t v, int flip) {
    int32_t a = v < -32767 ? -32767 : v;
    if (flip) a = -a;
    return (a > 7849 || a < -7849) ? a : 0;   /* XInput's dead zone, as nn_hid.c uses */
}

static DWORD WINAPI pad_main(void *arg) {
    (void)arg;
    hos_host_thread_register_self("gamepads (SDL)");
    /* poll in the background, whatever window is in front (nn_hid.c decides when input counts) */
    sdl.SetHint("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1");
    sdl.SetHint("SDL_JOYSTICK_HIDAPI_PS5_RUMBLE", "1");
    sdl.SetHint("SDL_JOYSTICK_HIDAPI_PS4_RUMBLE", "1");
    /* (the map below is by position; SDL2 reports Nintendo pads by their labels unless told not to, which put a Switch Pro
       Controller's A, on the right, at the bottom button's place: A acted as B and X as Y: SkyboundTotk issue 12) */
    sdl.SetHint("SDL_GAMECONTROLLER_USE_BUTTON_LABELS", "0");
    if (sdl.Init(SDL_INIT_GAMECONTROLLER) != 0) {
        hos_log("hid: SDL could not start its controller support: %s", sdl.GetError());
        return 0;
    }
    hos_log("hid: SDL controller support ready (DualSense, DualShock 4, Switch Pro, Xbox and others)");
    int64_t next_scan = 0;
    uint16_t last_lo = 0, last_hi = 0;
    for (;;) {
        sdl.Update();
        const int64_t now = hos_now_ns();
        if (now >= next_scan) {   /* new controllers (and ones that went away) */
            next_scan = now + 1000000000LL;
            for (int i = 0; i < MAX_PADS; i++)
                if (pads[i].gc && !sdl.GetAttached(pads[i].gc)) {
                    hos_log("hid: controller \"%s\" disconnected (SDL)", sdl.Name(pads[i].gc));
                    sdl.Close(pads[i].gc);
                    pads[i].gc = NULL;
                }
            const int nj = sdl.NumJoysticks();
            for (int j = 0; j < nj; j++) {
                if (!sdl.IsGameController(j)) continue;
                const int32_t id = sdl.JoystickGetDeviceInstanceID(j);
                int have = 0, slot = -1;
                for (int i = 0; i < MAX_PADS; i++) {
                    if (pads[i].gc && pads[i].id == id) have = 1;
                    if (!pads[i].gc && slot < 0) slot = i;
                }
                if (have || slot < 0) continue;
                void *gc = sdl.Open(j);
                if (!gc) continue;
                pads[slot].gc = gc;
                pads[slot].id = sdl.JoystickInstanceID(sdl.GetJoystick(gc));
                hos_log("hid: controller \"%s\" connected (SDL)", sdl.Name(gc));
            }
        }
        uint64_t b = 0;
        int32_t s[4] = {0};
        int count = 0;
        static const struct { int sdl, hid; } map[] = {
            {BTN_A, H_B}, {BTN_B, H_A}, {BTN_X, H_Y}, {BTN_Y, H_X}, {BTN_LSTICK, H_LS}, {BTN_RSTICK, H_RS},
            {BTN_LSHOULDER, H_L}, {BTN_RSHOULDER, H_R}, {BTN_START, H_PLUS}, {BTN_BACK, H_MINUS},
            {BTN_LEFT, H_LEFT}, {BTN_UP, H_UP}, {BTN_RIGHT, H_RIGHT}, {BTN_DOWN, H_DOWN}};
        const uint16_t lo = (uint16_t)rumble_low, hi = (uint16_t)rumble_high;
        for (int i = 0; i < MAX_PADS; i++) {
            void *gc = pads[i].gc;
            if (!gc) continue;
            count++;
            for (unsigned k = 0; k < sizeof map / sizeof *map; k++)
                if (sdl.GetButton(gc, map[k].sdl)) b |= 1ULL << map[k].hid;
            if (sdl.GetAxis(gc, AX_LT) > 8000) b |= 1ULL << H_ZL;
            if (sdl.GetAxis(gc, AX_RT) > 8000) b |= 1ULL << H_ZR;
            const int32_t v[4] = {stick_value(sdl.GetAxis(gc, AX_LX), 0), stick_value(sdl.GetAxis(gc, AX_LY), 1),
                                  stick_value(sdl.GetAxis(gc, AX_RX), 0), stick_value(sdl.GetAxis(gc, AX_RY), 1)};
            for (int k = 0; k < 4; k++)
                if ((v[k] < 0 ? -v[k] : v[k]) > (s[k] < 0 ? -s[k] : s[k])) s[k] = v[k];
            if (lo != last_lo || hi != last_hi) sdl.Rumble(gc, lo, hi, lo || hi ? 1000 : 0);
        }
        last_lo = lo;
        last_hi = hi;
        AcquireSRWLockExclusive(&state_lock);
        state_buttons = b;
        memcpy(state_stick, s, sizeof s);
        state_count = count;
        ReleaseSRWLockExclusive(&state_lock);
        Sleep(4);
    }
}

/* The controllers' merged state (nn_hid.c): ORs the buttons in, keeps the larger stick deflections; returns how many
   controllers SDL has open. Starts the SDL thread on the first call. */
int hos_sdl_pads(uint64_t *buttons, int32_t stick[4]) {
    static volatile LONG started;
    if (!started && !InterlockedExchange(&started, 1)) {
        if (getenv("HOS_NO_SDL")) return 0;
        if (!load()) {
            hos_log("hid: SDL2.dll not found next to the exe: only XInput controllers (Xbox pads) work");
            return 0;
        }
        CreateThread(NULL, 0, pad_main, NULL, 0, NULL);
    }
    AcquireSRWLockShared(&state_lock);
    const int n = state_count;
    *buttons |= state_buttons;
    for (int k = 0; k < 4; k++)
        if ((state_stick[k] < 0 ? -state_stick[k] : state_stick[k]) > (stick[k] < 0 ? -stick[k] : stick[k]))
            stick[k] = state_stick[k];
    ReleaseSRWLockShared(&state_lock);
    return n;
}

/* Vibration from the game (0..1 amplitudes of the low and high bands), for the pads that rumble. */
void hos_sdl_rumble(float low, float high) {
    if (low < 0) low = 0;
    if (low > 1) low = 1;
    if (high < 0) high = 0;
    if (high > 1) high = 1;
    InterlockedExchange(&rumble_low, (LONG)(low * 65535));
    InterlockedExchange(&rumble_high, (LONG)(high * 65535));
}

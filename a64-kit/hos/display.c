/* The game's window (Windows). With the renderer (gpu/render/, WebGPU) it is the renderer's surface: each window
   buffer the game queues is presented from the GPU's copy (nn_vi.c -> rx_present). Without it (RX_OFF=1, or no
   WebGPU device) the window shows the window buffer the game last presented, read from guest memory and
   deswizzled from the GPU's block-linear layout. The title bar carries live numbers.
   HOS_HEADLESS=1: no window (scripted runs; the renderer draws offscreen). Closing the window ends the run.
   HOS_WINDOW_NOACTIVATE=1: the window opens behind the one in front (test runs do not take the keyboard).
   Mouse: a click in the window captures the mouse (hidden, kept in the window) and its motion turns the camera
   (nn_hid.c reads it as the right stick); Esc or leaving the window releases it. */
#include "hos.h"
#include "perf_menu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../gpu/render/rx.h"
#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0
#endif

uint64_t hos_window_buffer_handle(int slot);   /* nn_vi.c */
uint64_t hos_nvmap_address(uint32_t handle);   /* nvdrv.c */
int64_t hos_gpu_stat_draws(void);              /* gpu.c */
extern volatile int hos_present_slot;
extern volatile LONG64 hos_present_count;

enum { FRAME_W = 1920, FRAME_H = 1080 };
static uint32_t *frame;               /* BGRA, FRAME_W x FRAME_H */
static int frame_w, frame_h, frame_blank = 1;
static HWND hwnd;
static int renderer;   /* the renderer presents into the window */

/* mouse capture: raw motion accumulated here, taken by nn_hid.c */
static volatile LONG mouse_dx, mouse_dy;
static volatile LONG mouse_captured;
static int cursor_hidden;

int hos_mouse_take(int32_t *dx, int32_t *dy) {
    *dx = InterlockedExchange(&mouse_dx, 0);
    *dy = InterlockedExchange(&mouse_dy, 0);
    return mouse_captured;
}

static int fullscreen;
static WINDOWPLACEMENT windowed_place;

static void set_fullscreen(HWND w, int on) {
    if (on == fullscreen) return;
    const LONG style = GetWindowLongA(w, GWL_STYLE);
    if (on) {
        MONITORINFO mi;
        mi.cbSize = sizeof mi;
        windowed_place.length = sizeof windowed_place;
        if (!GetWindowPlacement(w, &windowed_place) ||
            !GetMonitorInfoA(MonitorFromWindow(w, MONITOR_DEFAULTTONEAREST), &mi))
            return;
        fullscreen = 1;
        SetWindowLongA(w, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
        SetWindowPos(w, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    } else {
        fullscreen = 0;
        SetWindowLongA(w, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(w, &windowed_place);
        SetWindowPos(w, NULL, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    }
    hos_log("display: %s", on ? "fullscreen (F11 for a window)" : "windowed");
}

static void dpi_aware_thread(void) {
    if (getenv("HOS_DPI_UNAWARE")) return;
    typedef void *(WINAPI * SetCtx)(void *);
    SetCtx set = (SetCtx)(void *)GetProcAddress(GetModuleHandleA("user32.dll"), "SetThreadDpiAwarenessContext");
    if (set) set((void *)(intptr_t)-4);
}

static UINT window_dpi(HWND w) {
    typedef UINT(WINAPI * GetDpi)(HWND);
    GetDpi get = (GetDpi)(void *)GetProcAddress(GetModuleHandleA("user32.dll"), "GetDpiForWindow");
    const UINT dpi = get ? get(w) : 96;
    return dpi ? dpi : 96;
}

static void mouse_capture(HWND w, int on) {
    if (on == (int)mouse_captured) return;
    mouse_captured = on;
    InterlockedExchange(&mouse_dx, 0);
    InterlockedExchange(&mouse_dy, 0);
    if (on) {
        RECT rc;
        GetClientRect(w, &rc);
        MapWindowPoints(w, NULL, (POINT *)&rc, 2);
        ClipCursor(&rc);
        if (!cursor_hidden) { ShowCursor(FALSE); cursor_hidden = 1; }
    } else {
        ClipCursor(NULL);
        if (cursor_hidden) { ShowCursor(TRUE); cursor_hidden = 0; }
    }
}

/* One window buffer, described by its nvgr handle (NVIDIA gralloc, magic 0xdaffcaff at +0x18): nvmap handle at
   +0x10, width/height at +0x40/+0x44, pitch at +0x54, kind at +0x60 (0xfe block-linear), log2 block height at
   +0x64. RGBA8. */
static void grab(void) {
    int slot = hos_present_slot;
    uint64_t h = hos_window_buffer_handle(slot);
    if (!h || G32(h + 0x18) != 0xdaffcaff) return;
    uint64_t mem = hos_nvmap_address(G32(h + 0x10));
    uint32_t w = G32(h + 0x40), hgt = G32(h + 0x44), pitch = G32(h + 0x54), kind = G32(h + 0x60), bh = G32(h + 0x64);
    if (!mem || !w || !hgt || w > FRAME_W || hgt > FRAME_H || bh > 5) return;
    const uint8_t *src = (const uint8_t *)GP(mem);
    uint32_t gobs_x = (pitch + 63) / 64, block_rows = 8u << bh;
    int blank = 1;
    for (uint32_t y = 0; y < hgt; y++) {
        uint32_t *row = frame + (size_t)y * FRAME_W;
        uint64_t yblock = (uint64_t)(y / block_rows) * gobs_x * (512ull << bh) + ((uint64_t)((y >> 3) & ((1u << bh) - 1)) << 9);
        for (uint32_t x = 0; x < w; x++) {
            uint32_t xb = x * 4;
            uint64_t off;
            if (kind == 0xfe || kind == 0xdb)
                off = yblock + (uint64_t)(xb >> 6) * (512ull << bh) + (((xb & 63) >> 5) << 8 | ((y & 7) >> 1) << 6 |
                                                                        ((xb & 31) >> 4) << 5 | (y & 1) << 4 | (xb & 15));
            else
                off = (uint64_t)y * pitch + xb;
            uint32_t rgba;
            memcpy(&rgba, src + off, 4);
            if (rgba & 0xFFFFFF) blank = 0;
            row[x] = (rgba & 0xFF00FF00) | (rgba & 0xFF) << 16 | (rgba >> 16 & 0xFF);   /* RGBA -> BGRA */
        }
    }
    frame_w = (int)w;
    frame_h = (int)hgt;
    frame_blank = blank;
}

const char *rx_present_hotkey(int key);   /* gpu/render/rx_present_fx.cpp */
static int64_t present_notice_until;         /* the present settings shown in the title until then (a hotkey) */
static void update_title(void) {
    static int64_t last_t, last_presents, last_draws;
    int64_t t = hos_now_ns(), p = hos_present_count, d = hos_gpu_stat_draws();
    if (last_t && t - last_t < 1000000000LL) return;
    double dt = last_t ? (t - last_t) / 1e9 : 1.0;
    char title[384];
    if (renderer) {
        static int64_t last_drawn;
        int64_t drawn, skipped, pipelines;
        rx_stats(&drawn, &skipped, &pipelines);
        snprintf(title, sizeof title,
                 "Tears of the Kingdom - Rexcomp a64-kit  |  %.1f fps  |  %lld frames  |  %.0fk draws/s (%.0fk drawn, "
                 "%lld skipped, %lld pipelines)  |  WebGPU: %s",
                 last_t ? (p - last_presents) / dt : 0.0, (long long)p, last_t ? (d - last_draws) / dt / 1000.0 : 0.0,
                 last_t ? (drawn - last_drawn) / dt / 1000.0 : 0.0, (long long)skipped, (long long)pipelines,
                 rx_describe());
        last_drawn = drawn;
    } else {
        snprintf(title, sizeof title,
                 "Tears of the Kingdom - Rexcomp a64-kit  |  %.1f fps  |  %lld frames  |  %.0fk draws/s  |  renderer: off",
                 last_t ? (p - last_presents) / dt : 0.0, (long long)p, last_t ? (d - last_draws) / dt / 1000.0 : 0.0);
    }
    extern double hos_mouse_sens;   /* nn_hid.c */
    if (mouse_captured) {
        char m[96];
        snprintf(m, sizeof m, "  |  mouse camera, sensitivity %.2f (- / = change, Esc releases)", hos_mouse_sens);
        strncat(title, m, sizeof title - strlen(title) - 1);
    }
    else if (renderer) strncat(title, "  |  click for mouse camera", sizeof title - strlen(title) - 1);
    if (renderer && t < present_notice_until) {
        strncat(title, "  |  ", sizeof title - strlen(title) - 1);
        strncat(title, rx_present_hotkey(0), sizeof title - strlen(title) - 1);
    }
    SetWindowTextA(hwnd, title);
    last_t = t;
    last_presents = p;
    last_draws = d;
}

static LRESULT CALLBACK wndproc(HWND w, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_TIMER:
        if (!renderer) {
            grab();
            InvalidateRect(w, NULL, FALSE);
        }
        update_title();
        return 0;
    case WM_INPUT: {
        RAWINPUT ri;
        UINT size = sizeof ri;
        if (mouse_captured && GetRawInputData((HRAWINPUT)lp, RID_INPUT, &ri, &size, sizeof(RAWINPUTHEADER)) != (UINT)-1 &&
            ri.header.dwType == RIM_TYPEMOUSE && !(ri.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)) {
            InterlockedExchangeAdd(&mouse_dx, ri.data.mouse.lLastX);
            InterlockedExchangeAdd(&mouse_dy, ri.data.mouse.lLastY);
        }
        return DefWindowProcA(w, msg, wp, lp);
    }
    case WM_LBUTTONDOWN:
        if (renderer && !mouse_captured) mouse_capture(w, 1);
        return 0;
    case WM_KEYDOWN:
        if (wp == VK_F1) mouse_capture(w, 0);
        if (perf_menu_key((int)wp, w)) return 0;
        if (wp == VK_ESCAPE) mouse_capture(w, 0);
        if (wp == VK_F11 && !(lp & (1 << 30))) set_fullscreen(w, !fullscreen);
        if (renderer && wp >= VK_F6 && wp <= VK_F10 && !(lp & (1 << 30))) {   /* the present settings (F6-F10) */
            hos_log("present: %s", rx_present_hotkey((int)(wp - VK_F6) + 6));
            if (wp == VK_F6) {   /* (ultrawide: the aspect from the window's size again) */
                RECT rc;
                GetClientRect(w, &rc);
                rx_resize(rc.right, rc.bottom);
            }
            present_notice_until = hos_now_ns() + 4000000000LL;
            update_title();
            perf_menu_sync();
        }
        return 0;
    case WM_DPICHANGED:
        if (!fullscreen) {
            const RECT *s = (const RECT *)lp;
            SetWindowPos(w, NULL, s->left, s->top, s->right - s->left, s->bottom - s->top,
                         SWP_NOZORDER | SWP_NOACTIVATE);
        }
        return 0;
    case WM_KILLFOCUS:
    case WM_CANCELMODE:
        mouse_capture(w, 0);
        break;
    case WM_SETCURSOR:
        if (mouse_captured && LOWORD(lp) == HTCLIENT) {
            SetCursor(NULL);
            return TRUE;
        }
        break;
    case WM_MOVE:
        if (mouse_captured) { mouse_capture(w, 0); mouse_capture(w, 1); }   /* (the clip follows the window) */
        break;
    case WM_SIZE:
        if (mouse_captured) { mouse_capture(w, 0); mouse_capture(w, 1); }
        if (renderer) rx_resize(LOWORD(lp), HIWORD(lp));
        return 0;
    case WM_PAINT: {
        if (renderer) {   /* the renderer's surface: nothing to paint */
            ValidateRect(w, NULL);
            return 0;
        }
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(w, &ps);
        RECT rc;
        GetClientRect(w, &rc);
        if (frame_w) {
            BITMAPINFO bi;
            memset(&bi, 0, sizeof bi);
            bi.bmiHeader.biSize = sizeof bi.bmiHeader;
            bi.bmiHeader.biWidth = FRAME_W;
            bi.bmiHeader.biHeight = -FRAME_H;   /* top-down */
            bi.bmiHeader.biPlanes = 1;
            bi.bmiHeader.biBitCount = 32;
            bi.bmiHeader.biCompression = BI_RGB;
            SetStretchBltMode(dc, HALFTONE);
            StretchDIBits(dc, 0, 0, rc.right, rc.bottom, 0, FRAME_H - frame_h, frame_w, frame_h, frame, &bi,
                          DIB_RGB_COLORS, SRCCOPY);
        } else {
            FillRect(dc, &rc, (HBRUSH)GetStockObject(BLACK_BRUSH));
        }
        if (frame_blank) {
            char msg2[256];
            snprintf(msg2, sizeof msg2,
                     "The game is running: %lld frames presented. Its GPU commands execute, but the renderer that "
                     "turns them into pixels is not written yet, so the frames are black.",
                     (long long)hos_present_count);
            SetBkMode(dc, TRANSPARENT);
            SetTextColor(dc, RGB(200, 200, 200));
            RECT tr = {24, rc.bottom - 64, rc.right - 24, rc.bottom - 16};
            DrawTextA(dc, msg2, -1, &tr, DT_LEFT | DT_WORDBREAK);
        }
        EndPaint(w, &ps);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_CLOSE:
        hos_log("display: window closed, exiting");
        hos_log_flush();
        ExitProcess(0);
        return 0;
    }
    return DefWindowProcA(w, msg, wp, lp);
}

static DWORD WINAPI display_main(void *p) {
    (void)p;
    WNDCLASSA wc;
    memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleA(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "RexcompTotk";
    RegisterClassA(&wc);
    dpi_aware_thread();
    /* HOS_WINDOW_SIZE=<w>x<h>: the client area (default 1280x720, at 96 dpi; e.g. 2560x1080 for an ultrawide) */
    int win_w = 1280, win_h = 720;
    if (getenv("HOS_WINDOW_SIZE")) {
        int w = 0, h = 0;
        if (sscanf(getenv("HOS_WINDOW_SIZE"), "%dx%d", &w, &h) == 2 && w >= 64 && h >= 64) win_w = w, win_h = h;
    }
    RECT r = {0, 0, win_w, win_h};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    const int noactivate = getenv("HOS_WINDOW_NOACTIVATE") != NULL;
    hwnd = CreateWindowA("RexcompTotk", "Tears of the Kingdom - Rexcomp a64-kit", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                         CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, NULL, NULL, wc.hInstance, NULL);
    if (window_dpi(hwnd) != 96) {
        const UINT dpi = window_dpi(hwnd);
        RECT s = {0, 0, (LONG)(win_w * dpi / 96), (LONG)(win_h * dpi / 96)};
        AdjustWindowRect(&s, WS_OVERLAPPEDWINDOW, FALSE);
        SetWindowPos(hwnd, NULL, 0, 0, s.right - s.left, s.bottom - s.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    }
    /* (no activate: shown at the bottom of the z-order. SW_SHOWNOACTIVATE still put it on top, over a full-screen
       game, which then lost focus; and a process's first ShowWindow may take the launcher's STARTUPINFO instead) */
    if (noactivate)
        SetWindowPos(hwnd, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
    else
        ShowWindow(hwnd, SW_SHOW);
    if (getenv("HOS_FULLSCREEN") && atoi(getenv("HOS_FULLSCREEN"))) set_fullscreen(hwnd, 1);
    RAWINPUTDEVICE rid = {1, 2, 0, hwnd};   /* generic desktop mouse: WM_INPUT while the window is in front */
    RegisterRawInputDevices(&rid, 1, sizeof rid);
    renderer = rx_start(hwnd);
    perf_menu_init(hwnd);   /* F1 overlay menu */
    SetTimer(hwnd, 1, renderer ? 250 : 33, NULL);
    MSG m;
    while (GetMessageA(&m, NULL, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageA(&m);
    }
    return 0;
}

void hos_display_start(void) {
    if (getenv("HOS_HEADLESS")) {   /* no window: the renderer still draws, offscreen (screenshots, frame dumps) */
        rx_start(NULL);
        return;
    }
    frame = (uint32_t *)calloc((size_t)FRAME_W * FRAME_H, 4);
    CreateThread(NULL, 0, display_main, NULL, 0, NULL);
}

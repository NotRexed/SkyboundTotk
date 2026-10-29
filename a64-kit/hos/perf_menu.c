/* perf_menu.c: F1 performance overlay menu for SkyboundTotk.
   Press F1 while the game window is in focus: a small semi-transparent overlay appears with
   checkboxes for all rendering options (FSR, sharpness, frame gen, ultrawide, aspect, 30-fps
   cap).  Use the mouse, arrow keys, numbers 1-6, or shortcut hotkeys (F6-F10) + Space/Enter to
   toggle them.  Press F1 or Escape to close.

   The menu is a borderless, always-on-top, click-through-except-for-its-client child window
   so the game window keeps focus and its input handling is unaffected.  All settings are
   forwarded to rx_present_hotkey() and perf_menu_fps_cap. */

#include "perf_menu.h"
#include "hos.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* ---- extern helpers from rx_present_fx.cpp -------------------------------------------- */
extern const char *rx_present_hotkey(int key);   /* key 6..10, 0 for query */
extern void rx_present_get_settings(int *fsr, int *sharpness_idx, int *framegen, int *ultrawide, int *stretch);
extern void rx_resize(int width, int height);

/* ---- FPS cap: read by nn_vi.c's frame_limit() ----------------------------------------- */
volatile int perf_menu_fps_cap = 0;   /* 0 = no override; 30 = hard 30, etc. */

/* ---- menu geometry & colours ---------------------------------------------------------- */
enum {
    MENU_W   = 360,
    ITEM_H   = 26,
    PAD_X    = 14,
    PAD_Y    = 10,
    CB_SIZE  = 14,   /* checkbox square */
};
#define COL_BG        RGB(18,  18,  24)
#define COL_BG_HOVER  RGB(38,  38,  52)
#define COL_BORDER    RGB(80,  80, 120)
#define COL_TEXT      RGB(220, 220, 230)
#define COL_CHECK_BG  RGB(50,  50,  80)
#define COL_CHECK_FG  RGB(100, 180, 255)
#define COL_TITLE     RGB(140, 200, 255)
#define COL_DIM       RGB(120, 120, 140)
#define MENU_ALPHA    230   /* 0-255 */

/* ---- menu items ----------------------------------------------------------------------- */
typedef enum { IK_TOGGLE, IK_CYCLE } ItemKind;

typedef struct {
    const char *label;
    const char *hint;    /* shown right-aligned in dim colour */
    ItemKind    kind;
    int        *value;   /* pointer to a local int that mirrors the setting */
    int         hotkey;  /* rx_present_hotkey key index (0 = not routed there) */
    /* IK_CYCLE only: */
    const char **cycle_labels;
    int          cycle_n;
} MenuItem;

/* local mirrors of rx settings (updated when the menu is built / when a hotkey fires) */
static int s_fsr      = 1;
static int s_framegen = 0;
static int s_ultra    = 0;
static int s_stretch  = 0;
static int s_sharp    = 1;   /* index 0-4 into kSharpness[] in rx_present_fx.cpp */
static int s_fps30    = 0;   /* our own cap */

static const char *sharp_labels[] = {"0.00 (sharpest)", "0.25", "0.50", "1.00", "2.00 (softest)"};

static MenuItem items[] = {
    /* label                hint          kind        *value     hotkey */
    {"FSR 1 upscaling",   "F8 / 1",     IK_TOGGLE, &s_fsr,      8, NULL, 0},
    {"Sharpness",         "F9 / 2",     IK_CYCLE,  &s_sharp,    9, sharp_labels, 5},
    {"Frame interpolation","F7 / 3",    IK_TOGGLE, &s_framegen,  7, NULL, 0},
    {"Ultrawide",         "F6 / 4",     IK_TOGGLE, &s_ultra,     6, NULL, 0},
    {"Stretch aspect",    "F10 / 5",    IK_TOGGLE, &s_stretch,  10, NULL, 0},
    {"Cap to 30 FPS",     "(arc) / 6",  IK_TOGGLE, &s_fps30,     0, NULL, 0},
};
#define N_ITEMS ((int)(sizeof items / sizeof *items))

extern const char *hos_user_dir;

static void load_fps_cap(void) {
    char path[512];
    snprintf(path, sizeof path, "%s/fps_cap.txt", hos_user_dir ? hos_user_dir : "user");
    FILE *f = fopen(path, "r");
    if (f) {
        int val = 0;
        if (fscanf(f, "%d", &val) == 1) {
            perf_menu_fps_cap = val;
            s_fps30 = (val == 30);
        }
        fclose(f);
    } else if (getenv("HOS_FPS_CAP")) {
        int val = atoi(getenv("HOS_FPS_CAP"));
        if (val == 30) {
            perf_menu_fps_cap = 30;
            s_fps30 = 1;
        }
    }
}

static void save_fps_cap(void) {
    char path[512];
    snprintf(path, sizeof path, "%s/fps_cap.txt", hos_user_dir ? hos_user_dir : "user");
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "%d\n", perf_menu_fps_cap);
        fclose(f);
    }
}

/* ---- sync: pull the current settings directly from rx_present_fx ----------------------- */
static void sync_from_rx(void) {
    rx_present_get_settings(&s_fsr, &s_sharp, &s_framegen, &s_ultra, &s_stretch);
}

/* ---- overlay window ------------------------------------------------------------------- */
static HWND  g_menu_hwnd   = NULL;
static HWND  g_game_hwnd   = NULL;
static int   g_menu_open   = 0;
static int   g_hot         = 0;   /* hovered / keyboard-focused item index (defaults to 0: FSR) */
static HFONT g_font        = NULL;
static HFONT g_font_title  = NULL;

static int menu_height(void) { return PAD_Y * 2 + 24 + N_ITEMS * ITEM_H + PAD_Y; }

/* Draw the menu into a DC that covers the whole overlay window */
static void menu_paint(HWND w, HDC dc) {
    RECT rc;
    GetClientRect(w, &rc);
    const int W = rc.right, H = rc.bottom;

    /* --- background --- */
    HBRUSH bg_br = CreateSolidBrush(COL_BG);
    FillRect(dc, &rc, bg_br);
    DeleteObject(bg_br);

    /* --- border --- */
    HPEN border_pen = CreatePen(PS_SOLID, 1, COL_BORDER);
    HPEN old_pen    = (HPEN)SelectObject(dc, border_pen);
    HBRUSH null_br  = (HBRUSH)GetStockObject(NULL_BRUSH);
    HBRUSH old_br   = (HBRUSH)SelectObject(dc, null_br);
    Rectangle(dc, 0, 0, W, H);
    SelectObject(dc, old_pen);
    SelectObject(dc, old_br);
    DeleteObject(border_pen);

    SetBkMode(dc, TRANSPARENT);

    /* --- title --- */
    SelectObject(dc, g_font_title);
    SetTextColor(dc, COL_TITLE);
    RECT tr = {PAD_X, PAD_Y, W - PAD_X, PAD_Y + 22};
    DrawTextA(dc, "Performance Options  (F1 to close)", -1, &tr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    /* separator */
    HPEN sep = CreatePen(PS_SOLID, 1, COL_BORDER);
    SelectObject(dc, sep);
    MoveToEx(dc, PAD_X, PAD_Y + 24, NULL);
    LineTo(dc, W - PAD_X, PAD_Y + 24);
    SelectObject(dc, (HPEN)GetStockObject(NULL_PEN));
    DeleteObject(sep);

    SelectObject(dc, g_font);

    /* --- items --- */
    for (int i = 0; i < N_ITEMS; i++) {
        const MenuItem *it = &items[i];
        const int y = PAD_Y + 28 + i * ITEM_H;
        RECT row = {0, y, W, y + ITEM_H};

        /* hover / keyboard focus highlight */
        if (i == g_hot) {
            HBRUSH hov = CreateSolidBrush(COL_BG_HOVER);
            FillRect(dc, &row, hov);
            DeleteObject(hov);

            HPEN focus_pen = CreatePen(PS_SOLID, 1, COL_TITLE);
            HPEN old_p = (HPEN)SelectObject(dc, focus_pen);
            HBRUSH old_b = (HBRUSH)SelectObject(dc, GetStockObject(NULL_BRUSH));
            Rectangle(dc, row.left + 2, row.top + 1, row.right - 2, row.bottom);
            SelectObject(dc, old_p);
            SelectObject(dc, old_b);
            DeleteObject(focus_pen);
        }

        /* checkbox */
        RECT cb = {PAD_X, y + (ITEM_H - CB_SIZE) / 2, PAD_X + CB_SIZE, y + (ITEM_H + CB_SIZE) / 2};
        HBRUSH cb_bg = CreateSolidBrush(COL_CHECK_BG);
        FillRect(dc, &cb, cb_bg);
        DeleteObject(cb_bg);
        HPEN cb_pen = CreatePen(PS_SOLID, 1, COL_BORDER);
        SelectObject(dc, cb_pen);
        SelectObject(dc, (HBRUSH)GetStockObject(NULL_BRUSH));
        Rectangle(dc, cb.left, cb.top, cb.right, cb.bottom);
        SelectObject(dc, (HPEN)GetStockObject(NULL_PEN));
        DeleteObject(cb_pen);

        /* tick / value indicator */
        if (it->kind == IK_TOGGLE) {
            int checked = *it->value;
            if (checked) {
                HPEN tick = CreatePen(PS_SOLID, 2, COL_CHECK_FG);
                SelectObject(dc, tick);
                MoveToEx(dc, cb.left + 2, cb.top + CB_SIZE/2, NULL);
                LineTo(dc,   cb.left + CB_SIZE/2 - 1, cb.bottom - 3);
                LineTo(dc,   cb.right - 2,             cb.top + 3);
                SelectObject(dc, (HPEN)GetStockObject(NULL_PEN));
                DeleteObject(tick);
            }
        } else {
            /* cycle: draw a right-arrow to show it's cycling */
            HPEN arr = CreatePen(PS_SOLID, 1, COL_CHECK_FG);
            SelectObject(dc, arr);
            const int mx = cb.left + CB_SIZE/2, my = cb.top + CB_SIZE/2;
            MoveToEx(dc, mx-3, my-3, NULL); LineTo(dc, mx+3, my);
            MoveToEx(dc, mx-3, my+3, NULL); LineTo(dc, mx+3, my);
            SelectObject(dc, (HPEN)GetStockObject(NULL_PEN));
            DeleteObject(arr);
        }

        /* label */
        SetTextColor(dc, COL_TEXT);
        RECT lr = {PAD_X + CB_SIZE + 8, y, W - 60, y + ITEM_H};
        DrawTextA(dc, it->label, -1, &lr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

        /* cycle value or hint */
        const char *right_txt = it->hint;
        char cycle_buf[64] = "";
        if (it->kind == IK_CYCLE && it->cycle_labels) {
            int idx = *it->value;
            if (idx < 0) idx = 0;
            if (idx >= it->cycle_n) idx = it->cycle_n - 1;
            snprintf(cycle_buf, sizeof cycle_buf, "%s", it->cycle_labels[idx]);
            right_txt = cycle_buf;
        }
        if (right_txt) {
            SetTextColor(dc, COL_DIM);
            RECT hr = {W/2, y, W - PAD_X, y + ITEM_H};
            DrawTextA(dc, right_txt, -1, &hr, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
        }
    }

    /* bottom tip */
    SetTextColor(dc, COL_DIM);
    RECT tip = {PAD_X, H - PAD_Y - 16, W - PAD_X, H - PAD_Y};
    DrawTextA(dc, "Arrows/Space/1-6/F6-F10 to toggle  \x95  Esc/F1 to close", -1, &tip,
              DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

/* ---- apply item action ---------------------------------------------------------------- */
static void item_activate(int idx) {
    if (idx < 0 || idx >= N_ITEMS) return;
    MenuItem *it = &items[idx];

    if (it->hotkey) {
        hos_log("present: %s", rx_present_hotkey(it->hotkey));
        if (it->hotkey == 6 && g_game_hwnd) {
            RECT rc;
            GetClientRect(g_game_hwnd, &rc);
            rx_resize(rc.right, rc.bottom);
        }
        sync_from_rx();
    } else if (it->value == &s_fps30) {
        /* 30 fps cap toggle */
        s_fps30 = !s_fps30;
        perf_menu_fps_cap = s_fps30 ? 30 : 0;
        save_fps_cap();
        hos_log("perf_menu: 30 fps cap %s", s_fps30 ? "on" : "off");
    }
    if (g_menu_hwnd) {
        InvalidateRect(g_menu_hwnd, NULL, FALSE);
        UpdateWindow(g_menu_hwnd);
    }
    if (g_game_hwnd) {
        SetFocus(g_game_hwnd);
    }
}

/* ---- hit-test: which item is under a client-area point? ------------------------------- */
static int item_at(int cx, int cy) {
    const int top = PAD_Y + 28;
    if (cy < top) return -1;
    int i = (cy - top) / ITEM_H;
    if (i < 0 || i >= N_ITEMS) return -1;
    return i;
}

/* ---- position the overlay next to the game window ------------------------------------ */
static void reposition(void) {
    if (!g_game_hwnd || !g_menu_hwnd) return;
    RECT gr;
    GetWindowRect(g_game_hwnd, &gr);
    int x = gr.left + 20;
    int y = gr.top  + 60;
    SetWindowPos(g_menu_hwnd, HWND_TOPMOST, x, y, MENU_W, menu_height(),
                 SWP_NOACTIVATE);
}

/* ---- window proc ---------------------------------------------------------------------- */
static LRESULT CALLBACK menu_proc(HWND w, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(w, &ps);
        HDC mdc = CreateCompatibleDC(dc);
        RECT rc; GetClientRect(w, &rc);
        HBITMAP bmp = CreateCompatibleBitmap(dc, rc.right, rc.bottom);
        HBITMAP old = (HBITMAP)SelectObject(mdc, bmp);
        menu_paint(w, mdc);
        BitBlt(dc, 0, 0, rc.right, rc.bottom, mdc, 0, 0, SRCCOPY);
        SelectObject(mdc, old);
        DeleteObject(bmp);
        DeleteDC(mdc);
        EndPaint(w, &ps);
        return 0;
    }
    case WM_MOUSEMOVE: {
        const int hot = item_at(LOWORD(lp), HIWORD(lp));
        if (hot >= 0 && hot != g_hot) {
            g_hot = hot;
            InvalidateRect(w, NULL, FALSE);
            UpdateWindow(w);
            TRACKMOUSEEVENT tme = { sizeof tme, TME_LEAVE, w, 0 };
            TrackMouseEvent(&tme);
        }
        return 0;
    }
    case WM_MOUSELEAVE:
        /* Retain valid keyboard selection when mouse cursor leaves */
        if (g_hot < 0 || g_hot >= N_ITEMS) g_hot = 0;
        InvalidateRect(w, NULL, FALSE);
        UpdateWindow(w);
        return 0;
    case WM_MOUSEACTIVATE:
        return MA_NOACTIVATE;
    case WM_LBUTTONDOWN: {
        int clicked = item_at(LOWORD(lp), HIWORD(lp));
        if (clicked >= 0) {
            g_hot = clicked;
            item_activate(clicked);
        }
        if (g_game_hwnd) SetFocus(g_game_hwnd);
        return 0;
    }
    case WM_KEYDOWN:
        if (perf_menu_key((int)wp, g_game_hwnd)) return 0;
        if (g_game_hwnd) {
            SendMessageA(g_game_hwnd, WM_KEYDOWN, wp, lp);
        }
        return 0;
    case WM_KEYUP:
        if (g_game_hwnd) {
            SendMessageA(g_game_hwnd, WM_KEYUP, wp, lp);
        }
        return 0;
    case WM_ACTIVATE:
        if (LOWORD(wp) != WA_INACTIVE && g_game_hwnd)
            SetFocus(g_game_hwnd);
        return 0;
    case WM_ERASEBKGND:
        return 1;
    }
    return DefWindowProcA(w, msg, wp, lp);
}

/* ---- public API ----------------------------------------------------------------------- */

void perf_menu_sync(void) {
    sync_from_rx();
    if (g_menu_hwnd && g_menu_open) {
        InvalidateRect(g_menu_hwnd, NULL, FALSE);
        UpdateWindow(g_menu_hwnd);
    }
}

void perf_menu_init(void *hwnd_game) {
    g_game_hwnd = (HWND)hwnd_game;
    load_fps_cap();
    sync_from_rx();

    WNDCLASSA wc;
    memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc   = menu_proc;
    wc.hInstance     = GetModuleHandleA(NULL);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "TotKPerfMenu";
    RegisterClassA(&wc);

    g_font = CreateFontA(14, 0, 0, 0, FW_NORMAL, 0, 0, 0,
                         DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                         CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    g_font_title = CreateFontA(15, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0,
                               DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                               CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");

    g_menu_hwnd = CreateWindowExA(
        WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED | WS_EX_TOPMOST,
        "TotKPerfMenu", "",
        WS_POPUP,
        0, 0, MENU_W, menu_height(),
        NULL, NULL, GetModuleHandleA(NULL), NULL);

    /* semi-transparent background via layered window alpha */
    SetLayeredWindowAttributes(g_menu_hwnd, 0, MENU_ALPHA, LWA_ALPHA);
}

int perf_menu_key(int vk, void *hwnd_game) {
    g_game_hwnd = (HWND)hwnd_game;

    if (vk == VK_F1) {
        if (!g_menu_open) {
            sync_from_rx();
            if (g_hot < 0 || g_hot >= N_ITEMS) g_hot = 0;
            reposition();
            ShowWindow(g_menu_hwnd, SW_SHOWNOACTIVATE);
            g_menu_open = 1;
            InvalidateRect(g_menu_hwnd, NULL, FALSE);
            UpdateWindow(g_menu_hwnd);
        } else {
            ShowWindow(g_menu_hwnd, SW_HIDE);
            g_menu_open = 0;
        }
        return 1;   /* consumed */
    }

    if (g_menu_open) {
        if (vk == VK_ESCAPE) {
            ShowWindow(g_menu_hwnd, SW_HIDE);
            g_menu_open = 0;
            return 1;
        }
        if (vk == VK_UP) {
            g_hot = (g_hot <= 0) ? (N_ITEMS - 1) : (g_hot - 1);
            InvalidateRect(g_menu_hwnd, NULL, FALSE);
            UpdateWindow(g_menu_hwnd);
            return 1;
        }
        if (vk == VK_DOWN) {
            g_hot = (g_hot + 1) % N_ITEMS;
            InvalidateRect(g_menu_hwnd, NULL, FALSE);
            UpdateWindow(g_menu_hwnd);
            return 1;
        }
        if (vk == VK_SPACE || vk == VK_RETURN || vk == VK_LEFT || vk == VK_RIGHT) {
            if (g_hot < 0 || g_hot >= N_ITEMS) g_hot = 0;
            item_activate(g_hot);
            return 1;
        }
        /* Direct number shortcuts: 1..6 */
        if (vk >= '1' && vk <= '6') {
            int idx = vk - '1';
            g_hot = idx;
            item_activate(idx);
            return 1;
        }
        /* Numpad 1..6 */
        if (vk >= VK_NUMPAD1 && vk <= VK_NUMPAD6) {
            int idx = vk - VK_NUMPAD1;
            g_hot = idx;
            item_activate(idx);
            return 1;
        }
        /* Function keys F6..F10 while menu is open */
        if (vk >= VK_F6 && vk <= VK_F10) {
            int hotkey = (int)(vk - VK_F6) + 6;
            for (int i = 0; i < N_ITEMS; i++) {
                if (items[i].hotkey == hotkey) {
                    g_hot = i;
                    item_activate(i);
                    return 1;
                }
            }
        }
    }
    return 0;
}

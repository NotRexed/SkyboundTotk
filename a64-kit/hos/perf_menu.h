/* perf_menu.h: F1 performance overlay menu (display.c calls perf_menu_key on WM_KEYDOWN,
   perf_menu_draw_overlay for WM_PAINT / update_title; nn_vi.c reads perf_menu_fps_cap). */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/* Called from display.c's WM_KEYDOWN: opens/closes the menu on F1, navigates with arrow keys
   and Space/Enter. Returns 1 when the key was consumed by the menu (caller should skip other
   hotkey handling for that key). */
int perf_menu_key(int vk, void *hwnd_game);

/* Initialise the menu (called once after the game window is created). */
void perf_menu_init(void *hwnd_game);

/* Synchronise menu checkboxes with current rx settings and repaint if visible. */
void perf_menu_sync(void);

/* Optional FPS cap chosen in the menu (0 = no override, positive = cap in Hz). Read by nn_vi.c */
extern volatile int perf_menu_fps_cap;   /* 0, 30, 60, or other user-set value */

#ifdef __cplusplus
}
#endif

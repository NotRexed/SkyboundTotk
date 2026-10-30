# SkyboundTOTK

SkyboundTOTK is an experimental Windows x64 static recompilation project for *The Legend of Zelda: Tears of the Kingdom*, **base game version 1.0.0**. It is not affiliated with Nintendo. Expect graphical problems and incomplete compatibility; a successful build does not guarantee a complete playthrough.

The project's AArch64 game code is translated ahead of time into C and compiled locally into a native executable. A runtime implements the services the game expects, and the renderer translates its graphics work and shaders for WebGPU. This is a game-specific port, not a general-purpose Switch emulator. Some graphics translation still happens at runtime.

## Game files and distribution

You must supply your own legally obtained, already-extracted, **unmodified 1.0.0 base-game ExeFS and RomFS**. No game files, generated game code, keys, firmware or saves are included in this source project. The build does not extract NSP or XCI files, and other versions or updates are not supported by the game-specific hooks.

Keep your game files, generated code, executable package and saves private. The `game/`, `build/` and `dist/` directories are Git-ignored, but check what you are publishing rather than relying on ignore rules alone. The generated standalone package contains copyrighted game-derived data and must not be shared or committed.

## Build on Windows

### Requirements

- Windows 10/11 x64 and a GPU and driver supported by wgpu-native. Allow **at least 45 GB free in addition to the original game dump** for intermediate files and the separate packaged RomFS; the build is CPU- and memory-intensive.
- Python 3.10 or newer on PATH, with `lz4` installed: `python -m pip install lz4`.
- Visual Studio 2022 Build Tools with the MSVC C++ tools and Windows SDK. Use an **x64 Native Tools Command Prompt** so the compiler can find the Windows headers and libraries.
- Clang, clang++ and lld. The build previously used emsdk 6.0.9: install it separately and set `EMSDK` to its directory (the lookup defaults to `C:\emsdk`), or put these LLVM tools on PATH. The build script does not download compilers or Python packages.

### Prepare and build

1. Put the extracted files in the project's `game` folder. The paths must look like this, not `game/exefs/exefs/main` or `game/romfs/romfs/Event`:

   ```text
   SkyboundTOTK/
     build.bat
     game/
       exefs/
         main
         main.npdm
         rtld
         sdk
         ...
       romfs/
         ActorSystem/
         Event/
         ...
   ```

2. Open an **x64 Native Tools Command Prompt**, change to the SkyboundTOTK directory and run `build.bat`. The launcher pauses at the end so you can read any errors.
3. The script checks the `main` and `sdk` module build IDs and a few RomFS directories, lifts the game code, compiles the native executable and copies the runtime and RomFS into `dist/SkyboundTOTK/`. The checks do **not** verify the entire RomFS or guarantee that a dump is complete.
4. Open `dist/SkyboundTOTK/SkyboundTOTK.exe` to run it. No command-line arguments are needed. Keep `image.bin`, `image.json`, `romfs/`, `wgpu_native.dll` and `SDL2.dll` alongside the executable. The standalone folder can be moved as a unit and does not need the original `game/` folder to launch.

The first lift and compile can take a long time. The builder reuses a completed lift only when its recorded game and lifter inputs match. If the lift was interrupted or those inputs change, move the existing `build/` directory aside before retrying. The script refuses to overwrite a nonempty `dist/SkyboundTOTK/`: move the existing package aside before a new build. Copy its `user/` folder into the new package if you want to retain its saves and settings. Do not delete your only save backup.

The package normally creates its own `user/` directory beside the executable; it does not automatically use the saves from another installation. Native renderer pipeline cache data is kept in `rxcache/` beside the executable. The executable also accepts an image path and `--romfs` / `--user` overrides for advanced use, but they are not needed for normal launching.

### Build and launch troubleshooting

- **Python or `lz4` missing:** install them for the Python on PATH, then reopen the tools prompt. Run `python -m pip install lz4` with that same Python.
- **Compiler, headers or libraries missing:** open the x64 Native Tools prompt and check that the C++ tools, Windows SDK, Clang, clang++ and lld are installed and discoverable. Set `EMSDK` if the LLVM tools are in a non-default emsdk directory.
- **Wrong module or missing RomFS:** check the extracted folder layout and version. A base-game ID match is not a complete integrity check.
- **Existing output or stale/incomplete lift:** move the indicated package or build directory aside, preserving `user/` first, then retry. An interrupted copy can also leave a nonempty package that needs to be moved aside.
- **First launch or new scene stutters:** shader translation and pipeline compilation may happen as scenes are encountered. The native pipeline cache helps prepare previously encountered pipelines on later launches; it cannot guarantee stutter-free play.
- **GPU device lost or wrong GPU:** AMD cards lost the WebGPU device at startup while the GPU ASTC decoder was created (issue #1; fixed). The log now includes wgpu's own warnings and errors (`RX_WGPU_LOG=info|debug|trace` for more, `off` for none). `RX_ADAPTER=<part of the GPU's name>` (for example `RX_ADAPTER=Radeon`) picks a GPU on a PC with two, and `RX_ASTC_CPU=1` decodes ASTC textures on the CPU instead of the GPU.
- **Crash or graphics issue:** note the game version, hardware, driver, location and settings. The standalone startup test initialized the renderer and loaded the packaged RomFS, but did not verify full gameplay or graphics correctness.

## Controls and display features

Focus the game window before using the keys. The renderer briefly reports display settings in the window title; changes are saved in `user/present.txt` for the next launch.

| Key | Action |
| --- | --- |
| F6 | Toggle experimental ultrawide. A window wider than 16:9 is required; see the known issue below. |
| F7 | Toggle experimental frame interpolation. |
| F8 | Toggle FSR 1 upscaling. |
| F9 | Cycle FSR sharpening. |
| F10 | Toggle aspect-preserving presentation versus stretching the image to fill the window. |
| F11 | Toggle fullscreen/windowed mode. |

**FSR:** The current implementation is FidelityFX Super Resolution **1**, with EASU upscaling and RCAS sharpening in the presentation pass. It is enabled by default and is used when the presented frame is enlarged; otherwise the renderer uses the normal sampling path. FSR 1 is spatial upscaling, not FSR 2/3 temporal upscaling or AMD frame generation, and does not increase the game's internal render resolution. F9 cycles sharpening from sharper to softer values.

**Frame interpolation:** This is the project's own experimental image-based optical-flow interpolation, not AMD FSR frame generation. F7 is off by default and inserts estimated frames between real frames, normally targeting twice the displayed frame rate when the hardware can keep up. It does not make the game simulate more frames or reduce input latency; the display runs about one game frame behind. UI and fast motion can smear or distort, and long frame gaps are not interpolated.

**Ultrawide:** F6 attempts to adjust the camera and HUD and fill a wider window. It is off by default. A wider-than-16:9 window is needed, but a user reported that F6 stretches the image instead of providing a correctly proportioned wider view. Camera hooks are present, but their behavior in that report has not been resolved or independently verified. F10 stretching alone is not true ultrawide.

Keyboard and mouse controls are experimental. WASD moves, IJKL controls the camera, Space jumps, E is action, Shift or Q is run/cancel, F attacks, Z/R are L/R, C/V are ZL/ZR, Ctrl crouches, Enter is Plus, Tab is Minus and the arrow keys are the D-pad. Click the window to capture the mouse, Esc to release it, and use `-` / `=` to adjust sensitivity. XInput and SDL-supported controllers are also handled; buttons map by physical position, Nintendo controllers included (the right face button is A).

The game speaks English (US) by default. To play in another of its languages, put a `language.txt` next to `SkyboundTOTK.exe` whose first line is one of `ja`, `en-US`, `en-GB`, `fr`, `fr-CA`, `de`, `it`, `es`, `es-419`, `nl`, `ru`, `ko`, `zh-CN` (or `zh-Hans`), `zh-TW` (or `zh-Hant`); the environment variable `HOS_LANGUAGE` does the same. Anything else falls back to English (US).

### Why WebGPU?

The renderer targets the WebGPU API and WGSL so the same general renderer and shader backend can be used by native wgpu and by browser WebGPU experiments rather than maintaining wholly separate graphics implementations. On Windows, wgpu-native translates this API to an available native backend, such as Vulkan or Direct3D 12; choosing WebGPU does not mean the desktop game runs inside a browser. It also has portability and validation tradeoffs, and translation, pipeline compilation and API overhead still require optimization. The root `build.bat` described here builds the **Windows native** package; browser functionality and packaging are not promised or validated by that launcher.

### Intel Arc Graphics (Windows 11)

SkyboundTOTK includes dedicated support and mitigations for Intel Arc GPUs (Alchemist A-Series like A770/A750/A380 and Battlemage B-Series):

- **Automatic hardware detection:** The renderer identifies Intel Arc adapters (`vendorID 0x8086`) and applies architecture-specific shader compilation profiles and driver mitigations.
- **Driver and swapchain stabilization:** Redundant swapchain reconfigurations are prevented to avoid Vulkan/D3D12 device lost errors (`0xc000041d`) during window resize, minimize, or focus changes on Windows 11.
- **Shader precision & math mitigations:** Fast-math optimizations that cause half-float NaN/overflow issues on Intel Execution Units are automatically disabled, and floating-point clamping is sanitized (`OpFMax`+`OpFMin`).
- **Dual-source blending bypass:** Hardware dual-source blending requests are bypassed by default to avoid Intel Arc Naga shader translation crashes, utilizing the renderer's built-in blend approximation.
- **Xe Cacheline alignment:** Storage buffer offset alignment is matched to 64 bytes for Intel Xe cacheline efficiency.

**Recommended Intel Arc Settings:**
- **Resizable BAR (ReBAR):** Ensure *Above 4G Decoding* and *Resizable BAR* are enabled in your motherboard BIOS. Intel Arc architecture relies heavily on ReBAR for stable frame rates and memory streaming.
- **Backend Selection:** If you experience driver issues with the default backend, you can switch between Vulkan and Direct3D 12 by setting the environment variable in your command prompt before launching:
  ```cmd
  set RX_BACKEND=vulkan
  :: or
  set RX_BACKEND=dx12
  dist\SkyboundTOTK\SkyboundTOTK.exe
  ```

## Known issues and limitations

These are reports and investigation notes, not a complete compatibility list. Some were observed in earlier TOTKPort builds rather than independently reproduced in standalone Skybound.

| Issue | Status | Details |
| --- | --- | --- |
| Shadows / shaders | **Reported** | Shadow shader problems have been reported. The standalone startup log also showed three shader-decoder assertions; their visual effect and relationship to the shadow issue are unknown. |
| Stretched terrain | **Reported** | Terrain can appear randomly stretched; cause and reproducibility need investigation. |
| Incorrect terrain textures | **Reported** | Some areas have wrong terrain textures; locations and causes need investigation. |
| Weapon and shield wheel background | **Fixed** | The d-pad quick menus opened over a black background; they now take the ability wheel's 1/30 s step while held (PR #11). |
| Ultrahand cannot grab | **Reproduced; open** | Aiming Ultrahand at an object outlines it orange and shows "A Grab", but it never turns green and A does not pick it up (issue #9). Reproduced on both GPU paths; the cause is being investigated. |
| Ultrawide | **Unresolved** | F6 has been reported to stretch the image instead of providing a correctly proportioned wider view. The exact failure is unknown. |
| Zonai Portable Pot particles | **Workaround; retest** | A workaround greatly reduced disappearing-effect spikes in an earlier build, but short stray lines remained and the underlying cause is unknown. Retest in Skybound. |
| Depths gloom | **Earlier fix; retest** | Missing or incorrect ground gloom was reported on the path to Ganon. Earlier notes record a fix for a reproduced case; standalone Skybound and other locations need verification. |
| Fused-weapon inventory icons | **Workaround; retest** | Black icons were seen in earlier testing. A cache-retry workaround was recorded, but the underlying timing issue remains unexplained; retest in Skybound. |
| Browser new-game loading | **Previously reported; retest** | Earlier web experiments had an intermittent loading-screen hang. This is not a verified Windows-launcher issue; the root build does not package a browser release. |
| Browser interpolation judder | **Earlier fix; retest** | Earlier web experiments had interpolation judder; notes record a pacing fix. Browser behavior is not validated by the Windows launcher. |
| Performance and compatibility | **Ongoing limitation** | Frame pacing, shader warm-up and rendering vary by scene and hardware. A build and startup test are not a full-game compatibility test. |

## To-do

- [ ] Add support for other TOTK versions beyond the 1.0.0 base game.
- [ ] Fix all current issues and bugs.

## Source and licenses

NotRexed's original project code is offered under the [MIT License](LICENSE). The required AArch64 lifting/runtime and renderer sources originated in NotRexed's Rexcomp-based TOTKPort; unrelated Rexcomp kits and applications are not included.

**The whole repository and combined build are not MIT-only.** The AArch64 kit includes GPL-licensed suyu/yuzu components and other third-party code under their own licenses. Their licenses and applicable GPL obligations still apply; see [NOTICE.md](NOTICE.md) and the individual license files. MIT does not grant rights to the game. Before publishing, verify that every contributor whose code is being offered under MIT has agreed, and review the applicable source and binary distribution obligations.

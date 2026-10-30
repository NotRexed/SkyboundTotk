"""Build SkyboundTOTK from an already-extracted TOTK 1.0.0 dump."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import logging

ROOT = Path(__file__).resolve().parent
GAME = ROOT / "game"
BUILD = ROOT / "build"
KIT = ROOT / "a64-kit"
WGPU = ROOT / "third_party" / "wgpu-native"
DIST = ROOT / "dist" / "SkyboundTOTK"
# The original base-game module IDs used by this port; this is not a full content-integrity check.
EXPECTED_IDS = {
    "main": "082ce09b06e33a123cb1e2770f5f9147709033db",
    "sdk": "b9046c31eb5d31271be970fe732d38df49c6aa21",
}


def setup_logging():
    log_file = ROOT / "build.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    logging.info("=== build started ===")


def log_and_print(message):
    print(message, flush=True)
    logging.info(message)


def fail(message):
    logging.error(message)
    raise SystemExit("ERROR: " + message)


def safe_job_count(ram_per_job_gb: float = 3.5) -> int:
    """Return a parallel-job count that keeps RAM under ~80% of physical memory.

    Each clang -O3 job on TOTK's giant lifted C chunks consumes 3-8 GB of RAM.
    Running all 22 logical threads at once on a 16 GB machine causes an OOM
    crash/BSOD.  We cap at floor(0.80 * total_ram / ram_per_job_gb), clamped
    to [1, cpu_count].  Override with the RX_JOBS environment variable.
    """
    override = os.environ.get("RX_JOBS")
    if override:
        return max(1, int(override))
    try:
        import ctypes
        class _MEMSTATEX(ctypes.Structure):
            _fields_ = [
                ("dwLength",               ctypes.c_ulong),
                ("dwMemoryLoad",           ctypes.c_ulong),
                ("ullTotalPhys",           ctypes.c_ulonglong),
                ("ullAvailPhys",           ctypes.c_ulonglong),
                ("ullTotalPageFile",       ctypes.c_ulonglong),
                ("ullAvailPageFile",       ctypes.c_ulonglong),
                ("ullTotalVirtual",        ctypes.c_ulonglong),
                ("ullAvailVirtual",        ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = _MEMSTATEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_gb = stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        total_gb = 16.0  # safe fallback
    usable_gb = total_gb * 0.80
    jobs = max(1, int(usable_gb / ram_per_job_gb))
    capped = min(jobs, os.cpu_count() or 1)
    print(f"[build] RAM-safe job count: {capped} "
          f"(~{total_gb:.1f} GB total, {ram_per_job_gb} GB/job budget; "
          f"set RX_JOBS=N to override)")
    return capped


def check_game():
    exefs, romfs = GAME / "exefs", GAME / "romfs"
    for name, expected in EXPECTED_IDS.items():
        path = exefs / name
        if not path.is_file():
            fail(f"Missing {path}. Put your extracted TOTK 1.0.0 ExeFS and RomFS in game/. See README.md.")
        with path.open("rb") as f:
            header = f.read(0x60)
        if len(header) < 0x60 or header[:4] != b"NSO0" or header[0x40:0x60].rstrip(b"\0").hex() != expected:
            fail(f"{path} is not the expected TOTK 1.0.0 module. Updates and other versions are unsupported.")
    if not (romfs / "Event").is_dir() or not (romfs / "ActorSystem").is_dir():
        fail(f"{romfs} does not look like an extracted TOTK RomFS. See README.md.")
    return exefs, romfs


def check_tools():
    if os.name != "nt":
        fail("This launcher builds the Windows x64 version; run it on Windows.")
    if sys.version_info < (3, 10):
        fail("Python 3.10 or newer is required.")
    try:
        import lz4.block
    except ImportError:
        fail(f"Python package lz4 is missing. Install it with: \"{sys.executable}\" -m pip install lz4")
    sys.path.insert(0, str(KIT / "tools"))
    import toolchain
    for tool in ("clang", "clang++", "lld"):
        toolchain.need(toolchain.find(tool))
    for path in (WGPU / "include/webgpu/webgpu.h", WGPU / "lib/wgpu_native.dll", WGPU / "lib/wgpu_native.dll.lib",
                 ROOT / "third_party/sdl2/SDL2.dll"):
        if not path.is_file():
            fail(f"Missing bundled wgpu-native file: {path}")
    # Fail before the lengthy lift if the compiler cannot build a native Windows program.
    run(sys.executable, KIT / "lift/build_emit.py")


def run(*args):
    cmd_str = " ".join(str(x) for x in args)
    log_and_print(f"\n== {cmd_str}")
    
    process = subprocess.Popen(
        [str(x) for x in args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=ROOT,
        bufsize=1,
        universal_newlines=True
    )

    for line in process.stdout:
        print(line, end="", flush=True)
        logging.info(line.rstrip("\n"))

    process.wait()
    if process.returncode != 0:
        logging.error(f"Command failed with exit code {process.returncode}")
        raise subprocess.CalledProcessError(process.returncode, process.args)


def lift_identity(exefs):
    digest = hashlib.sha256()
    for root in (KIT / "lift", KIT / "hos", KIT / "rt"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".c", ".cpp", ".h"):
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    for path in (KIT / "build.py", KIT / "tools/nso.py", ROOT / "ports/zelda-totk/hot_funcs.txt"):
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    for path in sorted(exefs.iterdir()):
        if path.is_file() and path.name in {"rtld", "main", "sdk"} | {f"subsdk{i}" for i in range(1, 10)}:
            digest.update(path.name.encode())
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def package(romfs):
    native = BUILD / "totk/native"
    image = BUILD / "totk/image"
    DIST.mkdir(parents=True, exist_ok=True)
    # Refuse to silently replace a previous installation, including saves or mods.
    if any(DIST.iterdir()):
        fail(f"Standalone output {DIST} is not empty. Move it aside before rebuilding; keep its user folder.")
    for src, name in ((native / "totk.exe", "SkyboundTOTK.exe"),
                      (native / "wgpu_native.dll", "wgpu_native.dll"),
                      (ROOT / "third_party/sdl2/SDL2.dll", "SDL2.dll"),
                      (image / "image.bin", "image.bin"),
                      (image / "image.json", "image.json")):
        if not src.is_file():
            fail(f"Required build output is missing: {src}")
        shutil.copy2(src, DIST / name)
    log_and_print(f"Copying RomFS to {DIST / 'romfs'}; this may take several minutes.")
    shutil.copytree(romfs, DIST / "romfs", symlinks=False)
    log_and_print(f"\nBuild complete. Open {DIST / 'SkyboundTOTK.exe'} to play.")


def main():
    setup_logging()
    exefs, romfs = check_game()
    check_tools()
    # A matching completed lift is reusable; changed inputs or an interrupted lift need a fresh directory.
    identity = lift_identity(exefs)
    stamp = BUILD / "totk/lift-input.sha256"
    image = BUILD / "totk/image/image.json"
    generated = BUILD / "totk/gen"
    complete = image.is_file() and (generated / "a64_suyu.h").is_file() and any(generated.glob("code_*.c"))
    
    if complete and stamp.is_file() and stamp.read_text(encoding="ascii").strip() == identity:
        with image.open(encoding="utf-8") as f:
            modules = {m["name"]: m["build_id"] for m in json.load(f)["modules"]}
        if any(modules.get(name) != build_id for name, build_id in EXPECTED_IDS.items()):
            fail("Existing build contains a different game version. Move the build folder aside before retrying.")
        log_and_print("Reusing the existing lifted base-game code.")
    else:
        if BUILD.exists() and any(BUILD.iterdir()):
            fail("Build inputs changed or the lift is incomplete. Move the build folder aside before rebuilding.")
        run(sys.executable, KIT / "build.py", "lift", "totk", "--exefs", exefs,
            "--skip", "subsdk0", "--build", BUILD)
        stamp.write_text(identity + "\n", encoding="ascii")

    # ── RAM-safe parallel compile ─────────────────────────────────────────────
    # Each clang -O3 job on TOTK's lifted C files uses 3-8 GB of RAM.
    # Without a cap, all 22 logical cores compile at once → OOM crash on 16 GB.
    # RX_JOBS overrides; RX_OPT overrides the optimisation level (O1 is faster
    # to compile; O2/O3 give better runtime performance).
    jobs = safe_job_count(ram_per_job_gb=3.5)
    opt  = os.environ.get("RX_OPT", "O3")   # O1 builds faster; O3 is default

    run(sys.executable, KIT / "build.py", "native", "totk",
        "--build", BUILD, "--wgpu", WGPU,
        "--opt", opt, "-j", str(jobs))
    package(romfs)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        logging.error(f"ERROR: command failed with exit code {error.returncode}")
        print(f"ERROR: command failed with exit code {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode or 1)

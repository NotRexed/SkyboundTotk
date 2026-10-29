"""a64-kit build: Switch ExeFS -> lifted C -> native exe (web comes later).

  python build.py lift   <name> --exefs <dir> [--skip subsdk0] [--hle hle.txt] [--build <root>] [--comments]
  python build.py native <name> [--build <root>] [--opt O1] [-j N]
  python build.py stubs  <name> [--build <root>]       regenerate import stubs after adding HLE functions
  python build.py run    <name> [--build <root>] [--romfs <dir>] [-- args]

Output goes to <root>/<name>/ (default root: a64-kit/build): image/ (image.bin, image.json), gen/ (lifted C,
dispatch table, generated suyu runtime, import stubs), native/ (objects, <name>.exe).
Every import without an HLE definition in hos/ gets a generated stub (gen/a64_stubs.c) that logs its first call
and returns 0, so the game links from day one and the log says what to implement next (smm's approach).
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("lift", "hos", "tools"):
    sys.path.insert(0, os.path.join(HERE, sub))
import toolchain


def _safe_job_count(ram_per_job_gb=3.5):
    """Cap parallel jobs so peak RAM stays under ~80% of physical memory.

    Each clang -O3 job on a large TOTK lifted C file consumes 3-8 GB RAM.
    Running all logical cores in parallel (22 on Ultra 7 155H) on 16 GB
    causes an OOM crash.  Override with the RX_JOBS environment variable.
    """
    override = os.environ.get("RX_JOBS")
    if override:
        return max(1, int(override))
    try:
        import ctypes
        class _MS(ctypes.Structure):
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        s = _MS()
        s.dwLength = ctypes.sizeof(s)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        total_gb = s.ullTotalPhys / (1024 ** 3)
    except Exception:
        total_gb = 16.0
    jobs = max(1, int(total_gb * 0.80 / ram_per_job_gb))
    return min(jobs, os.cpu_count() or 1)

RT = os.path.join(HERE, "rt")
HOS = os.path.join(HERE, "hos")
GPU = os.path.join(HERE, "gpu")


def dirs(a):
    root = os.path.join(os.path.abspath(a.build or os.path.join(HERE, "build")), a.name)
    return root, os.path.join(root, "image"), os.path.join(root, "gen"), os.path.join(root, "native")


def entry_hooks(img_dir=None):
    """Lifted functions with an entry hook in hos/*.c, address -> the hook's C name: ENTRY_HOOK(0x0xxxxxxx), and
    ENTRY_HOOK_SYM(name) for a function the image exports by that name (image/image.json)."""
    out = {}
    exports = None
    for f in sorted(os.listdir(HOS)):
        if not f.endswith(".c"):
            continue
        src = open(os.path.join(HOS, f)).read()
        for a in re.findall(r"^ENTRY_HOOK\((0x[0-9a-fA-F]+)\)", src, re.M):
            if len(a) != 10:
                sys.exit("%s: ENTRY_HOOK(%s): write the address with 8 hex digits" % (f, a))
            out[int(a, 16)] = "entry_" + a
        for sym in re.findall(r"^ENTRY_HOOK_SYM\((\w+)\)", src, re.M):
            if exports is None:
                path = os.path.join(img_dir, "image.json") if img_dir else ""
                exports = json.load(open(path)).get("exports", {}) if os.path.exists(path) else {}
            if sym in exports:
                out[exports[sym]] = "entry_sym_" + sym
            else:
                print("%s: ENTRY_HOOK_SYM(%s): the image exports no function by that name (left out)" % (f, sym))
    return out


def apply_entry_hooks(gen, img_dir=None):
    """Splice `if (entry_...(c)) return;` into the lifted functions that have an entry hook, and take it out of
    the ones that no longer do. Files are only rewritten when the set changed (gen/entry_hooks.json)."""
    want = entry_hooks(img_dir)
    state = os.path.join(gen, "entry_hooks.json")
    have = {}
    if os.path.exists(state):
        old = json.load(open(state))
        have = {int(k): v for k, v in old.items()} if isinstance(old, dict) else {a: "entry_0x%08x" % a for a in old}
    if want == have:
        return
    for f in sorted(os.listdir(gen)):
        if not (f.startswith("code_") and f.endswith(".c")):
            continue
        path = os.path.join(gen, f)
        src = open(path, encoding="utf-8", errors="surrogateescape").read()
        if not (set(want) | set(have)) & {int(m, 16) for m in re.findall(r"^void f_([0-9a-f]{8})\(Ctx \*__restrict (?:A64_NOESC )?c\) \{$",
                                                                          src, re.M)}:
            continue
        lines = [l for l in src.split("\n") if not l.startswith("    if (entry_") and not l.startswith("int entry_")]
        out, decls = [], []
        for l in lines:
            out.append(l)
            m = re.match(r"^void f_([0-9a-f]{8})\(Ctx \*__restrict (?:A64_NOESC )?c\) \{$", l)
            if m and int(m.group(1), 16) in want:
                name = want[int(m.group(1), 16)]
                out.append("    if (%s(c)) return;" % name)
                decls.append("int %s(Ctx *A64_NOESC);" % name)
        out[1:1] = decls
        open(path, "w", encoding="utf-8", errors="surrogateescape").write("\n".join(out))
        print("entry hooks in %s: %s" % (f, ", ".join(d[4:-8] for d in decls) or "none"))
    json.dump({str(k): v for k, v in sorted(want.items())}, open(state, "w"))


def splices():
    """SPLICE_AFTER(0x0xxxxxxx, "text") in hos/*.c: address -> the anchor text (the C string literal's contents)."""
    out = {}
    for f in sorted(os.listdir(HOS)):
        if not f.endswith(".c"):
            continue
        src = open(os.path.join(HOS, f)).read()
        for each, a, text in re.findall(r'^SPLICE_AFTER(_EACH)?\((0x[0-9a-fA-F]+),\s*"((?:[^"\\]|\\.)*)"\)', src, re.M):
            if len(a) != 10:
                sys.exit("%s: SPLICE_AFTER(%s): write the address with 8 hex digits" % (f, a))
            # (SPLICE_AFTER_EACH: after every line with the text; kept with a \x01 in front)
            out[int(a, 16)] = ("\x01" if each else "") + bytes(text, "utf-8").decode("unicode_escape")
    return out


def apply_splices(gen):
    """Put `splice_0x...(c);` after the first line of each spliced lifted function that contains its anchor, and
    take out the ones no longer wanted. Files are only rewritten when the set changed (gen/splices.json)."""
    want = splices()
    state = os.path.join(gen, "splices.json")
    have = {int(k): v for k, v in json.load(open(state)).items()} if os.path.exists(state) else {}
    if want == have:
        return
    found = set()
    for f in sorted(os.listdir(gen)):
        if not (f.startswith("code_") and f.endswith(".c")):
            continue
        path = os.path.join(gen, f)
        src = open(path, encoding="utf-8", errors="surrogateescape").read()
        funcs = {int(m, 16) for m in re.findall(r"^void f_([0-9a-f]{8})\(Ctx \*__restrict (?:A64_NOESC )?c\) \{$", src, re.M)}
        if not (set(want) | set(have)) & funcs:
            continue
        lines = [l for l in src.split("\n") if not l.startswith("    splice_0x") and not l.startswith("void splice_0x")]
        out, decls, cur, done = [], [], None, False
        for l in lines:
            out.append(l)
            m = re.match(r"^void f_([0-9a-f]{8})\(Ctx \*__restrict (?:A64_NOESC )?c\) \{$", l)
            if m:
                cur, done = int(m.group(1), 16), False
                continue
            each = cur in want and want[cur].startswith("\x01")
            if cur in want and (not done or each) and want[cur].lstrip("\x01") in l:
                name = "splice_0x%08x" % cur
                out.append("    %s(c);" % name)
                if not done:
                    decls.append("void %s(Ctx *A64_NOESC);" % name)
                found.add(cur)
                done = True
        out[1:1] = decls
        open(path, "w", encoding="utf-8", errors="surrogateescape").write("\n".join(out))
        print("splices in %s: %s" % (f, ", ".join(d[5:-8] for d in decls) or "none"))
    missing = set(want) - found
    if missing:
        sys.exit("SPLICE_AFTER: anchor not found in lifted %s" % ", ".join("f_%08x" % a for a in sorted(missing)))
    json.dump({str(k): v for k, v in sorted(want.items())}, open(state, "w"))


def implemented():
    """C names of the HLE functions in hos/*.c: `void imp_x(Ctx` definitions and `HLE(x)` macros."""
    names = set()
    for f in os.listdir(HOS):
        if f.endswith(".c"):
            src = open(os.path.join(HOS, f)).read()
            names |= set(re.findall(r"^void (imp_\w+)\(Ctx", src, re.M))
            names |= {"imp_" + n for n in re.findall(r"^HLE\((\w+)\)", src, re.M)}
    return names


def demangle(syms):
    try:
        out = subprocess.run([toolchain.find("llvm-cxxfilt")], input="\n".join(syms), capture_output=True,
                             text=True).stdout.splitlines()
        return dict(zip(syms, out)) if len(out) == len(syms) else {s: s for s in syms}
    except OSError:
        return {s: s for s in syms}


def gen_stubs(gen, meta):
    have = implemented()
    hooks = [h for h in meta["hooks"] if h["c"] not in have]
    dm = demangle([h["sym"] for h in hooks])
    lines = ['#include "a64.h"', "void a64_unimplemented(Ctx *A64_NOESC c, const char *sym, int *seen);", ""]
    for h in hooks:
        lines.append("/* %s */" % dm[h["sym"]].replace("*/", "* /"))
        lines.append('void %s(Ctx *A64_NOESC c) { static int seen; a64_unimplemented(c, "%s", &seen); }' % (h["c"], h["sym"]))
    open(os.path.join(gen, "a64_stubs.c"), "w", newline="\n").write("\n".join(lines) + "\n")
    return len(hooks)


# SDK namespaces that talk to system services (IPC we do not emulate): every one of the game's imports from them is
# hooked, to the HLE in hos/ if there is one, else to a generated stub that logs and returns 0. The rest of the SDK
# (libc, nn::util, nn::image, nn::fontll, nn::mem, std) is pure computation and runs as lifted code.
OS_NAMESPACES = {"audio", "hid", "oe", "prepo", "nfp", "time", "account", "vi", "swkbd", "pl", "err", "codec", "diag",
                 "detail", "settings", "ns", "applet", "album", "friends", "nifm", "ssl", "socket", "ec", "bcat",
                 "web", "ldn", "irsensor", "aoc", "olsc", "pctl", "apm", "ro"}


def nested_name(sym):
    """The components of an Itanium nested name (_ZN[K]<len><id>...E), or None if it is not that simple shape."""
    m = re.match(r"_ZNK?", sym)
    if not m:
        return None
    i, parts = m.end(), []
    while i < len(sym) and sym[i].isdigit():
        j = i
        while sym[j].isdigit():
            j += 1
        n = int(sym[i:j])
        parts.append(sym[j:j + n])
        i = j + n
    return parts if i < len(sym) and sym[i] == "E" else None


def os_facing(sym):
    """A free function of an OS-facing namespace: nn::<OS namespace>::f. Class members (constructors set vtables up)
    and data stay with the SDK's own code. NVN and nv:: are not here: the real NVN driver runs, over hos/nvdrv.c."""
    parts = nested_name(sym)
    return bool(parts) and parts[0] == "nn" and len(parts) == 3 and parts[1] in OS_NAMESPACES


def check_hle(img, hle):
    """Every imp_ in hos/ must name a symbol some module defines or imports: a typo would silently leave the
    real function in place."""
    known = set(img.exports)
    for m in img.modules:
        known |= {s["name"] for s in m.symbols if s["name"]}
    bad = sorted(h for h in hle if h not in known)
    if bad:
        sys.exit("hos/ defines imp_ functions for symbols no module has (typo?):\n  " + "\n  ".join(bad))


def thread_hooks(img, meta):
    """The SDK thread entry's start and exit hooks: the variables nnmusl's __nnmusl_init_dso fills with its
    per-thread TLS setup and teardown. The entry (nn::os::detail::ThreadManagerImplByHorizon's, unexported) starts
    `mrs x1, tpidrro_el0; add x1, x1, #0x1f8; str x0, [x1]` and loads each hook through a GOT slot:
    `adrp x8, page; ldr x8, [x8, #off]; ldr x8, [x8]`. Our threads (hos_thread.c) run the hooks the same way."""
    import struct
    for m in meta["modules"]:
        for pc in range(m["base"], m["base"] + m["size"] - 12, 4):
            if not img.in_code(pc) or img.word(pc) != 0xD53BD061:
                continue
            if img.word(pc + 4) != 0x9107E021 or img.word(pc + 8) != 0xF9000020:
                continue
            hooks, page = [], {}
            for q in range(pc + 12, pc + 12 + 4 * 40, 4):
                w = img.word(q)
                if (w & 0x9F000000) == 0x90000000:                                   # ADRP
                    imm = ((w >> 3) & 0x1FFFFC) | ((w >> 29) & 3)
                    imm = imm - (1 << 21) if imm & (1 << 20) else imm
                    page[w & 31] = (q & ~0xFFF) + (imm << 12)
                elif (w & 0xFFC00000) == 0xF9400000 and ((w >> 5) & 31) in page:     # LDR Xt, [Xn, #imm]
                    got = page.pop((w >> 5) & 31) + ((w >> 10) & 0xFFF) * 8
                    hooks.append(struct.unpack("<Q", img.read(got, 8))[0])
                if len(hooks) == 2:
                    return hooks
    return [0, 0]


def write_image_c(gen, img, meta, exefs):
    """Boot constants for hos_main.c: the image end, each module's DT_INIT in rtld's order, the entry points, and
    the main thread's stack, priority and core from main.npdm (META header)."""
    ex = meta["exports"]
    inits = [m["init"] for m in meta["modules"] if m["init"]]
    npdm = open(os.path.join(exefs, "main.npdm"), "rb").read()
    if npdm[:4] != b"META":
        sys.exit("main.npdm: no META header")
    prio, core = npdm[0xE], npdm[0xF]
    stack = int.from_bytes(npdm[0x1C:0x20], "little")
    lines = ["/* Generated by a64-kit/build.py lift. */", "#include <stdint.h>",
             "const uint64_t hos_main_stack_size = %#xULL;" % stack,
             "const int hos_main_priority = %d, hos_main_core = %d;" % (prio, core),
             "const uint64_t hos_image_end = %#xULL;" % meta["end"],
             "const int hos_module_count = %d;" % len(inits),
             "const uint64_t hos_module_init[] = {%s};" % ", ".join("%#xULL" % i for i in inits),
             "const char *const hos_module_name[] = {%s};" % ", ".join('"%s"' % m["name"] for m in meta["modules"] if m["init"]),
             "const uint64_t hos_entry_nninitStartup = %#xULL;" % ex.get("nninitStartup", 0),
             "const uint64_t hos_entry_nnMain = %#xULL;" % ex.get("nnMain", 0)]
    # the SDK startup nn::init::Start runs (sdk+0xfa060): libc stages, the OS core, the SDK module
    for sym in ("_init_libc0", "_init_libc1", "_init_libc2", "nnosInitialize", "nninitInitializeSdkModule",
                "nvnBootstrapLoaderInternal"):
        lines.append("const uint64_t hos_entry_%s = %#xULL;" % (sym.lstrip("_"), ex.get(sym, 0)))
    start_hook, exit_hook = thread_hooks(img, meta)
    lines.append("const uint64_t hos_thread_start_hook = %#xULL, hos_thread_exit_hook = %#xULL;" % (start_hook, exit_hook))
    open(os.path.join(gen, "hos_image.c"), "w", newline="\n").write("\n".join(lines) + "\n")


def mod_patches(img, mods_dir):
    """exefs patches of the mods under mods_dir (hos/mods.c's layout: any folder holding exefs/, '-' or '_' names off,
    sorted) written into the image before it is lifted: code patches only take effect compiled in. .pchtxt
    (@nsobid-<id>, @flag offset_shift, @enabled/@disabled/@stop, "<offset> <hex>" or "<offset> \"text\"") and .ips
    / IPS32 named <build id>.ips; offsets count the 0x100-byte NSO header unless offset_shift says otherwise."""
    import nx_image
    found = []
    def walk(d, name, depth):
        if depth and (os.path.isdir(os.path.join(d, "exefs")) or os.path.isdir(os.path.join(d, "romfs"))):
            found.append((name, d))
            return
        if depth >= 4:
            return
        for e in sorted(os.listdir(d)):
            if os.path.isdir(os.path.join(d, e)) and e[0] not in "-_.":
                walk(os.path.join(d, e), (name + "/" if name else "") + e, depth + 1)
    walk(mods_dir, "", 0)
    def module(bid):
        want = "".join(ch for ch in bid.lower() if ch in "0123456789abcdef").rstrip("0")
        return next((m for m in img.modules if len(want) >= 16 and m.build_id.startswith(want)), None)
    applied = []
    def put(m, off, data, label):
        if off < 0 or off + len(data) > len(m.image):
            print("  %s: %s+%#x outside the module: left out" % (label, m.name, off))
            return
        at = m.base + off - nx_image.MAIN_BASE
        img.mem[at:at + len(data)] = data
        applied.append("%s %s+%#x %s" % (label, m.name, off, data.hex()))
    for name, d in sorted(found, key=lambda x: x[0].lower()):
        ex = os.path.join(d, "exefs")
        if not os.path.isdir(ex):
            continue
        for f in sorted(os.listdir(ex)):
            path, label = os.path.join(ex, f), name + "/" + f
            if f.lower().endswith(".pchtxt"):
                m, shift, on = None, 0, False
                for line in open(path, encoding="utf-8", errors="replace"):
                    t = line.strip()
                    if t.startswith("@nsobid-"):
                        m = module(t[8:])
                        if not m:
                            print("  %s: build id %s matches no module (another game version?)" % (label, t[8:48]))
                    elif t.startswith("@flag offset_shift"):
                        shift = int(t.split()[2], 0)
                    elif t.startswith("@enabled"):
                        on = True
                    elif t.startswith("@disabled") or t.startswith("@stop"):
                        on = False
                    elif on and m and t[:1] in "0123456789abcdefABCDEF" and t[:1]:
                        off_s, _, rest = t.partition(" ")
                        rest = rest.strip()
                        if rest.startswith('"'):
                            data = rest[1:rest.index('"', 1)].encode() + b"\0"
                        else:
                            hx = re.match(r"[0-9a-fA-F]*", rest).group(0)
                            data = bytes.fromhex(hx[:len(hx) // 2 * 2])
                        if data:
                            put(m, int(off_s, 16) + shift - 0x100, data, label)
            elif f.lower().endswith(".ips"):
                m = module(f)
                if not m:
                    print("  %s: names no module (another game version?)" % label)
                    continue
                b = open(path, "rb").read()
                ips32 = b[:5] == b"IPS32"
                if not ips32 and b[:5] != b"PATCH":
                    print("  %s: not an IPS file" % label)
                    continue
                i, ob = 5, 4 if ips32 else 3
                while i + ob <= len(b) and b[i:i + (4 if ips32 else 3)] != (b"EEOF" if ips32 else b"EOF"):
                    off = int.from_bytes(b[i:i + ob], "big"); i += ob
                    size = int.from_bytes(b[i:i + 2], "big"); i += 2
                    if size:
                        data = b[i:i + size]; i += size
                    else:
                        data = bytes([b[i + 2]]) * int.from_bytes(b[i:i + 2], "big"); i += 3
                    if off >= 0x100 and data:
                        put(m, off - 0x100, data, label)
    print("mods: %d exefs patch records from %d mods in %s written into the image" % (len(applied), len(found), mods_dir))
    return applied


def cmd_lift(a):
    import nx_image, a64_lift, a64_rtgen
    root, img_dir, gen, _ = dirs(a)
    hot = a.hot or os.path.join(HERE, "..", "ports", "zelda-" + a.name, "hot_funcs.txt")
    if hot != "none" and os.path.isfile(hot) and not os.environ.get("A64_HOT_FUNCS"):
        os.environ["A64_HOT_FUNCS"] = hot
    t0 = time.time()
    hle = {n[4:] for n in implemented()}
    if a.hle:
        hle |= {l.strip() for l in open(a.hle) if l.strip() and not l.startswith("#")}
    import nso
    mods = nso.load_exefs(a.exefs)
    main = next(m for m in mods if m.name == "main")
    kind = {s["name"]: s["type"] for m in mods for s in m.symbols if s["defined"] and s["name"]}
    stubbed = {s["name"] for s in main.symbols if not s["defined"] and s["name"] and os_facing(s["name"])
               and kind.get(s["name"], "func") == "func"} - hle
    img = nx_image.NxImage(a.exefs, a.skip.split(",") if a.skip else [], hle | stubbed)
    applied = mod_patches(img, a.mods) if a.mods else []   # (--mods <dir>: exefs patches compiled in)
    check_hle(img, hle)
    print("OS-facing imports without an HLE yet: %d (generated stubs)" % len(stubbed))
    meta = img.write(img_dir)
    os.makedirs(gen, exist_ok=True)
    with open(os.path.join(img_dir, "mod_patches.txt"), "w") as f:   # (what this lift compiled in)
        f.write("".join(l + "\n" for l in applied))
    write_image_c(gen, img, meta, a.exefs)
    print("HLE: %d functions from hos/, %d hooked at a definition" % (len(hle), sum(1 for n in img.hooks.values()
                                                                                if n[4:] in hle)))
    print("image: %s, %d entries, %d hooks (%.0fs)" % (", ".join("%s@%#x" % (m["name"], m["base"]) for m in meta["modules"]),
                                                       len(img.entries), len(img.hooks), time.time() - t0))
    L = a64_lift.Lifter(img, comments=a.comments)
    L.discover(max_insns=a.max_func)
    print("discovered (%.0fs)" % (time.time() - t0))
    for f in os.listdir(gen):
        if re.fullmatch(r"code_\d+\.c", f):
            os.remove(os.path.join(gen, f))
    # (the new code files have no entry hooks or splices: the native build must put them in again, so their state
    # files go too; kept, they said "applied" and every hook and splice was missing after a lift: 2026-10-20)
    for f in ("entry_hooks.json", "splices.json"):
        if os.path.exists(os.path.join(gen, f)):
            os.remove(os.path.join(gen, f))
    files = L.write(gen)
    a64_rtgen.write(gen)
    stubs = gen_stubs(gen, meta)
    L.report()
    print("wrote %d code files, %d import stubs to %s (%.0fs)" % (len(files), stubs, gen, time.time() - t0))
    with open(os.path.join(gen, "indirect_br.txt"), "w") as f:
        f.write("".join("%08x\n" % pc for pc in sorted(L.ijump)))
    with open(os.path.join(gen, "jump_tables_missing.txt"), "w") as f:   # BRs whose table address was not found
        f.write("".join("%08x\n" % pc for pc in sorted(L.jt_missing)))


def headers_key(*folders):
    """One hash over every header in the given folders: a header change must rebuild what includes it."""
    h = hashlib.sha1()
    for d in folders:
        for f in sorted(os.listdir(d)):
            if f.endswith(".h"):
                h.update(f.encode() + open(os.path.join(d, f), "rb").read())
    return h.hexdigest()


def tree_headers_key(root):
    """headers_key for a source tree (suyu's headers sit in many folders)"""
    h = hashlib.sha1()
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            if f.endswith((".h", ".hpp", ".inc")):
                h.update(f.encode() + open(os.path.join(d, f), "rb").read())
    return h.hexdigest()


def gpu_sources():
    """The renderer (gpu/render/) and what it builds on, from suyu (gpu/third_party/suyu): the shader recompiler
    (Maxwell frontend, IR passes, the WGSL backend), surface formats, texture decoders (block linear, ASTC)."""
    suyu = os.path.join(GPU, "third_party", "suyu")
    out = []
    for d, dirs, files in os.walk(os.path.join(suyu, "shader_recompiler")):
        dirs[:] = [x for x in dirs if x != "glsl"]   # the GLSL backend: the shader tests' reference only
        out += [os.path.join(d, f) for f in files if f.endswith(".cpp") and f not in ("attribute.cpp", "flow_test.cpp")]
    vc = os.path.join(suyu, "video_core")
    out += [os.path.join(vc, *f.split("/")) for f in ("surface.cpp", "texture_cache/format_lookup_table.cpp",
                                                      "textures/astc.cpp", "textures/decoders.cpp")]
    render = os.path.join(GPU, "render")
    out += [os.path.join(render, f) for f in os.listdir(render) if f.endswith(".cpp")]
    return sorted(out)


def capture_tracking_flag():
    """The lifted code's context parameter is A64_NOESC (rt/a64.h), but LLVM looks at only a pointer's first 100 uses
    to see whether it escapes: past that every guest store may have changed the context. 5000 covers the hot castle
    function (~4200 uses: context loads 535 -> 280); the giant loop functions in code_004/005 pass any limit and
    made each limit's compile cost explode (100000: code_004 496 -> 1625 s native, > 90 min on wasm).
    RX_CAPTURE_USES=<n> (0: LLVM's default of 100)."""
    n = os.environ.get("RX_CAPTURE_USES", "5000")
    return "-capture-tracking-max-uses-to-explore=" + n if n != "0" else None


def compile_one(job):
    cc, src, obj, flags, deps = job
    key = hashlib.sha1((open(src, "rb").read() + " ".join(flags).encode() + deps.encode())).hexdigest()
    stamp = obj + ".sha1"
    if os.path.exists(obj) and os.path.exists(stamp) and open(stamp).read() == key:
        return src, 0.0, None
    t = time.time()
    r = subprocess.run([cc] + flags + ["-c", src, "-o", obj], capture_output=True, text=True)
    if r.returncode:
        return src, time.time() - t, r.stderr[-4000:]
    open(stamp, "w").write(key)
    return src, time.time() - t, None


def cmd_native(a):
    root, img_dir, gen, nat = dirs(a)
    # (RX_OBJ_DIR=<name>: objects in native/<name>, so builds with other flags do not recompile each other's)
    obj_dir = os.path.join(nat, os.environ.get("RX_OBJ_DIR", "obj"))
    # RX_LIFTED_LTO=1 (experiment): the lifted code as ThinLTO bitcode, so small functions inline across the 42 code
    # files (a call between lifted functions stores and reloads the caller's guest registers in the context)
    lifted_lto = ["-flto=thin"] if os.environ.get("RX_LIFTED_LTO") else []
    # (RX_NO_GUEST_MARKS=1, measurement only: lifted stores mark no page, so the renderer misses guest writes)
    if os.environ.get("RX_NO_GUEST_MARKS"):
        lifted_lto = lifted_lto + ["-DA64_NO_MARKS"]
    # Sample PGO of the lifted code (experiment): RX_PROFILE_LINES=1 builds it with line tables, so HOS_PROFILE_PCS
    # samples map to lifted source lines (scripts/sample_profile.py writes an LLVM sample profile from them);
    # RX_SAMPLE_PROFILE=<file> builds with that profile (line tables again: the profile is matched by line)
    if os.environ.get("RX_PROFILE_LINES") or os.environ.get("RX_SAMPLE_PROFILE"):
        lifted_lto = lifted_lto + ["-gline-tables-only", "-fdebug-info-for-profiling"]
    if os.environ.get("RX_SAMPLE_PROFILE"):
        lifted_lto = lifted_lto + ["-fprofile-sample-use=" + os.path.abspath(os.environ["RX_SAMPLE_PROFILE"])]
    # The lifted code's context parameter is A64_NOESC (rt/a64.h), but LLVM only looks at a pointer's first 100 uses to
    # see whether it escapes and assumes it does past that: then every guest store may have changed the context.
    # RX_CAPTURE_USES=<n> (0: LLVM's default); with ThinLTO the link needs it too.
    capture_flag = capture_tracking_flag()
    os.makedirs(obj_dir, exist_ok=True)
    cc = toolchain.need(toolchain.find("clang"))
    base = toolchain.native_target() + ["-I" + gen, "-I" + RT, "-Wno-everything", "-fno-strict-aliasing", "-mcx16"]
    # RX_MARCH (e.g. x86-64-v3: AVX2/BMI2/FMA; the default runs on any x86-64). Measured twice without a gain: 10-16
    # (rain) and 10-22 (castle, game threads bound: 3 interleaved rounds, mixed, median 69.4 vs 73.2 fps).
    if os.environ.get("RX_MARCH"):
        base.append("-march=" + os.environ["RX_MARCH"])
    # Guest stores mark their pages (rt/a64.h A64_MARK, as the web build does) instead of Windows' write watch telling
    # the renderer which pages changed: GetWriteWatch over the ~2 GB the GPU sees took 20-50 ms while the game streamed
    # (the game waits for that scan at its fences: ~55 ms hitches every few seconds walking on Great Sky Island; the
    # scan is ~0.08 ms now). RX_SOFT_WRITE_WATCH=0: Windows' write watch, as before 2026-10-20.
    soft_watch = [] if os.environ.get("RX_SOFT_WRITE_WATCH") == "0" else ["-DA64_WRITE_TRACK"]
    base += soft_watch
    jobs = []
    lifted = sorted(os.path.join(gen, f) for f in os.listdir(gen) if f.endswith(".c"))
    runtime = [os.path.join(RT, f) for f in os.listdir(RT) if f.endswith(".c")] + \
              [os.path.join(HOS, f) for f in os.listdir(HOS) if f.endswith(".c")]
    # lifted chunks include only rt/ and generated headers; the OS layer also includes hos/*.h and the renderer's API
    apply_entry_hooks(gen, img_dir)
    apply_splices(gen)
    deps_lifted = headers_key(RT, gen)
    deps_runtime = headers_key(RT, HOS, gen, os.path.join(GPU, "render"))
    # the renderer: WebGPU through wgpu-native (a release of github.com/gfx-rs/wgpu-native: include/, lib/)
    wgpu = a.wgpu or os.environ.get("WGPU_NATIVE")
    if wgpu and not os.path.isfile(os.path.join(wgpu, "include", "webgpu", "webgpu.h")):
        sys.exit("%s is not a wgpu-native release (include/webgpu/webgpu.h)" % wgpu)
    if not wgpu:
        print("building without the renderer (--wgpu <wgpu-native release> or WGPU_NATIVE to build it)")
        runtime.append(os.path.join(GPU, "render", "rx_none.c"))
    for src in lifted + runtime:
        deps = deps_lifted if src in lifted else deps_runtime
        # lifted code: no debug info unless asked (it is a quarter of the object size and most of the link time).
        # LoopRotate duplicates loop headers; in TOTK's goto-dense giant functions it was 120 s of a 306 s compile,
        # and capping the header it may copy at 0 brought that chunk to 197 s.
        # (RX_FPX_INLINE=1: each float op's fast path inlined instead of a call. Tried 2026-10-16: a full lifted build
        # 12 -> 80 min, the exe 277 -> 388 MB, and uncapped rain 45.5 -> 42.3 fps (code size): calls stay.
        # Also tried 2026-10-20: helpers without the context, declared const (FPCR/FPSR state passed by value, the
        # exact bodies on a thread-local context), so that a call no longer spills and reloads the caller's guest
        # registers: build 12 -> 20 min, exe 277 -> 316 MB, castle benchmark 67.5 -> ~65 fps: not kept)
        fpx = ["-DA64_FPX_FN=static inline __attribute__((always_inline))"] if os.environ.get("RX_FPX_INLINE") else []
        # (-ffunction-sections: each lifted function in a section of its own, so RX_LINK_ORDER can place them)
        flags = ["-" + a.opt, "-g" if a.debug else "-g0", "-mllvm", "-rotation-max-header-size=0",
                 "-ffunction-sections"] + (["-mllvm", capture_flag] if capture_flag else []) + fpx + lifted_lto \
            if src in lifted else ["-O2", "-g", "-I" + os.path.join(HERE, "..", "third_party", "opus", "include")]
        # (the profile's hot functions, lifted with A64_HOT_CHUNK=1: RX_HOT_CFLAGS, e.g. a higher inline threshold)
        if os.path.basename(src) == "code_hot.c" and os.environ.get("RX_HOT_CFLAGS"):
            flags = flags + os.environ["RX_HOT_CFLAGS"].split()
        jobs.append((cc, src, os.path.join(obj_dir, os.path.basename(src)[:-2] + ".o"), base + flags, deps))
    # libopus (third_party/opus, the 1.5.2 release sources: nn::codec's hardware Opus decoder in hos/nn_codec.c)
    opus = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "opus")
    if os.path.isfile(os.path.join(opus, "sources.txt")):
        oflags = toolchain.native_target() + ["-O2", "-g0", "-Wno-everything", "-DOPUS_BUILD", "-DUSE_ALLOCA",
                                              "-DHAVE_LRINTF", "-DHAVE_LRINT"] +                  ["-I" + os.path.join(opus, d) for d in ("include", "celt", "silk", "silk/float", "src")]
        okey = hashlib.sha1(b"opus-1.5.2" + " ".join(oflags).encode()).hexdigest()
        for rel in open(os.path.join(opus, "sources.txt")).read().split():
            jobs.append((cc, os.path.join(opus, rel), os.path.join(obj_dir, "opus_" + rel.replace("/", "_")[:-2] + ".o"),
                         oflags, okey))
    if wgpu:
        cxx = toolchain.need(toolchain.find("clang++"))
        # (NDEBUG: no asserts in the renderer and suyu; ThinLTO: the renderer's small helpers inline across files.
        # RX_NO_LTO=1 builds without it)
        lto = [] if os.environ.get("RX_NO_LTO") else ["-flto=thin"]
        gflags = toolchain.native_target() + ["-std=c++20", "-O2", "-g", "-DNDEBUG", "-Wno-everything"] + lto + soft_watch + [
                                              "-I" + os.path.join(GPU, "compat"),
                                              "-I" + os.path.join(GPU, "third_party", "suyu"),
                                              "-I" + os.path.join(wgpu, "include")]
        gdeps = tree_headers_key(GPU) + hashlib.sha1(open(os.path.join(wgpu, "include", "webgpu", "webgpu.h"),
                                                          "rb").read()).hexdigest()
        for src in gpu_sources():
            name = os.path.relpath(src, GPU).replace(os.sep, "_")[:-4]
            jobs.append((cxx, src, os.path.join(obj_dir, "gpu_" + name + ".o"), gflags, gdeps))
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))   # biggest first
    t0 = time.time()
    failed = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for i, (src, dt, err) in enumerate(ex.map(compile_one, jobs), 1):
            if err:
                failed += 1
                print("FAILED %s\n%s" % (os.path.basename(src), err))
            elif dt:
                print("[%d/%d] %s %.0fs" % (i, len(jobs), os.path.basename(src), dt), flush=True)
    if failed:
        sys.exit("%d files failed to compile" % failed)
    exe = os.path.join(nat, (a.output or a.name) + ".exe")
    objs = [j[2] for j in jobs]
    rsp = os.path.join(nat, "link.rsp")
    open(rsp, "w").write("\n".join('"%s"' % o.replace("\\", "/") for o in objs))
    libs = ["-luser32", "-lkernel32", "-ldbghelp", "-lwinmm", "-lgdi32", "-lole32", "-lmfplat", "-lmfreadwrite", "-lshell32",
            "-lmfuuid"] if os.name == "nt" else []
    if wgpu:
        libs.append(os.path.join(wgpu, "lib", "wgpu_native.dll.lib"))
        dll, dst = os.path.join(wgpu, "lib", "wgpu_native.dll"), os.path.join(nat, "wgpu_native.dll")
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(dll):   # (in use while the game runs)
            shutil.copyfile(dll, dst)
    # SDL2 (zlib license, third_party/sdl2): game controllers beyond XInput (DualSense, DualShock 4, Switch Pro...),
    # loaded at run time by hos/pad_sdl.c; the game runs without it (XInput, keyboard and mouse only)
    sdl = os.path.join(HERE, "..", "third_party", "sdl2", "SDL2.dll")
    if os.name == "nt" and os.path.exists(sdl):
        dst = os.path.join(nat, "SDL2.dll")
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(sdl):
            shutil.copyfile(sdl, dst)
    # RX_LINK_ORDER=<file>: lifted functions laid out in that order first (hos_prof.c's HOS_PROFILE_ORDER writes one:
    # the sampled functions, hottest first), so the hot code shares pages and cache lines. Default: the port's
    # link_order.txt when there is one (TOTK: castle profile, 2966 functions; game threads -1.4% CPU a frame, 3 rounds,
    # 2026-10-22); RX_LINK_ORDER=none: the compiler's order
    order = os.environ.get("RX_LINK_ORDER")
    if order is None:
        port_order = os.path.join(HERE, "..", "ports", "zelda-" + a.name, "link_order.txt")
        order = port_order if os.path.isfile(port_order) else None
    if order == "none":
        order = None
    order_flags = ["-Wl,/ORDER:@" + os.path.abspath(order).replace("\\", "/"), "-Wl,/IGNORE:4037"] if order else []
    if lifted_lto:   # (the lifted code's optimization runs at the link now: its LoopRotate cap too; a cache for relinks)
        order_flags += ["-Wl,/mllvm:-rotation-max-header-size=0", "-Wl,/opt:lldltojobs=8",   # (~3 GB a big module)
                        "-Wl,/lldltocache:" + os.path.join(nat, "ltocache").replace("\\", "/")]
        if a.opt == "O3":   # (the link's optimization of the lifted code at that level too)
            order_flags.append("-Wl,/opt:lldlto=3")
        if capture_flag:
            order_flags.append("-Wl,/mllvm:" + capture_flag)
        # RX_LTO_LINK_FLAGS="<flags>" (experiments): more link flags, e.g. -Wl,/mllvm:-import-instr-limit=200
        order_flags += os.environ.get("RX_LTO_LINK_FLAGS", "").split()
    subprocess.run([cc] + toolchain.native_target() + toolchain.native_link() +
                   ["-g", "-o", exe, "@" + rsp, "-Wl,/STACK:268435456", "-Wl,/SUBSYSTEM:CONSOLE"] + order_flags + libs,
                   check=True)
    print("linked %s (%.0f MB) in %.0fs" % (exe, os.path.getsize(exe) / 2 ** 20, time.time() - t0))


def cmd_web(a):
    """The browser build (build/<name>/web): lifted code + rt/ + hos/ (hos/web/ stands in for Windows) + the renderer
    on the browser's WebGPU (emdawnwebgpu), compiled by emcc into one wasm with pthreads. Guest memory is wasm
    memory 0..4 GB (A64_MEM_ZERO): the host's own data is linked above the guest window (-sGLOBAL_BASE)."""
    root, img_dir, gen, _ = dirs(a)
    web = os.path.join(root, "web")
    obj_dir = os.path.join(web, "obj")
    os.makedirs(obj_dir, exist_ok=True)
    emcc = toolchain.need(toolchain.find("emcc"))
    empp = toolchain.need(toolchain.find("em++"))
    # (-msse2: Emscripten's SSE2 headers on wasm SIMD; the lifted float helpers take their fast path with them)
    # (A64_WRITE_TRACK: guest stores mark their pages: the renderer's write watch, rt/a64.h)
    common = ["-pthread", "-mtail-call", "-msimd128", "-msse2", "-DA64_MEM_ZERO", "-DA64_WRITE_TRACK", "-fno-delete-null-pointer-checks",
              "-fno-strict-aliasing", "-Wno-everything", "-I" + gen, "-I" + RT]
    jobs = []
    # RX_WEB_MT=1: the threaded web build (gpu/render/rx_web_mt.h: a WebGPU thread replays what the renderer's thread
    # records), totk_mt.js + index_mt.html beside the other: its runtime and renderer objects are its own (mt_*), the
    # lifted code's are shared (a full lifted compile for the web takes hours)
    mt = os.environ.get("RX_WEB_MT") == "1"
    mt_flags, mt_pre = (["-DRX_WEB_MT=1"], "mt_") if mt else ([], "")
    apply_entry_hooks(gen, img_dir)
    apply_splices(gen)
    deps_lifted = headers_key(RT, gen)
    lifted = sorted(os.path.join(gen, f) for f in os.listdir(gen) if f.endswith(".c"))
    fpx = ["-DA64_FPX_FN=static inline __attribute__((always_inline))"] if os.environ.get("RX_FPX_INLINE") else []
    capture_flag = capture_tracking_flag()   # (as in cmd_native: on wasm the hot castle function's context stores -7%)
    for src in lifted:
        flags = common + ["-" + a.opt, "-g0", "-mllvm", "-rotation-max-header-size=0"] + \
            (["-mllvm", capture_flag] if capture_flag else []) + fpx
        jobs.append((emcc, src, os.path.join(obj_dir, os.path.basename(src)[:-2] + ".o"), flags, deps_lifted))
    if not a.lifted_only:
        web_hos = os.path.join(HOS, "web")
        runtime = [os.path.join(RT, f) for f in os.listdir(RT) if f.endswith(".c")] +                   [os.path.join(HOS, f) for f in os.listdir(HOS) if f.endswith(".c") and f not in WEB_SKIP] +                   [os.path.join(web_hos, f) for f in os.listdir(web_hos) if f.endswith(".c")]
        deps_runtime = headers_key(RT, HOS, web_hos, gen, os.path.join(GPU, "render"))
        opus = os.path.join(HERE, "..", "third_party", "opus")
        rflags = common + mt_flags + ["-I" + web_hos, "-I" + HOS, "-I" + os.path.join(opus, "include"), "-O2", "-g"]
        for src in runtime:
            name = mt_pre + ("web_" if os.path.dirname(src) == web_hos else "") + os.path.basename(src)[:-2]
            jobs.append((emcc, src, os.path.join(obj_dir, name + ".o"), rflags, deps_runtime))
        oflags = ["-pthread", "-msimd128", "-O2", "-g0", "-Wno-everything", "-DOPUS_BUILD", "-DUSE_ALLOCA",
                  "-DHAVE_LRINTF", "-DHAVE_LRINT"] + ["-I" + os.path.join(opus, d) for d in
                                                       ("include", "celt", "silk", "silk/float", "src")]
        okey = hashlib.sha1(b"opus-1.5.2" + " ".join(oflags).encode()).hexdigest()
        for rel in open(os.path.join(opus, "sources.txt")).read().split():
            jobs.append((emcc, os.path.join(opus, rel), os.path.join(obj_dir, "opus_" + rel.replace("/", "_")[:-2] + ".o"),
                         oflags, okey))
        # (no -mtail-call for the renderer and suyu: LLVM's wasm tail calls miscompiled suyu's IR emitter, a return_call
        # whose callee read the caller's freed stack temporaries: shader immediates became other operands, 2026-10-15)
        # (the renderer and suyu as LLVM bitcode, optimized across files at link, as the native build's ThinLTO: the
        # render thread 10% faster, 22.6 -> 25.0 fps at Lookout Landing, the link ~2 min longer; RX_WEB_LTO=0: off)
        web_lto = os.environ.get("RX_WEB_LTO", "1") != "0"
        lto = ["-flto"] if web_lto else []
        gflags = [f for f in common if f != "-mtail-call"] + mt_flags + lto + ["-I" + web_hos, "-std=c++20", "-O2", "-g", "-DNDEBUG", "--use-port=emdawnwebgpu",
                           "-I" + os.path.join(GPU, "compat"), "-I" + os.path.join(GPU, "third_party", "suyu")]
        gdeps = tree_headers_key(GPU) + headers_key(web_hos)
        for src in gpu_sources():
            name = os.path.relpath(src, GPU).replace(os.sep, "_")[:-4]
            jobs.append((empp, src, os.path.join(obj_dir, mt_pre + "gpu_" + name + ".o"), gflags, gdeps))
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    t0 = time.time()
    failed = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for i, (src, dt, err) in enumerate(ex.map(compile_one, jobs), 1):
            if err:
                failed += 1
                print("FAILED %s\n%s" % (os.path.basename(src), err))
            elif dt:
                print("[%d/%d] %s %.0fs" % (i, len(jobs), os.path.basename(src), dt), flush=True)
    if failed:
        sys.exit("%d files failed to compile" % failed)
    if a.lifted_only:
        print("lifted code compiled in %.0fs" % (time.time() - t0))
        return
    rsp = os.path.join(web, "link_mt.rsp" if mt else "link.rsp")
    open(rsp, "w").write("\n".join('"%s"' % j[2].replace("\\", "/") for j in jobs))
    out = os.path.join(web, "totk_mt.js" if mt else "totk.js")
    guest_mb = a.guest_mb
    link = [empp, "@" + rsp, "-o", out, "-pthread", "-mtail-call", "-msimd128", "--use-port=emdawnwebgpu",
            # (all of wasm32's memory up front, no growth: with threads, growing leaves JS views of the heap stale)
            "-sGLOBAL_BASE=%d" % (guest_mb << 20), "-sINITIAL_MEMORY=4032MB", "-sALLOW_MEMORY_GROWTH=0",
            # (dlmalloc: mimalloc's per-thread arenas ran the host's ~1.2 GB out at boot)
            "-sSTACK_SIZE=16MB", "-sDEFAULT_PTHREAD_STACK_SIZE=2MB",
            "-sPROXY_TO_PTHREAD=1", "-sPTHREAD_POOL_SIZE=64", "-sEXIT_RUNTIME=0", "-sMALLOC=dlmalloc",
            "-sENVIRONMENT=web,worker", "-sEXPORTED_RUNTIME_METHODS=['FS','HEAPU8','HEAP32','ENV','addRunDependency','removeRunDependency']", "-sFORCE_FILESYSTEM=1",
            "-sJSPI=1", "-sOFFSCREENCANVAS_SUPPORT=1", "--js-library", os.path.join(HOS, "web", "library.js"), "-g0",
            # (RX_WEB_FASTLINK=1: no Binaryen pass over the module, ~5 min of a link)
            "-O0" if os.environ.get("RX_WEB_FASTLINK") else "-O2", "--profiling-funcs"]
    link += os.environ.get("RX_WEB_LINKFLAGS", "").split()   # (debugging: e.g. -sASSERTIONS=1)
    if os.environ.get("RX_WEB_LTO", "1") != "0":
        link.append("-flto")
    subprocess.run(link, check=True)
    if mt:   # (the same page, loading the threaded build)
        page = open(os.path.join(HOS, "web", "index.html"), encoding="utf-8").read()
        open(os.path.join(web, "index_mt.html"), "w", encoding="utf-8", newline="\n").write(
            page.replace('<script src="totk.js"></script>', '<script src="totk_mt.js"></script>'))
    else:
        shutil.copyfile(os.path.join(HOS, "web", "index.html"), os.path.join(web, "index.html"))
    print("linked %s (%.0f MB wasm) in %.0fs" % (out, os.path.getsize(out[:-3] + ".wasm") / 2 ** 20, time.time() - t0))


# hos files with no place in the browser (hos/web/ has their stand-ins)
WEB_SKIP = {"hos_prof.c", "movie_mf.c", "display.c", "pad_sdl.c"}


def cmd_stubs(a):
    """Regenerate gen/a64_stubs.c after adding HLE functions to hos/: hooks are named imp_<symbol> whether they are
    an HLE or a stub, so nothing else changes (no re-lift). A symbol that was never hooked needs a re-lift."""
    root, img_dir, gen, _ = dirs(a)
    meta = json.load(open(os.path.join(img_dir, "image.json")))
    hooked = {h["c"] for h in meta["hooks"]}
    missing = sorted(n for n in implemented() if n not in hooked)
    n = gen_stubs(gen, meta)
    print("%d stubs left; %d HLE functions in hos/" % (n, len(implemented())))
    if missing:
        print("not hooked in this lift (re-lift to use them):\n  " + "\n  ".join(missing))


def cmd_run(a):
    root, img_dir, gen, nat = dirs(a)
    exe = os.path.join(nat, a.name + ".exe")
    args = [exe, os.path.join(img_dir, "image.bin"), "--user", os.path.join(root, "user")]
    if a.romfs:
        args += ["--romfs", a.romfs]
    sys.exit(subprocess.call(args + [r for r in a.rest if r != "--"]))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    for name in ("lift", "native", "web", "stubs", "run"):
        s = sp.add_parser(name)
        s.add_argument("name")
        s.add_argument("--build")
        if name == "lift":
            s.add_argument("--exefs", required=True)
            s.add_argument("--skip", default="")
            s.add_argument("--hle")
            s.add_argument("--comments", action="store_true")
            s.add_argument("--mods", help="a mods folder (hos/mods.c): its exefs patches are written into the image "
                           "before lifting (code patches only work compiled in)")
            s.add_argument("--hot", help="a profile's hot functions, one name a line (default: ports/zelda-<name>/"
                                         "hot_funcs.txt when it exists; 'none' for none): they get the FP fast paths "
                                         "inline (a64_lift.py HOT_FPX)")
            s.add_argument("--max-func", type=int, default=0,
                           help="split functions longer than this many instructions where no branch crosses the cut "
                                "(0: never; on TOTK it finds only 60 cuts, its giant functions are loops end to end)")
        if name == "web":
            s.add_argument("--opt", default="O2")
            s.add_argument("-j", "--jobs", type=int, default=_safe_job_count())
            s.add_argument("--lifted-only", action="store_true", help="compile the lifted code and stop")
            s.add_argument("--guest-mb", type=int, default=3456, help="the guest window (wasm memory below the host's; hos_mem.c sizes the game's heap to leave its regions room)")
        if name == "native":
            # (O2: the game's own threads are the frame rate's limit in the open world; rain 58.8 -> 60 fps against O1,
            # a full build of the lifted code ~12 min on 12 threads. O3, with the ThinLTO link at level 3 too: castle
            # 66.7 -> 69.7 fps, ahead in each of 3 interleaved rounds, 2026-10-21; exe +11 MB)
            s.add_argument("--opt", default="O3", help="optimization of the lifted code (O1 builds faster)")
            s.add_argument("--debug", action="store_true", help="debug info for the lifted code too")
            s.add_argument("-j", "--jobs", type=int, default=_safe_job_count())
            s.add_argument("--wgpu", help="a wgpu-native release (include/, lib/): builds the WebGPU renderer")
            s.add_argument("-o", "--output", help="the executable's name (default: the game's; e.g. to link while it runs)")
        if name == "run":
            s.add_argument("--romfs", help="the game's extracted RomFS (tools/nx.py romfs)")
            s.add_argument("rest", nargs=argparse.REMAINDER)
    a = p.parse_args()
    {"lift": cmd_lift, "native": cmd_native, "web": cmd_web, "stubs": cmd_stubs, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()

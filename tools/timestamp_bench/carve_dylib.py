#!/usr/bin/env python3
# Carve a dylib from the split dyld cache into a sparse composite file:
# each segment placed at (vmaddr - text_base) so r2 sees correct addresses.
import struct, re, os, sys

D = "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld"
SUBCACHE = os.path.join(D, "dyld_shared_cache_arm64e.30")

def file_maps(path):
    data = open(path, "rb").read(0x4000)
    mo, mc = struct.unpack_from("<II", data, 0x10)
    out = []
    for i in range(mc):
        addr, size, foff, mp, ip = struct.unpack_from("<QQQII", data, mo + i * 32)
        out.append((addr, size, foff))
    return out

# build global vmaddr -> (file, fileoff) resolver over all cache files
caches = [os.path.join(D, f) for f in sorted(os.listdir(D))
          if re.fullmatch(r"dyld_shared_cache_arm64e(\.\d+(\.dyld(data|readonly|linkedit))?)?", f)]
allmaps = []  # (vmaddr, size, path, fileoff)
for c in caches:
    for addr, size, foff in file_maps(c):
        allmaps.append((addr, size, c, foff))

def resolve(vmaddr, length):
    for addr, size, path, foff in allmaps:
        if addr <= vmaddr and vmaddr + length <= addr + size:
            return path, foff + (vmaddr - addr)
    return None, None

def main():
    hdr_off = int(sys.argv[1], 16)  # mach header file offset in SUBCACHE
    out_path = sys.argv[2]
    data30 = open(SUBCACHE, "rb").read()
    magic, cpu, sub, filetype, ncmds, sizeofcmds, flags, reserved = struct.unpack_from("<8I", data30, hdr_off)
    off = hdr_off + 32
    segs = []
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data30, off)
        if cmd == 0x19:
            segname = data30[off + 8:off + 24].rstrip(b"\0").decode()
            vmaddr, vmsize, fileoff, filesize = struct.unpack_from("<QQQQ", data30, off + 24)
            segs.append((segname, vmaddr, vmsize, filesize))
        off += cmdsize
    text_base = [s[1] for s in segs if s[0] == "__TEXT"][0]
    print(f"__TEXT base {text_base:#x}")
    with open(out_path, "wb") as out:
        for name, vmaddr, vmsize, filesize in segs:
            if name == "__LINKEDIT":
                print(f"  skip {name} ({vmsize:#x})")
                continue
            path, foff = resolve(vmaddr, filesize)
            if path is None:
                print(f"  UNRESOLVED {name} vmaddr={vmaddr:#x}")
                continue
            with open(path, "rb") as f:
                f.seek(foff)
                blob = f.read(filesize)
            outoff = vmaddr - text_base
            out.seek(outoff)
            out.write(blob)
            print(f"  {name:16s} vmaddr={vmaddr:#x} -> out+{outoff:#x} ({filesize:#x} bytes from {os.path.basename(path)}+{foff:#x})")
    print(f"wrote {out_path}")

main()

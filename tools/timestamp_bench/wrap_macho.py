#!/usr/bin/env python3
# Wrap raw arm64 code bytes in a minimal Mach-O object so llvm-objdump can disassemble it.
# Usage: wrap_macho.py <in.bin> <out.o> <vmaddr_hex>
import struct, sys

src, dst, vma = sys.argv[1], sys.argv[2], int(sys.argv[3], 16)
code = open(src, "rb").read()

MH_MAGIC_64 = 0xFEEDFACF
CPU_TYPE_ARM64 = 0x0100000C
CPU_SUBTYPE_ARM64E = 2 | 0x80000000  # ABI64 | ptrauth
LC_SEGMENT_64 = 0x19
S_ATTR_PURE_INSTRUCTIONS = 0x80000000
S_ATTR_SOME_INSTRUCTIONS = 0x00000400

# header (32 bytes) + one LC_SEGMENT_64 (72 + 80 for one section)
segname = b"__text".ljust(16, b"\0")
sectname = b"__text".ljust(16, b"\0")

sizeofcmds = 72 + 80
header = struct.pack(
    "<IIIIIIII",
    MH_MAGIC_64, CPU_TYPE_ARM64, CPU_SUBTYPE_ARM64E,
    0x1,  # MH_OBJECT
    1,    # ncmds
    sizeofcmds,
    0,    # flags
    0,    # reserved
)
segcmd = struct.pack(
    "<II16sQQQQiiII",
    LC_SEGMENT_64, 72 + 80,
    b"__TEXT".ljust(16, b"\0"),
    vma,            # vmaddr
    len(code),      # vmsize
    32 + sizeofcmds,  # fileoff
    len(code),      # filesize
    5, 5,           # maxprot, initprot (r-x)
    1,              # nsects
    0,              # flags
)
section = struct.pack(
    "<16s16sQQIIIIIIII",
    sectname, b"__TEXT".ljust(16, b"\0"),
    vma, len(code),
    32 + sizeofcmds,  # offset
    2,                # align (2^2)
    0, 0,             # reloff, nreloc
    S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
    0, 0, 0,
)
with open(dst, "wb") as f:
    f.write(header + segcmd + section + code)
print(f"wrote {dst} ({len(code)} bytes @ {vma:#x})")

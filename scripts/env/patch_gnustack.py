"""Clear the PF_X (executable) bit from PT_GNU_STACK on torch/torchvision .so files.

torch/torchvision 1.12/0.13 wheels ship ``libtorch*.so`` with a ``GNU_STACK RWE``
program header.  Modern hardened containers reject ``dlopen`` of such libraries
with ``cannot enable executable stack as shared object requires``.  These libs
do not actually need an executable stack (old-binutils artifact), so clearing
PF_X makes them loadable without ``execstack``/``patchelf``.

Run it *inside* the target virtualenv so ``sysconfig`` resolves the right
site-packages:

    .venv-real/bin/python scripts/env/patch_gnustack.py
"""
import glob
import struct
import sysconfig


def _so_patterns():
    sp = sysconfig.get_paths()["purelib"]
    return [
        f"{sp}/torch/lib/*.so",
        f"{sp}/torchvision/lib/*.so",
        f"{sp}/torchvision/*.so",
    ]


def patch(path):
    with open(path, "rb") as f:
        data = bytearray(f.read())
    if data[:4] != b"\x7fELF" or data[4] != 2:  # ELF64 only
        return False
    e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum = struct.unpack_from("<H", data, 0x38)[0]
    changed = False
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from("<I", data, off)[0]
        if p_type == 0x6474E551:  # PT_GNU_STACK
            p_flags = struct.unpack_from("<I", data, off + 4)[0]
            if p_flags & 1:
                struct.pack_into("<I", data, off + 4, p_flags & ~1)
                changed = True
    if changed:
        with open(path, "wb") as f:
            f.write(data)
    return changed


if __name__ == "__main__":
    n = 0
    for pattern in _so_patterns():
        for so in sorted(glob.glob(pattern)):
            if patch(so):
                print("patched", so)
                n += 1
    print(f"{n} file(s) patched")

#!/usr/bin/env python3
"""Generate fixed OSD HUD text assets in SmartSens .ssbmp format.

The OSD SDK documents SSBMP as:
  magic "SSBM", uint32 width, uint32 height, uint32 colorNum, uint8 index data.
Index 31 is transparent. Index 0 is used as inverse/high-contrast foreground.

The board-side libosd loader compares the first 4 bytes as a little-endian
uint32 value 0x5353424D, so the physical file bytes must be "MBSS".
"""

from __future__ import annotations

import struct
from pathlib import Path


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "W": ["10001", "10001", "10001", "10001", "10101", "11011", "10001"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def render_text(text: str, scale: int, pad_x: int, pad_y: int) -> tuple[int, int, bytearray]:
    chars = list(text)
    glyph_w = 5
    glyph_h = 7
    gap = 1
    width_cells = sum(glyph_w if c != " " else 3 for c in chars) + gap * max(0, len(chars) - 1)
    width = pad_x * 2 + width_cells * scale
    height = pad_y * 2 + glyph_h * scale
    bg = 31
    fg = 0
    data = bytearray([bg] * (width * height))

    x_cell = 0
    for ch in chars:
        if ch == " ":
            x_cell += 3 + gap
            continue
        pattern = FONT.get(ch.upper(), FONT["_"])
        for gy, row in enumerate(pattern):
            for gx, bit in enumerate(row):
                if bit != "1":
                    continue
                x0 = pad_x + (x_cell + gx) * scale
                y0 = pad_y + gy * scale
                for yy in range(y0, y0 + scale):
                    base = yy * width
                    for xx in range(x0, x0 + scale):
                        data[base + xx] = fg
        x_cell += glyph_w + gap
    return width, height, data


def write_ssbmp(path: Path, text: str, scale: int, pad_x: int, pad_y: int) -> None:
    width, height, data = render_text(text, scale, pad_x, pad_y)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<I", 0x5353424D))
        f.write(struct.pack("<III", width, height, 32))
        f.write(data)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "app_assets" / "osd"
    for word in ["STOP", "SLOW", "CLEAR", "LEFT", "RIGHT"]:
        write_ssbmp(root / f"{word}.ssbmp", word, scale=10, pad_x=8, pad_y=6)

    dirs = ["L", "C", "R", "WIDE", "LC", "CR"]
    risks = ["NEAR", "WARN", "FAR", "UNK"]
    for d in dirs:
        for r in risks:
            write_ssbmp(root / f"{d}_{r}.ssbmp", f"{d} {r}", scale=7, pad_x=6, pad_y=5)


if __name__ == "__main__":
    main()

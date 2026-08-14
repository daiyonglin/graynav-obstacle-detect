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
import argparse
from pathlib import Path


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10001", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
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


def write_multiline_ssbmp(
    path: Path,
    lines: list[str],
    scale: int = 5,
    pad_x: int = 7,
    pad_y: int = 5,
    line_gap: int = 5,
) -> None:
    rendered = [render_text(line, scale, pad_x, pad_y) for line in lines]
    width = max(item[0] for item in rendered)
    height = sum(item[1] for item in rendered) + line_gap * (len(rendered) - 1)
    data = bytearray([31] * (width * height))
    y_offset = 0
    for line_width, line_height, line_data in rendered:
        x_offset = (width - line_width) // 2
        for y in range(line_height):
            src = y * line_width
            dst = (y_offset + y) * width + x_offset
            data[dst:dst + line_width] = line_data[src:src + line_width]
        y_offset += line_height + line_gap
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<I", 0x5353424D))
        handle.write(struct.pack("<III", width, height, 32))
        handle.write(data)


def generate_info_assets(root: Path) -> int:
    labels = [
        "PERSON", "CHAIR", "TABLE", "BAG", "COUCH", "BENCH",
        "STAIR", "STEP_CHECK", "BLOCKED", "PATH", "UNKNOWN", "AI_FAIL",
    ]
    ranges = ["NEAR", "MID", "FAR", "UNKNOWN"]
    directions = ["LEFT", "FRONT", "RIGHT"]
    count = 0
    for label in labels:
        first_line = "STEP CHECK" if label == "STEP_CHECK" else label.replace("_", " ")
        for range_name in ranges:
            for direction in directions:
                path = root / f"INFO_{label}_{range_name}_{direction}.ssbmp"
                write_multiline_ssbmp(path, [first_line, f"{range_name} {direction}"])
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only generate the new two-line INFO_ assets; preserve legacy files.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "app_assets" / "osd"
    if args.info_only:
        count = generate_info_assets(root)
        print(f"generated_info_assets={count}")
        return
    for word in ["STOP", "SLOW", "CLEAR", "LEFT", "RIGHT"]:
        write_ssbmp(root / f"{word}.ssbmp", word, scale=10, pad_x=8, pad_y=6)

    for word in [
        "PERSON", "CHAIR", "TABLE", "BAG", "COUCH", "BENCH",
        "STAIR", "BLOCKED", "PATH", "UNKNOWN", "AI_FAIL", "ITEM",
    ]:
        write_ssbmp(root / f"{word}.ssbmp", word, scale=7, pad_x=6, pad_y=5)

    dirs = ["L", "C", "R", "WIDE", "LC", "CR"]
    risks = ["NEAR", "WARN", "FAR", "UNK"]
    for d in dirs:
        for r in risks:
            write_ssbmp(root / f"{d}_{r}.ssbmp", f"{d} {r}", scale=7, pad_x=6, pad_y=5)
    count = generate_info_assets(root)
    print(f"generated_info_assets={count}")


if __name__ == "__main__":
    main()

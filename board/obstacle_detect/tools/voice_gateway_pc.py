#!/usr/bin/env python3
"""
PC-side voice gateway for the obstacle detector JSON Lines stream.

Current intended use is offline replay:
    python tools/voice_gateway_pc.py --jsonl runs/offline_obstacle_detect/detections.jsonl --dry-run

Later, after the Aurora preview path is stable, install pyserial and read the
board COM port:
    python tools/voice_gateway_pc.py --port COM7 --baud 115200
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class SpeechState:
    last_key: str = ""
    last_time: float = 0.0
    last_action: str = "clear"
    stable_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, help="Replay JSONL file")
    parser.add_argument("--port", help="Serial COM port, for future realtime test")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without speaking")
    parser.add_argument("--cooldown", type=float, default=4.0)
    parser.add_argument("--stable", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_serial(port: str, baud: int) -> Iterable[dict]:
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyserial is required for --port mode: pip install pyserial") from exc

    with serial.Serial(port, baudrate=baud, timeout=1.0) as ser:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw.startswith("{"):
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def make_prompt(packet: dict) -> tuple[str, str]:
    nav = packet.get("nav") or {}
    action = str(nav.get("action", "clear"))
    nearest = packet.get("nearest") or {}
    sector = str(nearest.get("dir") or nav.get("sector") or "center")
    semantic = str(nearest.get("semantic_class") or nearest.get("label") or "obstacle")
    risk = str(nearest.get("risk") or "unknown")
    dist = nearest.get("dist_m", -1)

    if action == "stop":
        text = f"Stop. {semantic} ahead."
    elif action == "turn_left":
        text = f"Go left. {semantic} on {sector}."
    elif action == "turn_right":
        text = f"Go right. {semantic} on {sector}."
    elif action == "slow":
        text = f"Slow. {semantic} {risk}."
    else:
        text = "Clear."

    if isinstance(dist, (int, float)) and dist >= 0:
        text += f" {dist:.1f} meters."
    return action, text


def should_speak(state: SpeechState, action: str, text: str, cooldown: float, stable_needed: int) -> bool:
    now = time.monotonic()
    key = f"{action}:{text}"

    if action == state.last_action:
        state.stable_count += 1
    else:
        state.last_action = action
        state.stable_count = 1

    if action == "stop":
        if key != state.last_key or now - state.last_time >= 1.0:
            state.last_key = key
            state.last_time = now
            return True
        return False

    if action == "clear":
        stable_gate = max(stable_needed * 3, 8)
    else:
        stable_gate = stable_needed

    if state.stable_count < stable_gate:
        return False
    if key == state.last_key and now - state.last_time < cooldown:
        return False

    state.last_key = key
    state.last_time = now
    return True


def speak(text: str, dry_run: bool) -> None:
    if dry_run:
        print(text)
        return
    try:
        import pyttsx3  # type: ignore
    except ImportError:
        print(text)
        return
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main() -> None:
    args = parse_args()
    if not args.jsonl and not args.port:
        raise SystemExit("Use --jsonl for offline replay or --port for future realtime COM input.")

    packets = read_jsonl(args.jsonl) if args.jsonl else read_serial(args.port, args.baud)
    state = SpeechState()
    for packet in packets:
        action, text = make_prompt(packet)
        if should_speak(state, action, text, args.cooldown, args.stable):
            speak(text, args.dry_run)
        if args.jsonl:
            time.sleep(0.05)


if __name__ == "__main__":
    main()


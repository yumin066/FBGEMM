"""Latency-hint sweep for the D256 cuTile FP8 prototype."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Iterable


_LATENCY_ENV = {
    "q": "HSTU_CUTILE_FP8_Q_LATENCY",
    "k": "HSTU_CUTILE_FP8_K_LATENCY",
    "v": "HSTU_CUTILE_FP8_V_LATENCY",
    "scale": "HSTU_CUTILE_FP8_SCALE_LATENCY",
    "o": "HSTU_CUTILE_FP8_O_LATENCY",
}


def _parse_latency_values(text: str) -> list[int | None]:
    values: list[int | None] = []
    for item in text.split(","):
        token = item.strip().lower()
        if token in ("", "auto", "none"):
            values.append(None)
        else:
            values.append(int(token))
    return values


def _format_latency(value: int | None) -> str:
    return "auto" if value is None else str(value)


def _latency_label(cfg: dict[str, int | None]) -> str:
    return (
        f"q={_format_latency(cfg['q'])},"
        f"k={_format_latency(cfg['k'])},"
        f"v={_format_latency(cfg['v'])},"
        f"s={_format_latency(cfg['scale'])},"
        f"o={_format_latency(cfg['o'])}"
    )


def _dedupe(configs: Iterable[dict[str, int | None]]) -> list[dict[str, int | None]]:
    seen: set[tuple[int | None, ...]] = set()
    unique: list[dict[str, int | None]] = []
    for cfg in configs:
        key = (cfg["q"], cfg["k"], cfg["v"], cfg["scale"], cfg["o"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(cfg)
    return unique


def _grouped_configs(
    q_values: list[int | None],
    kv_values: list[int | None],
    scale_values: list[int | None],
    o_values: list[int | None],
) -> list[dict[str, int | None]]:
    auto = {"q": None, "k": None, "v": None, "scale": None, "o": None}
    configs: list[dict[str, int | None]] = [auto]
    configs.extend({**auto, "q": q} for q in q_values)
    configs.extend({**auto, "k": kv, "v": kv} for kv in kv_values)
    configs.extend({**auto, "scale": scale} for scale in scale_values)
    configs.extend({**auto, "o": o} for o in o_values)
    return _dedupe(configs)


def _grid_configs(
    q_values: list[int | None],
    kv_values: list[int | None],
    scale_values: list[int | None],
    o_values: list[int | None],
) -> list[dict[str, int | None]]:
    configs = []
    for q, kv, scale, o in product(q_values, kv_values, scale_values, o_values):
        configs.append({"q": q, "k": kv, "v": kv, "scale": scale, "o": o})
    return _dedupe(configs)


def _apply_latency_env(env: dict[str, str], cfg: dict[str, int | None]) -> None:
    for name, env_name in _LATENCY_ENV.items():
        value = cfg[name]
        if value is None:
            env.pop(env_name, None)
        else:
            env[env_name] = str(value)


def _parse_result(line: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^ ]+)", line):
        parsed[key] = value
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--masks", nargs="+", choices=("full", "causal"), default=["full", "causal"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mode", choices=("grouped", "grid"), default="grouped")
    parser.add_argument("--q-values", default="auto,2,4,6")
    parser.add_argument("--kv-values", default="auto,3,5,7")
    parser.add_argument("--scale-values", default="auto,2,3,4")
    parser.add_argument("--o-values", default="auto,1,2,3")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    q_values = _parse_latency_values(args.q_values)
    kv_values = _parse_latency_values(args.kv_values)
    scale_values = _parse_latency_values(args.scale_values)
    o_values = _parse_latency_values(args.o_values)
    if args.mode == "grid":
        configs = _grid_configs(q_values, kv_values, scale_values, o_values)
    else:
        configs = _grouped_configs(q_values, kv_values, scale_values, o_values)

    rows: list[dict[str, str]] = []
    print(
        "LATENCY_SWEEP "
        f"mode={args.mode} configs={len(configs)} "
        f"batch={args.batch} seqlen={args.seqlen} heads={args.heads} "
        f"masks={','.join(args.masks)}"
    )
    for cfg in configs:
        label = _latency_label(cfg)
        for mask in args.masks:
            env = os.environ.copy()
            _apply_latency_env(env, cfg)
            cmd = [
                sys.executable,
                "-m",
                "hstu_blackwell_sm120_cudatile_fp8_ws.sweep_d256",
                "--batch",
                str(args.batch),
                "--seqlen",
                str(args.seqlen),
                "--heads",
                str(args.heads),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
            ]
            if mask == "causal":
                cmd.append("--causal")

            print(f"=== {label} mask={mask} ===")
            proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)

            result_line = ""
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT "):
                    result_line = line
            row = {
                "status": str(proc.returncode),
                "mask": mask,
                "latency_q": _format_latency(cfg["q"]),
                "latency_k": _format_latency(cfg["k"]),
                "latency_v": _format_latency(cfg["v"]),
                "latency_scale": _format_latency(cfg["scale"]),
                "latency_o": _format_latency(cfg["o"]),
            }
            if result_line:
                row.update(_parse_result(result_line))
            rows.append(row)

    if args.csv:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

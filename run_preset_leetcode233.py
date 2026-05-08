"""
One-shot generation for LeetCode 233 with a chosen preset.
Streams output to stdout, saves full log + extracted ```cpp``` block.

Usage:
  python run_preset_leetcode233.py [preset_name] [global_alpha] [--think|--no-think]

  preset_name can be:
    - a named preset from chat/neurons.json (e.g. math_anchor_v3)
    - a path to a YAML file (e.g. math_anchor_v4.yaml)

Default: math_anchor_v2  alpha=1.0  --think
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "chat"))
sys.stdout.reconfigure(encoding="utf-8")

import yaml  # new dependency

from runtime import ClampChatRuntime  # noqa: E402

CONFIG_PATH = ROOT / "chat" / "neurons.json"
OUT_DIR = ROOT / "outputs" / "preset_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PROMPT = """这是问题"233. Number of Digit One

# Hard

# Topics

# premium lock icon

# Companies

# Hint

# Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.



# Example 1:

# Input: n = 13

# Output: 6

# Example 2:

# Input: n = 0

# Output: 0



# Constraints:

# 0 <= n <= 109



# "



# 这是给定的class模板

# 'class Solution {

# public:

#     int countDigitOne(int n) {



#     }

# };'



# 根据模板和问题给出一个可用的c++代码."""


PROMPT = """**题目背景：**
LeetCode 233. Number of Digit One (Hard)
给定一个整数 $n$，计算所有小于等于 $n$ 的非负整数中数字 1 出现的总次数。
数据范围：$0 \le n \le 10^9$。

**核心挑战：**
由于 $n$ 高达 $10^9$，传统的暴力遍历（枚举每个数字并统计 1）的时间复杂度为 $O(n \log n)$，这在 $10^9$ 的量级下必然超时。我们需要一个 $O(\log_{10} n)$ 的算法。

**思考引导：**
请不要尝试逐个检查数字，而是尝试**计算每一位（个位、十位、百位...）上 1 出现的频率贡献**。请遵循以下思考路径：
1. **位置独立性**：假设我们要计算百位上出现 1 的次数。这个次数取决于百位左边的数字（高位）、百位本身的数字、以及百位右边的数字（低位）。
2. **分类讨论**：请深入思考并推导：当“当前位”的数字分别是 **0**、**1**、或 **大于 1** 时，该位置出现 1 的次数分别受高位和低位怎样的影响？
3. **周期规律**：观察每一位 1 出现的周期性（例如，个位每 10 个数出现一次 1，十位每 100 个数连续出现 10 次 1）。

**任务要求：**
1. 首先，请在内部逻辑中推导出每一位出现 1 的数学公式。
2. 随后，请直接输出一个完整的 C++ `class Solution {}` 代码块。
3. **实现细节要求**：
   - 使用 `long long` 处理计数和位因子（factor），以防止 $10^9$ 带来的潜在溢出。
   - 使用一个循环从个位开始向高位处理，循环条件建议为 `factor <= n`。
   - 只输出代码，不输出冗长的解释。"""




def load_preset_from_yaml(yaml_path: Path, runtime):
    """Load a preset definition from a standalone YAML file.

    Expected YAML structure:
        preset_name: "math_anchor_v4"
        enabled:
          - id: "L20#2810"
            gain: -2.0
            region: "thought"
          - id: "L23#1867"
            gain: -1.5
            region: "thought"
          ...
        overrides: {}        # optional, global overrides for gain
        think_mode: true     # optional, default true
        temperature: 0.0     # optional
        max_new_tokens: 8192 # optional

    If a neuron's 'id' contains '#' (e.g. 'L20#2810'), it is parsed directly.
    Otherwise we fall back to runtime.known_neurons lookup by id.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    preset_name = data.get("preset_name", yaml_path.stem)
    enabled_list = data.get("enabled", [])
    overrides = data.get("overrides", {})
    think_mode = data.get("think_mode", True)
    temperature = float(data.get("temperature", 0.0))
    max_new_tokens = int(data.get("max_new_tokens", 8192))

    # Build lookup from id -> known neuron entry
    known_by_id = {n["id"]: n for n in runtime.known_neurons}

    neurons = []
    enabled_ids = set()

    # First pass: collect enabled IDs and their gains from YAML
    yaml_gains = {}
    yaml_regions = {}
    for entry in enabled_list:
        nid = entry["id"]
        enabled_ids.add(nid)
        yaml_gains[nid] = float(entry.get("gain", 1.0))
        if "region" in entry:
            yaml_regions[nid] = entry["region"]

    # Apply overrides on top
    for nid, g in overrides.items():
        yaml_gains[nid] = float(g)

    # Second pass: build full neuron list
    for n in runtime.known_neurons:
        nid = n["id"]
        enabled = nid in enabled_ids
        gain = yaml_gains.get(nid, float(n.get("default_gain", 1.0)))
        region = yaml_regions.get(nid, n.get("region", "answer"))
        neurons.append({
            "id": nid,
            "layer": int(n["layer"]),
            "index": int(n["index"]),
            "enabled": enabled,
            "gain": gain,
            "region": region,
        })

    return {
        "enabled": True,
        "global_alpha": 1.0,  # will be overridden by CLI arg
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "think_mode": think_mode,
        "neurons": neurons,
        "preset": preset_name,
    }, preset_name


def build_snapshot(runtime, preset_name, global_alpha=1.0,
                   max_new_tokens=8192, temperature=0.0, think_mode=True):
    # Check if preset_name is a YAML file path
    yaml_path = Path(preset_name)
    if yaml_path.suffix in (".yaml", ".yml") and yaml_path.exists():
        snapshot, _ = load_preset_from_yaml(yaml_path, runtime)
        # Override with CLI args
        snapshot["global_alpha"] = global_alpha
        if think_mode is not None:
            snapshot["think_mode"] = think_mode
        return snapshot

    # Original logic for named presets from neurons.json
    preset = runtime.get_preset(preset_name)
    if preset is None:
        raise SystemExit(f"preset '{preset_name}' not found in {CONFIG_PATH}")

    enabled_ids = set(preset.get("enabled", []))
    overrides = preset.get("overrides", {})

    neurons = []
    for n in runtime.known_neurons:
        if n["id"] in enabled_ids:
            gain = float(overrides.get(n["id"], n.get("default_gain", 1.0)))
            neurons.append({
                "id": n["id"],
                "layer": int(n["layer"]),
                "index": int(n["index"]),
                "enabled": True,
                "gain": gain,
            })
        else:
            neurons.append({
                "id": n["id"],
                "layer": int(n["layer"]),
                "index": int(n["index"]),
                "enabled": False,
                "gain": float(n.get("default_gain", 1.0)),
            })

    return {
        "enabled": preset_name != "baseline",
        "global_alpha": global_alpha,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "think_mode": think_mode,
        "neurons": neurons,
        "preset": preset_name,
    }


def extract_last_cpp_block(text: str) -> str | None:
    matches = re.findall(r"```cpp\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"```c\+\+\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"```\s*\n(class Solution.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def main():
    # Parse CLI args: preset alpha [--think|--no-think]
    args = sys.argv[1:]
    preset_name = "math_anchor_v2"
    global_alpha = 1.0
    think_mode = True  # default

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--think",):
            think_mode = True
        elif arg in ("--no-think",):
            think_mode = False
        elif i == 0:
            preset_name = arg
        elif i == 1:
            global_alpha = float(arg)
        i += 1

    print(f"[init] preset={preset_name} global_alpha={global_alpha} think={think_mode}")
    runtime = ClampChatRuntime(str(CONFIG_PATH))
    runtime.load()

    snapshot = build_snapshot(runtime, preset_name, global_alpha=global_alpha, think_mode=think_mode)
    enabled_count = sum(1 for n in snapshot["neurons"] if n["enabled"])
    print(f"[init] enabled neurons: {enabled_count}")
    print(f"[init] think_mode={snapshot['think_mode']} temp={snapshot['temperature']} max_new={snapshot['max_new_tokens']}")
    print()
    print("=" * 60)
    print("STREAMING OUTPUT")
    print("=" * 60)

    chunks = []

    def stream_cb(text):
        chunks.append(text)
        sys.stdout.write(text)
        sys.stdout.flush()

    t0 = time.time()
    reply = runtime.generate_reply([], PROMPT, snapshot, stream_callback=stream_cb)
    dt = time.time() - t0
    n_chars = len(reply)
    n_tok_est = n_chars // 2
    print()
    print("=" * 60)
    print(f"[done] {dt:.1f}s, {n_chars} chars (~{n_tok_est} tok), {n_tok_est/max(dt,0.001):.1f} tok/s est")

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{preset_name}_a{global_alpha:.2f}_{ts}"
    # Sanitize base name for file paths (remove path separators etc.)
    base = base.replace("/", "_").replace("\\", "_").replace(".yaml", "").replace(".yml", "")
    log_path = OUT_DIR / f"{base}.txt"
    cpp_path = OUT_DIR / f"{base}.cpp"

    log_content = (
        f"# Preset: {preset_name}\n"
        f"# Global alpha: {global_alpha}\n"
        f"# Enabled neurons: {enabled_count}\n"
        f"# Think mode: {snapshot['think_mode']}  Temperature: {snapshot['temperature']}\n"
        f"# Duration: {dt:.1f}s\n"
        f"# Output chars: {n_chars}\n\n"
        f"## Prompt\n```\n{PROMPT}\n```\n\n"
        f"## Output\n{reply}\n"
    )
    log_path.write_text(log_content, encoding="utf-8")
    print(f"[save] log -> {log_path}")

    cpp = extract_last_cpp_block(reply)
    if cpp:
        cpp_path.write_text(cpp, encoding="utf-8")
        print(f"[save] cpp -> {cpp_path}")
        print(f"[cpp] {len(cpp)} chars, {cpp.count(chr(10))} lines")
    else:
        print("[warn] no ```cpp``` block detected; only the .txt log saved")


if __name__ == "__main__":
    main()
from __future__ import annotations

import torch


class ClampGate:
    """Mutable flag controlling whether clamp pre-hooks apply."""
    __slots__ = ("allow",)

    def __init__(self, allow: bool):
        self.allow = allow


def install_clamp_hooks(layers, selections, global_alpha: float, gate: "ClampGate | None" = None):
    by_layer = {}
    for item in selections:
        li = int(item["layer"])
        ni = int(item["index"])
        gain = float(item["gain"])
        by_layer.setdefault(li, []).append((ni, global_alpha * gain))

    hooks = []
    for li, items in by_layer.items():
        layer_dev = layers[li].mlp.down_proj.weight.device
        idx_dev = torch.tensor([n for n, _ in items], dtype=torch.long, device=layer_dev)
        shift_cpu = torch.tensor([s for _, s in items], dtype=torch.float32)
        shift_cache: dict = {}

        def make_fn(idx_local, shift_local_cpu, cache_local, gate_local):
            def pre_hook(module, inputs):
                if gate_local is not None and not gate_local.allow:
                    return
                x = inputs[0]
                key = (x.dtype, x.device)
                shift_dev = cache_local.get(key)
                if shift_dev is None:
                    shift_dev = shift_local_cpu.to(dtype=x.dtype, device=x.device)
                    cache_local[key] = shift_dev
                x = x.clone()
                x[:, :, idx_local] = x[:, :, idx_local] + shift_dev
                return (x,) + inputs[1:]

            return pre_hook

        hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(
            make_fn(idx_dev, shift_cpu, shift_cache, gate)
        ))
    return hooks


def install_segment_gate(model, gate: ClampGate, channel_close_id: int, on_seen_set_to: bool):
    """Watch embedding-layer input_ids; when channel_close_id appears,
    set gate.allow = on_seen_set_to (one-shot, then no-op)."""
    embed = model.model.language_model.embed_tokens
    seen = {"v": False}

    def pre_hook(module, inputs):
        if seen["v"]:
            return
        ids = inputs[0] if isinstance(inputs, tuple) else inputs
        if (ids == channel_close_id).any():
            gate.allow = on_seen_set_to
            seen["v"] = True

    return embed.register_forward_pre_hook(pre_hook)


def remove_hooks(hooks):
    for h in hooks:
        h.remove()

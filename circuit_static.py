"""
A. Static weight-overlap graph for known neurons.

For each pair (n_L_src, n'_L_sink) with L_src < L_sink, compute:
    write_dir  = down_proj[L_src].weight[:, n] * post_ff_norm[L_src].weight        [hidden]
    read_g     = gate_proj[L_sink].weight[n', :] * pre_ff_norm[L_sink].weight     [hidden]
    read_u     = up_proj[L_sink].weight[n', :] * pre_ff_norm[L_sink].weight       [hidden]
    edge       = max( |cos(write, read_g)|, |cos(write, read_u)| )

Outputs:
  - inventory <-> inventory adjacency (top edges by |cos|)
  - per-known-sink: top novel upstream feeders from full space
  - .pt with full results
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from transformers import Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
OUT_DIR = ROOT / "outputs" / "circuit_static"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
TOP_INV_EDGES = 50
TOP_NOVEL_PER_SINK = 8


def main():
    print(f"[load] {MODEL_PATH}  (BF16)")
    m = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True,
    )
    m.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(m.model, attr):
            setattr(m.model, attr, None)
    layers = m.model.language_model.layers
    n_layers = len(layers)
    print(f"  layers={n_layers}, alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB")

    # collect weights as float32 GPU tensors (max 35*3*1536*12288*4 = ~7.9 GB; use bf16)
    Ws_down  = []   # [hidden=1536, intermediate=12288]
    Ws_gate  = []   # [intermediate, hidden]
    Ws_up    = []
    norm_pre = []   # [hidden]
    norm_post = []  # [hidden]
    for L in layers:
        Ws_down.append(L.mlp.down_proj.weight.detach())
        Ws_gate.append(L.mlp.gate_proj.weight.detach())
        Ws_up.append(L.mlp.up_proj.weight.detach())
        norm_pre.append(L.pre_feedforward_layernorm.weight.detach())
        norm_post.append(L.post_feedforward_layernorm.weight.detach())
    print(f"[weights] cached")

    # ---------- load known neurons ----------
    nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
    kn = nj["known_neurons"]
    print(f"[neurons] loaded {len(kn)} known")
    known_ids = {f"L{n['layer']:02d}#{n['index']}" for n in kn}

    # build per-known write_dir, read_g_dir, read_u_dir (all [hidden], BF16 GPU)
    write_dirs = {}
    read_g_dirs = {}
    read_u_dirs = {}
    for nrn in kn:
        li, ni = nrn["layer"], nrn["index"]
        write_dirs[(li, ni)] = (Ws_down[li][:, ni] * norm_post[li]).float()
        read_g_dirs[(li, ni)] = (Ws_gate[li][ni, :] * norm_pre[li]).float()
        read_u_dirs[(li, ni)] = (Ws_up[li][ni, :] * norm_pre[li]).float()

    # ---------- inventory <-> inventory ----------
    print(f"\n[score] inventory→inventory edges (L_src < L_sink)")
    edges = []
    for src in kn:
        sli, sni = src["layer"], src["index"]
        w = write_dirs[(sli, sni)]
        for snk in kn:
            tli, tni = snk["layer"], snk["index"]
            if tli <= sli:
                continue
            rg = read_g_dirs[(tli, tni)]
            ru = read_u_dirs[(tli, tni)]
            sg = F.cosine_similarity(w.unsqueeze(0), rg.unsqueeze(0)).item()
            su = F.cosine_similarity(w.unsqueeze(0), ru.unsqueeze(0)).item()
            via = "gate" if abs(sg) >= abs(su) else "up"
            signed = sg if abs(sg) >= abs(su) else su
            edges.append({
                "src": src["id"], "sink": snk["id"],
                "src_layer": sli, "sink_layer": tli,
                "src_tier": src["tier"], "sink_tier": snk["tier"],
                "src_gain": src["default_gain"], "sink_gain": snk["default_gain"],
                "score": abs(signed), "signed": signed, "via": via,
                "dl": tli - sli,
            })
    edges.sort(key=lambda e: -e["score"])

    print(f"\n=== TOP {TOP_INV_EDGES} INVENTORY EDGES (by |cos|) ===")
    print(f"  {'rk':>3}  {'src':>10} → {'sink':<10}  {'cos':>7}  via   {'src_tier':>15} → {'sink_tier':<15}  Δ")
    for k, e in enumerate(edges[:TOP_INV_EDGES]):
        print(f"   {k+1:>2}  {e['src']:>10} → {e['sink']:<10}  {e['signed']:+.3f}  {e['via']:>4}  "
              f"{e['src_tier']:>15} → {e['sink_tier']:<15}  Δ{e['dl']}")

    # ---------- full-space upstream feeders for each known sink ----------
    print(f"\n[score] full-space → known-sink (top {TOP_NOVEL_PER_SINK} novel feeders per sink)")
    upstream_per_sink = {}
    sink_dirs = []
    for snk in kn:
        tli, tni = snk["layer"], snk["index"]
        sink_dirs.append((snk, read_g_dirs[(tli, tni)], read_u_dirs[(tli, tni)]))

    # for each src layer, do batched matmul with all sink read_dirs
    # write_all[sli] = (Ws_down[sli] * norm_post[sli][:, None]).float()  [hidden, intermediate]
    print(f"  building per-layer write tensors...")
    write_layer_norm = []
    for sli in range(n_layers):
        W = (Ws_down[sli].float() * norm_post[sli].float().unsqueeze(1))
        # normalize each column
        Wn = F.normalize(W, dim=0)
        write_layer_norm.append(Wn)

    # for each sink, scan src layers
    for snk, rg, ru in sink_dirs:
        tli, tni = snk["layer"], snk["index"]
        rg_n = F.normalize(rg, dim=0)
        ru_n = F.normalize(ru, dim=0)
        all_scores = []   # (|score|, sli, ni, signed, via)
        for sli in range(tli):
            Wn = write_layer_norm[sli]   # [hidden, intermediate]
            cg = (Wn.T @ rg_n)   # [intermediate]
            cu = (Wn.T @ ru_n)
            score = torch.maximum(cg.abs(), cu.abs())
            sgn = torch.where(cg.abs() >= cu.abs(), cg, cu)
            via = torch.where(cg.abs() >= cu.abs(), 0, 1)   # 0=gate, 1=up
            top_k = torch.topk(score, 30)
            for j in range(30):
                idx = top_k.indices[j].item()
                v_score = top_k.values[j].item()
                v_sgn = sgn[idx].item()
                v_via = "gate" if via[idx].item() == 0 else "up"
                all_scores.append((v_score, sli, idx, v_sgn, v_via))
        all_scores.sort(reverse=True)
        upstream_per_sink[snk["id"]] = all_scores

    print(f"\n=== TOP NOVEL UPSTREAM FEEDERS (per known sink, |cos|, novel = not in inventory) ===")
    for snk in kn:
        if snk["tier"] == "candidate":
            continue
        sid = snk["id"]
        feeders = upstream_per_sink[sid]
        novel = []
        in_inv = []
        for v_score, sli, ni, v_sgn, v_via in feeders:
            tag = f"L{sli:02d}#{ni}"
            if tag in known_ids:
                in_inv.append((v_score, sli, ni, v_sgn, v_via))
            else:
                novel.append((v_score, sli, ni, v_sgn, v_via))
            if len(novel) >= TOP_NOVEL_PER_SINK and len(in_inv) >= 5:
                break
        print(f"\n{sid}  ({snk['tier']:>15}, gain={snk['default_gain']:+.1f}, layer={snk['layer']})")
        if in_inv:
            txt = "  ".join(
                f"L{sli:02d}#{ni}({v_sgn:+.2f},{v_via})" for _, sli, ni, v_sgn, v_via in in_inv[:5]
            )
            print(f"  inv  : {txt}")
        if novel:
            txt = "  ".join(
                f"L{sli:02d}#{ni}({v_sgn:+.2f},{v_via})" for _, sli, ni, v_sgn, v_via in novel[:TOP_NOVEL_PER_SINK]
            )
            print(f"  novel: {txt}")

    # ---------- save ----------
    out = {
        "edges_inv": edges,
        "upstream_per_sink": {k: v[:50] for k, v in upstream_per_sink.items()},
        "n_layers": n_layers,
        "n_known": len(kn),
    }
    torch.save(out, OUT_DIR / "circuit_static.pt")
    print(f"\n[save] {OUT_DIR / 'circuit_static.pt'}")


if __name__ == "__main__":
    main()

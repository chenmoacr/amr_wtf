"""
Sanity check: load all QA JSONs, verify labeled sentences can be located in
the Opus output and mapped to token positions.

No model loaded — just tokenizer + offset mapping.

Uses difflib fuzzy matching: Gemini's annotations are not verbatim quotes —
they often paraphrase, swap commas for periods, or drop middle pieces.
We accept any sentence whose total matched-block length covers >= 50% of the
annotation, with each block >= MIN_BLOCK chars to avoid spurious noise.
"""
from __future__ import annotations

import difflib
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
DATA_DIR = ROOT / "data"
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"

REGION_NAMES = ["unlabeled", "glue", "d1_surface", "d2_structural", "d3_thematic"]
MIN_BLOCK = 6
MIN_COVERAGE = 0.5


def _exact_spans(opus_text: str, needle: str):
    spans, idx = [], 0
    if not needle:
        return spans
    while True:
        pos = opus_text.find(needle, idx)
        if pos == -1:
            break
        spans.append((pos, pos + len(needle)))
        idx = pos + len(needle)
    return spans


def find_fuzzy_spans(opus_text: str, sentence: str):
    """Return list of (char_lo, char_hi) spans in opus_text matching sentence.
    Short sentences: exact match (with optional trailing-punct strip).
    Long sentences: difflib block matching with MIN_BLOCK + MIN_COVERAGE filters.
    Returns None on miss."""
    if len(sentence) < 12:
        spans = _exact_spans(opus_text, sentence)
        if spans:
            return spans
        stripped = sentence.rstrip("。，？！；：.,\"'")
        if stripped != sentence and len(stripped) >= 4:
            spans = _exact_spans(opus_text, stripped)
            if spans:
                return spans
        return None

    sm = difflib.SequenceMatcher(None, opus_text, sentence, autojunk=False)
    blocks = [m for m in sm.get_matching_blocks() if m.size >= MIN_BLOCK]
    if not blocks:
        return None
    matched = sum(m.size for m in blocks)
    if matched < MIN_COVERAGE * len(sentence):
        return None
    return [(m.a, m.a + m.size) for m in blocks]


def label_tokens(opus_text, offsets, conclusion, glue):
    n = len(offsets)
    labels = torch.zeros(n, dtype=torch.uint8)
    found = {k: 0 for k in REGION_NAMES[1:]}
    missed = {k: [] for k in REGION_NAMES[1:]}

    def mark_span(label_id, char_lo, char_hi):
        for i, off in enumerate(offsets):
            cs, ce = off
            if cs < char_hi and ce > char_lo:
                if int(labels[i]) < label_id:
                    labels[i] = label_id

    def mark(sentence_list, label_id, key):
        for s in sentence_list:
            s = s.strip()
            if not s:
                continue
            spans = find_fuzzy_spans(opus_text, s)
            if spans is None:
                missed[key].append(s[:50])
                continue
            for lo, hi in spans:
                mark_span(label_id, lo, hi)
            found[key] += 1

    mark(glue, 1, "glue")
    mark(conclusion.get("depth_1_technical_surface", []), 2, "d1_surface")
    mark(conclusion.get("depth_2_structural_narrative", []), 3, "d2_structural")
    mark(conclusion.get("depth_3_core_thematic", []), 4, "d3_thematic")
    return labels, found, missed


def main():
    print("[load] tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"[load] is_fast={tok.is_fast}")

    qa_files = sorted(DATA_DIR.glob("claudeopusQA*.json"))
    print(f"[scan] {len(qa_files)} QAs found\n")

    for f in qa_files:
        qa = json.loads(f.read_text(encoding="utf-8"))
        out = qa["output"]
        ca = qa.get("conclusion_analysis", {})
        glue = qa.get("glue_sentences", [])

        try:
            enc = tok(out, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            ids = enc["input_ids"]
        except Exception as e:
            print(f"[{f.name}] tokenizer offset_mapping failed: {e}")
            continue

        n_tokens = len(ids)
        labels, found, missed = label_tokens(out, offsets, ca, glue)

        counts = torch.bincount(labels.long(), minlength=5).tolist()
        n_labeled = sum(counts[1:])
        coverage = n_labeled / max(1, n_tokens)
        print(f"[{f.stem}] model={qa.get('model','?')}  tokens={n_tokens}  output_chars={len(out)}")
        print(f"  region tokens: " + ", ".join(f"{REGION_NAMES[i]}={counts[i]}" for i in range(5)))
        print(f"  labeled coverage: {coverage*100:.1f}% ({n_labeled}/{n_tokens})")
        print(f"  found sentences: " + ", ".join(f"{k}={v}" for k, v in found.items()))
        if any(missed.values()):
            for k, v in missed.items():
                if v:
                    print(f"  MISSED {k}:")
                    for s in v:
                        print(f"    - {s!r}")
        print()


if __name__ == "__main__":
    main()

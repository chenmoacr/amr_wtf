from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from steering import ClampGate, install_clamp_hooks, install_segment_gate, remove_hooks

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class _StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event is not None and self.event.is_set()


class ClampChatRuntime:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.model_path = self.config.get("model_path", "J:/amr/models/gemma-4-E2B-it")
        self.device = self.config.get("device", "cuda:0")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 900))
        self.known_neurons = self.config.get("known_neurons", [])
        self.presets = self.config.get("presets", [])
        self.region_lookup = {n["id"]: n.get("region", "answer") for n in self.known_neurons}

        self.tokenizer = None
        self.model = None
        self.layers = None
        self._eoc_id = None

    def load(self):
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                     "embed_vision", "embed_audio"):
            if hasattr(self.model.model, attr):
                setattr(self.model.model, attr, None)
        self.layers = self.model.model.language_model.layers
        self._eoc_id = self.tokenizer.convert_tokens_to_ids("<channel|>")

    def _split_by_region(self, selections):
        thought, answer, always = [], [], []
        for sel in selections:
            region = self.region_lookup.get(sel["id"], "answer")
            if region == "thought":
                thought.append(sel)
            elif region == "always":
                always.append(sel)
            else:
                answer.append(sel)
        return thought, answer, always

    def generate_reply(self, history, user_text: str, steering_snapshot: dict,
                       stream_callback=None, stop_event=None):
        self.load()

        think_mode = bool(steering_snapshot.get("think_mode", False))
        messages = history + [{"role": "user", "content": user_text}]
        chat_input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=think_mode,
        )
        input_ids = chat_input["input_ids"].to(self.device)
        attention_mask = chat_input["attention_mask"].to(self.device)

        max_new_tokens = int(steering_snapshot.get("max_new_tokens", self.max_new_tokens))
        temperature = float(steering_snapshot.get("temperature", 1.0))
        do_sample = temperature > 0
        global_alpha = float(steering_snapshot.get("global_alpha", 1.0))

        hooks = []
        try:
            if steering_snapshot.get("enabled", False):
                selected = [x for x in steering_snapshot.get("neurons", []) if x.get("enabled", False)]
                if selected:
                    thought_sel, answer_sel, always_sel = self._split_by_region(selected)

                    if think_mode:
                        if thought_sel:
                            tg = ClampGate(allow=True)
                            hooks += install_clamp_hooks(self.layers, thought_sel, global_alpha, gate=tg)
                            hooks.append(install_segment_gate(self.model, tg, self._eoc_id, on_seen_set_to=False))
                        if answer_sel:
                            ag = ClampGate(allow=False)
                            hooks += install_clamp_hooks(self.layers, answer_sel, global_alpha, gate=ag)
                            hooks.append(install_segment_gate(self.model, ag, self._eoc_id, on_seen_set_to=True))
                        if always_sel:
                            hooks += install_clamp_hooks(self.layers, always_sel, global_alpha, gate=None)
                    else:
                        merged = answer_sel + always_sel
                        if merged:
                            hooks += install_clamp_hooks(self.layers, merged, global_alpha, gate=None)

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            if stop_event is not None:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnEvent(stop_event)])

            if stream_callback is not None:
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=False,
                )
                gen_kwargs["streamer"] = streamer

                gen_error = {"e": None}

                def _gen_in_thread():
                    try:
                        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                            self.model.generate(**gen_kwargs)
                    except Exception as e:
                        gen_error["e"] = e

                gen_thread = threading.Thread(target=_gen_in_thread, daemon=True)
                gen_thread.start()
                collected = []
                try:
                    for new_text in streamer:
                        collected.append(new_text)
                        try:
                            stream_callback(new_text)
                        except Exception:
                            pass
                finally:
                    gen_thread.join()
                if gen_error["e"] is not None:
                    raise gen_error["e"]
                return "".join(collected)

            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = self.model.generate(**gen_kwargs)
            gen_ids = out[0][input_ids.shape[1]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            return text
        finally:
            if hooks:
                remove_hooks(hooks)

    def get_preset(self, preset_name: str):
        for p in self.presets:
            if p.get("name") == preset_name:
                return p
        return None

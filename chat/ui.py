from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


class ClampChatUI:
    def __init__(self, root, runtime):
        self.root = root
        self.runtime = runtime
        self.history = []
        self._busy = False
        self._stop_event = None

        self.root.title("AMR 神经元开关聊天")
        self.root.geometry("1400x900")

        self.enable_var = tk.BooleanVar(value=False)
        self.compare_var = tk.BooleanVar(value=False)
        self.think_var = tk.BooleanVar(value=False)
        self.global_alpha_var = tk.StringVar(value="1.0")
        self.temperature_var = tk.StringVar(value="0.0")
        self.max_new_tokens_var = tk.StringVar(value="65535")

        self.neuron_rows = []
        self.preset_var = tk.StringVar(value="baseline")

        self._build_layout()

    def _build_layout(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        right = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)
        left.rowconfigure(2, weight=0)
        left.columnconfigure(0, weight=1)

        self.chat_box = ScrolledText(left, wrap="word", font=("Consolas", 11))
        self.chat_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.chat_box.configure(state="disabled")

        self.input_box = tk.Text(left, height=8, font=("Consolas", 11))
        self.input_box.grid(row=1, column=0, sticky="ew")

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        btn_frame.columnconfigure(0, weight=0)
        btn_frame.columnconfigure(1, weight=0)
        btn_frame.columnconfigure(2, weight=0)
        btn_frame.columnconfigure(3, weight=1)

        self.send_btn = ttk.Button(btn_frame, text="发送", command=self.on_send)
        self.send_btn.grid(row=0, column=0, padx=(0, 8))

        self.stop_btn = ttk.Button(btn_frame, text="终止生成", command=self.on_stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, 8))

        clear_btn = ttk.Button(btn_frame, text="清空对话", command=self.on_clear)
        clear_btn.grid(row=0, column=2, padx=(0, 8))

        ttk.Checkbutton(btn_frame, text="对比模式（基线+注入）", variable=self.compare_var).grid(row=0, column=3, sticky="w")

        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        top_ctrl = ttk.LabelFrame(right, text="推理与注入设置", padding=8)
        top_ctrl.grid(row=0, column=0, sticky="ew")
        top_ctrl.columnconfigure(1, weight=1)

        ttk.Checkbutton(top_ctrl, text="启用神经元注入", variable=self.enable_var).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(top_ctrl, text="全局 Alpha").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top_ctrl, textvariable=self.global_alpha_var, width=12).grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(top_ctrl, text="温度 Temperature").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top_ctrl, textvariable=self.temperature_var, width=12).grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Label(top_ctrl, text="输出长度(max_new_tokens)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top_ctrl, textvariable=self.max_new_tokens_var, width=12).grid(row=3, column=1, sticky="w", pady=(8, 0))

        ttk.Checkbutton(top_ctrl, text="开启 Think 模式", variable=self.think_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        preset_frame = ttk.LabelFrame(right, text="预设组合", padding=8)
        preset_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        preset_names = [p.get("name", "") for p in self.runtime.presets] or ["baseline"]
        if self.preset_var.get() not in preset_names:
            self.preset_var.set(preset_names[0])
        ttk.Combobox(preset_frame, textvariable=self.preset_var, values=preset_names, state="readonly").grid(row=0, column=0, sticky="ew")
        ttk.Button(preset_frame, text="应用预设", command=self.on_apply_preset).grid(row=0, column=1, padx=(8, 0))
        preset_frame.columnconfigure(0, weight=1)

        neuron_frame = ttk.LabelFrame(right, text="已知神经元", padding=8)
        neuron_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        neuron_frame.rowconfigure(0, weight=1)
        neuron_frame.columnconfigure(0, weight=1)

        canvas = tk.Canvas(neuron_frame)
        scroll = ttk.Scrollbar(neuron_frame, orient="vertical", command=canvas.yview)
        self.rows_holder = ttk.Frame(canvas)

        self.rows_holder.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.rows_holder, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        self._build_neuron_rows()

    def _build_neuron_rows(self):
        for i, n in enumerate(self.runtime.known_neurons):
            enabled_var = tk.BooleanVar(value=False)
            gain_var = tk.StringVar(value=str(n.get("default_gain", 1.0)))

            row = ttk.Frame(self.rows_holder)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.columnconfigure(1, weight=1)

            ttk.Checkbutton(row, variable=enabled_var).grid(row=0, column=0, padx=(0, 6))
            ttk.Label(row, text=f"{n['id']}  [{n.get('label', '')}]").grid(row=0, column=1, sticky="w")
            ttk.Entry(row, textvariable=gain_var, width=8).grid(row=0, column=2, padx=(6, 0))

            self.neuron_rows.append({
                "id": n["id"],
                "layer": int(n["layer"]),
                "index": int(n["index"]),
                "enabled_var": enabled_var,
                "gain_var": gain_var,
            })

    def on_apply_preset(self):
        preset = self.runtime.get_preset(self.preset_var.get())
        if not preset:
            messagebox.showerror("预设", "找不到该预设")
            return

        enabled_ids = set(preset.get("enabled", []))
        overrides = preset.get("overrides", {})

        for r in self.neuron_rows:
            rid = r["id"]
            r["enabled_var"].set(rid in enabled_ids)
            if rid in overrides:
                r["gain_var"].set(str(overrides[rid]))

    def _append_chat(self, role: str, text: str):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"[{role}]\n{text}\n\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _begin_stream(self, role: str):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"[{role}]\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _append_stream(self, text: str):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _end_stream(self):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", "\n\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def on_clear(self):
        self.history = []
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.configure(state="disabled")

    def _collect_snapshot(self):
        try:
            global_alpha = float(self.global_alpha_var.get().strip())
        except ValueError:
            raise ValueError("全局 Alpha 必须是数字")

        try:
            temperature = float(self.temperature_var.get().strip())
        except ValueError:
            raise ValueError("温度必须是数字")
        if temperature < 0:
            raise ValueError("温度不能小于 0")

        try:
            max_new_tokens = int(self.max_new_tokens_var.get().strip())
        except ValueError:
            raise ValueError("输出长度必须是正整数")
        if max_new_tokens <= 0:
            raise ValueError("输出长度必须大于 0")

        neurons = []
        for r in self.neuron_rows:
            try:
                gain = float(r["gain_var"].get().strip())
            except ValueError:
                raise ValueError(f"Gain 必须是数字: {r['id']}")
            neurons.append({
                "id": r["id"],
                "layer": r["layer"],
                "index": r["index"],
                "enabled": bool(r["enabled_var"].get()),
                "gain": gain,
            })

        return {
            "enabled": bool(self.enable_var.get()),
            "global_alpha": global_alpha,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "think_mode": bool(self.think_var.get()),
            "neurons": neurons,
            "preset": self.preset_var.get(),
        }

    def on_stop(self):
        if self._stop_event is not None:
            self._stop_event.set()
            self.stop_btn.configure(state="disabled")

    def on_send(self):
        if self._busy:
            return

        user_text = self.input_box.get("1.0", "end").strip()
        if not user_text:
            return

        try:
            snapshot = self._collect_snapshot()
        except Exception as e:
            messagebox.showerror("输入错误", str(e))
            return

        self.input_box.delete("1.0", "end")
        self._append_chat("用户", user_text)

        self._busy = True
        self._stop_event = threading.Event()
        self.send_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        compare = bool(self.compare_var.get())
        stop_event = self._stop_event

        def stream_cb(text):
            self.root.after(0, self._append_stream, text)

        def worker():
            try:
                if compare:
                    base_snapshot = dict(snapshot)
                    base_snapshot["enabled"] = False
                    base_snapshot["neurons"] = []
                    self.root.after(0, self._begin_stream, "助手-基线")
                    base_reply = self.runtime.generate_reply(
                        self.history, user_text, base_snapshot,
                        stream_callback=stream_cb, stop_event=stop_event,
                    )
                    self.root.after(0, self._end_stream)

                    if not stop_event.is_set():
                        self.root.after(0, self._begin_stream, "助手-注入")
                        steered_reply = self.runtime.generate_reply(
                            self.history, user_text, snapshot,
                            stream_callback=stream_cb, stop_event=stop_event,
                        )
                        self.root.after(0, self._end_stream)
                        final_reply = steered_reply
                    else:
                        final_reply = base_reply
                else:
                    self.root.after(0, self._begin_stream, "助手")
                    reply = self.runtime.generate_reply(
                        self.history, user_text, snapshot,
                        stream_callback=stream_cb, stop_event=stop_event,
                    )
                    self.root.after(0, self._end_stream)
                    final_reply = reply

                self.history.append({"role": "user", "content": user_text})
                self.history.append({"role": "assistant", "content": final_reply})
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("运行错误", str(e)))
            finally:
                def done():
                    self._busy = False
                    self._stop_event = None
                    self.send_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")

                self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

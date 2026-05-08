from __future__ import annotations

import sys
from pathlib import Path

import tkinter as tk

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from runtime import ClampChatRuntime
from ui import ClampChatUI


def main():
    config_path = ROOT / "neurons.json"
    runtime = ClampChatRuntime(str(config_path))

    root = tk.Tk()
    ClampChatUI(root, runtime)
    root.mainloop()


if __name__ == "__main__":
    main()

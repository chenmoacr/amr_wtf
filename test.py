import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, master):
        self.master = master
        master.title("井字棋 (Tic-Tac-Toe)")

        # --- 游戏状态变量 ---
        self.board = [""] * 9  # 存储棋盘状态，长度为9
        self.current_player = "X"
        self.game_active = True

        # --- UI 元素 ---
        # 状态标签
        self.status_label = tk.Label(master, text=f"当前玩家: {self.current_player}", font=('Arial', 14))
        self.status_label.grid(row=0, column=0, columnspan=3, pady=10)

        # 棋盘框架 (使用一个Frame来更好地组织布局)
        self.board_frame = tk.Frame(master, padx=10, pady=10, bg='lightgray')
        self.board_frame.grid(row=1, column=0, columnspan=3)

        # 创建 9 个按钮
        self.buttons = []
        for i in range(9):
            button = tk.Button(
                self.board_frame,
                text="",
                font=('Arial', 24, 'bold'),
                width=5,
                height=2,
                command=lambda index=i: self.handle_click(index)
            )
            # 按钮在棋盘框架内使用 grid 布局
            row = i // 3
            col = i % 3
            button.grid(row=row, column=col, padx=5, pady=5)
            self.buttons.append(button)

        # 重置按钮
        self.reset_button = tk.Button(master, text="重新开始", command=self.reset_game, font=('Arial', 12))
        self.reset_button.grid(row=2, column=0, columnspan=3, pady=10)

    def handle_click(self, index):
        """处理用户点击按钮的逻辑"""
        if not self.game_active or self.board[index] != "":
            return  # 如果游戏结束或位置已占，则忽略点击

        # 1. 更新状态
        self.board[index] = self.current_player
        self.buttons[index].config(text=self.current_player, state=tk.DISABLED, 
                                    fg="blue" if self.current_player == "X" else "red")

        # 2. 检查游戏结果
        if self.check_win():
            self.status_label.config(text=f"恭喜 {self.current_player} 赢了！", fg="green")
            self.game_active = False
            messagebox.showinfo("游戏结束", f"{self.current_player} 获胜！")
            return

        if self.check_draw():
            self.status_label.config(text="游戏平局！", fg="orange")
            self.game_active = False
            messagebox.showinfo("游戏结束", "游戏平局！")
            return

        # 3. 切换玩家
        self.current_player = "O" if self.current_player == "X" else "X"
        self.status_label.config(text=f"当前玩家: {self.current_player}", fg="black")

    def check_win(self):
        """检查是否有玩家获胜"""
        winning_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 行
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 列
            (0, 4, 8), (2, 4, 6)             # 对角线
        ]

        for combo in winning_combinations:
            a, b, c = combo
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != "":
                # 标记获胜的格子
                self.buttons[a].config(text=self.board[a], state=tk.DISABLED, fg="green")
                self.buttons[b].config(text=self.board[b], state=tk.DISABLED, fg="green")
                self.buttons[c].config(text=self.board[c], state=tk.DISABLED, fg="green")
                return True
        return False

    def check_draw(self):
        """检查棋盘是否已满"""
        return "" not in self.board

    def reset_game(self):
        """重置游戏状态"""
        self.board = [""] * 9
        self.current_player = "X"
        self.game_active = True

        # 清理所有按钮
        for button in self.buttons:
            button.config(text="", state=tk.NORMAL, fg="black")

        # 重置状态标签
        self.status_label.config(text=f"当前玩家: {self.current_player}", fg="black")


if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()
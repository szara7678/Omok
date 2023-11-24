import numpy as np
import tkinter as tk
from keras.models import load_model

# 오목 보드 크기
board_size = 15
cell_size = 40

# 모델 로드
model1 = load_model('model1.h5', compile=False)
model2 = load_model('model2.h5', compile=False)

class OmokSimulator:
    def __init__(self, model1, model2):
        self.root = tk.Tk()
        self.root.title("Omok Simulator")
        self.canvas = tk.Canvas(self.root, width=board_size * cell_size, height=board_size * cell_size)
        self.canvas.pack()
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.model1 = model1
        self.model2 = model2
        self.player_stone_color = {1: "black", 2: "white"}

        self.draw_board()
        self.root.after(100, self.play_model_move)  # 초기화 후 100ms 뒤에 모델이 두도록 설정

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(board_size):
            for j in range(board_size):
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = (j + 1) * cell_size, (i + 1) * cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")

                if self.board[i][j] != 0:
                    self.canvas.create_oval(x1, y1, x2, y2, fill=self.player_stone_color[self.board[i][j]])

    def make_move(self, row, col):
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            self.draw_board()
            winner = self.check_winner()
            if winner is not None:
                self.show_winner_message(winner)
            else:
                self.current_player = 3 - self.current_player
                self.root.after(100, self.play_model_move)  # 100ms 뒤에 모델이 두도록 설정

    def play_model_move(self):
        if self.current_player == 1:
            model = self.model1
        else:
            model = self.model2

        input_data = np.zeros((board_size, board_size, 2), dtype=int)
        input_data[:, :, 0] = (self.board == 1).astype(int)
        input_data[:, :, 1] = (self.board == 2).astype(int)
        move_probabilities = model.predict(np.expand_dims(input_data, axis=0))[0]

        legal_moves = np.where(self.board.flatten() == 0)[0]
        move = np.argmax(move_probabilities[legal_moves])
        row, col = divmod(legal_moves[move], board_size)

        self.make_move(row, col)

    def check_winner(self):
        for i in range(board_size):
            for j in range(board_size):
                if self.board[i][j] != 0:
                    # 가로, 세로, 대각선 체크
                    for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = 0
                        for k in range(5):
                            ni, nj = i + k * di, j + k * dj
                            if 0 <= ni < board_size and 0 <= nj < board_size and self.board[ni][nj] == self.board[i][j]:
                                count += 1
                        if count == 5:
                            return self.board[i][j]
        return None

    def show_winner_message(self, winner):
        message = f"Player {winner} wins!"
        print(message)
        tk.messagebox.showinfo("Game Over", message)
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# 시뮬레이터 실행
simulator = OmokSimulator(model1, model2)
simulator.run()

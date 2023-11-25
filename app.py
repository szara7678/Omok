# app.py

from flask import Flask, render_template, jsonify, request
import numpy as np
from keras.models import load_model

app = Flask(__name__)

board_size = 15
cell_size = 40

model1 = load_model('model1.h5', compile=False)

class OmokSimulator:
    def __init__(self, model1):
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.model1 = model1

    def play_model_move(self):
        input_data = np.zeros((board_size, board_size, 2), dtype=int)
        input_data[:, :, 0] = (self.board == 1).astype(int)
        input_data[:, :, 1] = (self.board == 2).astype(int)
        move_probabilities = self.model1.predict(np.expand_dims(input_data, axis=0))[0]

        legal_moves = np.where(self.board.flatten() == 0)[0]
        move = np.argmax(move_probabilities[legal_moves])
        row, col = divmod(legal_moves[move], board_size)

        self.make_move(row, col)

    def make_move(self, row, col):
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            winner = self.check_winner()
            if winner is not None:
                return {'status': 'win', 'winner': winner, 'board': self.board.tolist()}
            else:
                self.current_player = 3 - self.current_player
                if self.current_player == 1:
                    self.play_model_move()
                return {'status': 'success', 'board': self.board.tolist()}

    def check_winner(self):
        for i in range(board_size):
            for j in range(board_size):
                if self.board[i][j] != 0:
                    for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = 0
                        for k in range(5):
                            ni, nj = i + k * di, j + k * dj
                            if 0 <= ni < board_size and 0 <= nj < board_size and self.board[ni][nj] == self.board[i][j]:
                                count += 1
                        if count == 5:
                            return self.board[i][j]
        return None

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_board')
def get_board():
    return jsonify({'status': 'success', 'board': simulator.board.tolist()})

@app.route('/make_move/<int:row>/<int:col>', methods=['POST'])
def make_move(row, col):
    result = simulator.make_move(row, col)
    if result['status'] == 'win':
        return jsonify(result)
    return jsonify({'status': 'success', 'board': result['board']})

# Start the simulator
simulator = OmokSimulator(model1)

if __name__ == '__main__':
    app.run(debug=True)

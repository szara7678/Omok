import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# 오목 보드 크기
board_size = 15

# 전역 변수로 승리 횟수 초기화
total_wins_model1 = 0
total_wins_model2 = 0

# 오목 보드 클래스
class OmokBoard:
    def __init__(self):
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1

    def make_move(self, row, col):
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            self.current_player = 3 - self.current_player  # 플레이어 전환
            return True
        else:
            return False

# 훈련 데이터 생성 함수
def generate_data(board):
    input_data = np.zeros((board_size, board_size, 2), dtype=int)
    input_data[:, :, 0] = (board.board == 1).astype(int)
    input_data[:, :, 1] = (board.board == 2).astype(int)
    return input_data

# 오목 승자 체크 함수
def check_winner(board):
    for i in range(board_size):
        for j in range(board_size):
            if board.board[i][j] != 0:  # 수정된 부분: board 속성을 통해 배열 요소에 접근
                # 가로, 세로, 대각선 체크
                for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count = 0
                    for k in range(5):
                        ni, nj = i + k * di, j + k * dj
                        if 0 <= ni < board_size and 0 <= nj < board_size and board.board[ni][nj] == board.board[i][j]:  # 수정된 부분
                            count += 1
                    if count == 5:
                        return board.board[i][j]  # 수정된 부분
    return None


# 훈련 데이터 생성 및 모델 훈련
def train_model(model, opponent_model, epochs=10, games_per_epoch=100):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_wins_model1 = 0
        total_wins_model2 = 0

        for game in range(games_per_epoch):
            board = OmokBoard()
            training_data = []
            winner = None  # 수정된 부분: 초기화 추가
            while True:
                if board.current_player == 1:
                    input_data = generate_data(board)
                    move_probabilities = model.predict(np.expand_dims(input_data, axis=0))[0]
                else:
                    input_data = generate_data(board)
                    move_probabilities = opponent_model.predict(np.expand_dims(input_data, axis=0))[0]

                legal_moves = np.where(board.board.flatten() == 0)[0]
                move = np.random.choice(legal_moves, p=move_probabilities[legal_moves] / np.sum(move_probabilities[legal_moves]))
                row, col = divmod(move, board_size)
                if not board.make_move(row, col):
                    break
                winner = check_winner(board)
                if winner is not None:
                    break
                training_data.append((input_data, move))

            # 수정된 부분: 루프 종료 후에 승자를 확인하고 훈련 데이터를 생성하여 모델을 훈련
            if winner == 1:
                total_wins_model1 += 1
            elif winner == 2:
                total_wins_model2 += 1

            if winner is not None:
                input_data_array = np.array([data[0] for data in training_data])
                move_array = np.array([data[1] for data in training_data])
                model.train_on_batch(input_data_array, move_array)

        print(f"Model 1 Win Rate: {total_wins_model1 / games_per_epoch:.2%}")
        print(f"Model 2 Win Rate: {total_wins_model2 / games_per_epoch:.2%}")

# 모델 생성
model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(board_size, board_size, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(board_size * board_size, activation='softmax')
])

model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(board_size, board_size, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(board_size * board_size, activation='softmax')
])

# 모델 컴파일
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping 콜백 설정
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ModelCheckpoint 콜백 설정
checkpoint_callback = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# 모델 훈련
train_model(model1, model2, epochs=5, games_per_epoch=5)

# 각 모델의 훈련 결과 저장
model1.save('model1.h5')
model2.save('model2.h5')

# 총 승률 저장
total_win_rate_model1 = total_wins_model1 / (5 * 2)  # 총 게임 횟수 = 에포크 수 * 게임 횟수
total_win_rate_model2 = total_wins_model2 / (5 * 2)

with open('total_win_rate.txt', 'w') as f:
    f.write(f"Model 1 Total Win Rate: {total_win_rate_model1:.2%}\n")
    f.write(f"Model 2 Total Win Rate: {total_win_rate_model2:.2%}")
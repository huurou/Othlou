import numpy as np
import copy
import glob
from keras.models import Sequential
from keras.layers import Dense, Activation

FIRSTBAN = np.array([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                     [-1, 0, 0, 0, 2, 1, 0, 0, 0,-1],
                     [-1, 0, 0, 0, 1, 2, 0, 0, 0,-1],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                     [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])


class Play:
  def count_turn_over(self, board, color, y, x, d, e):  # ひっくり返る石の数を数える
    i = 1
    while board[y + i * d][x + i * e] == (3 - color):
      i += 1
    if board[y + i * d][x + i * e] == color:
      return i - 1
    else:
      return 0

  def is_legal(self, board, color, y, x):  # 打った石が合法手かどうか判定
    if x < 1 or y > 8 or x < 1 or y > 8:
      return 0
    if board[y][x] != 0:
      return 0
    if self.count_turn_over(board, color, y, x, -1, 0):
      return 1
    if self.count_turn_over(board, color, y, x, 1, 0):
      return 1
    if self.count_turn_over(board, color, y, x, 0, -1):
      return 1
    if self.count_turn_over(board, color, y, x, 0, 1):
      return 1
    if self.count_turn_over(board, color, y, x, -1, -1):
      return 1
    if self.count_turn_over(board, color, y, x, -1, 1):
      return 1
    if self.count_turn_over(board, color, y, x, 1, -1):
      return 1
    if self.count_turn_over(board, color, y, x, 1, 1):
      return 1
    return 0

  def legal_list(self, board, color):  # その盤面と手番での合法手一覧のリストを返す
    legal = np.empty((0, 2), int)
    for i in range(1, 9):
      for j in range(1, 9):
        if self.is_legal(board, color, i, j):
          legal = np.append(legal, np.array([[i, j]]), axis=0)

    return legal

  def exist_legal_move(self, board, color):  # 合法手の有無を1 or 0で返す
    for i in range(1, 9):
      for j in range(1, 9):
        if self.is_legal(board, color, i, j):
          return 1
    return 0

  def set_turn(self, board, color, y, x):  # 渡された盤面に石を打ち石をひっくり返す
    for d in range(-1, 2):
      for e in range(-1, 2):
        if d == e == 0:
          continue
        count = self.count_turn_over(board, color, y, x, d, e)
        for i in range(1, count + 1):
          board[y + i * d][x + i * e] = color
    board[y][x] = color


class Kifu:
  def readkifu(self, kifufile):
      ban = copy.deepcopy(FIRSTBAN)
      color = 1
      kifu = np.load(kifufile)
      black = np.empty((0, 64), int)
      white = np.empty((0, 64), int)
      label_1 = kifu[-1]
      for i in range(kifu.shape[0]):
        reban = ban[1:9, 1:9]
        b = np.zeros((8, 8))
        w = np.zeros((8, 8))
        b[reban == 1] = 1
        w[reban == 2] = 1
        black = np.append(black, np.reshape(b, (1, 64)), axis=0)
        white = np.append(white, np.reshape(w, (1, 64)), axis=0)
        if np.allclose(kifu[i], [[0, 0]]):
          color = 3 - color
          if np.allclose(kifu[i + 1], [[0, 0]]):
            break
          i += 1
        y, x = kifu[i]
        Play().set_turn(ban, color, y, x)
        color = 3 - color
      banlist = np.c_[black, white]
      label_2 = label_1[np.newaxis, :]
      labelist = np.empty((0, 2))
      for _ in range(banlist.shape[0]):
        labelist = np.append(labelist, label_2, axis=0)
      return banlist, labelist


if __name__ == '__main__':
    banlist, labelist = Kifu().readkifu()
    model = Sequential()
    model.add(Dense(200, input_dim=128, activation='relu'))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(banlist, labelist, epochs=10, batch_size=32)


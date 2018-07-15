import copy
import random
from datetime import datetime
import numpy as np
from tqdm import tqdm

PLAYOUT_NUM = 10
KIFU_NUM = 10000
BLACK =              np.array([[0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 0, 1, 0, 0, 0
                             , 0, 0, 0, 1, 0, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0]])
WHITE =              np.array([[0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 1, 0, 0, 0, 0
                             , 0, 0, 0, 0, 1, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0
                             , 0, 0, 0, 0, 0, 0, 0, 0]])

class Ban:
  def init_ban(self):
    first_board = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                             , [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1]
                             , [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1]
                             , [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1]
                             , [-1, 0, 0, 0, 2, 1, 0, 0, 0, -1]
                             , [-1, 0, 0, 0, 1, 2, 0, 0, 0, -1]
                             , [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1]
                             , [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1]
                             , [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1]
                             , [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    return first_board


class Play:
  def count_turn_over(self, board, color, y, x, d, e):
    i = 1
    while board[y + i * d][x + i * e] == (3 - color):
      i += 1
    if board[y + i * d][x + i * e] == color:
      return i - 1
    else:
      return 0

  def is_legal(self, board, color, y, x):
    if x < 1 or y > 8 or x < 1 or y > 8: return 0
    if board[y][x] != 0: return 0
    if self.count_turn_over(board, color, y, x, -1, 0): return 1
    if self.count_turn_over(board, color, y, x, 1, 0): return 1
    if self.count_turn_over(board, color, y, x, 0, -1): return 1
    if self.count_turn_over(board, color, y, x, 0, 1): return 1
    if self.count_turn_over(board, color, y, x, -1, -1): return 1
    if self.count_turn_over(board, color, y, x, -1, 1): return 1
    if self.count_turn_over(board, color, y, x, 1, -1): return 1
    if self.count_turn_over(board, color, y, x, 1, 1): return 1
    return 0

  def legal_list(self, board, color):
    legal = np.empty((0, 2), int)
    for i in range(1, 9):
      for j in range(1, 9):
        if self.is_legal(board, color, i, j):
          legal = np.append(legal, np.array([[i, j]]), axis=0)

    return legal

  def exist_legal_move(self, board, color):
    for i in range(1, 9):
      for j in range(1, 9):
        if self.is_legal(board, color, i, j): return 1
    return 0

  def set_turn(self, board, color, y, x):
    count = 0
    for d in range(-1, 2):
      for e in range(-1, 2):
        if d == 0 and e == 0:
          continue
        count = self.count_turn_over(board, color, y, x, d, e)
        for i in range(1, count + 1):
          board[y + i * d][x + i * e] = color
    board[y][x] = color


class Playout:
  def randmove(self, board, color):
    l = Play().legal_list(board, color)
    if l.shape[0] == 1:
      y, x = l[0]
    else:
      y, x = l[int(np.random.randint(l.shape[0], size=1))]
    return y, x

  def playout(self, board, color):
    tsugiban = copy.deepcopy(board)
    while True:
      if not Play().exist_legal_move(tsugiban, color):
        color = 3 - color
        if not Play().exist_legal_move(tsugiban, color):
          c_1 = np.sum(tsugiban == 1)
          c_2 = np.sum(tsugiban == 2)
          if c_1 >= c_2:
            return np.array([[1, 0]])
          else:
            return np.array([[0, 1]])
      y, x = self.randmove(tsugiban, color)
      Play().set_turn(tsugiban, color, y, x)
      color = 3 - color

  def playout_result(self, board, color, y, x):
    num = PLAYOUT_NUM
    tsugiban = copy.deepcopy(board)
    player = color
    result = np.empty((0, 2), int)
    Play().set_turn(tsugiban, color, y, x)
    color = 3 - color
    for i in range(num):
      result = np.append(result, self.playout(tsugiban, color), axis=0)
    if player == 1:
      return np.sum(result[:, 0] == 1) / float(num)
    elif player == 2:
      return np.sum(result[:, 1] == 1) / float(num)


class Cpu:
  def decide_move(self, board, color):
    rate = []
    l = Play().legal_list(board, color)
    if l.shape[0] == 1:
      y, x = l[0]
    for i in range(l.shape[0]):
      q, p =l[i]
      rate.append(Playout().playout_result(board, color, q, p))
      maxindex = [i for i, x in enumerate(rate) if x == max(rate)]
    y, x = l[random.choice(maxindex)]
    return  y, x



class Othello:
  def cvc_kifu(self, count):
    ban = Ban().init_ban()
    black = BLACK
    white = WHITE
    color = 1
    while True:
      if not Play().exist_legal_move(ban, color):
        color = 3- color
        if not Play().exist_legal_move(ban, color):
          break
      tsugiban = copy.deepcopy(ban)
      y, x = Playout().randmove(tsugiban, color)
      Play().set_turn(ban, color, y, x)
      color = 3- color
      reban = ban[1:9, 1:9]
      b = np.zeros((8, 8))
      w = np.zeros((8, 8))
      b[reban == 1] = 1
      w[reban == 2] = 1
      black = np.append(black, np.reshape(b, (1, 64)), axis=0)
      white = np.append(white, np.reshape(w, (1, 64)), axis=0)
    kifu = np.c_[black, white]
    c_1 = np.sum(ban == 1)
    c_2 = np.sum(ban == 2)
    if c_1 >= c_2:
      kifu = np.append(kifu, np.ones((1, 128)), axis=0)
    elif c_1 < c_2:
      kifu = np.append(kifu, np.zeros((1, 128)), axis=0)
    np.save('./kifu/{0}-{1}'.format(datetime.now().strftime('%y_%m_%d_%H_%M_%S'), count), kifu)


if __name__ == '__main__':
  for count in tqdm(range(1, KIFU_NUM + 1)):
    Othello().cvc_kifu(count)

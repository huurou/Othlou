import numpy as np
import copy
import math


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

  def stone(self, ishi):
    if ishi == 0:
      return '・'
    elif ishi == 1:
      return '●'
    else:
      return '○'

  def draw_ban(self, board):
    print('  a  b  c  d  e  f  g  h ')
    print('  ------------------------')
    for i in range(1, 9):
      print('%d|' % i, end='')
      for j in range(1, 9):
        print(self.stone(board[i][j]), end=' ')
      print('|')
    print('  ------------------------')


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


class Cpu:
  def cpu_randmove(self, board, color):
    l = Play().legal_list(board, color)
    if l.shape[0] == 1:
      y, x = l[0]
    else:
      y, x = l[int(np.random.randint(l.shape[0] - 1, size=1))]
    return y, x

  def playout(self, board, color):
    while True:
      if not Play().exist_legal_move(board, color):
        color = (3 - color)
        if not Play().exist_legal_move(board, color):
          c_1 = np.sum(board == 1)
          c_2 = np.sum(board == 2)
          if c_1 >= c_2:
            return np.array([[1, 0]])
          else:
            return np.array([[0, 1]])
      y, x = self.cpu_randmove(board, color)
      Play().set_turn(board, color, y, x)
      color = (3 - color)

  def ucb_playout(self, node):
    board = copy.deepcopy(node.board)
    color = node.color
    while True:
      if not Play().exist_legal_move(board, color):
        color = (3 - color)
        if not Play().exist_legal_move(board, color):
          c_1 = np.sum(board == 1)
          c_2 = np.sum(board == 2)
          if c_1 >= c_2:
            return np.array([[1, 0]])
          else:
            return np.array([[0, 1]])
      y, x = self.cpu_randmove(board, color)
      Play().set_turn(board, color, y, x)
      color = (3 - color)


class Node:
  def __init__(self, board, color):
    self.board = copy.deepcopy(board)  # ノードの局面
    self.color = color  # ノードの手番
    self.child_num = 0  # 子ノードの数
    self.child_move = None  # 子ノードの指し手のリスト
    self.ucb = 0.00  # 子ノードとしてのucb値
    self.child_ucb = []  # 子ノードのucb値
    self.child_node = []  # 子ノード
    self.total_play = 0  # 各子ノードの総プレイアウト数
    self.each_play = 0  # 子ノードとしてのプレイアウト数
    self.rate = 0.00  # 子ノードとしてのプレイアウト総勝率


class Entry_node:
  def __init__(self, node):
    self.node = node

  def make_child(self):
    self.node.child_move = Play().legal_list(self.node.board, self.node.color)
    self.node.child_num = self.node.child_move.shape[0]
    for i in range(self.node.child_num):
      tsugiban = copy.deepcopy(self.node.board)
      y, x = self.node.child_move[i]
      Play().set_turn(tsugiban, self.node.color, y, x)
      aite_color = 3 - self.node.color
      self.node.child_node.append(Node(tsugiban, aite_color))  # 子ノードに渡すのは盤面と手番の情報


class Ucb:
  def ucb(self, rate, each_play, total_play):  # rate:平均勝率, each_play:各プレイ回数 ucb値を返す
    return rate + math.sqrt(math.log(total_play) / each_play)

  def traial(self, node):  # ucb試行
    for child in range(node.child_node):
      Cpu.ucb_playout(child)

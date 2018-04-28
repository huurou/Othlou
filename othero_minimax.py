# -*- coding: utf-8 -*-
"""othero_minimax.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16duMI0PvWC06YyP78UFgQa8m-Gm5ot37
"""

import numpy as np
import copy
import math

PLAYOUT_NUM = 100  # プレイアウト試行回数
DEPTH = 1  # 探索深さ


class Ban:
  def init_ban(self):  # 初期盤面
    first_board = np.array([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                            [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                            [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                            [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                            [-1, 0, 0, 0, 2, 1, 0, 0, 0,-1],
                            [-1, 0, 0, 0, 1, 2, 0, 0, 0,-1],
                            [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                            [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                            [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
    return first_board

  def stone(self, ishi):  # 1,2に対応した石を表示
    if ishi == 0:
      return '・'
    elif ishi == 1:
      return '●'
    else:
      return '○'

  def draw_ban(self, board):  # 盤面描写
    print('  a  b  c  d  e  f  g  h ')
    print('  ------------------------')
    for i in range(1, 9):
      print('%d|' % i, end='')
      for j in range(1, 9):
        print(self.stone(board[i][j]), end=' ')
      print('|')
    print('  ------------------------')


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

  def exist_legal_move(self, board, color):  # 合法手の有る無しを1 or 0で返す
    for i in range(1, 9):
      for j in range(1, 9):
        if self.is_legal(board, color, i, j):
          return 1
    return 0

  def set_turn(self, board, color, y, x):  # 渡された盤面に石を打ち石をひっくり返す
    for d in range(-1, 2):
      for e in range(-1, 2):
        if d == 0 and e == 0:
          continue
        count = self.count_turn_over(board, color, y, x, d, e)
        for i in range(1, count + 1):
          board[y + i * d][x + i * e] = color
    board[y][x] = color


class Playout:
  def randmove(self, board, color):  # 合法手の中からランダムな手を選んで返す
    l = Play().legal_list(board, color)
    if l.shape[0] == 1:
      y, x = l[0]
    else:
      y, x = l[int(np.random.randint(l.shape[0] - 1, size=1))]
    return y, x

  def playout(self, board, color):  # 終局までランダムな手を打ち黒勝ちか引き分けなら[[1,0]]を
    tsugiban = copy.deepcopy(board)  # 白勝ちなら[[0,1]]を返す
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

  def playout_result(self, board, color, y, x):  # PLAY_OUT回数分プレイアウトした勝率を返す
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


class Node:  # 探索木のノード
  def __init__(self, board, color):
    self.board = copy.deepcopy(board)  # ノードの盤面
    self.color = color  # ノードの手番
    self.child_node = []  # 子ノードをリスト形式で格納
    self.child_num = 0  # 子ノードの数
    self.child_eval = []  # 各子ノードの指し手の評価値
    self.child_move = None  # 各子ノードの指し手のリスト（np.array[[]]）
    self.child_color = None  # 子ノードの手番


class Entry_node:  # ノードに情報を追加
  def __init__(self, node):
    self.node = node

  def make_child(self):  # 子ノードを展開
    self.node.child_move = Play().legal_list(self.node.board, self.node.color)
    self.node.child_num = self.node.child_move.shape[0]
    for i in range(self.node.child_num):
      tsugiban = copy.deepcopy(self.node.board)
      y, x = self.node.child_move[i]
      Play().set_turn(tsugiban, self.node.color, y, x)
      aite_color = 3 - self.node.color
      self.node.child_color = aite_color
      self.node.child_node.append(Node(tsugiban, aite_color))  # 子ノードに渡すのは盤面と手番の情報

  def node_eval(self):  # 与ノードの合法手すべてをプレイアウトしその勝率を評価値に変換
    eval = []            # 黒番ならその中で最大のものを白番ならそのなかで最小のものを返す
    l = Play().legal_list(self.node.board, self.node.color)
    print(self.node.color)
    if self.node.color == 1:
      for i in range(l.shape[0]):
        y, x = l[i]
        tsugiban = copy.deepcopy(self.node.board)
        r = Playout().playout_result(tsugiban, self.node.color, y, x)
        eval.append(int(600 * (math.log(r) - math.log(1 - r))))
      print(eval)
      return max(eval)
    elif self.node.color == 2:
      for i in range(l.shape[0]):
        y, x = l[i]
        tsugiban = copy.deepcopy(self.node.board)
        r = Playout().playout_result(tsugiban, self.node.color, y, x)
        eval.append(int(-600 * (math.log(r) - math.log(1 - r))))
      print(eval)
      return min(eval)


class Mmcpu:
  def mini_max(self, node, depth):  # ミニマックス法
    if depth == 0:
      return Entry_node(node).node_eval()
    Entry_node(node).make_child()
    if node.child_color == 1:
      maximum = float('-inf')
      for i in node.child_node:
        score = self.mini_max(i, depth - 1)
        if score > maximum:
          maximum = score
        print('max = {0}'.format(maximum))
      return maximum
    elif node.child_color == 2:
      minimum = float('inf')
      for i in node.child_node:
        score = self.mini_max(i, depth - 1)
        if score < minimum:
          minimum = score
          print('min = {0}'.format(minimum))
      return minimum

  def decide_move(self, board, color):
    tsugiban = copy.deepcopy(board)
    node = Node(tsugiban, color)
    Entry_node(node).make_child()
    for i in range(node.child_num):
      node.child_eval.append(self.mini_max(node.child_node[i], DEPTH))
    print(np.reshape(node.child_move, (1, -1)))
    print(node.child_eval)
    if color == 1:
      y, x = node.child_move[node.child_eval.index(max(node.child_eval))]
      return y, x
    elif color == 2:
      y, x = node.child_move[node.child_eval.index(min(node.child_eval))]
      return y, x


class Othello:
  def cvc_mm_game(self):
    ban = Ban().init_ban()
    color = 1
    p = {1: '●', 2: '○'}
    for count in range(2):  # 1,2手目はランダム
      Ban().draw_ban(ban)
      y, x = Playout().randmove(ban, color)
      Play().set_turn(ban, color, y, x)
      color = 3 - color
    while True:
      Ban().draw_ban(ban)
      if not Play().exist_legal_move(ban, color):
        print('打つ手が無いのでパスします')
        color = (3 - color)
        if not Play().exist_legal_move(ban, color):
          print('打つ手が無いのでパスします')
          break
      if color == 1:
        print('Player {0[1]}'.format(p))
      elif color == 2:
        print('Player {0[2]}'.format(p))
      y, x = Mmcpu().decide_move(ban, color)
      Play().set_turn(ban, color, y, x)
      color = int(3 - color)
    c_1 = np.sum(ban == 1)
    c_2 = np.sum(ban == 2)
    print('ゲーム終了')
    print('Player 1 %d' % c_1)
    print('Player 2 %d' % c_2)
    if c_1 > c_2:
      print('勝者:Player 1')
    elif c_1 < c_2:
      print('勝者:Player 2')
    elif c_1 == c_2:
      print('引き分け')


if __name__ == '__main__':
  Othello().cvc_mm_game()
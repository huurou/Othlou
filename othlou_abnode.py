import numpy as np
import copy
import concurrent.futures

PLAYOUT_NUM = 1000  # プレイアウト試行回数
DEPTH = 3  # 探索深さ


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

  def playout_result(self,node):# PLAYOUT_NUM回プレイアウトした先手の勝ち数からPLAYOUT回数/2を引いたものを返す
      num = PLAYOUT_NUM
      board = node.board
      color = node.color
      result = np.empty((0, 2), int)
      for _ in range(num):
          result = np.append(result, self.playout(board, color), axis=0)
      return (np.sum(result[:, 0] == 1) - num/2) * 20000 / num  # -10000～+10000



class Node:  # 探索木のノード
  def __init__(self, board, color):
    self.board = copy.deepcopy(board)  # ノードの盤面
    self.color = color  # ノードの手番
    self.child_node = []  # 子ノード格納リスト
    self.child_num = 0  # 子ノードの数
    self.child_move = None  # 子ノードの指し手


class Makenode:
    def make_child(self, node):  # 子ノードを展開
        node.child_move = Play().legal_list(node.board, node.color)  # 合法手から子ノードの指し手決定
        node.child_num = node.child_move.shape[0]
        for i in range(node.child_num):  # 指し手によって得られる盤面と相手の手番情報を持った子ノードを作成しchild_nodeに格納
            tsugiban = copy.deepcopy(node.board)
            y, x = node.child_move[i]
            Play().set_turn(tsugiban, node.color, y, x)
            aite_color = 3 - node.color
            node.child_node.append(Node(tsugiban, aite_color))


class Cpu:
    def abpruning(self, node, depth, alpha, beta):
        print(depth)
        print(alpha, beta)
        if not Play().exist_legal_move(node.board, node.color) or depth == 0:
            return Playout().playout_result(node)
        Makenode().make_child(node)
        if node.color == 1:
            for child in node.child_node:
                alpha = max(alpha, self.abpruning(child, depth-1, alpha, beta))
                if alpha >= beta:
                    print(f'α = {alpha}, β = {beta}  βカット！')
                    break  # βカット
            return alpha
        elif node.color == 2:
            for child in node.child_node:
                beta = min(beta, self.abpruning(child, depth - 1, alpha, beta))
                if alpha >= beta:
                    print(f'α = {alpha}, β = {beta}  αカット！')
                    break  # αカット
            return beta

    def decide_move(self, board, color):  # 現局面の合法手のうちもっとも評価値の高いものを選ぶ
        node = Node(board, color)
        Makenode().make_child(node)
        list = []
        for child in node.child_node:
            list.append(self.abpruning(child, DEPTH, float('-inf'), float('inf')))
            print(list)
        if color == 1:
            y, x = node.child_move[list.index(max(list))]
            return y, x
        elif color == 2:
            y, x = node.child_move[list.index(min(list))]
            return y, x
class Othello:
  def cvc_ab_game(self):
    ban = Ban().init_ban()
    color = 1
    p = {1: '●', 2: '○'}
    for _ in range(2):  # 1,2手目はランダム
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
      y, x = Cpu().decide_move(ban, color)
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
    Othello().cvc_ab_game()
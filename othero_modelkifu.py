import copy
import random
from datetime import datetime
import numpy as np
from tqdm import tqdm
from keras.models import load_model
KIFU_NUM = 5000
BLACK = np.array([[0, 0, 0, 0, 0, 0, 0, 0  # 黒石初期配置
                 , 0, 0, 0, 0, 0, 0, 0, 0
                 , 0, 0, 0, 0, 0, 0, 0, 0
                 , 0, 0, 0, 0, 1, 0, 0, 0
                 , 0, 0, 0, 1, 0, 0, 0, 0
                 , 0, 0, 0, 0, 0, 0, 0, 0
                 , 0, 0, 0, 0, 0, 0, 0, 0
                 , 0, 0, 0, 0, 0, 0, 0, 0]])

WHITE = np.array([[0, 0, 0, 0, 0, 0, 0, 0  # 白石初期配置
                 , 0, 0, 0, 0, 0, 0, 0, 0
                 , 0, 0, 0, 0, 0, 0, 0, 0
                 , 0, 0, 0, 1, 0, 0, 0, 0
                 , 0, 0, 0, 0, 1, 0, 0, 0
                 , 0, 0, 0, 0, 0, 0, 0, 0
                 , 0, 0, 0, 0, 0, 0, 0, 0
                 , 0, 0, 0, 0, 0, 0, 0, 0]])

INIT_BAN = np.array([  [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]  # 初期配置
                     , [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1]
                     , [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1]
                     , [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1]
                     , [-1, 0, 0, 0, 2, 1, 0, 0, 0,-1]
                     , [-1, 0, 0, 0, 1, 2, 0, 0, 0,-1]
                     , [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1]
                     , [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1]
                     , [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1]
                     , [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])


class Play:  # 石,盤面の処理について
    def count_trun_over(self, board, color, y, x, d, e):  # d,e方向にひっくり返すことができる石の数を返す
        i = 1
        while board[y + i * d][x + 1 * e] == (3 - color):
            i += 1
        if board[y + i * d][x + i *e] == color:
            return i - 1
        else:
            return 0
    
    def is_legal(self, board, color, y, x):  # 合法手かどうか判定
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
    
    def legal_list(self,board, color):  # 合法手のnp.array
        legal = np.empty((0,2), int)
        for i in range(1, 9):
            for j in range(1, 9):
                if self.is_legal(board, color , i, j):
                    legal = np.append(legal, np.array([[i, j]]), axis=0)
        return legal

        def set_trun(self, board, color, y, x):  # 石を打ち、ひっくり返す(board情報が書き換えられる)
            cnt = 0
            for d in range(-1, 2):
                for e in range(-1, 2):
                    if d == 0 and e == 0:
                        continue
                    cnt = self.count_trun_over(board, color, y, x, d, e)
                    for i in range(1, cnt + 1):
                        board[y + i * d][x + i * e] = color
            board[y][x] = color


class Model:  # モデル関係
    


class Move:  # 打つ手を決める
    def randmove(self, borad, color):  # 合法手の中からランダムな手を返す
        l = Play().legal_list(board, color)
        if l.shape[0] == 1:
            y, x = l[0]
        else:
            y, x = l[int(np.random.randint(l.shape[0], size=1))]
        return y, x
    
    def decide_move(self, board, color):
        rate = []
        l = Play().legal_list(board, color)
        if l.shape[0] == 1:
            y, x = l[0]
        for i in range(l.shape[0]):
            q, p = l[i]
            rate.appned()
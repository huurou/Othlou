import copy
import math
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm

KIFU_NUM = 1000
SEARCH_NUM = 200

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


class Move:
    def randmove(self, board, color):
        l = Play().legal_list(board, color)
        if l.shape[0] == 1:
            y, x = l[0]
        else:
            y, x = l[int(np.random.randint(l.shape[0], size=1))]
        return y, x
    
    def playout(self, board, color, q, p):
        tsugiban = copy.deepcopy(board)
        Play().set_turn(tsugiban, color, q, p)
        color = 3 - color
        while True:
            if not Play().exist_legal_move(tsugiban, color):
                color = 3 - color
                if not Play().exist_legal_move(tsugiban, color):
                    c_1 = np.sum(tsugiban == 1)
                    c_2 = np.sum(tsugiban == 2)
                    if c_1 >= c_2:
                        return 1
                    else:
                        return 0
            y, x = self.randmove(tsugiban, color)
            Play().set_turn(tsugiban, color, y, x)
            color = 3 - color

    
    def mcmove(self, board, color):
        l = Play().legal_list(board, color)
        m = l.shape[0]
        if m == 1:
            y, x = l[0]
        w = [0] * m
        for j in range(m):
            q, p = l[j]
            w[j] += Move().playout(board, color, q, p)
        v = [1] * m
        u = [0] * m
        for n in range(m):
                    u[n] = w[n]/v[n] + math.sqrt(math.log(m)/v[n])
        if color == 1:
            for i in range(m + 1, SEARCH_NUM + 1):
                s = u.index(max(u))
                q, p = l[s]
                w[s] += Move().playout(board, color, q, p)
                v[s] += 1
                for n in range(m):
                    u[n] = w[n]/v[n] + math.sqrt(math.log(i)/v[n])
            y, x = l[u.index(max(u))]
        if color == 2:
            for i in range(m + 1, SEARCH_NUM + 1):
                s = u.index(max(u))
                q, p = l[s]
                w[s] += (1 - Move().playout(board, color, q, p))
                v[s] += 1
                for n in range(m):
                    u[n] = w[n]/v[n] + math.sqrt(math.log(i)/v[n])
            y, x = l[u.index(max(u))]
        return y, x


class Othello:
    def cvc_kifu(self, count):
        ban = Ban().init_ban()
        color = 1
        kifulist = np.empty((0,130))
        while True:
            if not Play().exist_legal_move(ban, color):
                color = 3- color
                if not Play().exist_legal_move(ban, color):
                    break
            y, x = Move().mcmove(ban, color)
            move = np.array([[(y - 1) * 8 + x - 1]])
            teban = np.array([[color - 1]])
            reban = ban[1:9, 1:9]
            b = np.zeros((8, 8))
            w = np.zeros((8, 8))
            b[reban == 1] = 1
            w[reban == 2] = 1
            black = np.reshape(b, (1, 64))
            white = np.reshape(w, (1, 64))
            kifu = np.concatenate([black, white, teban, move], axis=1)
            kifulist = np.append(kifulist, kifu, axis=0)
            Play().set_turn(ban, color, y, x)
            color = 3- color
        np.save('./kifu/{0}-{1}'.format(datetime.now().strftime('%y_%m_%d_%H_%M_%S'), count), kifulist)


if __name__ == '__main__':
    for count in tqdm(range(1, KIFU_NUM + 1)):
        Othello().cvc_kifu(count)

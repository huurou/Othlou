import copy
import random
from datetime import datetime
import numpy as np
from tqdm import tqdm
KIFU_NUM = 10000
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


class Othello:
    def cvc_kifu(self, count):
        ban = Ban().init_ban()
        color = 1
        kifulist = np.empty((0, 193))
        while True:
            if not Play().exist_legal_move(ban, color):
                color = 3 - color
                if not Play().exist_legal_move(ban, color):
                    break
            y, x = Move().randmove(ban, color)
            teban = np.full((1, 1),color -1)
            m = (y - 1) * 8 + x - 1
            move = np.eye(64)[m].reshape(1, 64)
            reban = ban[1:9, 1:9]
            b = np.zeros((8, 8))
            w = np.zeros((8, 8))
            b[reban == 1] = 1
            w[reban == 2] = 1
            b_ban = np.reshape(b, (1, 64))
            w_ban = np.reshape(w, (1, 64))
            kifu = np.concatenate([b_ban, w_ban, teban, move], axis=1)
            kifulist = np.append(kifulist, kifu, axis=0)
            Play().set_turn(ban, color, y, x)
            color = 3 - color
        b_num = np.sum(ban == 1)
        w_num = np.sum(ban == 2)
        row_num = kifulist.shape[0]
        if b_num >= w_num:
            r = np.zeros((row_num, 1))
            kifulist = np.append(kifulist, r, axis=1)
        elif b_num < w_num:
            r = np.ones((row_num, 1))
            kifulist = np.append(kifulist, r, axis=1)
        kifulist = kifulist.astype('int')
        np.save('./kifu/{0}-{1}'.format(datetime.now().strftime('%y_%m_%d_%H_%M_%S'), count), kifulist)


if __name__ == '__main__':
    for count in tqdm(range(1, KIFU_NUM + 1)):
        Othello().cvc_kifu(count)
import copy
import random
from datetime import datetime
import numpy as np
from tqdm import tqdm
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, load_model
KIFU_NUM = 5000
model = load_model('othlo.h5')
model.summary()
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

class Play:  # 石,盤面の処理について
    def count_turn_over(self, board, color, y, x, d, e):  # d,e方向にひっくり返すことができる石の数を返す
        i = 1
        while board[y + i * d][x + i * e] == (3 - color):
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

    def exist_legal_move(self, board, color):
        for i in range(1, 9):
            for j in range(1, 9):
                if self.is_legal(board, color, i, j):
                    return 1
        return 0

    def set_turn(self, board, color, y, x):  # 石を打ち、ひっくり返す(board情報が書き換えられる)
        cnt = 0
        for d in range(-1, 2):
            for e in range(-1, 2):
                if d == 0 and e == 0:
                        continue
                cnt = self.count_turn_over(board, color, y, x, d, e)
                for i in range(1, cnt + 1):
                        board[y + i * d][x + i * e] = color
        board[y][x] = color
    
    def ban_convert(self, board):  # boardから(1,128)のbanを返す
        reban = board[1:9, 1:9]
        b = np.zeros((8, 8))
        w = np.zeros((8, 8))
        b[reban == 1] = 1
        w[reban == 2] = 1
        black = np.reshape(b, (1, 64))
        white = np.reshape(w, (1, 64))
        return np.c_[black, white]



class Move:  # 打つ手を決める
    def randmove(self, borad, color):  # 合法手の中からランダムな手を返す
        l = Play().legal_list(board, color)
        if l.shape[0] == 1:
            y, x = l[0]
        else:
            y, x = l[int(np.random.randint(l.shape[0], size=1))]
        return y, x
    
    def decide_move(self, board, color):
        banlist = np.empty((0,128))
        l = Play().legal_list(board, color)
        if l.shape[0] == 1:
            y, x = l[0]
        for i in range(l.shape[0]):
            q, p = l[i]
            tsugiban = copy.deepcopy(board)
            Play().set_turn(tsugiban, color, q, p)
            ban = Play().ban_convert(tsugiban)
            banlist = np.append(banlist, ban, axis=0)
        pred = model.predict(banlist, batch_size=1)
        if color == 1:
            y, x = l[np.argmax(pred[:, 1], axis=0)]
        elif color == 2:
            y, x = l[np.argmax(pred[:, 0], axis=0)]
        return y, x


class Othllo:
    def cvc_kifu(self, count):
        ban = Ban().init_ban()
        black = BLACK
        white = WHITE
        color = 1
        while True:
            if not Play().exist_legal_move(ban, color):
                color = 3 - color
                if not Play().exist_legal_move(ban, color):
                    break
            tsugiban = copy.deepcopy(ban)
            y, x = Move().decide_move(tsugiban, color)
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
        Othllo().cvc_kifu(count)
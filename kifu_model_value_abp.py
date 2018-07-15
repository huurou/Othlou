import concurrent.futures
import copy
import os.path
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Activation, Concatenate, Conv2D, Dense, Dropout,
                          Flatten, Input, concatenate)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

model = load_model('model_value.h5')
KIFU_NUM = 100
DEPTH = 3


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


class Node:
    def __init__(self, board, color):
        self.board = copy.deepcopy(board)
        self.color = color
        self.child_node = []
        self.child_num = 0
        self.child_move = None 


class Makenode:
    def make_child(self, node):
        node.child_move = Play().legal_list(node.board,
                                             node.color)
        node.child_num  = node.child_move.shape[0]
        for i in range(node.child_num):
            tugiban = copy.deepcopy(node.board)
            y, x = node.child_move[i]
            Play().set_turn(tugiban, node.color, y, x)
            aite_color = 3 - node.color
            node.child_node.append(Node(tugiban, aite_color))


class Model:
    def pred(self, board, color):
        reban = board[1:9, 1:9]
        b = np.zeros((8, 8))
        w = np.zeros((8, 8))
        b[reban == 1] = 1
        w[reban == 2] = 1
        ban = np.stack([b, w], axis=0).reshape(1,2,8,8)
        teban = np.array([[color - 1]])
        return model.predict([ban, teban], batch_size=1)

class Move:
    def randmove(self, board, color):
        l = Play().legal_list(board, color)
        if l.shape[0] == 1:
            y, x = l[0]
        else:
            y, x = l[int(np.random.randint(l.shape[0], size=1))]
        return y, x
    
    def abpruning(self, node, depth, alpha, beta):
        if not Play().exist_legal_move(node.board, node.color) or depth == 0:
            return 1.0 - Model().pred(node.board, node.color)
        Makenode().make_child(node)
        if node.color == 1:
            for child in node.child_node:
                alpha = max(alpha, self.abpruning(child,
                                                  depth-1,
                                                  alpha,
                                                  beta))
                if alpha >= beta:
                    break  # βカット
            return alpha
        elif node.color == 2:
            for child in node.child_node:
                beta = min(beta, self.abpruning(child,
                                                depth-1,
                                                alpha,
                                                beta))
                if alpha >= beta:
                    break  # αカット
            return beta
    
    def decide_move(self, board, color):
        node = Node(board, color)
        Makenode().make_child(node)
        list = []
        for child in node.child_node:
            list.append(self.abpruning(child, DEPTH,
                                       float('-inf'),
                                       float('inf')))
        if color == 1:
            y, x = node.child_move[list.index(max(list))]
            return y, x
        elif color == 2:
            y, x = node.child_move[list.index(min(list))]
            return y, x

class Othello:
    def cvc_kifu(self, count):
        ban = Ban().init_ban()
        color = 1
        kifulist = np.empty((0, 129))
        pbar = tqdm(total=60)
        while True:
            if not Play().exist_legal_move(ban, color):
                color = 3 - color
                if not Play().exist_legal_move(ban, color):
                    break
            y, x = Move().decide_move(ban, color)
            teban = np.array([[color - 1]])
            reban = ban[1:9, 1:9]
            b = np.zeros((8, 8))
            w = np.zeros((8, 8))
            b[reban == 1] = 1
            w[reban == 2] = 1
            b_ban = np.reshape(b, (1, 64))
            w_ban = np.reshape(w, (1, 64))
            kifu = np.concatenate([b_ban, w_ban, teban], axis=1)
            kifulist = np.append(kifulist, kifu, axis=0)
            Play().set_turn(ban, color, y, x)
            color = 3 - color
            pbar.update(1)
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

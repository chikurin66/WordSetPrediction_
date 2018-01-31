#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn import metrics
# import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, \
    Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import sys
import argparse
import chainer.serializers as chaSerial
import copy
# from chainer import cuda

# np = cuda.cupy

import numpy as np

import json

import random


class RSChain(Chain):
    def __init__(self, inDim, hDim, outDim):
        super(RSChain, self).__init__(
            l1 = L.Linear(inDim, hDim),
            l2 = L.Linear(hDim, outDim),
        )
        self.inDim = inDim
        self.outDim = outDim

    def __call__(self, x_idx, y_idx):
        # idxのリストからベクトルにする
        # x_vec = self.idx2vec(x_idx, self.inDim)
        # y_vec = self.idx2vec(y_idx, self.outDim)
        x_vec, y_vec = x_idx, y_idx
        fv = self.fwd(x_vec)
        # mean squared errorでええのか？
        loss = F.mean_squared_error(fv, y_vec)
        return loss

    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2

    def idx2vec(self, idx_set, dim):
        vec = np.zeros(dim, dtype=np.int32)
        vec[list(idx_set)] = 1
        return vec


def generate_fileline(f):
    for l in f:
        yield l

def load_data(inputDataFile, outputDataFile, input_w2i, output_w2i):
    sys.stdout.write("data loading ... ")
    X, Y = [], []
    i = 0
    print(inputDataFile)
    with open(inputDataFile, 'r') as f, open(outputDataFile, 'r') as g:
        # 二つのファイルを同時に回したいから，generateする
        f_gen = generate_fileline(f)
        g_gen = generate_fileline(g)
        while True:
            try:
                # 強引だが、try and except
                # yieldで終わりの見極め方がわからないから
                input_line = f_gen.next()
                output_line = g_gen.next()
                in_idx_list = [input_w2i[w] if w in input_w2i else input_w2i["<unk>"] for w in input_line.strip().split(" ")]
                out_idx_list = [output_w2i[w] if w in output_w2i else output_w2i["<unk>"] for w in output_line.strip().split(" ")]

                X.append(in_idx_list)
                Y.append(out_idx_list)
            except:
                break

    sys.stdout.write("loading ends\n")
    return X, Y


def load_model(modelFile):
    chaSerial.load_npz(modelFile, model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-H',
        '--hidden-dim',
        dest='hDim',
        default=512,
        type=int,
        help='dimensions of all hidden layers [int] default=512')
    parser.add_argument(
        '-E',
        '--epoch',
        dest='epoch',
        default=13,
        type=int,
        help='number of epoch [int] default=13')
    parser.add_argument(
        '-B',
        '--batch-size',
        dest='batch_size',
        default=128,
        type=int,
        help='mini batch size [int] default=128')
    parser.add_argument(
        '--input-train-file',
        dest='inTrainFile',
        default='',
        help='filename of input-side data')
    parser.add_argument(
        '--output-train-file',
        dest='outTrainFile',
        default='',
        help='filename of output-side data')
    parser.add_argument(
        '--input-varid-file',
        dest='inVaridFile',
        default='',
        help='filename of input-side data')
    parser.add_argument(
        '--output-varid-file',
        dest='outVaridFile',
        default='',
        help='filename of output-side data')
    parser.add_argument(
        '--wsp-model-name',
        dest='modelName',
        default='',
        help='filename of output model')
    parser.add_argument(
        '--input-dict-file',
        dest='inputDictFile',
        default='',
        help='filename of word to index dictionary')
    parser.add_argument(
        '--output-dict-file',
        dest='outputDictFile',
        default='',
        help='filename of word to index dictionary')
    parser.add_argument(
        '--n-copy-word',
        dest='nCopy',
        default='',
        type=int,
        help='filename of word to index dictionary')

    
    ### start from here ###
    args = parser.parse_args()
    # 辞書データの読み込み
    input_w2i = json.load(open(args.inputDictFile, 'r'))
    output_w2i = json.load(open(args.outputDictFile, 'r'))
    # model setting 
    model = RSChain(len(input_w2i), args.hDim, len(output_w2i))
    # cuda.get_device(0).use # use gpu
    # model.to_gpu()         # use gpu
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    # データをすべて読み込む
    # Xの形式はidxのset
    # X_train_idx, Y_train_idx = load_data(args.inTrainFile, args.outTrainFile, input_w2i, output_w2i)
    X_varid_idx, Y_varid_idx = load_data(args.inVaridFile, args.outVaridFile, input_w2i, output_w2i)

    # n = len(Y_train_idx)
    # bs = int(args.batch_size)
    # X_train = np.zeros((len(X_train_idx), len(input_w2i)), dtype=np.float32)
    # for i, idx_list in enumerate(X_train_idx):
    #     X_train[i, idx_list] = 1
    # Y_train = np.zeros((len(Y_train_idx), len(output_w2i)), dtype=np.float32)
    # for i, idx_list in enumerate(Y_train_idx):
    #     Y_train[i, idx_list] = 1
    X_varid = np.zeros((len(X_varid_idx), len(input_w2i)), dtype=np.float32)
    for i, idx_list in enumerate(X_varid_idx):
        X_varid[i, idx_list] = 1
    Y_varid = np.zeros((len(Y_varid_idx), len(output_w2i)), dtype=np.float32)
    for i, idx_list in enumerate(Y_varid_idx):
        Y_varid[i, idx_list] = 1
    # sys.stdout.write("length of train data: {}\n".format(len(Y_train_idx)))
    sys.stdout.write("length of varid data: {}\n".format(len(Y_varid_idx)))
    sys.stdout.write("dimension of input vector: {}\n".format(len(X_varid[0])))
    sys.stdout.write("dimension of output vector: {}\n".format(len(Y_varid[0])))
    sys.stdout.write("dimension of input w2i dict: {}\n".format(len(input_w2i)))
    sys.stdout.write("dimension of output w2i dict: {}\n".format(len(output_w2i)))
    
    Y_varid_rav = np.ravel(Y_varid)
    chaSerial.load_npz(args.modelName, model)
    step = 10
    for th in [float(t)/step for t in range(step)]:
        pred_vec = model.fwd(Variable(X_varid)).data
        hyp_vec = pred_vec.copy()
        for i in range(len(hyp_vec)):
            hyp_vec[i, pred_vec[i] >= th] = 1.0
            hyp_vec[i, pred_vec[i] < th] = 0.0
        # precision, recall, fscore, tmp = precision_recall_fscore_support(np.ravel(Y_varid), np.ravel(hyp_vec))
        hyp_vec_rav = np.ravel(hyp_vec)
        precision = metrics.precision_score(Y_varid_rav, hyp_vec_rav)
        recall = metrics.recall_score(Y_varid_rav, hyp_vec_rav)
        fscore = metrics.f1_score(Y_varid_rav, hyp_vec_rav)
        print("orig {}\t{}\t{}\t{}".format(th, precision, recall, fscore))
        
        # 正解にあって正答していないインデックスをdiffとして、そこからランダムにとってきて１にする 
        seq = np.array(list(range(len(Y_varid_rav))))
        diff = [i for i, r_h in enumerate(zip(Y_varid_rav, hyp_vec_rav)) if r_h[0] == 1 and r_h[1] == 0]
        nCopy = min(len(diff), args.nCopy * len(Y_varid_idx))
        hyp_vec_rav[random.sample(diff, nCopy)] = 1.0
      
        precision = metrics.precision_score(Y_varid_rav, hyp_vec_rav)
        recall = metrics.recall_score(Y_varid_rav, hyp_vec_rav)
        fscore = metrics.f1_score(Y_varid_rav, hyp_vec_rav)
        print("new  {}\t{}\t{}\t{}".format(th, precision, recall, fscore))


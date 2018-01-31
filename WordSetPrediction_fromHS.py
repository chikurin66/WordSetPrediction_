#!/usr/bin/python
# -*- coding: utf-8 -*-

# this code is programmed based on ~/src/RemainLength/RemainLength_emb.py

from sklearn import datasets
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


class RSChain(Chain):
    def __init__(self):
        super(RSChain, self).__init__(
            l1 = L.Linear(1024, args.hDim),
            l2 = L.Linear(args.hDim, 27605),
        )

    def __call__(self, x, y):
        fv = self.fwd(x)
        loss = F.mean_squared_error(fv, y)
        return loss

    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2

def generate_fileline(f):
    for l in f:
        yield l

def load_data(DataFile, trainNum, testNum):
    dataNum = trainNum + testNum
    sys.stdout.write("data loading ... ")
    X_train, Y_train, X_test, Y_test = [], [], [], []
    i = 0
    j = 0
    STEP = 5 # step個とばしてデータを取得
    for i, line in enumerate(open(DataFile, 'r')):
        if (i/5) >= dataNum:
            break
        if i % STEP != 0:
            continue

        in_vec_line, length, out_idx = line.strip().split('|')

        in_vec = np.array(in_vec_line.split(','), dtype=np.float32)
        out_idx = [int(idx) for idx in out_idx.split(',')]
        out_vec = np.zeros(27605, dtype=np.float32)
        out_vec[out_idx] = 1.0

        if (i/5) % (trainNum / testNum) == 0: # テストデータとする
            X_test.append(in_vec)
            Y_test.append(out_vec)
        else:
            X_train.append(in_vec)
            Y_train.append(out_vec)

    sys.stdout.write("loading ends\n")
    sys.stdout.write("# of train: {}, # of test: {}\n".format(len(Y_train), len(Y_test)))
    return X_train, Y_train, X_test, Y_test


def load_model(modelFile):
    chaSerial.load_npz(modelFile, model)
    return model

def invertDict(d):
    inv_d = {}
    for k, v in d.items():
        inv_d[v] = k
    return inv_d

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
        '--data-file',
        dest='DataFile',
        default='',
        help='filename of data')
    parser.add_argument(
        '--output-model',
        dest='outModel',
        default='',
        help='filename of output model')
    parser.add_argument(
        '--show',
        dest='show',
        default='',
        help='filename of output model')
    parser.add_argument(
        '--continue-epoch',
        dest='contEP',
        default='',
        help='filename of output model')
    parser.add_argument(
        '--output-dict',
        dest='outputDictFile',
        default='',
        help='filename of output model')

    
    ### start from here ###
    args = parser.parse_args()

    # 辞書データの読み込み
    output_w2i = json.load(open(args.outputDictFile, 'r'))
    output_i2w = invertDict(output_w2i)


    # model setting 
    model = RSChain()
    # cuda.get_device(0).use # use gpu
    # model.to_gpu()         # use gpu
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    # データをすべて読み込む
    X_train, Y_train, X_test, Y_test = load_data(args.DataFile, 200000, 2000)
    # X_train, Y_train, X_test, Y_test = load_data(args.DataFile, 2000, 5)
    sys.stdout.write("len of input data: {}\n".format(len(X_train)))
    sys.stdout.write("len of input vector: {}\n".format(len(X_train[0])))
    sys.stdout.write("len of output vector: {}\n".format(len(Y_train[0])))
    n = len(Y_train)
    bs = int(args.batch_size)
    
    X_train_np = np.array(X_train)
    Y_train_np = np.array(Y_train)
    xt = Variable(np.array(X_test).astype(np.float32))
    loss_data = "None"
    test_loss = "None"
    first_ep = 0
    if not int(args.show) and args.contEP != 0:
        chaSerial.load_npz(args.outModel + ".epoch" + str(args.contEP), model)
        first_ep = int(args.contEP)

    if int(args.show):
        chaSerial.load_npz(args.outModel + ".epoch" + str(args.contEP), model)
        sent_id = list(range(5))
        pred = model.fwd(xt).data
        print("th = 0.15")
        for idx in sent_id:
            print("sent id {}".format(idx))
            word_list1, word_list2 = [], []
            for i, elm in enumerate(Y_test[idx]):
                if elm == 1:
                    word_list1.append(output_i2w[i])
            for i, elm in enumerate(pred[idx]):
                if elm >= 0.15:
                    word_list2.append(output_i2w[i])
            print("{}".format(" ".join([w.encode('utf-8') for w in word_list1])))
            print("{}".format(" ".join([w.encode('utf-8') for w in word_list2])))
        print("th = 0.1")
        for idx in sent_id:
            print("sent id {}".format(idx))
            word_list1, word_list2 = [], []
            for i, elm in enumerate(Y_test[idx]):
                if elm == 1:
                    word_list1.append(output_i2w[i])
            for i, elm in enumerate(pred[idx]):
                if elm >= 0.1:
                    word_list2.append(output_i2w[i])
            print("{}".format(" ".join([w.encode('utf-8') for w in word_list1])))
            print("{}".format(" ".join([w.encode('utf-8') for w in word_list2])))
        print("th = 0.05")
        for idx in sent_id:
            print("sent id {}".format(idx))
            word_list1, word_list2 = [], []
            for i, elm in enumerate(Y_test[idx]):
                if elm == 1:
                    word_list1.append(output_i2w[i])
            for i, elm in enumerate(pred[idx]):
                if elm >= 0.05:
                    word_list2.append(output_i2w[i])
            print("{}".format(" ".join([w.encode('utf-8') for w in word_list1])))
            print("{}".format(" ".join([w.encode('utf-8') for w in word_list2])))
    
    else:

        for epoch in range(first_ep, args.epoch):
            epoch = epoch + 1
            sys.stdout.write("Epoch: {} | train_loss: {} | test_loss: {}\n".format(epoch, loss_data, test_loss))
            sffindx = np.random.permutation(n)
            
            for i in range(0, n, bs):
                x = Variable(X_train_np[sffindx[i:(i+bs) if (i+bs) < n else n]].astype(np.float32))
                y = Variable(Y_train_np[sffindx[i:(i+bs) if (i+bs) < n else n]].astype(np.float32))

                model.zerograds()
                loss = model(x, y)
                loss.backward()
                optimizer.update()
                loss_data = loss.data
                # sys.stderr.write("Epoch: {}| i: {}| loss: {}\n".format(epoch, i, loss_data))
            test_loss = model(xt, Variable(np.array(Y_test).astype(np.float32))).data
            yt = model.fwd(xt[:min(50, len(xt)),:])
            ans = yt.data
            for j in range(min(50, len(xt))):
                sys.stderr.write("{} | {}\n".format(yt[j][0].data, Y_test[j][0]))
            
            # save the model
            if epoch % 5 == 0:
                sys.stdout.write("saving Model\n")
                modelName = args.outModel + ".epoch" + str(epoch)
                chaSerial.save_npz(modelName, copy.deepcopy(model).to_cpu(), compression=True)
     
        yt = model.fwd(xt)
        ans = yt.data
        nrow, ncol = ans.shape
        ok = 0
        sys.stdout.write("predict | correct\n")
        for i in range(nrow):
            sys.stdout.write("{} | {}\n".format(yt[i][0].data, Y_test[i][0]))
        sys.stdout.write("test loss: {}\n".format(model(xt, Variable(np.array(Y_test).astype(np.float32))).data))



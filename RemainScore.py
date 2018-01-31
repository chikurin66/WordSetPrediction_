#!/usr/bin/python
# -*- coding: utf-8 -*-

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


class RSChain(Chain):
    def __init__(self):
        super(RSChain, self).__init__(
            l1 = L.Linear(1024, args.hDim),
            l2 = L.Linear(args.hDim, 1),
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

def load_data(inputDataFile, outputDataFile, dataNum):
    sys.stdout.write("data loading ... ")
    X_train, Y_train, X_test, Y_test = [], [], [], []
    i = 0
    with open(inputDataFile, 'r') as f, open(outputDataFile, 'r') as g:
        f_gen = generate_fileline(f)
        g_gen = generate_fileline(g)
        while True:
            vec_line = f_gen.next()
            score_line = g_gen.next()

            vec = np.array(vec_line.strip().split(','))
            score = [float(score_line.strip())]
            if i % 100 == 0: # 1%はテストデータとする
                X_test.append(vec)
                Y_test.append(score)
            else:
                X_train.append(vec)
                Y_train.append(score)
            i += 1
            if i >= dataNum:
                break
    sys.stdout.write("loading ends\n")
    sys.stdout.write("# of train: {}, # of test: {}\n".format(len(Y_train), len(Y_test)))
    return X_train, Y_train, X_test, Y_test


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
        '--input-data-file',
        dest='inDataFile',
        default='',
        help='filename of input-side data')
    parser.add_argument(
        '--output-data-file',
        dest='outDataFile',
        default='',
        help='filename of output-side data')
    parser.add_argument(
        '--output-model',
        dest='outModel',
        default='',
        help='filename of output model')

    
    ### start from here ###
    args = parser.parse_args()
    # model setting 
    model = RSChain()
    # cuda.get_device(0).use # use gpu
    # model.to_gpu()         # use gpu
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    # データをすべて読み込む
    X_train, Y_train, X_test, Y_test = load_data(args.inDataFile, args.outDataFile, 500000)
    sys.stdout.write("len of input vector: {}\n".format(len(X_train[0])))
    sys.stdout.write("len of output vector: {}\n".format(len(Y_train[0])))
    n = len(Y_train)
    bs = int(args.batch_size)
    
    X_train_np = np.array(X_train)
    Y_train_np = np.array(Y_train)
    xt = Variable(np.array(X_test).astype(np.float32))
    loss_data = "None"
    test_loss = "None"
    for epoch in range(args.epoch):
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
        if (epoch+1) % 5 == 0:
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


    model_1 = load_model(args.outModel)
    yt_1 = model_1.fwd(xt)
    ans_1 = yt_1.data
    nrow, ncol = ans_1.shape
    ok_1 = 0
    sys.stdout.write("model_1\n")
    for i in range(nrow):
        sys.stdout.write("{} | {}\n".format(yt_1[i][0].data, Y_test[i][0]))


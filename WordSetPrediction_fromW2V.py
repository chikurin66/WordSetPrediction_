#!/usr/bin/python
# -*- coding: utf-8 -*-

# this code is programmed based on ~/src/WordSetPrediction_fromHS.py

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
from gensim.models import word2vec

sprint = sys.stdout.write


class RSChain(Chain):
    def __init__(self):
        super(RSChain, self).__init__(
            l1 = L.Linear(512, args.hDim),
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

def load_data(w2v, output_w2i, args):
    sys.stdout.write("data loading ")
    X_train, Y_train = [], []
    #Y_train = np.empty((0, 27605), dtype=np.float32)

    i = 0
    for in_line, out_line in zip(open(args.inDataFile, 'r'), open(args.outDataFile, 'r')):
        in_sentence = in_line.strip().split(" ")
        sent_vector = np.zeros(512)
        for word in in_sentence:
            word_u = word.decode('utf-8')
            if word_u not in w2v:
                if word_u != ' ':
                    #sprint("word {} is not in w2v model\n".format(word))
                    pass
                continue
            sent_vector += w2v[word_u]
        X_train.append(sent_vector)
        
        out_sentence = out_line.strip().split(" ")
        out_idx = []
        for word in out_sentence:
            if word not in output_w2i:
                #sprint("word {} is not in w2i dict\n".format(word))
                continue
            out_idx.append(output_w2i[word]) 
        #sprint("{}\n".format(out_sentence))
        #sprint("{}\n".format(out_idx))
        out_vec = np.zeros(27605)
        out_vec[out_idx] = 1.0
        if i%10000 == 0:
            sprint(".")
        i+=1
        #Y_train = np.append(Y_train, np.array([out_vec]), axis=0)
        Y_train.append(out_vec)

    sprint("loading ends\n")
    sprint("# of data: {}\n".format(len(Y_train)))
    X_train = np.array(X_train)
    #Y_train = np.array(Y_train)
    return X_train, Y_train


def load_model(modelFile):
    chaSerial.load_npz(modelFile, model)
    return model

def invertDict(d):
    inv_d = {}
    for k, v in d.items():
        inv_d[v] = k
    return inv_d


def split_train_test(X, Y, n_test=10):
    shuffle = np.random.permutation(len(X))
    X_test = X[shuffle[:n_test]]
    X_train = X[shuffle[n_test:]]
    Y_train, Y_test = [], []
    for i, idx in enumerate(shuffle):
        if i < n_test:
            Y_test.append(Y[idx])
        else:
            Y_train.append(Y[idx])

    Y_test = np.array(Y_test)
    return X_train, Y_train, X_test, Y_test


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
        '--in-data',
        dest='inDataFile',
        default='',
        help='filename of data')
    parser.add_argument(
        '--out-data',
        dest='outDataFile',
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
    parser.add_argument(
        '--w2v-model',
        dest='w2v_model',
        default='',
        help='w2v model name')

    
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

    # w2v
    w2v = word2vec.Word2Vec.load(args.w2v_model)

    # データをすべて読み込む
    X_train, Y_train = load_data(w2v, output_w2i, args)
    X_train, Y_train, X_test, Y_test = split_train_test(X_train, Y_train, n_test=100)
    sys.stdout.write("len of input data: {}\n".format(len(X_train)))
    sys.stdout.write("len of input vector: {}\n".format(len(X_train[0])))
    sys.stdout.write("len of output vector0: {}\n".format(len(Y_train[0])))
    n = len(Y_train)
    bs = int(args.batch_size)
    
    xt = Variable(np.array(X_test).astype(np.float32))
    loss_data = "None"
    test_loss = "None"
    first_ep = 0
    if int(args.show) == 1 and int(args.contEP) != 0:
        chaSerial.load_npz(args.outModel + ".epoch" + str(args.contEP), model)
        first_ep = int(args.contEP)

    if int(args.show):
        chaSerial.load_npz(args.outModel + ".epoch" + str(args.contEP), model)
        sent_id = list(range(5))
        pred = model.fwd(xt).data
        for th in [0.2, 0.15, 0.1, 0.05, 0.01]:
            print("")
            print("####th = {}####".format(th))
            for idx in sent_id:
                print("sent id {}".format(idx))
                word_list1, word_list2 = [], []
                for i, elm in enumerate(Y_test[idx]):
                    if elm == 1:
                        word_list1.append(output_i2w[i])
                for i, elm in enumerate(pred[idx]):
                    if elm >= th:
                        word_list2.append(output_i2w[i])
                tp = len(set(word_list1) & set(word_list2))
                fp = len(set(word_list2) - set(word_list1))
                fn = len(set(word_list1) - set(word_list2))
                sprint("  tp {}  fp {}  fn{}  ".format(tp, fp, fn))
                pre = float(tp) / (tp + fp) if (tp + fp) > 0 else -1
                rec = float(tp) / (tp + fn) if (tp + fn) > 0 else -1
                print("recall {}\t precision {}".format(rec, pre))
                print("  [refe]  {}".format(" ".join([w.encode('utf-8') for w in word_list1])))
                print("  [pred]  {}".format(" ".join([w.encode('utf-8') for w in word_list2])))
                print("")
    else:

        for epoch in range(first_ep, args.epoch):
            epoch = epoch + 1
            sys.stdout.write("Epoch: {} | train_loss: {} | test_loss: {}\n".format(epoch, loss_data, test_loss))
            sffindx = np.random.permutation(n)
            
            for i in range(0, n, bs):
                x = Variable(X_train[sffindx[i:(i+bs) if (i+bs) < n else n]].astype(np.float32))
                shuffle_y = sffindx[i:(i+bs) if (i+bs) < n else n]
                y = []
                for idx in shuffle_y:
                    y.append(Y_train[idx])
                y = Variable(np.array(y).astype(np.float32))
                print(len(y[0]))
                if len(x) != len(y):
                    print("######worning#####")
                model.zerograds()
                loss = model(x, y)
                loss.backward()
                optimizer.update()
                loss_data = loss.data
                sys.stderr.write("Epoch: {}| i: {}| loss: {}\n".format(epoch, i, loss_data))

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



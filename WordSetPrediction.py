#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import datasets
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

import numpy as np
import json
import time

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
    max_iter = min(len(open(inputDataFile, 'r').readlines()), len(open(outputDataFile, 'r').readlines()))
    with open(inputDataFile, 'r') as f, open(outputDataFile, 'r') as g:
        # 二つのファイルを同時に回したいから，generateする
        f_gen = generate_fileline(f)
        g_gen = generate_fileline(g)
        counter = 0
        while True:
            input_line = f_gen.next()
            output_line = g_gen.next()
            in_idx_list = [input_w2i[w.decode('utf-8')] if w.decode('utf-8') in input_w2i else input_w2i["<unk>"] for w in input_line.strip().split(" ")]
            out_idx_list = [output_w2i[w.decode('utf-8')] if w.decode('utf-8') in output_w2i else output_w2i["<unk>"] for w in output_line.strip().split(" ")]

            X.append(in_idx_list)
            Y.append(out_idx_list)
            counter += 1
            if max_iter <= counter:
                break

    sys.stdout.write("loading ends\n")
    return X, Y


def load_model(modelFile):
    chaSerial.load_npz(modelFile, model)
    return model

def invertDict(d):
    inv_d = {}
    for k, v in d.items():
        inv_d[v] = k
    return inv_d

def showPredResult(X_vecs, Y_vecs, output_i2w, model, optimizer, args, sent_id):
    modelName = args.outModel + ".epoch" + '50'
    serializers.load_npz(modelName, model)
    Pred_vecs = model.fwd(Variable(X_vecs)).data

    # print(Pred_vecs)
    # print(Y_vecs)
    # print(np.where(Pred_vecs[0] == Pred_vecs[0].max()))
 
    word_list = []
    for i, elm in enumerate(Y_vecs[sent_id]):
        if elm == 1:
            word_list.append(output_i2w[i])
    print("{}".format(" ".join([w.encode('utf-8') for w in word_list])))

    count_dict = {}    
    
    for sent_id in [0,1,2,3]:
        word_list = []
        for i, elm in enumerate(Pred_vecs[sent_id]):
            if elm >= 0.01:
                count_dict[i] = count_dict[i] + 1 if i in count_dict else 1
                word_list.append(output_i2w[i])
        print("")
        print("{}".format(" ".join([output_i2w[idx].encode('utf-8') for idx, num in enumerate(Y_vecs[sent_id]) if num == 1])))
        print("{}".format(" ".join([w.encode('utf-8') for w in word_list])))


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
        '--output-model',
        dest='outModel',
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

    
    ### start from here ###
    args = parser.parse_args()
    # 辞書データの読み込み
    input_w2i = json.load(open(args.inputDictFile, 'r'))
    output_w2i = json.load(open(args.outputDictFile, 'r'))
    
    input_i2w = invertDict(input_w2i)
    output_i2w = invertDict(output_w2i)
    
    # model setting 
    model = RSChain(len(input_w2i), args.hDim, len(output_w2i))
    ##################
    ### おやくそく ###
    #cuda.get_device(0).use()
    #model.to_gpu()
    #xp = cuda.cupy
    ##################
    print("{} -> {}".format(len(input_w2i), len(output_w2i)))
    # cuda.get_device(0).use # use gpu
    # model.to_gpu()         # use gpu
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    
    X_varid_idx, Y_varid_idx = load_data(args.inVaridFile, args.outVaridFile, input_w2i, output_w2i)
    X_varid = np.zeros((len(X_varid_idx), len(input_w2i)), dtype=np.float32)
    for i, idx_list in enumerate(X_varid_idx):
        X_varid[i, idx_list] = 1
    Y_varid = np.zeros((len(Y_varid_idx), len(output_w2i)), dtype=np.float32)
    for i, idx_list in enumerate(Y_varid_idx):
        Y_varid[i, idx_list] = 1
    
    sent_id = 2
    print("{}".format(" ".join([output_i2w[w].encode('utf-8') for w in Y_varid_idx[sent_id]]))) 
    showPredResult(X_varid, Y_varid, output_i2w, model, optimizer, args, sent_id)


    # データをすべて読み込む
    # Xの形式はidxのset
    """
    X_train_idx, Y_train_idx = load_data(args.inTrainFile, args.outTrainFile, input_w2i, output_w2i)
    X_varid_idx, Y_varid_idx = load_data(args.inVaridFile, args.outVaridFile, input_w2i, output_w2i)
    n = len(Y_train_idx)
    bs = int(args.batch_size)
    X_train = np.zeros((len(X_train_idx), len(input_w2i)), dtype=np.float32)
    for i, idx_list in enumerate(X_train_idx):
        X_train[i, idx_list] = 1
    Y_train = np.zeros((len(Y_train_idx), len(output_w2i)), dtype=np.float32)
    for i, idx_list in enumerate(Y_train_idx):
        Y_train[i, idx_list] = 1
    X_varid = np.zeros((len(X_varid_idx), len(input_w2i)), dtype=np.float32)
    for i, idx_list in enumerate(X_varid_idx):
        X_varid[i, idx_list] = 1
    Y_varid = np.zeros((len(Y_varid_idx), len(output_w2i)), dtype=np.float32)
    for i, idx_list in enumerate(Y_varid_idx):
        Y_varid[i, idx_list] = 1
    sys.stdout.write("length of train data: {}\n".format(len(Y_train_idx)))
    sys.stdout.write("length of varid data: {}\n".format(len(Y_varid_idx)))
    sys.stdout.write("dimension of input vector: {}\n".format(len(X_train[0])))
    sys.stdout.write("dimension of output vector: {}\n".format(len(Y_train[0])))
    sys.stdout.write("first sentence of input (train): {}\n".format(X_train_idx[0]))
    sys.stdout.write("first sentence of input (varid): {}\n".format(X_varid_idx[0]))    
    sys.stdout.write("first sentence of output (train): {}\n".format(Y_train_idx[0]))
    sys.stdout.write("first sentence of output (varid): {}\n".format(Y_varid_idx[0]))    

   
    for epoch in range(args.epoch):
        epoch = epoch + 1
        sffindx = np.random.permutation(n)

        for i in range(0, n, bs):
            # ミニバッチ
            x = Variable(X_train[sffindx[i:(i+bs) if (i+bs) < n else n]].astype(np.float32))
            y = Variable(Y_train[sffindx[i:(i+bs) if (i+bs) < n else n]].astype(np.float32))

            model.zerograds()
            loss = model(x, y)
            loss.backward()
            optimizer.update()
            train_loss = loss.data
            sys.stderr.write("Epoch: {}| i: {}| train_loss: {}\n".format(epoch, i, train_loss))
        varid_loss = model(Variable(X_varid), Variable(Y_varid)).data
        sys.stdout.write("Epoch: {} | train_loss: {} | test_loss: {}\n".format(epoch, train_loss, varid_loss))

        # save the model
        if epoch % 5 == 0:
            sys.stdout.write("saving the Model\n")
            modelName = args.outModel + ".epoch" + str(epoch)
            chaSerial.save_npz(modelName, copy.deepcopy(model).to_cpu(), compression=True)
    """

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def difference(reference, predicted, margin=5):
    # refe pred 共にnparrayを想定
    # refeに対してpredが何単語違うかを計算．marginだけ違いを許容
    diff = sum([1 if r==0 and p==1 else 0 for r, p in zip(reference, predicted)])
    if diff <= margin:
        return 0
    else:
        return diff - margin

def difference_set(reference, predicted, margin=5):
    # refe pred 共にsetを想定
    # refeに対してpredが何単語違うかを計算．marginだけ違いを許容
    diff = len(predicted - reference)
    if diff <= margin:
        return 0
    else:
        return diff - margin

def something():
    print ''


if __name__ == '__main__':
    refe = np.array([1,0,0,0,1,0,1,1])
    pred = np.array([0,1,0,0,1,0,0,0])
    print difference(refe, pred, margin=0)
    print difference(refe, pred, margin=1)
    print difference(refe, pred)

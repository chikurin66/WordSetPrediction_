#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

if __name__ == '__main__':
    origin_tp, origin_fp, origin_fn = 0, 0, 0
    new_tp, new_fp, new_fn = 0, 0, 0
    for line in sys.stdin:
        _, typ, _, i, _, pre, _, rec, _, ln, _, tp, _, fp, _, fn = line.strip().split(" ")
        if typ == 'origin':
            origin_tp += int(tp)
            origin_fp += int(fp)
            origin_fn += int(fn)
        if typ == 'new':
            new_tp += int(tp)
            new_fp += int(fp)
            new_fn += int(fn)
    print("   precision  recall  fscore")
    origin_pre = float(origin_tp)/(origin_tp + origin_fp)
    origin_rec = float(origin_tp)/(origin_tp + origin_fn)
    print("origin {} {} {}".format(origin_pre, origin_rec, 2.0 * origin_rec * origin_pre / (origin_rec + origin_pre) ))
    new_pre = float(new_tp)/(new_tp + new_fp)
    new_rec = float(new_tp)/(new_tp + new_fn)
    print("new {} {} {}".format(new_pre, new_rec, 2.0 * new_rec * new_pre / (new_rec + new_pre) ))


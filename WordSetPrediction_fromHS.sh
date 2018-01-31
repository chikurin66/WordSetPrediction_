#!/bin/sh

# this code is programmed based on ~/src/RemainLength/RemainLength_emb.sh

PROJ_DIR=/home/takebayashi/src/RemainScore
DATA_DIR=/media/takebayashi/064AD3034AD2EE85
MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85/WordSetPrediction/model_ja-en_fromHS
YEAR=2010

hDim=512

date


python -u ./WordSetPrediction_fromHS.py \
   --hidden-dim        ${hDim} \
   --data-file         ${DATA_DIR}/ \
   --epoch             100 \
   --batch-size        128 \
   --output-model      ${MEDIA_DIR}/WordSetPred.hDim${hDim}.en-ja.fromHS.model \
   --output-dict       /home/takebayashi/src/WordSetPrediction/w2i_dict.ja.json \
   --show              1 \
   --continue-epoch    100
date

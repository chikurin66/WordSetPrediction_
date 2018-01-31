#!/bin/sh

# 学習 (GPUが無い環境では以下でGPU=-1として実行)
SLAN=ja; TLAN=en; GPU=0;  EP=13 ;  \

DATA_DIR=/home/takebayashi/src/corpus/IWSLT
DATA_NAME=iwslt.tok

PROJ_DIR=/home/takebayashi/src/WordSetPrediction
MODEL=${PROJ_DIR}/model/${DATA_NAME}.model

OUT_DIR=/media/takebayashi/064AD3034AD2EE85/WordSetPrediction/model
date

python -u ${PROJ_DIR}/LSTMEncDecAttn_saveVocabDict.py -V2 \
   -T                      train \
   --gpu-enc               ${GPU} \
   --gpu-dec               ${GPU} \
   --enc-vocab-file        ${DATA_DIR}/${DATA_NAME}.${SLAN}.vocab_t3_tab \
   --dec-vocab-file        ${DATA_DIR}/${DATA_NAME}.${TLAN}.vocab_t3_tab \
   --enc-data-file         ${DATA_DIR}/${DATA_NAME}.${SLAN} \
   --dec-data-file         ${DATA_DIR}/${DATA_NAME}.${TLAN} \
   --enc-devel-data-file   ${DATA_DIR}/dev.${DATA_NAME}.${SLAN} \
   --dec-devel-data-file   ${DATA_DIR}/dev.${DATA_NAME}.${TLAN} \
   -D                          512 \
   -H                          512 \
   -N                          2 \
   --optimizer                 SGD \
   --lrate                     1.0 \
   --batch-size                32 \
   --out-each                  1 \
   --epoch                     ${EP} \
   --eval-accuracy             0 \
   --dropout-rate              0.3 \
   --attention-mode            1 \
   --gradient-clipping         5 \
   --initializer-scale         0.1 \
   --initializer-type          uniform \
   --merge-encoder-fwbw        0 \
   --use-encoder-bos-eos       0 \
   --use-decoder-inputfeed     1 \
   --length-normalized \
   -O                          ${OUT_DIR}/${DATA_NAME}.nlz.test.model 


date

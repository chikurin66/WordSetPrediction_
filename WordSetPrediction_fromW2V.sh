#!/bin/sh

# this code is programmed based on ~/src/RemainLength/RemainLength_emb.sh

SLAN=ja
TLAN=en
CORPUS_DIR=/home/takebayashi/src/corpus/IWSLT
MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85
PROJ_DIR=/home/takebayashi/src/WordSetPrediction
YEAR=2010

hDim=512

date


python -u ./WordSetPrediction_fromW2V.py \
   --hidden-dim        ${hDim} \
   --w2v-model         ${MEDIA_DIR}/w2v_model/iwslt/iwslt.tok.${SLAN}.model \
   --in-data           ${CORPUS_DIR}/iwslt.tok.${SLAN} \
   --out-data          ${CORPUS_DIR}/iwslt.tok.${TLAN} \
   --epoch             100 \
   --batch-size        64 \
   --output-model      ${MEDIA_DIR}/WordSetPrediction/model_fromW2V_${SLAN}-${TLAN}/WordSetPred.hDim${hDim}.${SLAN}-${TLAN}.fromW2V.model \
   --output-dict       ${PROJ_DIR}/w2i_dict.${TLAN}.json \
   --show              1 \
   --continue-epoch    80
date

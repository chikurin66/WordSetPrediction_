#!/bin/sh

PROJ_DIR=/home/takebayashi/src/WordSetPrediction
MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85/WordSetPrediction
DATA_DIR=/home/takebayashi/src/corpus/IWSLT
YEAR=2010

date

ep=50
n_cp=5
python -u ./searchThreshold.py \
   --hidden-dim         512 \
   --input-varid-file   ${DATA_DIR}/dev.iwslt.tok.ja \
   --output-varid-file  ${DATA_DIR}/dev.iwslt.tok.en \
   --input-dict-file    ${PROJ_DIR}/input_w2i_dict.json \
   --output-dict-file   ${PROJ_DIR}/output_w2i_dict.json \
   --wsp-model-name     ${MEDIA_DIR}/model/WordSetPred.model.epoch${ep} \
   --n-copy-word        ${n_cp}
date

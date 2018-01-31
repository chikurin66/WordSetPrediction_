#!/bin/sh

PROJ_DIR=/home/takebayashi/src/WordSetPrediction
MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85/WordSetPrediction
DATA_DIR=/home/takebayashi/src/corpus/IWSLT
YEAR=2010

date

SLAN=en
TLAN=ja

SW='.noSW'
BS=64
hDim=512
python -u ./WordSetPrediction.py \
   --hidden-dim         ${hDim} \
   --input-train-file   ${DATA_DIR}/iwslt.tok${SW}.${SLAN} \
   --output-train-file  ${DATA_DIR}/iwslt.tok${SW}.${TLAN} \
   --input-varid-file   ${DATA_DIR}/dev.iwslt.tok${SW}.${SLAN} \
   --output-varid-file  ${DATA_DIR}/dev.iwslt.tok${SW}.${TLAN} \
   --epoch              100 \
   --batch-size         ${BS} \
   --input-dict-file    ${PROJ_DIR}/w2i_dict.${SLAN}.json \
   --output-dict-file   ${PROJ_DIR}/w2i_dict.${TLAN}.json \
   --output-model       ${MEDIA_DIR}/model_${SLAN}-${TLAN}/WordSetPred.hDim${hDim}.${SLAN}-${TLAN}.bs${BS}${SW}.model
#   --output-model       /home/takebayashi/kuzu.model
date

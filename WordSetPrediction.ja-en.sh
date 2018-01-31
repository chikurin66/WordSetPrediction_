#!/bin/sh

PROJ_DIR=/home/takebayashi/src/WordSetPrediction
MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85/WordSetPrediction
DATA_DIR=/home/takebayashi/src/corpus/IWSLT
YEAR=2010

date
SLAN=ja
TLAN=en
echo ${SLAN} "->" ${TLAN}
hDim=512
python -u ./WordSetPrediction.py \
   --hidden-dim         ${hDim} \
   --input-train-file   ${DATA_DIR}/iwslt.tok.$SLAN \
   --output-train-file  ${DATA_DIR}/iwslt.tok.$TLAN \
   --input-varid-file   ${DATA_DIR}/dev.iwslt.tok.$SLAN \
   --output-varid-file  ${DATA_DIR}/dev.iwslt.tok.$TLAN \
   --epoch              100 \
   --batch-size         128 \
   --input-dict-file    ${PROJ_DIR}/w2i_dict.${SLAN}.json \
   --output-dict-file   ${PROJ_DIR}/w2i_dict.${TLAN}.json \
   --output-model       ${MEDIA_DIR}/model_${SLAN}-${TLAN}/WordSetPred.hDim${hDim}.${SLAN}-${TLAN}.model
date

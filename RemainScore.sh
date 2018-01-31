#!/bin/sh

PROJ_DIR=/home/takebayashi/src/RemainScore
MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85/RemainScore
YEAR=2010

date


python -u ./RemainScore.py \
   --hidden-dim        512 \
   --input-data-file   ${MEDIA_DIR}/data_h_1.txt \
   --output-data-file  ${MEDIA_DIR}/data_s_1.txt \
   --epoch             100 \
   --batch-size        128 \
   --output-model      ${MEDIA_DIR}/model/Rscore.model
date

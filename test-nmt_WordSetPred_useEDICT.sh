#!/bin/sh

# 評価
GPU=0; BEAM=5 ;  \
RED_EP=50
WEIGHT="
0.1
"
MARGIN=0
EP=13
DATA_DIR=/home/takebayashi/src/corpus/IWSLT
DATA_NAME=iwslt.tok

PROJ_DIR=/home/takebayashi/src/WordSetPrediction
MODEL=${PROJ_DIR}/model/${DATA_NAME}.model

LSTM_MODEL_DIR=/media/takebayashi/064AD3034AD2EE85/LSTMEncDecAttn_model

YEAR=2010

date

SLAN=ja
TLAN=en
T_DIRECT=ja-en

MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85/WordSetPrediction/model_useEDICT
OUT_DATA=${MEDIA_DIR}/iwslt.tok.nlz.model.epoch${EP}.decode_MAX${MAXLEN}_BEAM${BEAM}.WordSetPrediction.useEDICT


for red_w in ${WEIGHT}
do
ep=${RED_EP}
echo red_w ${red_w}
python -u ./LSTMEncDecAttn_WordSetPred_useEDICT.py \
   -T                  test \
   --gpu-enc           ${GPU} \
   --gpu-dec           ${GPU} \
   --enc-data-file     ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${SLAN} \
   --dec-data-file     ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${TLAN} \
   --init-model        ${LSTM_MODEL_DIR}/${T_DIRECT}/iwslt.tok.nlz.${T_DIRECT}.model.epoch13 \
   --setting           ${LSTM_MODEL_DIR}/${T_DIRECT}/iwslt.tok.nlz.${T_DIRECT}.model.setting \
   --beam-size         ${BEAM} \
   --max-length        150 \
   --red-weight        ${red_w} \
   --target-len        ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${TLAN}.len \
   --wsp-hidden-dim    512 \
   --trans-dir         ${T_DIRECT} \
   --j2e-dict          /home/takebayashi/src/corpus/EDICT/j2e_dict.json \
   > ${OUT_DATA}.red_w${red_w}.${T_DIRECT}.junk.txt
done
date


WEIGHT="
0.01
0.05
0.15
0.1
0.2
0.3
0.5
0.7
1.0
1.5
2.0
5.0
10.0
"

echo "Evaluation starts"

for red_w in ${WEIGHT}
do
ep=${RED_EP}
echo -n red_w ${red_w} " "
perl ~/src/Postprocess/multi-bleu.perl ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${TLAN} \
   < ${OUT_DATA}.red_w${red_w}.${T_DIRECT}.txt
done
date

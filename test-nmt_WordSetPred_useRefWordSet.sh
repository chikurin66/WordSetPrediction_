#!/bin/sh

# 評価
SLAN=ja
TLAN=en
GPU=0; BEAM=5 ;  \
RED_EP=50
'''
WEIGHT="
0.0
0.05
0.1
0.2
0.5
1.0
2.0
5.0
10.0
20.0
50.0
"
'''
WEIGHT="
0.0
"
#0.3
#0.5
#5.0
#10.0
#"
MARGIN="
0
"
EP=13

DATA_DIR=/home/takebayashi/src/corpus/IWSLT
DATA_NAME=iwslt.tok

PROJ_DIR=/home/takebayashi/src/WordSetPrediction
MODEL=${PROJ_DIR}/model/${DATA_NAME}.model

LSTM_MODEL_DIR=/media/takebayashi/064AD3034AD2EE85/LSTMEncDecAttn_model
MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85/WordSetPrediction

OUT_DATA=${MEDIA_DIR}/iwslt.tok.nlz.model.epoch${EP}.decode_MAX${MAXLEN}_BEAM${BEAM}.WordSetPrediction.useRefWordSet

YEAR=2010

date
#head -n 100 ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${SLAN} > ${DATA_DIR}/test${YEAR}.${DATA_NAME}.h100.${SLAN}

for red_w in ${WEIGHT}
do
for margin in ${MARGIN}
do
ep=${RED_EP}
python -u ./LSTMEncDecAttn_WordSetPred_useRefWordSet.py \
   -T                  test \
   --gpu-enc           ${GPU} \
   --gpu-dec           ${GPU} \
   --enc-data-file     ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${SLAN} \
   --dec-data-file     ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${TLAN} \
   --init-model        ${LSTM_MODEL_DIR}/iwslt.tok.nlz.model.epoch13 \
   --setting           ${LSTM_MODEL_DIR}/iwslt.tok.nlz.model.setting \
   --beam-size         ${BEAM} \
   --max-length        150 \
   --red-model         ${MEDIA_DIR}/model/WordSetPred.model.epoch${RED_EP} \
   --red-weight        ${red_w} \
   --target-len        ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${TLAN}.len \
   --wsp-hidden-dim    512 \
   --diff-margin       ${margin} \
   > ${OUT_DATA}.red_w${red_w}.margin${margin}.txt

done
done
date


echo "Evaluation starts"

for red_w in ${WEIGHT}
do
for margin in ${MARGIN}
do
ep=${RED_EP}
echo -n red_w ${red_w} " "
echo -n margin ${margin} " "
perl ~/src/Postprocess/multi-bleu.perl ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${TLAN} \
    < ${OUT_DATA}.red_w${red_w}.margin${margin}.txt

done
done
date

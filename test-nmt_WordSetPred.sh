#!/bin/sh

# 評価
GPU=0; BEAM=5 ;  \
RED_EP=50
WEIGHT="
0.2
0.5
1.0
2.0
"
MARGIN=0
EP=13
GIVEN_WORD="
10000
"
TH="
0.1
0.05
"
DATA_DIR=/home/takebayashi/src/corpus/IWSLT
DATA_NAME=iwslt.tok

PROJ_DIR=/home/takebayashi/src/WordSetPrediction
MODEL=${PROJ_DIR}/model/${DATA_NAME}.model

LSTM_MODEL_DIR=/media/takebayashi/064AD3034AD2EE85/LSTMEncDecAttn_model

YEAR=2010

date
#head -n 100 ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${SLAN} > ${DATA_DIR}/test${YEAR}.${DATA_NAME}.h100.${SLAN}

margin=${MARGIN}
for red_w in ${WEIGHT}
do
for th in ${TH}
do
for n_given_word in ${GIVEN_WORD}
do
for i in 0 1
do
if [ $i = 0 ]
then
    SLAN=ja
    TLAN=en
    T_DIRECT=ja-en
fi
if [ $i = 1 ]
then
    SLAN=en
    TLAN=ja
    T_DIRECT=en-ja
fi
MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85/WordSetPrediction/model_${T_DIRECT}
OUT_DATA=${MEDIA_DIR}/iwslt.tok.nlz.model.epoch${EP}.decode_MAX${MAXLEN}_BEAM${BEAM}.WordSetPrediction.useFFNN

ep=${RED_EP}
echo -n red_w ${red_w} " "
echo -n nGivenWord ${n_given_word} " "
echo -n th ${th} " "
echo ${T_DIRECT}
python -u ./LSTMEncDecAttn_WordSetPred.py \
   -T                  test \
   --gpu-enc           ${GPU} \
   --gpu-dec           ${GPU} \
   --enc-data-file     ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${SLAN} \
   --dec-data-file     ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${TLAN} \
   --init-model        ${LSTM_MODEL_DIR}/${T_DIRECT}/iwslt.tok.nlz.${T_DIRECT}.model.epoch13 \
   --setting           ${LSTM_MODEL_DIR}/${T_DIRECT}/iwslt.tok.nlz.${T_DIRECT}.model.setting \
   --beam-size         ${BEAM} \
   --max-length        150 \
   --red-model         ${MEDIA_DIR}/WordSetPred.hDim512.${T_DIRECT}.model.epoch${RED_EP} \
   --red-weight        ${red_w} \
   --target-len        ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${TLAN}.len \
   --wsp-hidden-dim    512 \
   --diff-margin       ${margin} \
   --n-given-word      ${n_given_word} \
   --wordset-th        ${th} \
   --trans-dir         ${T_DIRECT} \
   > ${OUT_DATA}.givenWord${n_given_word}.red_w${red_w}.margin${margin}.th${th}.${T_DIRECT}.txt
done
done
done
done
date


echo "Evaluation starts"

for n_given_word in ${GIVEN_WORD}
do
for red_w in ${WEIGHT}
do
for margin in ${MARGIN}
do
for th in ${TH}
do
for i in 0 1
do
if [ $i = 0 ]
then
    SLAN=ja
    TLAN=en
    T_DIRECT=ja-en
fi
if [ $i = 1 ]
then
    SLAN=en
    TLAN=ja
    T_DIRECT=en-ja
fi
MEDIA_DIR=/media/takebayashi/064AD3034AD2EE85/WordSetPrediction/model_${T_DIRECT}
OUT_DATA=${MEDIA_DIR}/iwslt.tok.nlz.model.epoch${EP}.decode_MAX${MAXLEN}_BEAM${BEAM}.WordSetPrediction.useFFNN
ep=${RED_EP}
echo -n red_w ${red_w} " "
echo -n margin ${margin} " "
echo -n nGivenWord ${n_given_word} " "
echo -n th ${th} " "
echo -n ${T_DIRECT} " "
perl ~/src/Postprocess/multi-bleu.perl ${DATA_DIR}/test${YEAR}.${DATA_NAME}.${TLAN} \
    < ${OUT_DATA}.givenWord${n_given_word}.red_w${red_w}.margin${margin}.th${th}.${T_DIRECT}.txt
done
done
done
done
done
date
